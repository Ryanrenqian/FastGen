# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 TI2V specialization of the official-aligned TFD objective."""

from __future__ import annotations

from typing import Any, Callable, Dict

import torch

from fastgen.methods.distribution_matching.teacher_feature_drifting import (
    TFDModel,
    _pool_and_flatten_feature,
    _repeat_condition,
    anchor_margin_loss,
    teacher_feature_drifting_loss,
)


class WanTFDModel(TFDModel):
    """One-step Wan student with CFG restricted to teacher positive filling."""

    def build_model(self):
        if not self.config.feature_indices:
            raise ValueError("Wan TFD requires Transformer feature_indices")
        if self.config.feature_layers:
            raise ValueError("Wan TFD does not use EDM feature_layers")
        if self.config.teacher_positive_batch_size < 1:
            raise ValueError("teacher_positive_batch_size must be positive")
        super().build_model()

    def _extract_teacher_features(
        self,
        samples: torch.Tensor,
        condition: Any,
        sigmas: torch.Tensor,
        *,
        keep_input_grad: bool,
    ) -> list[torch.Tensor]:
        noisy = self.teacher.noise_scheduler.forward_process(
            samples, torch.randn_like(samples), sigmas
        )
        context = torch.enable_grad() if keep_input_grad else torch.no_grad()
        with context:
            features = self.teacher(
                noisy,
                sigmas,
                condition=condition,
                return_features_early=True,
                feature_indices=set(self.config.feature_indices),
            )
        return [
            _pool_and_flatten_feature(feature, self.config.feature_pool_size)
            for feature in features
        ]

    def _get_outputs(
        self, generated: torch.Tensor, input_student: torch.Tensor
    ) -> Dict[str, torch.Tensor | Callable]:
        count = self.config.generated_samples_per_condition
        batch_size = generated.shape[0] // count
        return {
            "gen_rand": generated.reshape(
                batch_size, count, *generated.shape[1:]
            )[:, 0],
            "input_rand": input_student.reshape(
                batch_size, count, *input_student.shape[1:]
            )[:, 0],
        }

    @torch.no_grad()
    def _sample_teacher_positives(
        self,
        batch_size: int,
        count: int,
        condition: Any,
        neg_condition: Any,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        chunks = []
        for start in range(0, count, self.config.teacher_positive_batch_size):
            chunk_count = min(
                self.config.teacher_positive_batch_size, count - start
            )
            noise = torch.randn(
                batch_size * chunk_count,
                *self.input_shape,
                device=self.device,
                dtype=dtype,
            )
            samples = self.teacher.sample(
                noise,
                condition=_repeat_condition(condition, chunk_count),
                neg_condition=_repeat_condition(neg_condition, chunk_count),
                guidance_scale=self.config.teacher_positive_guidance_scale,
                num_steps=self.config.teacher_positive_sample_steps,
                **self.config.teacher_positive_sample_kwargs,
            )
            chunks.append(
                samples.reshape(
                    batch_size, chunk_count, *samples.shape[1:]
                ).detach()
            )
        return torch.cat(chunks, dim=1)

    def _prepare_positive_sets(
        self,
        real: torch.Tensor,
        condition: Any,
        neg_condition: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positives = real.unsqueeze(1)
        missing = self.config.positive_samples_per_condition - 1
        if missing > 0:
            if not self.config.teacher_positive_fill:
                positives = positives.expand(
                    -1, self.config.positive_samples_per_condition, *real.shape[1:]
                )
            else:
                supplements = self._sample_teacher_positives(
                    real.shape[0], missing, condition, neg_condition, real.dtype
                )
                positives = torch.cat([positives, supplements], dim=1)
        anchors = positives[:, : self.config.anchor_samples_per_condition]
        if anchors.shape[1] < self.config.anchor_samples_per_condition:
            repeats = self.config.anchor_samples_per_condition - anchors.shape[1]
            anchors = torch.cat([anchors, positives[:, :1].expand(-1, repeats, *real.shape[1:])], dim=1)
        return positives, anchors

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        del iteration
        real, condition, neg_condition = self._prepare_training_data(data)
        positives, anchors = self._prepare_positive_sets(
            real, condition, neg_condition
        )
        batch_size, num_positive = positives.shape[:2]
        num_anchor = anchors.shape[1]
        num_generated = self.config.generated_samples_per_condition

        condition_generated = _repeat_condition(condition, num_generated)
        noise = torch.randn(
            batch_size * num_generated,
            *self.input_shape,
            device=self.device,
            dtype=real.dtype,
        )
        input_student = self.net.noise_scheduler.latents(noise=noise)
        t_student = torch.full(
            (batch_size * num_generated,),
            self.net.noise_scheduler.max_t,
            device=self.device,
            dtype=self.net.noise_scheduler.t_precision,
        )
        generated = self.gen_data_from_net(
            input_student, t_student, condition=condition_generated
        )

        group_sigmas = self._sample_group_sigmas(batch_size, generated.device)
        generated_features = self._extract_teacher_features(
            generated,
            condition_generated,
            group_sigmas.repeat_interleave(num_generated),
            keep_input_grad=True,
        )
        positive_features = self._extract_teacher_features(
            positives.flatten(0, 1),
            _repeat_condition(condition, num_positive),
            group_sigmas.repeat_interleave(num_positive),
            keep_input_grad=False,
        )
        anchor_features = self._extract_teacher_features(
            anchors.flatten(0, 1),
            _repeat_condition(condition, num_anchor),
            group_sigmas.repeat_interleave(num_anchor),
            keep_input_grad=False,
        )

        drift_loss = generated.new_zeros((), dtype=torch.float32)
        anchor_loss = generated.new_zeros((), dtype=torch.float32)
        drift_norm = generated.new_zeros((), dtype=torch.float32)
        support = generated.new_zeros((), dtype=torch.float32)
        bandwidth = generated.new_zeros((), dtype=torch.float32)
        for generated_layer, positive_layer, anchor_layer in zip(
            generated_features, positive_features, anchor_features
        ):
            generated_grouped = generated_layer.reshape(
                batch_size, num_generated, -1
            )
            layer_drift, drift_info = teacher_feature_drifting_loss(
                generated_grouped,
                positive_layer.reshape(batch_size, num_positive, -1),
                self.config.drift_radii,
            )
            layer_anchor, anchor_info = anchor_margin_loss(
                generated_grouped,
                anchor_layer.reshape(batch_size, num_anchor, -1),
                bandwidth=self.config.anchor_bandwidth,
                alpha=self.config.anchor_margin,
            )
            drift_loss += layer_drift.mean()
            anchor_loss += self.config.anchor_weight * layer_anchor.mean()
            drift_norm += drift_info["drift_norm"]
            support += anchor_info["generated_support"]
            bandwidth += anchor_info["bandwidth"]

        num_layers = len(generated_features)
        loss_map = {
            "total_loss": drift_loss + anchor_loss,
            "tfd_loss": drift_loss,
            "anchor_loss": anchor_loss,
            "drift_norm": drift_norm / num_layers,
            "generated_support": support / num_layers,
            "anchor_bandwidth": bandwidth / num_layers,
            "feature_sigma_mean": group_sigmas.mean().detach(),
            "real_positive_count": generated.new_tensor(1),
            "teacher_positive_count": generated.new_tensor(num_positive - 1),
        }
        return loss_map, self._get_outputs(generated, input_student)
