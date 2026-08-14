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


def _repeat_first_frame_video(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 5 or video.shape[2] < 1:
        raise ValueError("video must have shape [B, C, T, H, W] with T >= 1")
    return video[:, :, :1].expand(-1, -1, video.shape[2], -1, -1).contiguous()


def _repeat_group_feature_noise(
    feature_noise: torch.Tensor, samples_per_condition: int
) -> torch.Tensor:
    """Reuse one noise realization for every sample of a condition."""
    if samples_per_condition < 1:
        raise ValueError("samples_per_condition must be positive")
    return feature_noise.repeat_interleave(samples_per_condition, dim=0)


def _mask_first_temporal_token_slot(
    feature: torch.Tensor,
    sample_shape: tuple[int, ...],
    patch_size: tuple[int, ...],
) -> torch.Tensor:
    """Drop the first temporal patch while preserving Wan token ordering."""
    if len(sample_shape) != 5 or len(patch_size) != 3:
        raise ValueError(
            "Wan feature masking expects 5D samples and 3D patches; "
            f"got feature={tuple(feature.shape)}, sample={sample_shape}, patch={patch_size}"
        )
    # FastGen's Wan override currently exposes block states as [B, C, T, H, W].
    # Keep token-layout support for upstream diffusers implementations that expose
    # [B, tokens, C] instead.
    if feature.ndim == 5:
        if feature.shape[2] < 2:
            raise ValueError("Cannot mask the only temporal feature slot")
        return feature[:, :, 1:]
    if feature.ndim != 3:
        raise ValueError(
            "Wan feature must be [B, C, T, H, W] or [B, tokens, C]; "
            f"got {tuple(feature.shape)}"
        )
    _, _, frames, height, width = sample_shape
    patch_t, patch_h, patch_w = patch_size
    if frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError("Wan latent shape must be divisible by transformer patch_size")
    token_frames = frames // patch_t
    token_height = height // patch_h
    token_width = width // patch_w
    expected_tokens = token_frames * token_height * token_width
    if feature.shape[1] != expected_tokens:
        raise ValueError(
            f"Wan feature has {feature.shape[1]} tokens; expected {expected_tokens}"
        )
    if token_frames < 2:
        raise ValueError("Cannot mask the only temporal token slot")
    return feature.reshape(
        feature.shape[0], token_frames, token_height, token_width, feature.shape[-1]
    )[:, 1:].reshape(feature.shape[0], -1, feature.shape[-1])


class WanTFDModel(TFDModel):
    """One-step Wan student with CFG restricted to teacher positive filling."""

    def build_model(self):
        if not self.config.feature_indices:
            raise ValueError("Wan TFD requires Transformer feature_indices")
        if self.config.feature_layers:
            raise ValueError("Wan TFD does not use EDM feature_layers")
        if self.config.teacher_positive_batch_size < 1:
            raise ValueError("teacher_positive_batch_size must be positive")
        if self.config.static_negative_guidance_scale < 1.0:
            raise ValueError("static_negative_guidance_scale must be at least 1")
        if self.config.feature_tap not in {
            "block_output", "post_self_attn", "self_attn_delta"
        }:
            raise ValueError(
                "feature_tap must be block_output, post_self_attn, or self_attn_delta"
            )
        if self.config.feature_taps and len(self.config.feature_taps) != len(
            self.config.feature_indices
        ):
            raise ValueError("feature_taps must align with feature_indices")
        if self.config.feature_loss_weights and len(
            self.config.feature_loss_weights
        ) != len(self.config.feature_indices):
            raise ValueError("feature_loss_weights must align with feature_indices")
        if any(weight <= 0 for weight in self.config.feature_loss_weights):
            raise ValueError("feature_loss_weights must be positive")
        super().build_model()

    @torch.no_grad()
    def _build_static_negative(self, data: Dict[str, Any], real: torch.Tensor) -> torch.Tensor:
        if "real_raw" not in data:
            raise ValueError("static negative construction requires pixel-space real_raw data")
        static_video = _repeat_first_frame_video(data["real_raw"])
        static_latent = self.net.vae.encode(static_video, mode="argmax")
        if static_latent.shape != real.shape:
            raise ValueError(
                f"static latent shape {tuple(static_latent.shape)} does not match real latent {tuple(real.shape)}"
            )
        return static_latent.to(device=real.device, dtype=real.dtype)

    def _extract_teacher_features(
        self,
        samples: torch.Tensor,
        condition: Any,
        sigmas: torch.Tensor,
        *,
        keep_input_grad: bool,
        feature_noise: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if feature_noise is None:
            feature_noise = torch.randn_like(samples)
        elif feature_noise.shape != samples.shape:
            raise ValueError(
                "feature_noise must have the same shape as samples; "
                f"got {tuple(feature_noise.shape)} and {tuple(samples.shape)}"
            )
        noisy = self.teacher.noise_scheduler.forward_process(
            samples, feature_noise.to(device=samples.device, dtype=samples.dtype), sigmas
        )
        context = torch.enable_grad() if keep_input_grad else torch.no_grad()
        feature_taps = (
            dict(zip(self.config.feature_indices, self.config.feature_taps))
            if self.config.feature_taps
            else None
        )
        with context:
            features = self.teacher(
                noisy,
                sigmas,
                condition=condition,
                return_features_early=True,
                feature_indices=set(self.config.feature_indices),
                feature_tap=self.config.feature_tap,
                feature_taps=feature_taps,
            )
        if self.config.feature_mask_first_temporal_slot:
            patch_size = tuple(int(value) for value in self.teacher.transformer.config.patch_size)
            features = [
                _mask_first_temporal_token_slot(feature, tuple(samples.shape), patch_size)
                for feature in features
            ]
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
        data: Dict[str, Any],
        real: torch.Tensor,
        condition: Any,
        neg_condition: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positives_raw = data.get("positive_raw")
        if positives_raw is None:
            positives = real.unsqueeze(1)
        else:
            if positives_raw.ndim != 6 or positives_raw.shape[0] != real.shape[0]:
                raise ValueError("positive_raw must have shape [B, N, C, T, H, W]")
            batch_size, count = positives_raw.shape[:2]
            with torch.no_grad():
                positive_latents = self.net.vae.encode(
                    positives_raw.flatten(0, 1), mode="argmax"
                )
            positives = positive_latents.reshape(
                batch_size, count, *positive_latents.shape[1:]
            ).to(device=real.device, dtype=real.dtype)
            # Reuse the trainer-encoded stride-1 latent exactly.
            positives[:, 0] = real
        positives = positives[:, : self.config.positive_samples_per_condition]
        missing = self.config.positive_samples_per_condition - positives.shape[1]
        if missing > 0:
            if not self.config.teacher_positive_fill:
                positives = torch.cat(
                    [positives, positives[:, :1].expand(-1, missing, *real.shape[1:])],
                    dim=1,
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
            data, real, condition, neg_condition
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
        shared_feature_noise = (
            torch.randn_like(real)
            if self.config.feature_shared_noise_per_condition
            else None
        )
        generated_features = self._extract_teacher_features(
            generated,
            condition_generated,
            group_sigmas.repeat_interleave(num_generated),
            keep_input_grad=True,
            feature_noise=(
                None
                if shared_feature_noise is None
                else _repeat_group_feature_noise(
                    shared_feature_noise, num_generated
                )
            ),
        )
        positive_features = self._extract_teacher_features(
            positives.flatten(0, 1),
            _repeat_condition(condition, num_positive),
            group_sigmas.repeat_interleave(num_positive),
            keep_input_grad=False,
            feature_noise=(
                None
                if shared_feature_noise is None
                else _repeat_group_feature_noise(
                    shared_feature_noise, num_positive
                )
            ),
        )
        static_features = None
        static_negative_weight = 0.0
        if self.config.static_negative_enabled:
            static_negative = self._build_static_negative(data, real)
            static_features = self._extract_teacher_features(
                static_negative,
                condition,
                group_sigmas,
                keep_input_grad=False,
                feature_noise=shared_feature_noise,
            )
            static_negative_weight = (
                (self.config.static_negative_guidance_scale - 1.0)
                * (num_generated - 1)
            )
        if num_anchor == num_positive and anchors.data_ptr() == positives.data_ptr():
            anchor_features = positive_features
        else:
            anchor_features = self._extract_teacher_features(
                anchors.flatten(0, 1),
                _repeat_condition(condition, num_anchor),
                group_sigmas.repeat_interleave(num_anchor),
                keep_input_grad=False,
                feature_noise=(
                    None
                    if shared_feature_noise is None
                    else _repeat_group_feature_noise(
                        shared_feature_noise, num_anchor
                    )
                ),
            )

        drift_loss = generated.new_zeros((), dtype=torch.float32)
        anchor_loss = generated.new_zeros((), dtype=torch.float32)
        drift_norm = generated.new_zeros((), dtype=torch.float32)
        support = generated.new_zeros((), dtype=torch.float32)
        bandwidth = generated.new_zeros((), dtype=torch.float32)
        layer_weights = self.config.feature_loss_weights or [1.0] * len(
            generated_features
        )
        for layer_index, (generated_layer, positive_layer, anchor_layer) in enumerate(
            zip(generated_features, positive_features, anchor_features)
        ):
            generated_grouped = generated_layer.reshape(
                batch_size, num_generated, -1
            )
            layer_drift, drift_info = teacher_feature_drifting_loss(
                generated_grouped,
                positive_layer.reshape(batch_size, num_positive, -1),
                self.config.drift_radii,
                fixed_negative=(
                    None
                    if static_features is None
                    else static_features[layer_index].reshape(batch_size, 1, -1)
                ),
                negative_weight=static_negative_weight if static_features is not None else None,
            )
            layer_anchor, anchor_info = anchor_margin_loss(
                generated_grouped,
                anchor_layer.reshape(batch_size, num_anchor, -1),
                bandwidth=self.config.anchor_bandwidth,
                alpha=self.config.anchor_margin,
            )
            layer_weight = layer_weights[layer_index]
            drift_loss += layer_weight * layer_drift.mean()
            anchor_loss += (
                layer_weight * self.config.anchor_weight * layer_anchor.mean()
            )
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
            "real_positive_count": generated.new_tensor(
                min(data.get("positive_raw", real.unsqueeze(1)).shape[1], num_positive)
            ),
            "teacher_positive_count": generated.new_tensor(
                max(0, num_positive - data.get("positive_raw", real.unsqueeze(1)).shape[1])
                if self.config.teacher_positive_fill
                else 0
            ),
            "static_negative_weight": generated.new_tensor(static_negative_weight),
        }
        return loss_map, self._get_outputs(generated, input_student)
