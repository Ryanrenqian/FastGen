# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Teacher-Feature Drifting (TFD), aligned with the authors' ImageNet code."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Sequence, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from fastgen.methods import FastGenModel

if TYPE_CHECKING:
    from fastgen.configs.methods.config_tfd import ModelConfig


def _repeat_condition(condition: Any, repeats: int) -> Any:
    if isinstance(condition, torch.Tensor):
        return condition.repeat_interleave(repeats, dim=0)
    if isinstance(condition, dict):
        return {key: _repeat_condition(value, repeats) for key, value in condition.items()}
    if isinstance(condition, tuple):
        return tuple(_repeat_condition(value, repeats) for value in condition)
    if isinstance(condition, list):
        return [_repeat_condition(value, repeats) for value in condition]
    return condition


def _slice_condition(condition: Any, start: int, end: int) -> Any:
    if isinstance(condition, torch.Tensor):
        return condition[start:end]
    if isinstance(condition, dict):
        return {key: _slice_condition(value, start, end) for key, value in condition.items()}
    if isinstance(condition, tuple):
        return tuple(_slice_condition(value, start, end) for value in condition)
    if isinstance(condition, list):
        return [_slice_condition(value, start, end) for value in condition]
    return condition


def _pool_and_flatten_feature(feature: torch.Tensor, pool_size: int) -> torch.Tensor:
    if feature.ndim == 4 and pool_size > 1:
        if feature.shape[-1] % pool_size or feature.shape[-2] % pool_size:
            raise ValueError(f"feature_pool_size={pool_size} must divide {feature.shape[-2:]}")
        feature = F.avg_pool2d(feature, pool_size, pool_size)
    return feature.flatten(start_dim=1)


def resolve_edm_feature_selectors(model: torch.nn.Module, tokens: Sequence[str]) -> list[tuple[str, str]]:
    """Resolve the official zero-based ModuleDict selectors."""
    encoder_keys = list(model.enc.keys())
    decoder_keys = list(model.dec.keys())
    selectors = []
    for token in tokens:
        if token == "bottleneck":
            selector = ("enc", encoder_keys[-1])
        else:
            try:
                branch, index_text = token.split(":", 1)
                keys = encoder_keys if branch == "enc" else decoder_keys if branch == "dec" else None
                if keys is None:
                    raise ValueError
                selector = (branch, keys[int(index_text)])
            except (ValueError, IndexError) as error:
                raise ValueError(f"Invalid EDM feature layer selector: {token!r}") from error
        selectors.append(selector)
    if len(set(selectors)) != len(selectors):
        raise ValueError("feature_layers resolve to duplicate EDM modules")
    return selectors


def teacher_feature_drifting_loss(
    generated: torch.Tensor,
    positive: torch.Tensor,
    radii: Sequence[float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Official mini-batch TFD objective from ``tfd/losses/drifting.py``."""
    if generated.ndim != 3 or positive.ndim != 3:
        raise ValueError("generated and positive features must have shape [B, N, D]")
    if generated.shape[0] != positive.shape[0] or generated.shape[2] != positive.shape[2]:
        raise ValueError("generated and positive feature shapes are incompatible")
    if generated.shape[1] < 2:
        raise ValueError("TFD requires at least two generated samples per condition")
    if not radii or any(radius <= 0 for radius in radii):
        raise ValueError("drift_radii must contain positive values")

    generated = generated.float()
    positive = positive.float()
    frozen_generated = generated.detach()
    targets = torch.cat([frozen_generated, positive], dim=1)
    num_generated = generated.shape[1]
    feature_dim = generated.shape[2]

    with torch.no_grad():
        distance = torch.sqrt(torch.clamp(torch.cdist(frozen_generated, targets).pow(2), min=1e-8))
        scale = distance.mean()
        input_scale = torch.clamp(scale / math.sqrt(float(feature_dim)), min=1e-3)
        generated_scaled = frozen_generated / input_scale
        targets_scaled = targets / input_scale
        normalized_distance = distance / torch.clamp(scale, min=1e-3)
        diagonal = F.pad(
            torch.eye(num_generated, device=generated.device, dtype=generated.dtype).unsqueeze(0),
            (0, positive.shape[1]),
        )
        normalized_distance = normalized_distance + diagonal * 1e6

        force = torch.zeros_like(generated_scaled)
        info = {"scale": scale.detach()}
        for radius in radii:
            logits = -normalized_distance / float(radius)
            affinity = torch.softmax(logits, dim=-1)
            reverse_affinity = torch.softmax(logits, dim=-2)
            affinity = torch.sqrt(torch.clamp(affinity * reverse_affinity, min=1e-6))
            negative_affinity = affinity[:, :, :num_generated]
            positive_affinity = affinity[:, :, num_generated:]
            coefficients = torch.cat(
                [
                    -negative_affinity * positive_affinity.sum(dim=-1, keepdim=True),
                    positive_affinity * negative_affinity.sum(dim=-1, keepdim=True),
                ],
                dim=2,
            )
            radius_force = torch.einsum("biy,byd->bid", coefficients, targets_scaled)
            radius_force -= coefficients.sum(dim=-1)[..., None] * generated_scaled
            force_norm = radius_force.pow(2).mean()
            info[f"loss_{radius}"] = force_norm.detach()
            force += radius_force / torch.sqrt(torch.clamp(force_norm, min=1e-8))
        target = generated_scaled + force

    loss = (generated / input_scale.detach() - target.detach()).pow(2).mean(dim=(-1, -2))
    info["drift_norm"] = force.pow(2).mean().sqrt().detach()
    return loss, info


def anchor_margin_loss(
    generated: torch.Tensor, anchors: torch.Tensor, *, bandwidth: float = 0.0, alpha: float = 0.5
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    distance = torch.cdist(anchors.float(), generated.float())
    kernel_bandwidth = (
        torch.clamp(torch.median(distance.detach()), min=1e-6)
        if bandwidth <= 0
        else distance.new_tensor(float(bandwidth))
    )
    generated_support = torch.exp(-distance / kernel_bandwidth).mean(dim=2)
    if anchors.shape[1] <= 1:
        anchor_support = torch.zeros_like(generated_support)
    else:
        anchor_distance = torch.cdist(anchors.float(), anchors.float())
        diagonal = torch.eye(anchors.shape[1], device=anchors.device, dtype=torch.bool).unsqueeze(0)
        anchor_kernel = torch.exp(-anchor_distance / (2.0 * kernel_bandwidth)).masked_fill(diagonal, 0.0)
        anchor_support = anchor_kernel.sum(dim=2) / float(anchors.shape[1] - 1)
    target_support = float(alpha) * anchor_support
    margin = torch.relu(target_support - generated_support)
    return margin.mean(dim=1), {
        "bandwidth": kernel_bandwidth.detach(),
        "generated_support": generated_support.mean().detach(),
        "covered_fraction": (generated_support >= target_support).float().mean().detach(),
    }


class TFDModel(FastGenModel):
    def build_model(self):
        if self.config.student_sample_steps != 1:
            raise ValueError("TFDModel supports one-step students only")
        if self.config.generated_samples_per_condition < 2:
            raise ValueError("generated_samples_per_condition must be at least 2")
        if not self.config.feature_layers:
            raise ValueError("feature_layers must select at least one teacher layer")
        if self.config.feature_noise_sigma_min <= 0:
            raise ValueError("feature_noise_sigma_min must be positive")
        if self.config.feature_noise_sigma_max < self.config.feature_noise_sigma_min:
            raise ValueError("feature noise sigma bounds are invalid")
        super().build_model()
        self.build_teacher()
        self.load_student_weights_and_ema()
        self.feature_selectors = resolve_edm_feature_selectors(
            self.teacher.model, self.config.feature_layers
        )

    def _sample_group_sigmas(self, batch_size: int, device: torch.device) -> torch.Tensor:
        cfg = self.config
        sigmas = (
            torch.randn(batch_size, device=device) * cfg.feature_noise_p_std + cfg.feature_noise_p_mean
        ).exp()
        valid = (sigmas >= cfg.feature_noise_sigma_min) & (sigmas <= cfg.feature_noise_sigma_max)
        for _ in range(cfg.feature_noise_trunc_resamples):
            if bool(valid.all()):
                break
            retry = (
                torch.randn(int((~valid).sum()), device=device) * cfg.feature_noise_p_std
                + cfg.feature_noise_p_mean
            ).exp()
            sigmas = sigmas.clone()
            sigmas[~valid] = retry
            valid = (sigmas >= cfg.feature_noise_sigma_min) & (sigmas <= cfg.feature_noise_sigma_max)
        return sigmas.clamp(cfg.feature_noise_sigma_min, cfg.feature_noise_sigma_max)

    def _extract_teacher_features(
        self, samples: torch.Tensor, condition: Any, sigmas: torch.Tensor, *, keep_input_grad: bool
    ) -> list[torch.Tensor]:
        noisy = self.teacher.noise_scheduler.forward_process(samples, torch.randn_like(samples), sigmas)

        def teacher_forward(
            noisy_chunk: torch.Tensor, sigma_chunk: torch.Tensor, condition_chunk: Any
        ) -> tuple[torch.Tensor, ...]:
            # The authors call the teacher with force_fp32=True even though the
            # generator recipe uses FP16. Preserve that behavior here.
            with torch.autocast(device_type=samples.device.type, enabled=False):
                return tuple(
                    self.teacher(
                        noisy_chunk.float(),
                        sigma_chunk.float(),
                        condition=(
                            condition_chunk.float()
                            if isinstance(condition_chunk, torch.Tensor)
                            else condition_chunk
                        ),
                        return_features_early=True,
                        feature_selectors=self.feature_selectors,
                    )
                )

        if keep_input_grad:
            if self.config.teacher_generated_checkpoint:
                features = checkpoint(
                    teacher_forward, noisy, sigmas, condition, use_reentrant=False
                )
            else:
                features = teacher_forward(noisy, sigmas, condition)
            return [_pool_and_flatten_feature(feature, self.config.feature_pool_size) for feature in features]

        chunk_size = int(self.config.teacher_reference_chunk_size)
        if chunk_size <= 0:
            chunk_size = samples.shape[0]
        layer_chunks: list[list[torch.Tensor]] | None = None
        with torch.no_grad():
            for start in range(0, samples.shape[0], chunk_size):
                end = min(start + chunk_size, samples.shape[0])
                chunk_features = teacher_forward(
                    noisy[start:end], sigmas[start:end], _slice_condition(condition, start, end)
                )
                pooled = [
                    _pool_and_flatten_feature(feature, self.config.feature_pool_size)
                    for feature in chunk_features
                ]
                if layer_chunks is None:
                    layer_chunks = [[] for _ in pooled]
                for chunks, feature in zip(layer_chunks, pooled):
                    chunks.append(feature)
        if layer_chunks is None:
            raise ValueError("Teacher feature extraction received an empty batch")
        return [torch.cat(chunks, dim=0) for chunks in layer_chunks]

    def _get_outputs(self, generated: torch.Tensor, input_student: torch.Tensor) -> Dict[str, torch.Tensor | Callable]:
        count = self.config.generated_samples_per_condition
        batch_size = generated.shape[0] // count
        return {
            "gen_rand": generated.reshape(batch_size, count, *generated.shape[1:])[:, 0],
            "input_rand": (input_student / self.config.conditioning_sigma).reshape(
                batch_size, count, *input_student.shape[1:]
            )[:, 0],
        }

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        del iteration
        condition = data["condition"]
        positives = data["positive"]
        anchors = data["anchor"]
        batch_size, num_positive = positives.shape[:2]
        num_anchor = anchors.shape[1]
        num_generated = self.config.generated_samples_per_condition

        condition_generated = _repeat_condition(condition, num_generated)
        noise = torch.randn(batch_size * num_generated, *self.input_shape, device=self.device, dtype=positives.dtype)
        input_student = noise * self.config.conditioning_sigma
        t_student = torch.full(
            (batch_size * num_generated,), self.config.conditioning_sigma, device=self.device,
            dtype=self.net.noise_scheduler.t_precision,
        )
        generated = self.gen_data_from_net(input_student, t_student, condition=condition_generated)

        group_sigmas = self._sample_group_sigmas(batch_size, generated.device)
        generated_features = self._extract_teacher_features(
            generated, condition_generated, group_sigmas.repeat_interleave(num_generated), keep_input_grad=True
        )
        positive_features = self._extract_teacher_features(
            positives.flatten(0, 1), _repeat_condition(condition, num_positive),
            group_sigmas.repeat_interleave(num_positive), keep_input_grad=False,
        )
        anchor_features = self._extract_teacher_features(
            anchors.flatten(0, 1), _repeat_condition(condition, num_anchor),
            group_sigmas.repeat_interleave(num_anchor), keep_input_grad=False,
        )

        drift_loss = generated.new_zeros((), dtype=torch.float32)
        anchor_loss = generated.new_zeros((), dtype=torch.float32)
        drift_norm = generated.new_zeros((), dtype=torch.float32)
        support = generated.new_zeros((), dtype=torch.float32)
        bandwidth = generated.new_zeros((), dtype=torch.float32)
        for generated_layer, positive_layer, anchor_layer in zip(
            generated_features, positive_features, anchor_features
        ):
            generated_grouped = generated_layer.reshape(batch_size, num_generated, -1)
            layer_drift, drift_info = teacher_feature_drifting_loss(
                generated_grouped, positive_layer.reshape(batch_size, num_positive, -1), self.config.drift_radii
            )
            layer_anchor, anchor_info = anchor_margin_loss(
                generated_grouped, anchor_layer.reshape(batch_size, num_anchor, -1),
                bandwidth=self.config.anchor_bandwidth, alpha=self.config.anchor_margin,
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
            "feature_sigma_min": group_sigmas.min().detach(),
            "feature_sigma_max": group_sigmas.max().detach(),
        }
        return loss_map, self._get_outputs(generated, input_student)
