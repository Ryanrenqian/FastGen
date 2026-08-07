# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Teacher-Feature Drifting (TFD) for one-step diffusion distillation."""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastgen.methods import FastGenModel

if TYPE_CHECKING:
    from fastgen.configs.methods.config_tfd import ModelConfig


def _repeat_condition(condition: Any, repeats: int) -> Any:
    """Repeat every batch-aligned tensor in a nested condition."""
    if isinstance(condition, torch.Tensor):
        return condition.repeat_interleave(repeats, dim=0)
    if isinstance(condition, dict):
        return {key: _repeat_condition(value, repeats) for key, value in condition.items()}
    if isinstance(condition, tuple):
        return tuple(_repeat_condition(value, repeats) for value in condition)
    if isinstance(condition, list):
        return [_repeat_condition(value, repeats) for value in condition]
    return condition


def _pool_and_flatten_feature(feature: torch.Tensor, pool_size: int) -> torch.Tensor:
    """Apply local average pooling and return one feature vector per sample."""
    if feature.ndim == 3:  # [B, tokens, channels]
        if pool_size > 1 and feature.shape[1] >= pool_size:
            feature = F.avg_pool1d(feature.transpose(1, 2), pool_size, pool_size).transpose(1, 2)
    elif feature.ndim == 4:  # [B, channels, height, width]
        if pool_size > 1 and min(feature.shape[-2:]) >= pool_size:
            feature = F.avg_pool2d(feature, pool_size, pool_size)
    elif feature.ndim == 5:  # [B, channels, frames, height, width]
        if pool_size > 1 and min(feature.shape[-2:]) >= pool_size:
            feature = F.avg_pool3d(feature, (1, pool_size, pool_size), (1, pool_size, pool_size))
    elif feature.ndim < 2:
        raise ValueError(f"Teacher feature must include a batch dimension, got shape {tuple(feature.shape)}")
    return feature.flatten(start_dim=1)


def teacher_feature_drifting_loss(
    generated: torch.Tensor,
    positive: torch.Tensor,
    radii: Sequence[float],
    anchor_weight: float = 1.0,
    anchor_temperature: float = 1.0,
    anchor_margin: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compute mini-batch drifting and anchor-margin losses.

    Args:
        generated: Teacher features with shape ``[B, N_gen, D]``.
        positive: Reference/anchor features with shape ``[B, N_pos, D]``.
    """
    if generated.ndim != 3 or positive.ndim != 3:
        raise ValueError("generated and positive features must have shape [B, N, D]")
    if generated.shape[0] != positive.shape[0] or generated.shape[2] != positive.shape[2]:
        raise ValueError("generated and positive feature shapes are incompatible")
    if generated.shape[1] < 2:
        raise ValueError("TFD requires at least two generated samples per condition for repulsion")
    if not radii or any(radius <= 0 for radius in radii):
        raise ValueError("drift_radii must contain positive values")
    if anchor_temperature <= 0:
        raise ValueError("anchor_temperature must be positive")

    # Kernel geometry is evaluated in FP32. The field is a stop-gradient target,
    # while the regression side retains the gradient through the frozen teacher.
    z = generated.float()
    r = positive.float()
    with torch.no_grad():
        z_fixed, r_fixed = z.detach(), r.detach()
        dist_pos = torch.cdist(z_fixed, r_fixed, p=2)
        dist_neg = torch.cdist(z_fixed, z_fixed, p=2)
        diagonal = torch.eye(z.shape[1], device=z.device, dtype=torch.bool).unsqueeze(0)
        dist_neg = dist_neg.masked_fill(diagonal, torch.inf)

        drift = torch.zeros_like(z_fixed)
        for radius in radii:
            weight_pos = torch.softmax(-dist_pos / radius, dim=-1)
            weight_neg = torch.softmax(-dist_neg / radius, dim=-1)
            mean_pos = torch.bmm(weight_pos, r_fixed)
            mean_neg = torch.bmm(weight_neg, z_fixed)
            drift.add_(mean_pos - mean_neg)
        drift.div_(len(radii))
        drift_target = z_fixed + drift

    drift_loss = F.mse_loss(z, drift_target)

    # Equations (12)-(14): anchors with weak generated support receive a margin.
    anchor_to_generated = torch.cdist(r, z, p=2).square()
    generated_support = torch.exp(-anchor_to_generated / anchor_temperature).mean(dim=-1)
    if r.shape[1] > 1:
        anchor_to_anchor = torch.cdist(r, r, p=2).square()
        eye = torch.eye(r.shape[1], device=r.device, dtype=torch.bool).unsqueeze(0)
        self_support = torch.exp(-anchor_to_anchor / (2.0 * anchor_temperature)).masked_fill(eye, 0.0)
        self_support = self_support.sum(dim=-1) / (r.shape[1] - 1)
        anchor_loss = F.relu(anchor_margin * self_support - generated_support).mean()
    else:
        anchor_loss = z.new_zeros(())

    metrics = {
        "drift_norm": drift.square().mean().sqrt(),
        "generated_support": generated_support.mean().detach(),
    }
    return drift_loss, anchor_weight * anchor_loss, metrics


class TFDModel(FastGenModel):
    """One-step distillation in the frozen diffusion teacher's feature space."""

    def __init__(self, config: ModelConfig):
        self._positive_bank: dict[int, list[torch.Tensor]] = {}
        super().__init__(config)

    def build_model(self):
        if self.config.student_sample_steps != 1:
            raise ValueError("TFDModel currently supports one-step students only")
        if self.config.generated_samples_per_condition < 2:
            raise ValueError("generated_samples_per_condition must be at least 2")
        if self.config.positive_samples_per_condition < 1:
            raise ValueError("positive_samples_per_condition must be positive")
        if not self.config.feature_indices:
            raise ValueError("feature_indices must select at least one teacher layer")
        if len(set(self.config.feature_indices)) != len(self.config.feature_indices):
            raise ValueError("feature_indices must not contain duplicates")
        if self.config.feature_noise_level < 0:
            raise ValueError("feature_noise_level must be non-negative")
        if self.config.feature_pool_size < 1:
            raise ValueError("feature_pool_size must be positive")
        if self.config.anchor_weight < 0 or self.config.anchor_margin < 0:
            raise ValueError("anchor_weight and anchor_margin must be non-negative")
        super().build_model()
        self.build_teacher()
        self.load_student_weights_and_ema()

    def _class_keys(self, condition: Any, batch_size: int) -> list[int] | None:
        if not isinstance(condition, torch.Tensor) or condition.shape[0] != batch_size:
            return None
        if condition.ndim == 1 and not condition.dtype.is_floating_point:
            return [int(value) for value in condition.detach().cpu()]
        if condition.ndim == 2:
            return [int(value) for value in condition.detach().argmax(dim=1).cpu()]
        return None

    def _prepare_positives(self, data: Dict[str, Any], real: torch.Tensor, condition: Any) -> torch.Tensor:
        positives = data.get("positive")
        if positives is not None:
            if positives.ndim == real.ndim:
                positives = positives.unsqueeze(1)
            if positives.ndim != real.ndim + 1 or positives.shape[0] != real.shape[0]:
                raise ValueError("positive must have shape [B, N_pos, ...] or [B, ...]")
            return positives

        count = self.config.positive_samples_per_condition
        keys = self._class_keys(condition, real.shape[0])
        if keys is None or self.config.positive_bank_size <= 0:
            return real.unsqueeze(1).expand(-1, count, *real.shape[1:])

        # A small per-class bank lets the standard class-conditional loader supply
        # multiple references without requiring a specialized grouped sampler.
        for key, sample in zip(keys, real.detach()):
            bank = self._positive_bank.setdefault(key, [])
            bank.append(sample.to(device="cpu", copy=True))
            del bank[: max(0, len(bank) - self.config.positive_bank_size)]

        result = []
        for key in keys:
            bank = self._positive_bank[key]
            selected = [bank[index % len(bank)] for index in range(count)]
            result.append(torch.stack(selected))
        return torch.stack(result).to(device=real.device, dtype=real.dtype)

    def _feature_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        schedule = self.teacher.noise_scheduler
        target = torch.as_tensor(self.config.feature_noise_level, device=device, dtype=schedule.t_precision)
        sigmas = schedule.sigmas.to(device=device, dtype=schedule.t_precision)
        sigma_index = (sigmas - target).abs().argmin().reshape(1)
        timestep = schedule.sigma_idx_to_t(sigma_index).reshape(1)
        return timestep.expand(batch_size)

    def _extract_teacher_features(
        self, samples: torch.Tensor, condition: Any, *, keep_input_grad: bool
    ) -> list[torch.Tensor]:
        t_feature = self._feature_timestep(samples.shape[0], samples.device)
        noisy = self.teacher.noise_scheduler.forward_process(samples, torch.randn_like(samples), t_feature)
        context = torch.enable_grad() if keep_input_grad else torch.no_grad()
        with context:
            features = self.teacher(
                noisy,
                t_feature,
                condition=condition,
                return_features_early=True,
                feature_indices=set(self.config.feature_indices),
            )
        if len(features) != len(self.config.feature_indices):
            raise RuntimeError("Teacher did not return every requested TFD feature layer")
        pooled = [_pool_and_flatten_feature(feature, self.config.feature_pool_size) for feature in features]
        if self.config.normalize_features:
            pooled = [F.normalize(feature.float(), p=2, dim=-1, eps=1e-6) for feature in pooled]
        return pooled

    def _get_outputs(
        self, gen_data: torch.Tensor, input_student: torch.Tensor, condition: Any = None
    ) -> Dict[str, torch.Tensor | Callable]:
        samples_per_condition = self.config.generated_samples_per_condition
        batch_size = gen_data.shape[0] // samples_per_condition
        gen_first = gen_data.reshape(batch_size, samples_per_condition, *gen_data.shape[1:])[:, 0]
        input_first = input_student.reshape(batch_size, samples_per_condition, *input_student.shape[1:])[:, 0]
        noise_scale = self.net.noise_scheduler.max_sigma
        noise_first = input_first / noise_scale if noise_scale > 0 else input_first
        return {"gen_rand": gen_first, "input_rand": noise_first}

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        real = data["real"]
        condition = data["condition"]
        positives = self._prepare_positives(data, real, condition)
        batch_size, num_positive = positives.shape[:2]
        num_generated = self.config.generated_samples_per_condition

        condition_generated = _repeat_condition(condition, num_generated)
        condition_positive = _repeat_condition(condition, num_positive)
        noise = torch.randn(batch_size * num_generated, *self.input_shape, device=self.device, dtype=real.dtype)
        input_student = self.net.noise_scheduler.latents(noise=noise)
        t_student = torch.full(
            (batch_size * num_generated,),
            self.net.noise_scheduler.max_t,
            device=self.device,
            dtype=self.net.noise_scheduler.t_precision,
        )
        generated = self.gen_data_from_net(input_student, t_student, condition=condition_generated)

        generated_features = self._extract_teacher_features(
            generated, condition_generated, keep_input_grad=True
        )
        positive_flat = positives.flatten(0, 1)
        positive_features = self._extract_teacher_features(
            positive_flat, condition_positive, keep_input_grad=False
        )

        drift_loss = generated.new_zeros((), dtype=torch.float32)
        anchor_loss = generated.new_zeros((), dtype=torch.float32)
        drift_norm = generated.new_zeros((), dtype=torch.float32)
        support = generated.new_zeros((), dtype=torch.float32)
        for generated_layer, positive_layer in zip(generated_features, positive_features):
            layer_drift, layer_anchor, metrics = teacher_feature_drifting_loss(
                generated_layer.reshape(batch_size, num_generated, -1),
                positive_layer.reshape(batch_size, num_positive, -1),
                radii=self.config.drift_radii,
                anchor_weight=self.config.anchor_weight,
                anchor_temperature=self.config.anchor_temperature,
                anchor_margin=self.config.anchor_margin,
            )
            drift_loss = drift_loss + layer_drift
            anchor_loss = anchor_loss + layer_anchor
            drift_norm = drift_norm + metrics["drift_norm"]
            support = support + metrics["generated_support"]

        num_layers = len(generated_features)
        drift_loss = drift_loss / num_layers
        anchor_loss = anchor_loss / num_layers
        total_loss = drift_loss + anchor_loss
        loss_map = {
            "total_loss": total_loss,
            "tfd_loss": drift_loss,
            "anchor_loss": anchor_loss,
            "drift_norm": drift_norm / num_layers,
            "generated_support": support / num_layers,
        }
        return loss_map, self._get_outputs(generated, input_student, condition=condition)
