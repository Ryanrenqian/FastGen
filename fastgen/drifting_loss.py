# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-independent drifting-field loss utilities."""

from __future__ import annotations

import math
from typing import Dict, Sequence

import torch


def _pairwise_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute batched Euclidean distances without materializing differences."""
    squared = (
        x.square().sum(dim=-1, keepdim=True)
        + y.square().sum(dim=-1).unsqueeze(-2)
        - 2 * torch.bmm(x, y.transpose(1, 2))
    )
    return squared.clamp_min(1e-8).sqrt()


def drifting_loss(
    generated: torch.Tensor,
    positive: torch.Tensor,
    *,
    negative: torch.Tensor | None = None,
    negative_weight: float | torch.Tensor = 1.0,
    group_weight: torch.Tensor | None = None,
    radii: Sequence[float] = (0.02, 0.05),
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Construct a normalized drifting field independently per token group.

    Args:
        generated: Trainable candidates with shape ``[G, N, D]``.
        positive: Fixed data samples with shape ``[G, P, D]``.
        negative: Optional fixed negatives with shape ``[G, Q, D]``.
        negative_weight: Scalar or ``[G, Q]`` importance for fixed negatives.
        radii: Kernel temperatures in normalized distance units.
    """
    if generated.ndim != 3 or positive.ndim != 3:
        raise ValueError("generated and positive must have shape [G, samples, features]")
    if generated.shape[0] != positive.shape[0] or generated.shape[2] != positive.shape[2]:
        raise ValueError("generated and positive group/feature dimensions must match")
    if generated.shape[1] < 2:
        raise ValueError("drifting requires at least two generated candidates")
    if positive.shape[1] < 1:
        raise ValueError("drifting requires at least one positive sample")
    if not radii or any(radius <= 0 for radius in radii):
        raise ValueError("radii must contain positive values")

    groups, num_generated, feature_dim = generated.shape
    generated_f = generated.float()
    old_generated = generated_f.detach()
    positive_f = positive.detach().float()
    if negative is None:
        negative_f = old_generated[:, :0]
    else:
        if negative.ndim != 3 or negative.shape[0] != groups or negative.shape[2] != feature_dim:
            raise ValueError("negative must have shape [G, Q, D] matching generated")
        negative_f = negative.detach().float()

    num_negative = negative_f.shape[1]
    generated_weights = generated_f.new_ones(groups, num_generated)
    positive_weights = generated_f.new_ones(groups, positive_f.shape[1])
    if isinstance(negative_weight, torch.Tensor):
        negative_weights = negative_weight.to(device=generated.device, dtype=torch.float32)
        if negative_weights.ndim == 0:
            negative_weights = negative_weights.expand(groups, num_negative)
        elif negative_weights.shape != (groups, num_negative):
            raise ValueError("negative_weight tensor must be scalar or have shape [G, Q]")
    else:
        if negative_weight < 0:
            raise ValueError("negative_weight must be non-negative")
        negative_weights = generated_f.new_full((groups, num_negative), float(negative_weight))

    targets = torch.cat((old_generated, negative_f, positive_f), dim=1)
    target_weights = torch.cat((generated_weights, negative_weights, positive_weights), dim=1)

    with torch.no_grad():
        distance = _pairwise_distance(old_generated, targets)
        scale = (distance * target_weights.unsqueeze(1)).mean(dim=(1, 2))
        scale = scale / target_weights.mean(dim=1).clamp_min(1e-8)
        scale = scale.clamp_min(1e-3)
        input_scale = (scale / math.sqrt(feature_dim)).clamp_min(1e-3).view(groups, 1, 1)
        old_scaled = old_generated / input_scale
        targets_scaled = targets / input_scale
        normalized_distance = distance / scale.view(groups, 1, 1)
        diagonal = torch.eye(num_generated, targets.shape[1], device=generated.device, dtype=torch.bool).unsqueeze(0)
        normalized_distance = normalized_distance.masked_fill(diagonal, 1e6)

        total_force = torch.zeros_like(old_scaled)
        force_rms = []
        positive_support = []
        split = num_generated + num_negative
        for radius in radii:
            logits = -normalized_distance / radius
            row_affinity = logits.softmax(dim=-1)
            column_affinity = logits.softmax(dim=-2)
            # Keep the same floor as the reference DriftWorld implementation.
            affinity = (row_affinity * column_affinity).clamp_min(1e-6).sqrt()
            affinity = affinity * target_weights.unsqueeze(1)

            negative_affinity = affinity[:, :, :split]
            positive_affinity = affinity[:, :, split:]
            sum_positive = positive_affinity.sum(dim=-1, keepdim=True)
            sum_negative = negative_affinity.sum(dim=-1, keepdim=True)
            coefficients = torch.cat((-negative_affinity * sum_positive, positive_affinity * sum_negative), dim=-1)
            force = torch.bmm(coefficients, targets_scaled)
            force = force - coefficients.sum(dim=-1, keepdim=True) * old_scaled
            radius_rms = force.square().mean(dim=(1, 2)).clamp_min(1e-8).sqrt()
            total_force.add_(force / radius_rms.view(groups, 1, 1))
            force_rms.append(radius_rms.mean())
            positive_support.append(sum_positive.mean())

        goal = old_scaled + total_force

    error = generated_f / input_scale - goal
    per_group_loss = error.square().mean(dim=(1, 2))
    metrics = {
        "drift_scale": scale.mean().detach(),
        "drift_force_norm": total_force.square().mean().sqrt().detach(),
        "positive_support": torch.stack(positive_support).mean().detach(),
        "radius_force_rms": torch.stack(force_rms).mean().detach(),
    }
    if group_weight is not None:
        if group_weight.shape != (groups,):
            raise ValueError("group_weight must have shape [G]")
        per_group_loss = per_group_loss * group_weight.to(
            device=generated.device, dtype=per_group_loss.dtype
        )
    return per_group_loss.mean(), metrics
