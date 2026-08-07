# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from fastgen.methods.distribution_matching.teacher_feature_drifting import (
    anchor_margin_loss,
    resolve_edm_feature_selectors,
    teacher_feature_drifting_loss,
)
from fastgen.configs.experiments.EDM.config_tfd_in64 import create_config
from fastgen.networks.EDM.network import DhariwalUNet


def test_teacher_feature_drifting_loss_has_generator_gradient():
    generated = torch.randn(2, 4, 8, requires_grad=True)
    positive = torch.randn(2, 4, 8)
    loss, metrics = teacher_feature_drifting_loss(generated, positive, [0.02, 0.05, 0.1, 0.2])
    loss.mean().backward()
    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()
    assert metrics["drift_norm"].ndim == 0


def test_teacher_feature_drifting_matches_official_equations():
    generated = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], requires_grad=True)
    positive = torch.tensor([[[0.0, 1.0], [1.0, 1.0]]])
    radius = 0.5
    loss, _ = teacher_feature_drifting_loss(generated, positive, [radius])

    frozen = generated.detach()
    targets = torch.cat([frozen, positive], dim=1)
    distance = torch.sqrt(torch.clamp(torch.cdist(frozen, targets).pow(2), min=1e-8))
    scale = distance.mean()
    input_scale = torch.clamp(scale / math.sqrt(2.0), min=1e-3)
    normalized = distance / torch.clamp(scale, min=1e-3)
    normalized += torch.nn.functional.pad(torch.eye(2).unsqueeze(0), (0, 2)) * 1e6
    affinity = torch.softmax(-normalized / radius, dim=-1)
    reverse = torch.softmax(-normalized / radius, dim=-2)
    affinity = torch.sqrt(torch.clamp(affinity * reverse, min=1e-6))
    negative, positive_affinity = affinity[:, :, :2], affinity[:, :, 2:]
    coefficients = torch.cat(
        [
            -negative * positive_affinity.sum(-1, keepdim=True),
            positive_affinity * negative.sum(-1, keepdim=True),
        ],
        dim=2,
    )
    scaled_generated, scaled_targets = frozen / input_scale, targets / input_scale
    force = torch.einsum("biy,byd->bid", coefficients, scaled_targets)
    force -= coefficients.sum(-1)[..., None] * scaled_generated
    force /= torch.sqrt(torch.clamp(force.pow(2).mean(), min=1e-8))
    expected = (generated / input_scale - (scaled_generated + force)).pow(2).mean((-1, -2))
    assert torch.allclose(loss, expected)


def test_anchor_margin_uses_median_euclidean_bandwidth():
    generated = torch.tensor([[[0.0], [2.0]]], requires_grad=True)
    anchors = torch.tensor([[[1.0], [3.0]]])
    loss, info = anchor_margin_loss(generated, anchors, bandwidth=0.0, alpha=0.5)
    assert torch.allclose(info["bandwidth"], torch.median(torch.cdist(anchors, generated)))
    loss.mean().backward()
    assert generated.grad is not None


def test_imagenet_feature_tokens_resolve_to_official_modules():
    model = DhariwalUNet(
        img_resolution=64, in_channels=3, out_channels=3, label_dim=1000,
        model_channels=8, channel_mult=[1, 2, 3, 4], num_blocks=3,
        attn_resolutions=[], dropout=0.0,
    )
    selectors = resolve_edm_feature_selectors(
        model, ["enc:6", "enc:11", "bottleneck", "dec:7", "dec:12"]
    )
    assert selectors == [
        ("enc", "32x32_block1"),
        ("enc", "16x16_block2"),
        ("enc", "8x8_block2"),
        ("dec", "16x16_block0"),
        ("dec", "32x32_block0"),
    ]


def test_edm_named_feature_extraction_preserves_requested_order():
    model = DhariwalUNet(
        img_resolution=8, in_channels=3, out_channels=3, label_dim=2,
        model_channels=8, channel_mult=[1], num_blocks=1, attn_resolutions=[], dropout=0.0,
    )
    selectors = [("enc", "8x8_block0"), ("dec", "8x8_block1")]
    features = model(
        torch.randn(2, 3, 8, 8), torch.ones(2), torch.eye(2),
        return_features_early=True, feature_indices=set(), feature_selectors=selectors,
    )
    assert len(features) == 2
    assert features[0].shape[0] == features[1].shape[0] == 2


def test_imagenet_recipe_uses_official_bfloat16_semantics_and_update_count():
    config = create_config()
    assert config.model.precision_amp == "bfloat16"
    assert config.model.grad_scaler_enabled is False
    assert config.model.use_ema is False
    assert config.trainer.max_iter == 200001
    assert config.model.net_scheduler.f_start == [0.0]
    assert config.trainer.save_ckpt_iter == 500
