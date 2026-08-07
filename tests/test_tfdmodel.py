# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest
import torch

from fastgen.configs.config_utils import override_config_with_opts
from fastgen.configs.methods.config_tfd import ModelConfig
from fastgen.methods import TFDModel
from fastgen.methods.distribution_matching.teacher_feature_drifting import teacher_feature_drifting_loss


def test_teacher_feature_drifting_loss_has_generator_gradient():
    generated = torch.randn(2, 4, 8, requires_grad=True)
    positive = torch.randn(2, 3, 8)

    drift_loss, anchor_loss, metrics = teacher_feature_drifting_loss(
        generated, positive, radii=[0.02, 0.05, 0.2]
    )
    (drift_loss + anchor_loss).backward()

    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()
    assert metrics["drift_norm"].ndim == 0


def test_teacher_feature_drifting_matches_mean_shift_equations():
    generated = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    positive = torch.tensor([[[0.0, 1.0], [1.0, 1.0]]])
    radius = 0.5

    drift_loss, _, metrics = teacher_feature_drifting_loss(
        generated, positive, radii=[radius], anchor_weight=0.0
    )

    dist_pos = torch.cdist(generated, positive)
    dist_neg = torch.cdist(generated, generated).masked_fill(
        torch.eye(2, dtype=torch.bool).unsqueeze(0), torch.inf
    )
    attraction = torch.bmm(torch.softmax(-dist_pos / radius, dim=-1), positive)
    repulsion = torch.bmm(torch.softmax(-dist_neg / radius, dim=-1), generated)
    expected_drift = attraction - repulsion

    assert torch.allclose(metrics["drift_norm"], expected_drift.square().mean().sqrt())
    assert torch.allclose(drift_loss, expected_drift.square().mean())


@pytest.fixture
def get_model_data():
    gc.collect()
    config = ModelConfig()
    config.net = override_config_with_opts(
        config.net, ["-", "img_resolution=8", "channel_mult=[1]", "channel_mult_noise=1"]
    )
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.precision = "float32"
    config.pretrained_model_path = ""
    config.input_shape = [3, 8, 8]
    # Exercise both the encoder and newly exposed decoder feature levels.
    config.feature_indices = [0, 1]
    config.feature_pool_size = 2
    config.generated_samples_per_condition = 2
    config.positive_samples_per_condition = 2

    model = TFDModel(config)
    model.on_train_begin()
    model.init_optimizers()

    labels = torch.nn.functional.one_hot(torch.tensor([1, 2]), num_classes=10)
    data = {
        "real": torch.randn(2, 3, 8, 8, device=model.device, dtype=model.precision),
        "condition": labels.to(device=model.device, dtype=model.precision),
    }
    return model, data


def test_single_train_step(get_model_data):
    model, data = get_model_data
    loss_map, outputs = model.single_train_step(data, iteration=0)

    assert {"total_loss", "tfd_loss", "anchor_loss", "drift_norm"} <= loss_map.keys()
    assert outputs["gen_rand"].shape == data["real"].shape
    loss_map["total_loss"].backward()
    assert any(parameter.grad is not None for parameter in model.net.parameters())
