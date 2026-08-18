# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from fastgen.drifting_loss import drifting_loss


def test_drifting_loss_is_finite_and_backpropagates():
    generated = torch.randn(6, 4, 8, requires_grad=True)
    positive = torch.randn(6, 1, 8)
    negative = torch.randn(6, 1, 8)

    loss, metrics = drifting_loss(
        generated,
        positive,
        negative=negative,
        negative_weight=1.0,
        radii=[0.02, 0.05],
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()
    assert generated.grad.abs().sum() > 0
    assert all(torch.isfinite(value) for value in metrics.values())


def test_drifting_groups_are_batch_independent():
    generated = torch.randn(2, 4, 5, requires_grad=True)
    positive = torch.randn(2, 1, 5)

    combined, _ = drifting_loss(generated, positive, radii=[0.05])
    separate = torch.stack(
        [
            drifting_loss(
                generated[index : index + 1],
                positive[index : index + 1],
                radii=[0.05],
            )[0]
            for index in range(2)
        ]
    ).mean()

    assert torch.allclose(combined, separate, rtol=1e-5, atol=1e-6)


def test_drifting_group_weight_matches_reference_weighting():
    generated = torch.randn(3, 4, 5, requires_grad=True)
    positive = torch.randn(3, 1, 5)
    weights = torch.tensor([1.0, 2.0, 4.0])

    weighted, _ = drifting_loss(
        generated, positive, group_weight=weights, radii=[0.05]
    )
    separate = torch.stack(
        [
            drifting_loss(
                generated[index : index + 1],
                positive[index : index + 1],
                radii=[0.05],
            )[0]
            * weights[index]
            for index in range(3)
        ]
    ).mean()

    assert torch.allclose(weighted, separate, rtol=1e-5, atol=1e-6)


def test_drifting_loss_matches_driftworld_reference_equations():
    torch.manual_seed(42)
    generated = torch.randn(5, 4, 7, requires_grad=True)
    positive = torch.randn(5, 1, 7)
    negative = torch.randn(5, 1, 7)

    loss, _ = drifting_loss(
        generated,
        positive,
        negative=negative,
        negative_weight=1.0,
        radii=[0.02, 0.05],
    )

    # Regression value produced by driftworld/drifting/drift_loss_indep.py.
    assert torch.allclose(loss, torch.tensor(3.8144168853759766), rtol=1e-5, atol=1e-6)


def test_bridge_field_objective_is_direct_sum():
    vae_loss = torch.tensor(2.0)
    dino_block_losses = torch.tensor([3.0, 5.0, 7.0])

    objective = vae_loss + 3.0 * dino_block_losses.mean()

    assert objective == vae_loss + dino_block_losses.sum()


def test_wan_objective_disables_previous_frame_negative(monkeypatch):
    if importlib.util.find_spec("omegaconf") is None:
        pytest.skip("FastGen framework dependencies are not installed")
    from fastgen.methods.distribution_matching import driftworld as driftworld_module

    calls = []
    losses = iter((2.0, 3.0, 5.0, 7.0))

    def fake_drifting_loss(generated, positive, **kwargs):
        calls.append((kwargs["negative"], kwargs["negative_weight"]))
        return generated.new_tensor(next(losses), requires_grad=True), {}

    monkeypatch.setattr(driftworld_module, "drifting_loss", fake_drifting_loss)
    fake = SimpleNamespace(
        config=SimpleNamespace(
            generated_samples_per_condition=2,
            mask_conditioning_latent_slot=True,
            local_drift_weight=1.0,
            trajectory_drift_weight=0.0,
            dinov3_drift_weight=3.0,
            static_negative_weight=1.0,
            use_dinov3_static_negative=False,
            drift_radii=[0.02, 0.05],
            compare_temporal_sample_force=False,
        ),
        input_shape=(1, 2, 1, 1),
        device=torch.device("cpu"),
    )
    fake.net = SimpleNamespace(
        noise_scheduler=SimpleNamespace(
            max_sigma=1.0,
            max_t=1.0,
            t_precision=torch.float32,
            latents=lambda noise: noise,
        )
    )
    real = torch.randn(1, 1, 2, 1, 1)
    fake._prepare_data = lambda data: (real, data["condition"])
    fake._positives = lambda data, value: value.unsqueeze(1)
    fake.gen_data_from_net = lambda noise, timestep, condition: noise
    fake._get_outputs = lambda *args: {}
    dino_negative = torch.randn(1, 1, 1)
    dino_negative_weight = torch.tensor([[0.75]])
    fake._dinov3_drifting_fields = lambda generated, positive: [
        (
            f"block_{index}",
            torch.randn(1, 2, 1, requires_grad=True),
            torch.randn(1, 1, 1),
            dino_negative,
            torch.ones(1),
            dino_negative_weight,
        )
        for index in (2, 5, 8)
    ]

    loss_map, _ = driftworld_module.DriftWorldModel.single_train_step(
        fake, {"condition": torch.zeros(1, 1)}, iteration=1
    )

    assert calls[0][0] is None
    assert [call[0] for call in calls[1:]] == [None, None, None]
    assert all(call[1] == 0.0 for call in calls[1:])
    assert loss_map["total_loss"] == 17.0
    assert loss_map["vae_weighted_loss"] == 2.0
    assert loss_map["dino_weighted_loss"] == 15.0


def test_dino_static_negative_weight_is_zero_without_motion():
    if importlib.util.find_spec("omegaconf") is None:
        pytest.skip("FastGen framework dependencies are not installed")
    from fastgen.features.dinov3 import DinoV3FeatureExtractor

    extractor = SimpleNamespace(
        motion_quantile=0.98,
        motion_threshold=0.35,
        motion_alpha=2.5,
    )
    extractor._motion_factor = lambda positive, previous: (
        DinoV3FeatureExtractor._motion_factor(extractor, positive, previous)
    )
    static = torch.ones(2, 4, 3)
    moving = static.clone()
    moving[:, 0] += 2.0

    static_weight = DinoV3FeatureExtractor.motion_negative_weights(
        extractor, static, static
    )
    moving_weight = DinoV3FeatureExtractor.motion_negative_weights(
        extractor, moving, static
    )

    assert torch.equal(static_weight, torch.zeros_like(static_weight))
    assert moving_weight[:, 0].min() > 0
    assert torch.equal(moving_weight[:, 1:], torch.zeros_like(moving_weight[:, 1:]))


def test_video_tokens_drop_ti2v_conditioning_slot():
    if importlib.util.find_spec("omegaconf") is None:
        pytest.skip("FastGen framework dependencies are not installed")
    from fastgen.methods.distribution_matching.driftworld import _video_tokens

    videos = torch.arange(2 * 3 * 4 * 5 * 2 * 2).reshape(2, 3, 4, 5, 2, 2)

    tokens = _video_tokens(videos, drop_first_frame=True)

    assert tokens.shape == (2 * 4 * 2 * 2, 3, 4)
    assert torch.equal(tokens[0, 0], videos[0, 0, :, 1, 0, 0])


def test_video_block_tokens_capture_complete_latent_trajectory():
    if importlib.util.find_spec("omegaconf") is None:
        pytest.skip("FastGen framework dependencies are not installed")
    from fastgen.methods.distribution_matching.driftworld import _video_block_tokens

    videos = torch.arange(2 * 3 * 4 * 5 * 4 * 6).reshape(2, 3, 4, 5, 4, 6)
    tokens = _video_block_tokens(
        videos, block_shape=(4, 2, 3), drop_first_frame=True
    )

    assert tokens.shape == (2 * 1 * 2 * 2, 3, 4 * 4 * 2 * 3)
    expected = videos[0, 0, :, 1:5, 0:2, 0:3].reshape(-1)
    assert torch.equal(tokens[0, 0], expected)


def test_dinov3_video_tokens_keep_candidates_aligned():
    if importlib.util.find_spec("omegaconf") is None:
        pytest.skip("FastGen framework dependencies are not installed")
    from fastgen.methods.distribution_matching.driftworld import _dinov3_video_tokens

    features = torch.arange(2 * 3 * 4 * 5 * 7).reshape(2 * 3 * 4, 5, 7)
    tokens = _dinov3_video_tokens(features, batch=2, candidates=3, frames=4)

    assert tokens.shape == (2 * 4 * 5, 3, 7)
    original = features.reshape(2, 3, 4, 5, 7)
    assert torch.equal(tokens[0, 2], original[0, 2, 0, 0])


def test_temporal_sample_force_comparison_is_finite_and_frame_aligned():
    if importlib.util.find_spec("omegaconf") is None:
        pytest.skip("FastGen framework dependencies are not installed")
    from fastgen.methods.distribution_matching.driftworld import (
        _compare_temporal_sample_force,
    )

    batch, frames, spatial, candidates, feature_dim = 2, 4, 3, 5, 7
    generated = torch.randn(batch * frames * spatial, candidates, feature_dim)
    positive = torch.randn(batch * frames * spatial, 1, feature_dim)
    negative = torch.randn(batch * frames * spatial, 1, feature_dim)
    _, sample_metrics, sample_force = drifting_loss(
        generated, positive, negative=negative, radii=[0.05], return_force=True
    )

    metrics = _compare_temporal_sample_force(
        generated,
        positive,
        negative,
        sample_force,
        sample_metrics,
        batch=batch,
        candidates=candidates,
        frames=frames,
        negative_weight=1.0,
        radii=[0.05],
    )

    assert all(torch.isfinite(value) for value in metrics.values())
    assert -1 <= metrics["force_cosine"] <= 1
    assert metrics["temporal_over_sample_force"] > 0
    assert all(f"frame_{index}_force_cosine" in metrics for index in range(1, 5))


def test_repeat_condition_keeps_ti2v_candidates_aligned():
    if importlib.util.find_spec("omegaconf") is None:
        pytest.skip("FastGen framework dependencies are not installed")
    from fastgen.methods.distribution_matching.driftworld import _repeat_condition

    condition = {
        "text_embeds": torch.tensor([[1.0], [2.0]]),
        "first_frame_cond": torch.tensor([[10.0], [20.0]]),
    }

    repeated = _repeat_condition(condition, 3)

    assert repeated["text_embeds"].flatten().tolist() == [1, 1, 1, 2, 2, 2]
    assert repeated["first_frame_cond"].flatten().tolist() == [10, 10, 10, 20, 20, 20]


def test_wan22_driftworld_config_uses_ti2v_pretrained_model():
    required = ("omegaconf", "webdataset", "diffusers", "torchvision", "av")
    if any(importlib.util.find_spec(name) is None for name in required):
        pytest.skip("full FastGen training dependencies are not installed")
    from fastgen.configs.experiments.WanI2V.config_driftworld_wan22_5b import (
        create_config,
    )
    from fastgen.methods.distribution_matching.driftworld import DriftWorldModel

    config = create_config()

    assert config.model_class["_target_"] is DriftWorldModel
    assert config.model.net.model_id_or_local_path == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert config.model.input_shape == [48, 5, 16, 16]
    assert config.model.generated_samples_per_condition == 64
    assert config.model.trajectory_drift_block == (4, 2, 2)
    assert config.model.local_drift_weight == 1.0
    assert config.model.trajectory_drift_weight == 0.0
    assert config.model.dinov3_drift_weight == 3.0
    assert config.model.use_dinov3_static_negative is False
    assert config.model.static_negative_weight == 0.0
    assert config.model.dinov3_block_indices == (2, 5, 8)
    assert config.dataloader_train.sequence_length == 5
    assert config.dataloader_train.frame_start == 30
    assert config.dataloader_train.frame_stride == 1
    assert config.trainer.callbacks.wandb.sample_logging_iter == 500
    assert config.dataloader_train.img_size == (256, 256)
    assert config.model.framewise_vae is True
    assert config.dataloader_train.index_path.endswith(
        "demo5_dataset/manifests/demo5_clean_10k_f49_seed10.csv"
    )


def test_wan22_driftworld_dinov3_only_is_single_variable_ablation():
    required = ("omegaconf", "webdataset", "diffusers", "torchvision", "av")
    if any(importlib.util.find_spec(name) is None for name in required):
        pytest.skip("full FastGen training dependencies are not installed")
    from fastgen.configs.experiments.WanI2V.config_driftworld_wan22_5b import (
        create_config as create_baseline_config,
    )
    from fastgen.configs.experiments.WanI2V.config_driftworld_wan22_5b_dinov3_only import (
        create_config,
    )

    baseline = create_baseline_config()
    config = create_config()

    assert baseline.model.local_drift_weight == 1.0
    assert config.model.local_drift_weight == 0.0
    assert config.model.trajectory_drift_weight == baseline.model.trajectory_drift_weight == 0.0
    assert config.model.dinov3_drift_weight == baseline.model.dinov3_drift_weight == 3.0
    assert config.model.static_negative_weight == baseline.model.static_negative_weight == 0.0
    assert config.model.use_dinov3_static_negative is baseline.model.use_dinov3_static_negative is False
    assert config.model.input_shape == baseline.model.input_shape == [48, 5, 16, 16]
    assert config.model.framewise_vae is baseline.model.framewise_vae is True
    assert config.model.generated_samples_per_condition == 64
    assert config.dataloader_train.img_size == baseline.dataloader_train.img_size == (256, 256)
    assert config.dataloader_train.sequence_length == 5
    assert config.dataloader_train.frame_start == 30
    assert config.dataloader_train.frame_stride == 1
    assert "dinov3_only" in config.log_config.name
