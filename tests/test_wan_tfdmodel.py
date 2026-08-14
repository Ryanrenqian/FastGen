# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from fastgen.configs.experiments.WanI2V.config_tfd_wan22_5b import create_config
from fastgen.methods.distribution_matching.wan_teacher_feature_drifting import (
    _mask_first_temporal_token_slot,
    _repeat_group_feature_noise,
    _repeat_first_frame_video,
)


def test_wan_tfd_recipe_separates_cfg_from_student():
    config = create_config()
    assert config.model.feature_layers == []
    assert config.model.feature_indices == [9, 19, 29]
    assert config.model.feature_noise_sigma == 0.1
    assert config.model.generated_samples_per_condition == 4
    assert config.model.positive_samples_per_condition == 4
    assert config.model.teacher_positive_fill is True
    assert config.model.teacher_positive_sample_steps == 50
    assert config.model.teacher_positive_guidance_scale == 6.0
    assert config.model.static_negative_enabled is True
    assert config.model.static_negative_guidance_scale == 6.0
    assert config.model.guidance_scale is None
    assert config.dataloader_train.negative_prompt == ""


def test_static_video_repeats_conditioning_frame_without_aliasing():
    video = torch.randn(2, 3, 5, 4, 6)
    static = _repeat_first_frame_video(video)
    assert static.shape == video.shape
    assert torch.equal(static, video[:, :, :1].expand_as(video))
    assert static.is_contiguous()


def test_mask_first_temporal_token_slot_drops_exact_first_grid():
    feature = torch.arange(2 * 20 * 3).reshape(2, 20, 3)
    masked = _mask_first_temporal_token_slot(
        feature, sample_shape=(2, 48, 5, 4, 4), patch_size=(1, 2, 2)
    )
    assert masked.shape == (2, 16, 3)
    assert torch.equal(masked, feature.reshape(2, 5, 2, 2, 3)[:, 1:].reshape(2, 16, 3))


def test_mask_first_temporal_slot_for_fastgen_wan_feature_layout():
    feature = torch.arange(2 * 3 * 5 * 4 * 6).reshape(2, 3, 5, 4, 6)
    masked = _mask_first_temporal_token_slot(
        feature, sample_shape=(2, 48, 5, 4, 6), patch_size=(1, 2, 2)
    )
    assert masked.shape == (2, 3, 4, 4, 6)
    assert torch.equal(masked, feature[:, :, 1:])


def test_stride123_experiment_configs_only_differ_by_first_slot_mask():
    from fastgen.configs.experiments.WanI2V.config_tfd_wan22_5b_10k_f17_stride123 import (
        create_config as create_a,
    )
    from fastgen.configs.experiments.WanI2V.config_tfd_wan22_5b_10k_f17_stride123_maskfirst import (
        create_config as create_b,
    )

    config_a, config_b = create_a(), create_b()
    assert config_a.dataloader_train.positive_frame_strides == [1, 2, 3]
    assert config_a.model.positive_samples_per_condition == 3
    assert config_a.model.anchor_samples_per_condition == 3
    assert config_a.model.teacher_positive_fill is False
    assert config_a.model.feature_mask_first_temporal_slot is False
    assert config_b.model.feature_mask_first_temporal_slot is True


def test_self_attn_delta_experiment_recipe():
    from fastgen.configs.experiments.WanI2V.config_tfd_wan22_5b_10k_f17_self_attn_delta import (
        create_config as create_self_attn_config,
    )

    config = create_self_attn_config()
    assert config.model.feature_tap == "self_attn_delta"
    assert config.model.feature_indices == [8, 10, 12]
    assert config.model.feature_noise_sigma == 0.1
    assert config.model.feature_mask_first_temporal_slot is True
    assert config.model.positive_samples_per_condition == 3
    assert config.model.anchor_samples_per_condition == 3
    assert config.model.teacher_positive_fill is False
    assert config.dataloader_train.positive_frame_strides == [1, 2, 3]


def test_motion_semantic_experiment_recipe():
    from fastgen.configs.experiments.WanI2V.config_tfd_wan22_5b_10k_f17_motion_semantic import (
        create_config as create_motion_semantic_config,
    )

    config = create_motion_semantic_config()
    assert config.model.feature_indices == [8, 10, 12, 20, 24]
    assert config.model.feature_taps == [
        "self_attn_delta",
        "self_attn_delta",
        "self_attn_delta",
        "block_output",
        "block_output",
    ]
    assert config.model.feature_loss_weights == pytest.approx(
        [1.0, 1.0, 1.0, 0.225, 0.225]
    )
    assert config.model.feature_noise_sigma == 0.1
    assert config.model.feature_mask_first_temporal_slot is True


def test_group_feature_noise_is_shared_within_each_condition():
    noise = torch.tensor([[1.0], [2.0]])
    repeated = _repeat_group_feature_noise(noise, 3)
    assert torch.equal(
        repeated, torch.tensor([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])
    )


def test_motion_semantic_shared_noise_experiment_recipe():
    from fastgen.configs.experiments.WanI2V.config_tfd_wan22_5b_10k_f17_motion_semantic_shared_noise import (
        create_config as create_shared_noise_config,
    )

    config = create_shared_noise_config()
    assert config.model.feature_shared_noise_per_condition is True
    assert config.model.feature_noise_sigma == 0.1
    assert config.model.feature_indices == [8, 10, 12, 20, 24]


def test_balanced_four_by_four_experiment_recipe():
    from fastgen.configs.experiments.WanI2V.config_tfd_wan22_5b_10k_f17_balanced_4x4 import (
        create_config as create_balanced_config,
    )

    config = create_balanced_config()
    assert config.model.generated_samples_per_condition == 4
    assert config.model.positive_samples_per_condition == 4
    assert config.model.anchor_samples_per_condition == 4
    assert config.model.teacher_positive_fill is False
    assert config.model.static_negative_enabled is True
    static_weight = (
        config.model.static_negative_guidance_scale - 1.0
    ) * (config.model.generated_samples_per_condition - 1)
    assert static_weight == pytest.approx(1.0)
    assert config.dataloader_train.positive_frame_strides == [1]
    assert config.dataloader_train.positive_random_walk_count == 3
    assert config.dataloader_train.positive_random_step_min == 1
    assert config.dataloader_train.positive_random_step_max == 3


def test_balanced_four_by_four_nopool_experiment_recipe():
    from fastgen.configs.experiments.WanI2V.config_tfd_wan22_5b_10k_f17_balanced_4x4_nopool import (
        create_config as create_nopool_config,
    )

    config = create_nopool_config()
    assert config.model.feature_pool_size == 1
    assert config.model.generated_samples_per_condition == 4
    assert config.model.positive_samples_per_condition == 4
    assert config.model.static_negative_enabled is True
    static_weight = (
        config.model.static_negative_guidance_scale - 1.0
    ) * (config.model.generated_samples_per_condition - 1)
    assert static_weight == pytest.approx(1.0)
