# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.experiments.WanI2V.config_tfd_wan22_5b import create_config


def test_wan_tfd_recipe_separates_cfg_from_student():
    config = create_config()
    assert config.model.feature_layers == []
    assert config.model.feature_indices == [9, 19, 29]
    assert config.model.generated_samples_per_condition == 4
    assert config.model.positive_samples_per_condition == 4
    assert config.model.teacher_positive_fill is True
    assert config.model.teacher_positive_sample_steps == 50
    assert config.model.teacher_positive_guidance_scale == 6.0
    assert config.model.guidance_scale is None
    assert config.dataloader_train.negative_prompt == ""
