# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 DriftWorld temporal-field weight-search configuration."""

from fastgen.configs.experiments.WanI2V.config_driftworld_wan22_5b import (
    create_config as create_base_config,
)


def create_config():
    config = create_base_config()
    # Defaults preserve the frame-field baseline except for scale-preserving
    # DINO motion reweighting. Temporal fields are enabled per experiment arm.
    config.model.normalize_dinov3_motion_weight = True
    config.model.latent_velocity_drift_weight = 0.0
    config.model.dinov3_velocity_drift_weight = 0.0
    config.model.log_component_gradient_iter = 1
    config.log_config.group = "wan22_5b_ti2v_driftworld_temporal_search"
    config.log_config.name = "wan22_driftworld_temporal_search"
    return config
