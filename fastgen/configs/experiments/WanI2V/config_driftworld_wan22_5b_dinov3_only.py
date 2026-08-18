# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framewise Wan2.2 DriftWorld ablation without latent-space drift loss."""

from fastgen.configs.experiments.WanI2V.config_driftworld_wan22_5b import (
    create_config as create_baseline_config,
)


def create_config():
    config = create_baseline_config()

    # Single-variable ablation: retain RGB decoding and DINOv3 supervision,
    # while removing both latent-space drifting objectives.
    config.model.local_drift_weight = 0.0
    config.model.trajectory_drift_weight = 0.0

    config.log_config.name = (
        "wan22_ti2v5b_driftworld_dinov3_only_256x256_framewise_f5_s1_n64_10k"
    )
    return config
