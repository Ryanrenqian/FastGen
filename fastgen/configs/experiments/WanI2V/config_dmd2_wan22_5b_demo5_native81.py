# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-step Wan2.2 DMD2 on eligible 81-frame Demo5 clean videos."""

from fastgen.configs.data import CSVVideoLoaderConfig
from fastgen.configs.experiments.WanI2V.config_dmd2_wan22_5b import (
    create_config as create_wan22_dmd2_config,
)
from fastgen.datasets.wds_dataloaders import PRESET_CONSTANTS


def create_config():
    config = create_wan22_dmd2_config()
    config.dataloader_train = CSVVideoLoaderConfig
    config.dataloader_train.index_path = (
        "/mnt/dataset/demo5_dataset/manifests/"
        "demo5_clean_10k_f49_seed10.csv"
    )
    # The frozen manifest contains 6,423 entries with at least 81 frames.
    config.dataloader_train.dataset_size = 6_423
    config.dataloader_train.batch_size = 1
    config.dataloader_train.sequence_length = 81
    config.dataloader_train.img_size = (1280, 704)
    config.dataloader_train.frame_start = None
    config.dataloader_train.frame_stride = 1
    config.dataloader_train.positive_frame_strides = []
    config.dataloader_train.negative_prompt = PRESET_CONSTANTS["neg_prompt_wan"]
    config.dataloader_train.num_workers = 2

    config.trainer.fsdp = True
    config.trainer.seed = 10
    config.trainer.logging_iter = 10
    config.trainer.callbacks.wandb.sample_logging_iter = 500
    config.log_config.group = "wan22_5b_ti2v_dmd2_native81"
    config.log_config.name = "wan22_ti2v5b_dmd2_2step_demo5_1280x704_nativevae_f81"
    return config
