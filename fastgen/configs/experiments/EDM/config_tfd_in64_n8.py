# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.experiments.EDM.config_tfd_in64 import create_config as create_base_config


def create_config():
    config = create_base_config()
    config.model.generated_samples_per_condition = 8
    config.model.positive_samples_per_condition = 8
    config.model.anchor_samples_per_condition = 8
    config.model.teacher_generated_checkpoint = True
    config.model.teacher_reference_chunk_size = 18
    config.model.teacher_attention_backend = "sdpa"
    config.dataloader_train.batch_size = 9
    config.dataloader_train.positives_per_condition = 8
    config.dataloader_train.anchors_per_condition = 8
    config.trainer.batch_size_global = 72
    config.trainer.max_iter = 1001
    return config
