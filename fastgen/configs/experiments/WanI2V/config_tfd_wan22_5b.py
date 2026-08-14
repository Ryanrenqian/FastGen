# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 TI2V-5B TFD on the Cosmos3 WebData CSV index."""

from fastgen.configs.data import CSVVideoLoaderConfig
from fastgen.configs.methods.config_tfd import create_config as create_tfd_config
from fastgen.configs.net import Wan22_I2V_5B_Config
from fastgen.methods import WanTFDModel
from fastgen.utils import LazyCall as L


def create_config():
    config = create_tfd_config()
    config.model_class = L(WanTFDModel)(config=None)
    config.model.net = Wan22_I2V_5B_Config
    config.model.fsdp_meta_init = True
    config.model.input_shape = [48, 21, 44, 80]
    config.model.precision = "bfloat16"
    config.model.precision_amp_enc = "bfloat16"
    config.model.feature_layers = []
    config.model.feature_indices = [9, 19, 29]
    config.model.feature_pool_size = 4
    config.model.feature_noise_sigma = 0.1
    config.model.generated_samples_per_condition = 4
    config.model.positive_samples_per_condition = 4
    config.model.anchor_samples_per_condition = 4
    config.model.teacher_positive_fill = True
    config.model.teacher_positive_sample_steps = 50
    config.model.teacher_positive_guidance_scale = 6.0
    config.model.teacher_positive_batch_size = 1
    config.model.teacher_positive_sample_kwargs = {
        "shift": 3.0,
        "show_progress": False,
    }
    config.model.static_negative_enabled = True
    config.model.static_negative_guidance_scale = 6.0
    config.model.anchor_weight = 1.0
    config.model.net_optimizer.optim_type = "adamw"
    config.model.net_optimizer.lr = 1e-6
    config.model.net_optimizer.weight_decay = 0.01
    config.model.net_scheduler.f_start = [0.0]

    config.dataloader_train = CSVVideoLoaderConfig
    config.dataloader_train.index_path = (
        "/mnt/dataset/cosmos3-dataset/data_index/webdata/"
        "all_final_selected_sixth.csv"
    )
    config.dataloader_train.dataset_size = 144_428
    config.dataloader_train.batch_size = 1
    config.dataloader_train.sequence_length = 81
    config.dataloader_train.img_size = (1280, 704)
    config.dataloader_train.negative_prompt = ""
    config.dataloader_train.num_workers = 2

    config.trainer.seed = 10
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 20
    config.trainer.max_iter = 21
    config.trainer.callbacks.grad_clip.grad_norm = 1.0
    config.log_config.group = "wan22_5b_ti2v_tfd"
    config.log_config.wandb_mode = "online"
    config.log_config.wandb_netrc = (
        "/mnt/home/renqian/oneNFE/FastGen-tfd/.secrets/wandb.netrc"
    )
    return config
