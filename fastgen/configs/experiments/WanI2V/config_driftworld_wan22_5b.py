# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DriftWorld-style one-step TI2V training from Wan2.2-TI2V-5B."""

from fastgen.configs.data import CSVVideoLoaderConfig
from fastgen.configs.methods.config_driftworld import create_config as create_driftworld_config
from fastgen.configs.net import Wan22_I2V_5B_Config
from fastgen.methods import DriftWorldModel
from fastgen.utils import LazyCall as L


def create_config():
    config = create_driftworld_config()
    config.model_class = L(DriftWorldModel)(config=None)
    config.model.net = Wan22_I2V_5B_Config
    config.model.fsdp_meta_init = True
    config.model.input_shape = [48, 5, 24, 40]
    config.model.precision = "bfloat16"
    config.model.precision_amp_enc = "bfloat16"
    config.model.student_sample_steps = 1
    config.model.framewise_vae = True
    config.model.generated_samples_per_condition = 64
    config.model.drift_radii = [0.02, 0.05]
    config.model.mask_conditioning_latent_slot = True
    # Match Bridge DriftWorld's n_neg=64. Weight 1 corresponds to CFG
    # alpha=64/63 in its static-transition-negative parameterization.
    config.model.static_negative_weight = 1.0
    # Match Bridge DriftWorld's local latent and DINOv3 feature objectives.
    config.model.local_drift_weight = 1.0
    config.model.trajectory_drift_weight = 0.0
    config.model.trajectory_drift_block = (4, 2, 2)
    config.model.dinov3_drift_weight = 1.0
    config.model.dinov3_repo_dir = (
        "/mnt/home/renqian/imgGen/runtime/MODEL/dinov3/repo"
    )
    config.model.dinov3_weights_path = (
        "/mnt/home/renqian/imgGen/runtime/MODEL/dinov3/"
        "model.safetensors"
    )
    config.model.dinov3_model_name = "dinov3_vitb16"
    config.model.dinov3_input_size = 256
    config.model.dinov3_block_indices = (2, 5, 8)
    config.model.dinov3_frame_chunk_size = 8
    config.model.dinov3_vae_chunk_size = 1
    config.model.dinov3_motion_alpha = 2.5
    config.model.dinov3_motion_lambda = 12.0
    config.model.dinov3_motion_quantile = 0.98
    config.model.dinov3_motion_threshold = 0.35
    config.model.net_optimizer.optim_type = "adamw"
    config.model.net_optimizer.lr = 1e-7
    config.model.net_optimizer.weight_decay = 0.01
    config.model.net_optimizer.betas = (0.9, 0.95)
    config.model.net_scheduler.warm_up_steps = [200]
    config.model.net_scheduler.f_start = [1e-6]

    config.dataloader_train = CSVVideoLoaderConfig
    config.dataloader_train.index_path = (
        "/mnt/dataset/demo5_dataset/manifests/" "demo5_clean_10k_f49_seed10.csv"
    )
    config.dataloader_train.dataset_size = 10_000
    config.dataloader_train.batch_size = 1
    config.dataloader_train.sequence_length = 5
    # Skip the initial static camera period, then use five adjacent frames.
    config.dataloader_train.frame_start = 30
    config.dataloader_train.frame_stride = 1
    config.dataloader_train.img_size = (640, 384)
    config.dataloader_train.negative_prompt = ""
    config.dataloader_train.num_workers = 2

    config.trainer.fsdp = True
    config.trainer.seed = 10
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 500
    config.trainer.max_iter = 10_001
    config.trainer.callbacks.grad_clip.grad_norm = 2.0
    # Keep scalar metrics frequent, but avoid encoding/uploading two MP4 files
    # (generated and ground truth) at every scalar logging interval.
    config.trainer.callbacks.wandb.sample_logging_iter = 500
    config.log_config.group = "wan22_5b_ti2v_driftworld"
    config.log_config.name = "wan22_ti2v5b_driftworld_framewise_f5_s1_n64_10k"
    return config
