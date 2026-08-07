# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.data import TFD_ImageNet64_Loader_Config
from fastgen.configs.methods.config_tfd import create_config as create_tfd_config
from fastgen.configs.net import CKPT_ROOT_DIR, EDM_ImageNet64_Config


def create_config():
    config = create_tfd_config()
    config.model.net = EDM_ImageNet64_Config
    config.model.input_shape = [3, 64, 64]
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/imagenet-64/edm-imagenet-64x64-cond-adm.pth"

    # Official Table 3 / configs/imagenet64/tfd.yaml recipe.
    config.model.feature_layers = ["enc:6", "enc:11", "bottleneck", "dec:7", "dec:12"]
    config.model.drift_radii = [0.02, 0.05, 0.1, 0.2]
    config.model.feature_pool_size = 4
    config.model.feature_noise_p_mean = -1.2
    config.model.feature_noise_p_std = 1.2
    config.model.feature_noise_sigma_min = 0.02
    config.model.feature_noise_sigma_max = 0.1
    config.model.feature_noise_trunc_resamples = 8
    config.model.generated_samples_per_condition = 4
    config.model.positive_samples_per_condition = 4
    config.model.anchor_samples_per_condition = 4
    config.model.conditioning_sigma = 80.0
    config.model.anchor_weight = 1.0
    config.model.anchor_temperature = 1.0
    config.model.anchor_margin = 0.5
    config.model.anchor_bandwidth = 0.0
    config.model.net_optimizer.optim_type = "adamw"
    config.model.net_optimizer.lr = 2e-6
    config.model.net_optimizer.betas = (0.9, 0.999)
    config.model.net_optimizer.weight_decay = 0.01
    # The authors' EDM flag is named use_fp16, but its implementation uses
    # torch.bfloat16 and Accelerator(mixed_precision="no") without scaling.
    config.model.precision_amp = "bfloat16"
    config.model.precision_amp_infer = "bfloat16"
    config.model.grad_scaler_enabled = False
    config.model.use_ema = False
    config.dataloader_train = TFD_ImageNet64_Loader_Config
    # Seven ranks x 10 class groups x 4 generations = 280 generated images/update.
    config.dataloader_train.batch_size = 10
    config.dataloader_train.positives_per_condition = 4
    config.dataloader_train.anchors_per_condition = 4
    config.dataloader_train.seed = 10
    config.trainer.batch_size_global = 70
    config.trainer.seed = 10
    # FastGen optimizes range(1, max_iter), hence +1 for 200,000 updates.
    config.trainer.max_iter = 200001
    config.trainer.logging_iter = 100
    config.trainer.callbacks.grad_clip.grad_norm = 10.0
    config.log_config.group = "edm_imagenet64_tfd"
    return config
