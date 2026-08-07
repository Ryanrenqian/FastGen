# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig

from fastgen.configs.callbacks import EMA_CONST_CALLBACKS
from fastgen.configs.data import ImageNet64_Loader_Config
from fastgen.configs.methods.config_tfd import create_config as create_tfd_config
from fastgen.configs.net import CKPT_ROOT_DIR, EDM_ImageNet64_Config


def create_config():
    config = create_tfd_config()
    config.model.net = EDM_ImageNet64_Config
    config.model.input_shape = [3, 64, 64]
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/imagenet-64/edm-imagenet-64x64-cond-adm.pth"

    # Two encoder levels, bottleneck, and two decoder levels.
    config.model.feature_indices = [1, 2, 3, 4, 6]
    config.model.drift_radii = [0.02, 0.05, 0.1, 0.2]
    config.model.net_optimizer.lr = 2e-6
    config.model.net_optimizer.weight_decay = 0.01
    config.model.precision_amp = "float16"
    config.model.precision_amp_infer = "float16"
    config.model.grad_scaler_enabled = True

    config.model.use_ema = ["ema_9999", "ema_99995", "ema_9996"]
    config.trainer.callbacks = DictConfig(
        {key: value for key, value in config.trainer.callbacks.items() if not key.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_CONST_CALLBACKS)
    config.dataloader_train = ImageNet64_Loader_Config
    # With eight ranks this is 128 conditions and 512 generated samples/update.
    config.dataloader_train.batch_size = 16
    config.trainer.batch_size_global = 128
    config.trainer.max_iter = 200000
    config.trainer.callbacks.grad_clip.grad_norm = 10.0
    config.log_config.group = "edm_imagenet64_tfd"
    return config
