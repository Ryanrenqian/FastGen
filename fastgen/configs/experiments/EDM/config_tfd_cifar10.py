# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig

from fastgen.configs.callbacks import EMA_CONST_CALLBACKS
from fastgen.configs.data import CIFAR10_Loader_Config
from fastgen.configs.methods.config_tfd import create_config as create_tfd_config
from fastgen.configs.net import CKPT_ROOT_DIR, EDM_CIFAR10_Config


def create_config():
    """Small, class-conditional recipe for fast TFD validation."""
    config = create_tfd_config()
    config.model.net = EDM_CIFAR10_Config
    config.model.input_shape = [3, 32, 32]
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/cifar10/edm-cifar10-32x32-cond-vp.pth"

    # Three encoder and two decoder feature levels exposed by EDMPrecond.
    config.model.feature_indices = [0, 1, 2, 3, 4]
    config.model.drift_radii = [0.02, 0.05, 0.2]
    config.model.net_optimizer.lr = 2e-6
    config.model.net_optimizer.weight_decay = 0.01
    config.model.precision_amp = "float16"
    config.model.precision_amp_infer = "float16"
    config.model.grad_scaler_enabled = True

    config.model.use_ema = ["ema_9999"]
    config.trainer.callbacks = DictConfig(
        {key: value for key, value in config.trainer.callbacks.items() if not key.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_CONST_CALLBACKS)
    config.dataloader_train = CIFAR10_Loader_Config
    config.dataloader_train.batch_size = 32
    config.trainer.batch_size_global = 256
    config.trainer.max_iter = 10000
    config.trainer.callbacks.grad_clip.grad_norm = 10.0
    config.log_config.group = "edm_cifar10_tfd"
    return config
