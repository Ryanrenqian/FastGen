# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-step DriftWorld initialized from the native-81 DMD2 student."""

from fastgen.configs.experiments.WanI2V.config_dmd2_wan22_5b_demo5_native81 import (
    create_config as create_dmd2_config,
)
from fastgen.configs.experiments.WanI2V.config_driftworld_wan22_5b import (
    create_config as create_driftworld_config,
)


def create_config():
    config = create_driftworld_config()
    dmd2 = create_dmd2_config()

    config.model.input_shape = list(dmd2.model.input_shape)
    config.model.framewise_vae = False
    config.model.use_ema = False
    config.model.generated_samples_per_condition = 2
    config.model.net_optimizer.lr = 3e-8
    config.model.net_scheduler.warm_up_steps = [200]
    config.model.net_scheduler.f_start = [0.01]
    config.model.dinov3_warmup_steps = 1_000
    config.model.dinov3_ramp_steps = 1_000

    config.dataloader_train = dmd2.dataloader_train
    config.dataloader_train.batch_size = 1
    config.trainer.checkpointer.pretrained_ckpt_path = ""
    config.trainer.checkpointer.pretrained_ckpt_key_map = {"net": "net"}
    config.trainer.resume = False
    config.trainer.max_iter = 10_001
    config.log_config.group = "wan22_5b_ti2v_dmd2_driftworld_native81"
    config.log_config.name = (
        "wan22_ti2v5b_dmd2_to_driftworld_1step_"
        "demo5_1280x704_nativevae_f81_n2"
    )
    return config
