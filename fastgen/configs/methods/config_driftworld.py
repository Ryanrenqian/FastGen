# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default configuration for latent-space DriftWorld training."""

import attrs
from omegaconf import DictConfig

from fastgen.configs.callbacks import (
    EMA_CALLBACK,
    GPUStats_CALLBACK,
    GradClip_CALLBACK,
    ParamCount_CALLBACK,
    TrainProfiler_CALLBACK,
    WANDB_CALLBACK,
)
from fastgen.configs.config import BaseConfig, BaseModelConfig
from fastgen.methods import DriftWorldModel
from fastgen.utils import LazyCall as L


@attrs.define(slots=False)
class ModelConfig(BaseModelConfig):
    # Official DriftWorld evaluates the exponentially averaged student.
    use_ema: bool = True
    framewise_vae: bool = False
    generated_samples_per_condition: int = 4
    drift_radii: list[float] = attrs.field(factory=lambda: [0.02, 0.05])
    mask_conditioning_latent_slot: bool = True
    static_negative_weight: float = 0.0
    use_dinov3_static_negative: bool = False
    compare_temporal_sample_force: bool = False
    local_drift_weight: float = 1.0
    trajectory_drift_weight: float = 0.0
    trajectory_drift_block: tuple[int, int, int] = (1, 1, 1)
    dinov3_drift_weight: float = 0.0
    dinov3_repo_dir: str = ""
    dinov3_weights_path: str = ""
    dinov3_model_name: str = "dinov3_vitb16"
    dinov3_input_size: int = 256
    dinov3_block_indices: tuple[int, ...] = (2, 5, 8)
    dinov3_frame_chunk_size: int = 8
    dinov3_vae_chunk_size: int = 1
    dinov3_motion_alpha: float = 2.5
    dinov3_motion_lambda: float = 12.0
    dinov3_motion_quantile: float = 0.98
    dinov3_motion_threshold: float = 0.35


@attrs.define(slots=False)
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = L(DriftWorldModel)(config=None)


def create_config():
    config = Config()
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **EMA_CALLBACK,
            **GPUStats_CALLBACK,
            **TrainProfiler_CALLBACK,
            **ParamCount_CALLBACK,
            **WANDB_CALLBACK,
        }
    )
    # Match the official Bridge DriftWorld EMA decay.
    config.trainer.callbacks.ema.beta = 0.999
    config.model.net_scheduler.warm_up_steps = [500]
    return config
