# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import attrs
from omegaconf import DictConfig

from fastgen.configs.config import BaseConfig, BaseModelConfig
from fastgen.configs.callbacks import (
    GPUStats_CALLBACK,
    GradClip_CALLBACK,
    ParamCount_CALLBACK,
    TrainProfiler_CALLBACK,
    WANDB_CALLBACK,
)
from fastgen.methods import TFDModel
from fastgen.utils import LazyCall as L


@attrs.define(slots=False)
class ModelConfig(BaseModelConfig):
    feature_layers: list[str] = attrs.field(
        factory=lambda: ["enc:6", "enc:11", "bottleneck", "dec:7", "dec:12"]
    )
    feature_pool_size: int = 4
    drift_radii: list[float] = attrs.field(factory=lambda: [0.02, 0.05, 0.1, 0.2])
    feature_noise_p_mean: float = -1.2
    feature_noise_p_std: float = 1.2
    feature_noise_sigma_min: float = 0.02
    feature_noise_sigma_max: float = 0.1
    feature_noise_trunc_resamples: int = 8

    generated_samples_per_condition: int = 4
    positive_samples_per_condition: int = 4
    anchor_samples_per_condition: int = 4
    teacher_generated_checkpoint: bool = False
    teacher_reference_chunk_size: int = 0
    conditioning_sigma: float = 80.0

    anchor_weight: float = 1.0
    anchor_temperature: float = 1.0
    anchor_margin: float = 0.5
    anchor_bandwidth: float = 0.0


@attrs.define(slots=False)
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = L(TFDModel)(config=None)


def create_config():
    config = Config()
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **GPUStats_CALLBACK,
            **TrainProfiler_CALLBACK,
            **ParamCount_CALLBACK,
            **WANDB_CALLBACK,
        }
    )
    config.model.net_scheduler.warm_up_steps = [500]
    return config
