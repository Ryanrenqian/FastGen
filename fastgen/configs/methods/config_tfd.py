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
    feature_indices: list[int] = attrs.field(factory=lambda: [0])
    feature_noise_level: float = 0.1
    feature_pool_size: int = 4
    normalize_features: bool = True
    drift_radii: list[float] = attrs.field(factory=lambda: [0.02, 0.05, 0.2])

    generated_samples_per_condition: int = 4
    positive_samples_per_condition: int = 4
    positive_bank_size: int = 4

    anchor_weight: float = 1.0
    anchor_temperature: float = 1.0
    anchor_margin: float = 0.5


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
