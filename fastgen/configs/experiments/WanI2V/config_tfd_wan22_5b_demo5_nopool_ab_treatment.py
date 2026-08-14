"""HYP-21 treatment: no-pool TFD with four identical trajectories."""

from fastgen.configs.experiments.WanI2V.config_tfd_wan22_5b_demo5_nopool_ab_control import (
    create_config as create_control_config,
)


def create_config():
    config = create_control_config()
    config.dataloader_train.positive_random_walk_count = 0
    config.dataloader_train.positive_stride1_repeat_count = 3
    config.log_config.name = "hyp21_treatment_repeat_stride1_demo5_nopool_f17_2k"
    return config
