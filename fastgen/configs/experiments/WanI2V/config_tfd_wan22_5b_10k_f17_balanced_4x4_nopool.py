"""Extreme no-spatial-pooling control for balanced 4:4 MSA-TFD."""

from fastgen.configs.experiments.WanI2V.config_tfd_wan22_5b_10k_f17_balanced_4x4 import (
    create_config as create_balanced_config,
)


def create_config():
    config = create_balanced_config()
    config.model.feature_pool_size = 1
    config.log_config.group = "wan22_5b_ti2v_msa_tfd_balanced_4x4_nopool"
    config.log_config.name = "msa_tfd_balanced_4x4_nopool_staticw1_f17_2k"
    return config
