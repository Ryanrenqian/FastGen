"""Motion-semantic MSA-TFD with condition-shared feature noise."""

from fastgen.configs.experiments.WanI2V.config_tfd_wan22_5b_10k_f17_motion_semantic import (
    create_config as create_motion_semantic_config,
)


def create_config():
    config = create_motion_semantic_config()
    config.model.feature_shared_noise_per_condition = True
    config.log_config.group = (
        "wan22_5b_ti2v_msa_tfd_motion_semantic_shared_noise"
    )
    config.log_config.name = (
        "msa_tfd_motion_semantic015_shared_noise_s01_f17_2k"
    )
    return config
