"""MSA-TFD with motion and high-level semantic teacher features."""

from fastgen.configs.experiments.WanI2V.config_tfd_wan22_5b_10k_f17_self_attn_delta import (
    create_config as create_motion_config,
)


def create_config():
    config = create_motion_config()
    semantic_weight = 0.15
    motion_indices = [8, 10, 12]
    semantic_indices = [20, 24]

    config.model.feature_indices = motion_indices + semantic_indices
    config.model.feature_taps = [
        "self_attn_delta",
        "self_attn_delta",
        "self_attn_delta",
        "block_output",
        "block_output",
    ]
    # Preserve the previous motion-loss scale. The semantic branch contributes
    # 0.15 times the mean motion branch: sum(motion) + 0.15*3*mean(semantic).
    semantic_layer_weight = (
        semantic_weight * len(motion_indices) / len(semantic_indices)
    )
    config.model.feature_loss_weights = [1.0] * len(motion_indices) + [
        semantic_layer_weight
    ] * len(semantic_indices)

    config.log_config.group = "wan22_5b_ti2v_msa_tfd_motion_semantic"
    config.log_config.name = "msa_tfd_motion_semantic015_b8_10_12_b20_24_f17_2k"
    return config
