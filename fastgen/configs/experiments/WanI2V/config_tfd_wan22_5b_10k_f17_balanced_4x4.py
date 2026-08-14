"""Balanced 4-positive/4-generated MSA-TFD experiment."""

from fastgen.configs.experiments.WanI2V.config_tfd_wan22_5b_10k_f17_motion_semantic_shared_noise import (
    create_config as create_shared_noise_config,
)


def create_config():
    config = create_shared_noise_config()
    config.model.generated_samples_per_condition = 4
    config.model.positive_samples_per_condition = 4
    config.model.anchor_samples_per_condition = 4
    config.model.teacher_positive_fill = False
    config.model.static_negative_enabled = True
    # WanTFD maps guidance scale s to fixed-negative weight
    # (s - 1) * (N_generated - 1). With N_generated=4, s=4/3 gives weight 1.
    config.model.static_negative_guidance_scale = 4.0 / 3.0

    # One natural-time reference plus three independently sampled variable-speed
    # trajectories. All four begin at the same TI2V conditioning frame.
    config.dataloader_train.positive_frame_strides = [1]
    config.dataloader_train.positive_random_walk_count = 3
    config.dataloader_train.positive_random_step_min = 1
    config.dataloader_train.positive_random_step_max = 3

    config.log_config.group = "wan22_5b_ti2v_msa_tfd_balanced_4x4"
    config.log_config.name = "msa_tfd_balanced_4x4_random123_shared_noise_f17_2k"
    return config
