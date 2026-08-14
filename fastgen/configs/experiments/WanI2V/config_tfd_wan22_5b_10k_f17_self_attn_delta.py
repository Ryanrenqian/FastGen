"""Wan2.2 TI2V-5B TFD with motion-sensitive self-attention features."""

from fastgen.configs.experiments.WanI2V.config_tfd_wan22_5b_10k_f17_stride123_maskfirst import (
    create_config as create_base_config,
)


def create_config():
    config = create_base_config()
    config.model.feature_tap = "self_attn_delta"
    config.model.feature_indices = [8, 10, 12]
    config.model.feature_noise_sigma = 0.1
    config.model.feature_mask_first_temporal_slot = True
    config.model.positive_samples_per_condition = 3
    config.model.anchor_samples_per_condition = 3
    config.model.teacher_positive_fill = False
    config.dataloader_train.positive_frame_strides = [1, 2, 3]
    config.trainer.max_iter = 2_001
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500
    config.trainer.resume = False
    config.log_config.group = "wan22_5b_ti2v_tfd_self_attn_delta"
    config.log_config.name = (
        "wan22_tfd_self_attn_delta_b8_10_12_s01_f17_2k"
    )
    return config
