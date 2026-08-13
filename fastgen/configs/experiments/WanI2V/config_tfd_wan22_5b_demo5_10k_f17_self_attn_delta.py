"""Wan2.2 TI2V TFD data-control run on clean demo5 simulation videos."""

from fastgen.configs.experiments.WanI2V.config_tfd_wan22_5b_10k_f17_self_attn_delta import (
    create_config as create_baseline_config,
)


def create_config():
    config = create_baseline_config()
    config.dataloader_train.index_path = (
        "/mnt/dataset/demo5_dataset/manifests/demo5_clean_10k_f49_seed10.csv"
    )
    config.dataloader_train.dataset_size = 10_000
    config.trainer.resume = False
    config.log_config.group = "wan22_5b_ti2v_tfd_demo5_clean"
    config.log_config.name = "wan22_tfd_demo5_clean_self_attn_delta_b8_10_12_s01_f17_10k"
    return config
