"""Wan2.2 TI2V-5B TFD quick experiment on 10k videos and 17 frames."""

from fastgen.configs.experiments.WanI2V.config_tfd_wan22_5b import (
    create_config as create_full_config,
)


def create_config():
    config = create_full_config()
    config.model.input_shape = [48, 5, 24, 40]
    config.dataloader_train.index_path = (
        "/mnt/dataset/cosmos3-dataset/data_index/webdata/"
        "all_final_selected_sixth_10k_f17_seed10.csv"
    )
    config.dataloader_train.dataset_size = 10_000
    config.dataloader_train.sequence_length = 17
    config.dataloader_train.img_size = (640, 384)
    config.log_config.group = "wan22_5b_ti2v_tfd_10k_f17"
    return config
