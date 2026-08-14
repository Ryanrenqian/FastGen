"""HYP-21 control: no-pool TFD with four mismatched trajectories."""

from fastgen.configs.experiments.WanI2V.config_tfd_wan22_5b_10k_f17_balanced_4x4_nopool import (
    create_config as create_nopool_config,
)


def create_config():
    config = create_nopool_config()
    config.dataloader_train.index_path = (
        "/mnt/dataset/demo5_dataset/manifests/demo5_clean_10k_f49_seed10.csv"
    )
    config.dataloader_train.dataset_size = 10_000
    config.dataloader_train.positive_frame_strides = [1]
    config.dataloader_train.positive_random_walk_count = 3
    config.dataloader_train.positive_stride1_repeat_count = 0
    config.dataloader_train.positive_random_step_min = 1
    config.dataloader_train.positive_random_step_max = 3
    config.dataloader_train.positive_decode_step_max = 3
    config.trainer.max_iter = 2_001
    config.trainer.resume = False
    config.log_config.group = "wan22_5b_ti2v_hyp21_demo5_nopool_ab"
    config.log_config.name = "hyp21_control_random123_demo5_nopool_f17_2k"
    return config
