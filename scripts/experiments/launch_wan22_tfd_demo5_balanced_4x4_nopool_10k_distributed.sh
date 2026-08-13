#!/usr/bin/env bash
# Distributed 10k-step rerun of msa_tfd_balanced_4x4_nopool_staticw1_f17_2k
# on the clean demo5 simulation manifest. DLC supplies WORLD_SIZE, RANK,
# MASTER_ADDR, MASTER_PORT, and GPU_NUM_PER_NODE on every node.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CONFIG="${CONFIG:-fastgen/configs/experiments/WanI2V/config_tfd_wan22_5b_10k_f17_balanced_4x4_nopool.py}"
export DATA_INDEX="${DATA_INDEX:-/mnt/dataset/demo5_dataset/manifests/demo5_clean_10k_f49_seed10.csv}"
export DATASET_SIZE="${DATASET_SIZE:-10000}"
export IMAGE_WIDTH="${IMAGE_WIDTH:-640}"
export IMAGE_HEIGHT="${IMAGE_HEIGHT:-384}"
export SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-17}"
export MAX_ITER="${MAX_ITER:-10001}"
export LOGGING_ITER="${LOGGING_ITER:-100}"
export SAVE_CKPT_ITER="${SAVE_CKPT_ITER:-1000}"
export RUN_NAME="${RUN_NAME:-msa_tfd_balanced_4x4_nopool_staticw1_f17_demo5_clean_10k}"
export RESUME="${RESUME:-False}"
export WANDB_MODE="${WANDB_MODE:-online}"

exec "$HERE/scripts/launch_dlc_wan22_tfd_multinode.sh" \
  model.generated_samples_per_condition=4 \
  model.positive_samples_per_condition=4 \
  model.anchor_samples_per_condition=4 \
  model.teacher_positive_fill=False \
  model.static_negative_enabled=True \
  model.static_negative_guidance_scale=1.3333333333333333 \
  model.feature_pool_size=1 \
  dataloader_train.positive_frame_strides=[1] \
  dataloader_train.positive_random_walk_count=3 \
  dataloader_train.positive_random_step_min=1 \
  dataloader_train.positive_random_step_max=3 \
  "$@"
