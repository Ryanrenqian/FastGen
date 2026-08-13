#!/usr/bin/env bash
# Data-only control: HYP-12 Wan TFD recipe trained on clean demo5 simulations.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CONFIG="${CONFIG:-fastgen/configs/experiments/WanI2V/config_tfd_wan22_5b_demo5_10k_f17_self_attn_delta.py}"
export DATA_INDEX="${DATA_INDEX:-/mnt/dataset/demo5_dataset/manifests/demo5_clean_10k_f49_seed10.csv}"
export DATASET_SIZE="${DATASET_SIZE:-10000}"
export MAX_ITER="${MAX_ITER:-10001}"
export LOGGING_ITER="${LOGGING_ITER:-100}"
export SAVE_CKPT_ITER="${SAVE_CKPT_ITER:-1000}"
export RUN_NAME="${RUN_NAME:-wan22_tfd_demo5_clean_self_attn_delta_b8_10_12_s01_f17_10k}"
export RESUME="${RESUME:-False}"

exec "$HERE/scripts/experiments/launch_wan22_tfd_self_attn_delta_8_10_12.sh" "$@"
