#!/usr/bin/env bash
# HYP-21 matched A/B: random trajectories vs exact stride-1 copies.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AB_ARM="${AB_ARM:-control}"

case "$AB_ARM" in
  control)
    default_config="fastgen/configs/experiments/WanI2V/config_tfd_wan22_5b_demo5_nopool_ab_control.py"
    default_run="hyp21_control_random123_demo5_nopool_f17_2k"
    ;;
  treatment)
    default_config="fastgen/configs/experiments/WanI2V/config_tfd_wan22_5b_demo5_nopool_ab_treatment.py"
    default_run="hyp21_treatment_repeat_stride1_demo5_nopool_f17_2k"
    ;;
  *)
    echo "[launch] ERROR: AB_ARM must be control or treatment, got '$AB_ARM'" >&2
    exit 2
    ;;
esac

export CONFIG="${CONFIG:-$default_config}"
export DATA_INDEX="${DATA_INDEX:-/mnt/dataset/demo5_dataset/manifests/demo5_clean_10k_f49_seed10.csv}"
export DATASET_SIZE="${DATASET_SIZE:-10000}"
export IMAGE_WIDTH="${IMAGE_WIDTH:-640}"
export IMAGE_HEIGHT="${IMAGE_HEIGHT:-384}"
export SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-17}"
export MAX_ITER="${MAX_ITER:-2001}"
export LOGGING_ITER="${LOGGING_ITER:-100}"
export SAVE_CKPT_ITER="${SAVE_CKPT_ITER:-500}"
export RUN_NAME="${RUN_NAME:-$default_run}"
export RESUME="${RESUME:-False}"
export WANDB_MODE="${WANDB_MODE:-online}"

exec "$HERE/scripts/launch_dlc_wan22_tfd_multinode.sh" "$@"
