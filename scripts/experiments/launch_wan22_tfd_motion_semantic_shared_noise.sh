#!/usr/bin/env bash
# MSA-TFD motion+semantic experiment with condition-shared feature noise.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CONFIG="${CONFIG:-fastgen/configs/experiments/WanI2V/config_tfd_wan22_5b_10k_f17_motion_semantic_shared_noise.py}"
export MAX_ITER="${MAX_ITER:-2001}"
export LOGGING_ITER="${LOGGING_ITER:-100}"
export SAVE_CKPT_ITER="${SAVE_CKPT_ITER:-500}"
export RUN_NAME="${RUN_NAME:-msa_tfd_motion_semantic015_shared_noise_s01_f17_2k}"
export RESUME="${RESUME:-False}"

exec "$HERE/scripts/experiments/launch_wan22_tfd_motion_semantic.sh" \
  model.feature_shared_noise_per_condition=True \
  "$@"
