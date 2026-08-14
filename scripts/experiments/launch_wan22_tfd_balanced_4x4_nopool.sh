#!/usr/bin/env bash
# Extreme feature-pooling control: preserve full Wan teacher feature grid.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CONFIG="${CONFIG:-fastgen/configs/experiments/WanI2V/config_tfd_wan22_5b_10k_f17_balanced_4x4_nopool.py}"
export MAX_ITER="${MAX_ITER:-2001}"
export LOGGING_ITER="${LOGGING_ITER:-100}"
export SAVE_CKPT_ITER="${SAVE_CKPT_ITER:-500}"
export RUN_NAME="${RUN_NAME:-msa_tfd_balanced_4x4_nopool_staticw1_f17_2k}"
export RESUME="${RESUME:-False}"

exec "$HERE/scripts/experiments/launch_wan22_tfd_balanced_4x4.sh" \
  model.feature_pool_size=1 \
  "$@"
