#!/usr/bin/env bash
# Balanced 4 generated : 4 positive MSA-TFD experiment.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CONFIG="${CONFIG:-fastgen/configs/experiments/WanI2V/config_tfd_wan22_5b_10k_f17_balanced_4x4.py}"
export MAX_ITER="${MAX_ITER:-2001}"
export LOGGING_ITER="${LOGGING_ITER:-100}"
export SAVE_CKPT_ITER="${SAVE_CKPT_ITER:-500}"
export RUN_NAME="${RUN_NAME:-msa_tfd_balanced_4x4_random123_shared_noise_f17_2k}"
export RESUME="${RESUME:-False}"

exec "$HERE/scripts/experiments/launch_wan22_tfd_motion_semantic_shared_noise.sh" \
  model.generated_samples_per_condition=4 \
  model.positive_samples_per_condition=4 \
  model.anchor_samples_per_condition=4 \
  model.teacher_positive_fill=False \
  model.static_negative_enabled=True \
  model.static_negative_guidance_scale=1.3333333333333333 \
  dataloader_train.positive_frame_strides=[1] \
  dataloader_train.positive_random_walk_count=3 \
  dataloader_train.positive_random_step_min=1 \
  dataloader_train.positive_random_step_max=3 \
  "$@"
