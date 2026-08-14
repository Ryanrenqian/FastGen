#!/usr/bin/env bash
# MSA-TFD: motion self-attention branch plus 0.15 high-level semantic branch.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CONFIG="${CONFIG:-fastgen/configs/experiments/WanI2V/config_tfd_wan22_5b_10k_f17_motion_semantic.py}"
export MAX_ITER="${MAX_ITER:-2001}"
export LOGGING_ITER="${LOGGING_ITER:-100}"
export SAVE_CKPT_ITER="${SAVE_CKPT_ITER:-500}"
export RUN_NAME="${RUN_NAME:-msa_tfd_motion_semantic015_b8_10_12_b20_24_f17_2k}"
export RESUME="${RESUME:-False}"

exec "$HERE/scripts/launch_dlc_wan22_tfd_multinode.sh" \
  model.feature_indices=[8,10,12,20,24] \
  model.feature_taps=[self_attn_delta,self_attn_delta,self_attn_delta,block_output,block_output] \
  model.feature_loss_weights=[1.0,1.0,1.0,0.225,0.225] \
  model.feature_noise_sigma=0.1 \
  model.feature_mask_first_temporal_slot=True \
  model.teacher_positive_fill=False \
  model.positive_samples_per_condition=3 \
  model.anchor_samples_per_condition=3 \
  dataloader_train.positive_frame_strides=[1,2,3] \
  "$@"
