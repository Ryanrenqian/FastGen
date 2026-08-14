#!/usr/bin/env bash
# Aliyun DLC multi-node launcher for Wan2.2 TI2V-5B TFD.
#
# DLC injects node-level topology before torchrun starts:
#   WORLD_SIZE = number of nodes
#   RANK       = node rank (0..WORLD_SIZE-1)
#   MASTER_ADDR / MASTER_PORT = rendezvous endpoint
#
# Submit the same command on every node:
#   bash /mnt/home/renqian/oneNFE/FastGen-tfd/scripts/launch_dlc_wan22_tfd_multinode.sh
set -uo pipefail

hard_nofile="$(ulimit -Hn 2>/dev/null || echo 65536)"
[[ "$hard_nofile" == "unlimited" ]] && hard_nofile=1048576
ulimit -n "$hard_nofile" 2>/dev/null || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HERE/.conda/envs/fastgen/bin/python}"
CONFIG="${CONFIG:-fastgen/configs/experiments/WanI2V/config_tfd_wan22_5b_10k_f17_stride123.py}"
if [[ "$CONFIG" == /* ]]; then
  CONFIG_PATH="$CONFIG"
else
  CONFIG_PATH="$HERE/$CONFIG"
fi
# FastGen historically converts config paths to module names. Pass a path
# relative to repo_root when possible so an absolute /mnt/... path never turns
# into the invalid relative module name `.mnt....`.
if [[ "$CONFIG_PATH" == "$HERE/"* ]]; then
  CONFIG_ARG="${CONFIG_PATH#"$HERE/"}"
else
  CONFIG_ARG="$CONFIG_PATH"
fi

# Capture DLC node-level values before torchrun replaces RANK/WORLD_SIZE with
# process-level values inside workers.
NNODES="${WORLD_SIZE:-1}"
NODE_RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23456}"
GPU_PER_NODE="${GPU_NUM_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}"
[[ "$GPU_PER_NODE" =~ ^[1-9][0-9]*$ ]] || GPU_PER_NODE=8

for value_name in NNODES NODE_RANK MASTER_PORT; do
  value="${!value_name}"
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "[launch] ERROR: $value_name must be an integer, got '$value'" >&2; exit 2; }
done
(( NNODES >= 1 )) || { echo "[launch] ERROR: NNODES must be positive" >&2; exit 2; }
(( NODE_RANK < NNODES )) || { echo "[launch] ERROR: NODE_RANK=$NODE_RANK is outside 0..$((NNODES-1))" >&2; exit 2; }
if (( NNODES > 1 )) && [[ "$MASTER_ADDR" == "127.0.0.1" || "$MASTER_ADDR" == "localhost" ]]; then
  echo "[launch] ERROR: multi-node training requires DLC MASTER_ADDR, not $MASTER_ADDR" >&2
  exit 2
fi

# Shared paths. The model cache, output, code, and Conda environment must be mounted at the
# same absolute paths on every node.
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/mnt/home/renqian/imgGen/runtime/fastgen}"
export DATA_ROOT_DIR="${DATA_ROOT_DIR:-/mnt/home/renqian/imgGen/runtime/DATA}"
export CKPT_ROOT_DIR="${CKPT_ROOT_DIR:-/mnt/home/renqian/imgGen/runtime/MODEL}"
export HF_HOME="${HF_HOME:-$CKPT_ROOT_DIR/huggingface/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-true}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_NETRC="${WANDB_NETRC:-$HERE/.secrets/wandb.netrc}"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# FastGen reads NCCL_TIMEOUT when creating its default process group.
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"

NET_BACKEND="${NET_BACKEND:-ib}"
if [[ "$NET_BACKEND" == "tcp" ]]; then
  export NCCL_IB_DISABLE=1
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth}"
elif [[ "$NET_BACKEND" == "ib" ]]; then
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
  export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
  export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-23}"
  export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"
else
  echo "[launch] ERROR: NET_BACKEND must be 'ib' or 'tcp', got '$NET_BACKEND'" >&2
  exit 2
fi

# Training defaults. Width and height must be multiples of 64 because Wan's
# spatial VAE ratio is 16 and TFD pools teacher features by another factor 4.
IMAGE_WIDTH="${IMAGE_WIDTH:-640}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-384}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-17}"
MAX_ITER="${MAX_ITER:-100001}"
LOGGING_ITER="${LOGGING_ITER:-100}"
SAVE_CKPT_ITER="${SAVE_CKPT_ITER:-1000}"
DATASET_SIZE="${DATASET_SIZE:-10000}"
RUN_NAME="${RUN_NAME:-wan22_tfd_10k_f17_s10}"
RESUME="${RESUME:-True}"

for value_name in IMAGE_WIDTH IMAGE_HEIGHT SEQUENCE_LENGTH MAX_ITER LOGGING_ITER SAVE_CKPT_ITER DATASET_SIZE; do
  value="${!value_name}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "[launch] ERROR: $value_name must be positive, got '$value'" >&2; exit 2; }
done
(( IMAGE_WIDTH % 64 == 0 && IMAGE_HEIGHT % 64 == 0 )) || {
  echo "[launch] ERROR: IMAGE_WIDTH/IMAGE_HEIGHT must be divisible by 64" >&2; exit 2; }
(( (SEQUENCE_LENGTH - 1) % 4 == 0 )) || {
  echo "[launch] ERROR: SEQUENCE_LENGTH must satisfy (T-1) % 4 == 0" >&2; exit 2; }

LATENT_T=$(( (SEQUENCE_LENGTH - 1) / 4 + 1 ))
LATENT_H=$(( IMAGE_HEIGHT / 16 ))
LATENT_W=$(( IMAGE_WIDTH / 16 ))

# HSDP: shard student and teacher within each node, replicate across nodes.
# Set FSDP_SHARD_GROUP_SIZE=0 to request full sharding over all ranks.
FSDP_SHARD_GROUP_SIZE="${FSDP_SHARD_GROUP_SIZE:-$GPU_PER_NODE}"
FSDP_OPT=()
if [[ "$FSDP_SHARD_GROUP_SIZE" != "0" ]]; then
  [[ "$FSDP_SHARD_GROUP_SIZE" =~ ^[1-9][0-9]*$ ]] || {
    echo "[launch] ERROR: FSDP_SHARD_GROUP_SIZE must be a positive integer or 0" >&2; exit 2; }
  (( NNODES * GPU_PER_NODE % FSDP_SHARD_GROUP_SIZE == 0 )) || {
    echo "[launch] ERROR: total ranks must be divisible by FSDP_SHARD_GROUP_SIZE" >&2; exit 2; }
  FSDP_OPT+=("trainer.fsdp_sharding_group_size=$FSDP_SHARD_GROUP_SIZE")
fi

DATA_INDEX="${DATA_INDEX:-/mnt/dataset/cosmos3-dataset/data_index/webdata/all_final_selected_sixth_10k_f49_seed10.csv}"
MODEL_CACHE="$HF_HOME/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers"
[[ -x "$PY" ]] || { echo "[node $NODE_RANK] ERROR: missing project Conda Python: $PY" >&2; exit 1; }
[[ -f "$CONFIG_PATH" ]] || { echo "[node $NODE_RANK] ERROR: missing config: $CONFIG_PATH" >&2; exit 1; }
[[ -f "$DATA_INDEX" ]] || { echo "[node $NODE_RANK] ERROR: missing data index: $DATA_INDEX" >&2; exit 1; }
[[ -d "$MODEL_CACHE" ]] || { echo "[node $NODE_RANK] ERROR: missing offline model cache: $MODEL_CACHE" >&2; exit 1; }
if [[ "$WANDB_MODE" == "online" ]]; then
  [[ -r "$WANDB_NETRC" ]] || {
    echo "[node $NODE_RANK] ERROR: W&B online mode requires readable netrc: $WANDB_NETRC" >&2
    exit 1
  }
  "$PY" - "$WANDB_NETRC" <<'PY' || exit 1
import netrc
import sys

path = sys.argv[1]
try:
    credentials = netrc.netrc(path).authenticators("api.wandb.ai")
except (OSError, netrc.NetrcParseError) as error:
    raise SystemExit(f"Invalid W&B netrc {path}: {error}")
if not credentials or not credentials[2]:
    raise SystemExit(f"W&B netrc has no api.wandb.ai credential: {path}")
print(f"[launch] W&B credential preflight passed: {path}")
PY
fi

# Fail once per node before torchrun fans out to all local GPUs. This catches
# bad config paths/imports without producing one traceback per rank.
(
  cd "$HERE"
  "$PY" -c \
    'import sys; from fastgen.configs.config_utils import import_config_from_python_file; import_config_from_python_file(sys.argv[1])' \
    "$CONFIG_ARG"
) || { echo "[node $NODE_RANK] ERROR: config preflight failed: $CONFIG_ARG" >&2; exit 1; }

LOG_DIR="${LOG_DIR:-$FASTGEN_OUTPUT_ROOT/launch_logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/wan22_tfd_dlc_${NNODES}n${GPU_PER_NODE}g_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"

OPTS=(
  "trainer.fsdp=True"
  "trainer.seed=10"
  "trainer.max_iter=$MAX_ITER"
  "trainer.logging_iter=$LOGGING_ITER"
  "trainer.save_ckpt_iter=$SAVE_CKPT_ITER"
  "trainer.resume=$RESUME"
  "log_config.name=$RUN_NAME"
  "log_config.wandb_mode=$WANDB_MODE"
  "log_config.wandb_netrc=$WANDB_NETRC"
  "model.teacher_positive_fill=False"
  "model.positive_samples_per_condition=3"
  "model.anchor_samples_per_condition=3"
  "model.input_shape=[48,$LATENT_T,$LATENT_H,$LATENT_W]"
  "dataloader_train.index_path=$DATA_INDEX"
  "dataloader_train.dataset_size=$DATASET_SIZE"
  "dataloader_train.sequence_length=$SEQUENCE_LENGTH"
  "dataloader_train.positive_frame_strides=[1,2,3]"
  "dataloader_train.img_size=[$IMAGE_WIDTH,$IMAGE_HEIGHT]"
  "${FSDP_OPT[@]}"
)

# Remove the empty array expansion produced by full-shard mode. Extra CLI
# overrides supplied after the script name take precedence over these defaults.
FINAL_OPTS=()
for opt in "${OPTS[@]}"; do [[ -n "$opt" ]] && FINAL_OPTS+=("$opt"); done
FINAL_OPTS+=("$@")

echo "[launch] node=$NODE_RANK/$((NNODES-1)) gpus/node=$GPU_PER_NODE total_gpus=$((NNODES*GPU_PER_NODE))"
echo "[launch] master=$MASTER_ADDR:$MASTER_PORT backend=$NET_BACKEND shard_group=${FSDP_SHARD_GROUP_SIZE}"
echo "[launch] video=${SEQUENCE_LENGTH}x${IMAGE_WIDTH}x${IMAGE_HEIGHT} latent=48x${LATENT_T}x${LATENT_H}x${LATENT_W}"
echo "[launch] config=$CONFIG_ARG python=$PY"
echo "[launch] run=$RUN_NAME max_iter=$MAX_ITER resume=$RESUME log=$LOG_FILE"

CMD=(
  "$PY" -m torch.distributed.run
  "--nnodes=$NNODES"
  "--node_rank=$NODE_RANK"
  "--nproc_per_node=$GPU_PER_NODE"
  "--master_addr=$MASTER_ADDR"
  "--master_port=$MASTER_PORT"
  "$HERE/train.py"
  "--config=$CONFIG_ARG"
  -
  "${FINAL_OPTS[@]}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[dry-run] %q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

cd "$HERE"
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
exit_code=${PIPESTATUS[0]}
echo "[launch] node $NODE_RANK exited with code $exit_code"
exit "$exit_code"
