#!/usr/bin/env bash
set -euo pipefail

# MI250X note:
# - Each MI250X has 2 GCDs; ROCm exposes each GCD as a "GPU" to PyTorch.
# - On an 8x MI250X node, you typically see 16 devices (0..15).
# - We therefore launch with nproc_per_node=16.

# -----------------------------------------------------------------------------
# ROCm environment knobs (safe defaults; adjust for your cluster)
# -----------------------------------------------------------------------------

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# If your container image was built for a different gfx target you MAY need this.
# MI250X is gfx90a.
# export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-"9.0.0"}

# RCCL uses NCCL_* env vars.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}

# Reduce fragmentation on long runs (often helpful on ROCm).
export PYTORCH_HIP_ALLOC_CONF=${PYTORCH_HIP_ALLOC_CONF:-"expandable_segments:True"}

# HF datasets: The Pile loader needs remote code.
export HF_DATASETS_TRUST_REMOTE_CODE=${HF_DATASETS_TRUST_REMOTE_CODE:-1}

# Point HF caches to fast local storage if you have it.
export HF_HOME=${HF_HOME:-"/data/hf"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"/data/hf/datasets"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"/data/hf/transformers"}

# Ensure we can import this repo as a module
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

CONFIG=${1:-"$REPO_ROOT/configs/qwen2_5_0_5b_moe_pile_pretrain_fsdp.yaml"}

echo "[launch] repo_root=$REPO_ROOT"
echo "[launch] config=$CONFIG"

cd "$REPO_ROOT"

# Choose a free port on your system if 29500 is in use.
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"29500"}

# torchtune's CLI wraps torchrun and forwards args to the recipe.
tune run \
  --nnodes 1 \
  --nproc_per_node 16 \
  recipes/moe_pretrain_distributed.py \
  --config "$CONFIG"

# --- Alternative: torchrun ---
# torchrun --nnodes=1 --nproc_per_node=16 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
#   recipes/moe_pretrain_distributed.py --config "$CONFIG"
