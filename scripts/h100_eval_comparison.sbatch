#!/bin/bash
#SBATCH --job-name=pedantix-eval
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/home/yguenn/pedantix/logs/eval-%j.out
# Sharded eval: trained v16 ckpt-200, qwen, 100 steps.
# Submit 5 shards in parallel:
#   for i in 0 1 2 3 4; do sbatch --container-image='nvcr.io#nvidia/pytorch:25.06-py3' --export=ALL,SHARD_ID=$i,NUM_SHARDS=5 scripts/h100_eval_comparison.sbatch; done

PROJ=/home/yguenn/pedantix
export HF_HOME=$PROJ/models/hf_cache
export HF_HUB_CACHE=$PROJ/models/hf_cache/hub
export TRANSFORMERS_CACHE=$PROJ/models/hf_cache/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p $PROJ/logs $PROJ/models/hf_cache $PROJ/models/eval_comparison

echo "[setup] installing deps..."
pip install --quiet --upgrade --break-system-packages \
  "transformers>=4.51.0" \
  "peft>=0.14.0" \
  "accelerate>=0.34.0" \
  "torchao>=0.16.0" \
  einops sentencepiece 2>&1 | tail -3

python -c "
import torch
print(f'[gpu] CUDA={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')
print(f'[gpu] VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"

export PYTHONPATH=$PROJ:${PYTHONPATH}
export SHARD_ID=${SHARD_ID:-0}
export NUM_SHARDS=${NUM_SHARDS:-5}

echo "[eval] shard ${SHARD_ID}/${NUM_SHARDS} starting at $(date)"
python -u -m pedantix_project.eval_comparison

echo "[eval] shard ${SHARD_ID}/${NUM_SHARDS} done at $(date)"
