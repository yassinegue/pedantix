#!/bin/bash
#SBATCH --job-name=pedantix-sft
#SBATCH --partition=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/yguenn/pedantix/logs/sft-%j.out
# NOTE: --container-image must be on the CLI, not here (#SBATCH strips the #).
# Submit with:
#   sbatch --container-image='nvcr.io#nvidia/pytorch:25.06-py3' scripts/h100_sft_warmstart.sbatch

# ── paths ─────────────────────────────────────────────────────────────────────
PROJ=/home/yguenn/pedantix
export HF_HOME=$PROJ/models/hf_cache
export HF_HUB_CACHE=$PROJ/models/hf_cache/hub
export TRANSFORMERS_CACHE=$PROJ/models/hf_cache/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p $PROJ/logs $PROJ/models/hf_cache

# ── install deps ───────────────────────────────────────────────────────────────
echo "[setup] installing/checking Python deps..."
pip install --quiet --upgrade --break-system-packages \
  "transformers>=4.51.0" \
  "peft>=0.14.0" \
  "accelerate>=0.34.0" \
  "trl>=0.12.0" \
  "torchao>=0.16.0" \
  tqdm einops sentencepiece 2>&1 | tail -3

# ── verify GPU ─────────────────────────────────────────────────────────────────
python -c "
import torch
print(f'[gpu] CUDA={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')
print(f'[gpu] VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"

# ── launch SFT ─────────────────────────────────────────────────────────────────
export PYTHONPATH=$PROJ:${PYTHONPATH}
cd $PROJ 2>/dev/null || echo "[warn] cd $PROJ failed — using PYTHONPATH"

# Required for SFTTrainer/accelerate on single-GPU without torchrun
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

RUN_NAME="${RUN_NAME:-v14_sft_warmstart}"
MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
TRAIN_DATA="${TRAIN_DATA:-$PROJ/data/sft_deepseek_nochain_20260526.jsonl}"
OUT_DIR="$PROJ/models/$RUN_NAME"
mkdir -p "$OUT_DIR" && chown -R yguenn "$OUT_DIR" 2>/dev/null || true

echo "[sft] run=$RUN_NAME model=$MODEL data=$TRAIN_DATA output=$OUT_DIR"
echo "[sft] $(date)"
echo "[sft] 5206 examples, batch=1, grad_accum=8 → 650 steps, ~5-6h"

python -u -m pedantix_project.cli llm-sft \
  --train           "$TRAIN_DATA" \
  --model           "$MODEL" \
  --tiny-model      $PROJ/models/tiny_model.json \
  --output-dir      "$OUT_DIR" \
  --max-steps       50 \
  --batch-size      1 \
  --gradient-accumulation-steps 8 \
  --learning-rate   2e-5 \
  --lora-rank       16 \
  --max-seq-length  768 \
  --save-steps      50 \
  --save-total-limit 2 \
  --eval-every      50 \
  --eval-corpus     $PROJ/data/filtered_pages.jsonl \
  --eval-pages      10 \
  --eval-max-game-steps 20 \
  --eval-chat-format qwen \
  --eval-num-generations 1 \
  --eval-batch-size 1 \
  --seed            42

echo "[sft] done at $(date)"
echo "[sft] output: $OUT_DIR"
echo "[sft] next: RUN_NAME=v15_grpo_sft_warmstart MODEL=$OUT_DIR sbatch ..."
