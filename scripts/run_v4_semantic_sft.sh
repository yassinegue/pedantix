#!/bin/bash
# SFT run with semantic soft labels (alpha=0.3) — comparison against v4 hard-CE run
# Same data, same hyperparams as v4; only difference is the loss function.
# 50% of training examples have semantic smoothing; the rest fall back to hard CE.
# Expected: similar loss curve but more robustness to near-synonyms at inference time.

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="/tmp/sft_v4_semantic_4b_${TIMESTAMP}.log"

echo "Launching semantic-smoothing SFT (v4-semantic) — log: $LOG"

nohup conda run -n py310 --no-capture-output \
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u -m pedantix_project.cli llm-sft \
    --train data/sft_poc_chat.jsonl \
    --model Qwen/Qwen3-4B \
    --output-dir models/v4_sft_semantic_4b_v2 \
    --max-steps 600 \
    --batch-size 1 \
    --gradient-accumulation-steps 32 \
    --learning-rate 2e-5 \
    --lora-rank 16 \
    --seed 42 \
    --log "models/v4_sft_semantic_4b_v2/train_log.json" \
    --tiny-model models/tiny_model.json \
    --eval-every 50 \
    --eval-pages 50 \
    --eval-corpus data/clean_pages.jsonl \
    --eval-max-game-steps 30 \
    --eval-chat-format qwen \
    --eval-num-generations 4 \
    --eval-batch-size 4 \
    --save-steps 100 \
    --save-total-limit 3 \
    --semantic-smoothing \
    --semantic-smoothing-alpha 0.3 \
    --fasttext-model models/fasttext_wiki_model.npz \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "Tail log: tail -f $LOG"
