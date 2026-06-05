#!/usr/bin/env bash
# Submit trajectory GRPO (qwen-think mode) to the H100 cluster.
# Syncs code+data to the login pod, then submits the sbatch job.
#
# Usage (from harpie):
#   bash scripts/run_h100_grpo.sh
#   bash scripts/run_h100_grpo.sh --dry-run   # print kubectl commands only
set -euo pipefail

YOU="yguenn"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

POD=$(kubectl get pod -n slurm -l "stanford/user=${YOU}" -o jsonpath='{.items[0].metadata.name}')
echo "[h100] login pod: $POD"

REMOTE_HOME="/home/${YOU}"
LOCAL_PROJ="/Data/yassine.guennoun/pedantix"

# ── 1. Sync project onto the cluster ─────────────────────────────────────────
echo "[h100] syncing code and data..."

# Create remote dirs
for dir in pedantix_project data models scripts; do
  kubectl exec -n slurm "$POD" -c login -- \
    runuser -l "$YOU" -c "mkdir -p ${REMOTE_HOME}/pedantix/${dir}"
done

# Python source
kubectl exec -n slurm "$POD" -c login -- \
  runuser -l "$YOU" -c "mkdir -p ${REMOTE_HOME}/pedantix/pedantix_project"
for f in "$LOCAL_PROJ"/pedantix_project/*.py; do
  kubectl cp "$f" "slurm/${POD}:${REMOTE_HOME}/pedantix/pedantix_project/$(basename $f)" -c login
done

# Sbatch script
kubectl cp "$LOCAL_PROJ/scripts/h100_grpo_job.sbatch" \
  "slurm/${POD}:${REMOTE_HOME}/pedantix/scripts/h100_grpo_job.sbatch" -c login

# FastText model (required for reward computation — must use FastTextWikiModel)
kubectl cp "$LOCAL_PROJ/models/fasttext_wiki_model.npz" \
  "slurm/${POD}:${REMOTE_HOME}/pedantix/models/fasttext_wiki_model.npz" -c login
kubectl cp "$LOCAL_PROJ/models/fasttext_wiki_model.json" \
  "slurm/${POD}:${REMOTE_HOME}/pedantix/models/fasttext_wiki_model.json" -c login
kubectl cp "$LOCAL_PROJ/models/tiny_model.json" \
  "slurm/${POD}:${REMOTE_HOME}/pedantix/models/tiny_model.json" -c login

# Data (GRPO training pages + eval pages)
kubectl cp "$LOCAL_PROJ/data/filtered_pages.jsonl" \
  "slurm/${POD}:${REMOTE_HOME}/pedantix/data/filtered_pages.jsonl" -c login
kubectl cp "$LOCAL_PROJ/data/clean_pages.jsonl" \
  "slurm/${POD}:${REMOTE_HOME}/pedantix/data/clean_pages.jsonl" -c login

echo "[h100] sync done"

# ── 2. Submit the job ─────────────────────────────────────────────────────────
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] would submit: sbatch ${REMOTE_HOME}/pedantix/scripts/h100_grpo_job.sbatch"
else
  kubectl exec -n slurm "$POD" -c login -- \
    runuser -l "$YOU" -c \
    "cd ${REMOTE_HOME}/pedantix && sbatch scripts/h100_grpo_job.sbatch"
fi
