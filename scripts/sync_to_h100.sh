#!/usr/bin/env bash
# Sync the minimum needed to run h100_grpo_job.sbatch on the cluster.
# Run from the repo root on harpie, then submit from inside the login pod.
set -euo pipefail

YOU="yguenn"
POD=$(kubectl get pod -n slurm -l "stanford/user=${YOU}" -o jsonpath='{.items[0].metadata.name}')
echo "[sync] pod: $POD"

LOCAL=/Data/yassine.guennoun/pedantix
REMOTE=/home/${YOU}/pedantix

kubectl exec -n slurm "$POD" -c login -- \
  runuser -l "$YOU" -c "mkdir -p ${REMOTE}/{pedantix_project,data,models,scripts,logs}"

# Python source (~2 MB)
echo "[sync] pedantix_project/*.py ..."
for f in "$LOCAL"/pedantix_project/*.py; do
  kubectl cp "$f" "slurm/${POD}:${REMOTE}/pedantix_project/$(basename "$f")" -c login
done

# Job script
echo "[sync] h100_grpo_job.sbatch ..."
kubectl cp "$LOCAL/scripts/h100_grpo_job.sbatch" \
  "slurm/${POD}:${REMOTE}/scripts/h100_grpo_job.sbatch" -c login

# Tiny model (5.9 MB)
echo "[sync] tiny_model.json ..."
kubectl cp "$LOCAL/models/tiny_model.json" \
  "slurm/${POD}:${REMOTE}/models/tiny_model.json" -c login

# FastText model (473 MB — needed for proper semantic reward)
echo "[sync] fasttext_wiki_model.npz (473 MB) ..."
kubectl cp "$LOCAL/models/fasttext_wiki_model.npz" \
  "slurm/${POD}:${REMOTE}/models/fasttext_wiki_model.npz" -c login

# Training + eval pages (101 MB)
echo "[sync] filtered_pages.jsonl (101 MB) ..."
kubectl cp "$LOCAL/data/filtered_pages.jsonl" \
  "slurm/${POD}:${REMOTE}/data/filtered_pages.jsonl" -c login

echo ""
echo "[sync] done. To submit:"
echo "  kubectl exec -n slurm $POD -c login -- \\"
echo "    runuser -l ${YOU} -c 'cd ${REMOTE} && sbatch scripts/h100_grpo_job.sbatch'"
