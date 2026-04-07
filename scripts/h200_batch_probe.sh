#!/usr/bin/env bash
# Find a large stable batch size on single-GPU JAX training (e.g. H200).
# Tries candidates high → low; stops at the first success = largest in this list that works.
# Disables W&B and subtask-eval holdout scan for speed.
#
# Usage (from repo root):
#   chmod +x scripts/h200_batch_probe.sh
#   XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 ./scripts/h200_batch_probe.sh
#
# Optional: CONFIG_NAME=my_config ./scripts/h200_batch_probe.sh

set -uo pipefail
CONFIG_NAME="${CONFIG_NAME:-twist_and_static_mixture_lora}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"

# High → low; adjust if you want finer steps near the limit.
CANDIDATES=(192 176 160 144 128 112 96 80 72 64 56 48 40 32)

echo "Config: $CONFIG_NAME  (repo: $REPO_ROOT)"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION"
echo ""

for bs in "${CANDIDATES[@]}"; do
  echo "========== trying batch_size=$bs =========="
  exp="_probe_bs_${bs}_$(date +%s)"
  log="/tmp/openpi_bs_probe_${bs}.log"
  if uv run scripts/train.py "$CONFIG_NAME" \
    --exp-name "$exp" \
    --overwrite \
    --num-train-steps 2 \
    --batch-size "$bs" \
    --no-wandb-enabled \
    --no-subtask-eval-enabled \
    2>&1 | tee "$log"
  then
    echo ""
    echo "=== RESULT: first passing batch_size (largest in default list): $bs ==="
    echo "Log: $log"
    echo "For full training, use a bit lower than the OOM edge (e.g. $((bs - 8))) and re-enable --subtask-eval-enabled true if desired."
    exit 0
  fi
  echo "FAIL batch_size=$bs (log: $log)"
  echo ""
done

echo "No candidate batch size passed. Lower the list in scripts/h200_batch_probe.sh or check logs in /tmp/openpi_bs_probe_*.log"
exit 1
