#!/usr/bin/env bash
# Start long LoRA training on a single GPU (tmux session).
#
# Usage:
#   export BATCH_SIZE=128  # default; override if needed
#   export EXP_NAME=my_run_$(date +%Y%m%d_%H%M)
#   ./scripts/start_h200_lora_train.sh
#
# Optional: W&B + port forward from laptop:
#   ssh -p 50415 root@HOST -L 8080:localhost:8080

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_NAME="${CONFIG_NAME:-twist_and_static_mixture_lora}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EXP_NAME="${EXP_NAME:-twist_static_lora_h200_$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION:-openpi_lora}"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"

cmd="cd '$REPO_ROOT' && uv run scripts/train.py '$CONFIG_NAME' \
  --exp-name '$EXP_NAME' \
  --batch-size $BATCH_SIZE"

if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s "$SESSION" bash -lc "$cmd"
  echo "Started tmux session: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
else
  echo "tmux not found; running in foreground..."
  eval "$cmd"
fi
