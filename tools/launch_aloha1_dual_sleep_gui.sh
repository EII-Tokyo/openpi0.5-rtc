#!/usr/bin/env bash
set -euo pipefail

# Persistent one-command launcher for the reviewed Isaac Sim 5.1 ALOHA GUI.
# It never starts ROS or a real-robot publisher; the GUI confirmation dialog
# remains the only path that can request the guarded bridge.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAGE="assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_z_up_meters_diagnostic.usda"
STAGE_SHA256="327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9"
REPORT="reports/aloha1_dual_sleep_gui_bridge_20260804.json"
LOG_DIR=".codex/artifacts/20260804-digital-real-live-bridge"

if pgrep -f '^\.venv_issac/bin/python tools/open_aloha1_runtime_sleep_gui\.py' >/dev/null; then
  echo "ALOHA1 GUI is already running; refusing to start a duplicate process." >&2
  exit 2
fi

if [[ ! -x ".venv_issac/bin/python" ]]; then
  echo "Missing Isaac Sim 5.1 environment: $ROOT_DIR/.venv_issac" >&2
  exit 2
fi

mkdir -p "$LOG_DIR" "$(dirname -- "$REPORT")"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=YES

exec .venv_issac/bin/python tools/open_aloha1_runtime_sleep_gui.py \
  --stage "$STAGE" \
  --stage-sha256 "$STAGE_SHA256" \
  --report "$REPORT" \
  --startup-workspace 2 \
  "$@"
