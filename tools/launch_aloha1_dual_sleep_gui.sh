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
REMOTE_HOST="192.168.1.103"
REMOTE_PROJECT="/home/eii/openpi0.5-rtc-reward-learning"
REMOTE_STATE_FILE="/tmp/aloha1_gui_ros_lifecycle_${$}.json"

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

REMOTE_ROS_STARTED=0
REMOTE_ROS_MASTER_PREEXISTING=0
REMOTE_ALOHA_PREEXISTING=0

remote_exec() {
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$REMOTE_HOST" "cd '$REMOTE_PROJECT' && $*"
}

remote_ros_preflight() {
  remote_exec "
    master=0; aloha=0;
    docker ps --format '{{.Names}}' | grep -q '^openpi_reward_learning_eii-ros_master-1$' && master=1 || true;
    docker ps --format '{{.Names}}' | grep -q '^openpi_reward_learning_eii-aloha_ros_nodes-1$' && aloha=1 || true;
    printf '%s %s\\n' \"\$master\" \"\$aloha\"
  "
}

start_remote_ros_if_needed() {
  local state
  state="$(remote_ros_preflight)"
  read -r REMOTE_ROS_MASTER_PREEXISTING REMOTE_ALOHA_PREEXISTING <<< "$state"
  if [[ "$REMOTE_ROS_MASTER_PREEXISTING" == "1" && "$REMOTE_ALOHA_PREEXISTING" == "1" ]]; then
    echo "103 ROS master and both follower SDKs are already running; launcher will not stop them on exit."
    return
  fi
  echo "Starting only ros_master and aloha_ros_nodes on 103..."
  remote_exec "docker compose up -d --no-build ros_master aloha_ros_nodes >/tmp/aloha1_gui_ros_start_${$}.log 2>&1"
  REMOTE_ROS_STARTED=1
  for _ in $(seq 1 45); do
    state="$(remote_ros_preflight || true)"
    read -r master aloha <<< "$state"
    if [[ "$master" == "1" && "$aloha" == "1" ]]; then
      echo "103 ROS services are running."
      return
    fi
    sleep 1
  done
  echo "103 ROS services did not become ready; refusing to launch Isaac GUI." >&2
  return 1
}

stop_remote_ros_started_by_launcher() {
  [[ "$REMOTE_ROS_STARTED" == "1" ]] || return 0
  if [[ "$REMOTE_ALOHA_PREEXISTING" != "1" ]]; then
    remote_exec "docker compose stop aloha_ros_nodes" || true
  fi
  if [[ "$REMOTE_ROS_MASTER_PREEXISTING" != "1" ]]; then
    remote_exec "docker compose stop ros_master" || true
  fi
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  stop_remote_ros_started_by_launcher
  exit "$code"
}
trap cleanup EXIT INT TERM

start_remote_ros_if_needed

.venv_issac/bin/python tools/open_aloha1_runtime_sleep_gui.py \
  --stage "$STAGE" \
  --stage-sha256 "$STAGE_SHA256" \
  --report "$REPORT" \
  --startup-workspace 2 \
  "$@"
status=$?
trap - EXIT INT TERM
stop_remote_ros_started_by_launcher
exit "$status"
