#!/usr/bin/env bash
set -euo pipefail

# Persistent one-command launcher for the reviewed Isaac Sim 5.1 ALOHA GUI.
# It starts only the project ROS services on 103, performs one explicit
# readback-driven Sleep alignment at startup, then opens the GUI.  The GUI
# confirmation dialog remains the only path for any later Sleep/Home replay.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAGE="assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_z_up_meters_diagnostic.usda"
STAGE_SHA256="327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9"
REPORT="reports/aloha1_dual_sleep_gui_bridge_20260804.json"
LOG_DIR=".codex/artifacts/20260804-digital-real-live-bridge"
LIFECYCLE_LOG="$LOG_DIR/launcher_lifecycle_${$}.log"
STARTUP_SLEEP_MANIFEST="$ROOT_DIR/reports/aloha1_mapping/aloha1_home_sleep_command_manifest.json"
REMOTE_HOST="192.168.1.103"
REMOTE_PROJECT="/home/eii/openpi0.5-rtc-reward-learning"
REMOTE_RUN_ID="aloha1_gui_${$}"

if pgrep -f '^\.venv_issac/bin/python tools/open_aloha1_runtime_sleep_gui\.py' >/dev/null; then
  echo "ALOHA1 GUI is already running; refusing to start a duplicate process." >&2
  exit 2
fi

if [[ ! -x ".venv_issac/bin/python" ]]; then
  echo "Missing Isaac Sim 5.1 environment: $ROOT_DIR/.venv_issac" >&2
  exit 2
fi

mkdir -p "$LOG_DIR" "$(dirname -- "$REPORT")"
touch "$LIFECYCLE_LOG"
log_lifecycle() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LIFECYCLE_LOG"
}
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=YES
# Isaac Sim 5.1 bundles its ROS 2 Jazzy libraries inside the Isaac Python
# environment.  The real ALOHA driver remains ROS1 on 103; these variables only
# make the local Isaac ROS2 extension load without relying on shell history.
ISAAC_ROS2_JAZZY_LIB="$ROOT_DIR/.venv_issac/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib"
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export LD_LIBRARY_PATH="$ISAAC_ROS2_JAZZY_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

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
    log_lifecycle "103 ROS master and follower SDKs already running; they will be preserved on exit."
    return
  fi
  log_lifecycle "Starting only ros_master and aloha_ros_nodes on 103..."
  remote_exec "docker compose up -d --no-build ros_master aloha_ros_nodes >/tmp/aloha1_gui_ros_start_${$}.log 2>&1"
  REMOTE_ROS_STARTED=1
  for _ in $(seq 1 45); do
    state="$(remote_ros_preflight || true)"
    read -r master aloha <<< "$state"
    if [[ "$master" == "1" && "$aloha" == "1" ]]; then
      log_lifecycle "103 ROS services are running."
      return
    fi
    sleep 1
  done
  log_lifecycle "ERROR: 103 ROS services did not become ready; refusing to launch Isaac GUI." >&2
  return 1
}

run_startup_sleep_alignment() {
  if [[ "${ALOHA1_SKIP_STARTUP_SLEEP_ALIGNMENT:-0}" == "1" ]]; then
    log_lifecycle "Startup real Sleep alignment skipped by ALOHA1_SKIP_STARTUP_SLEEP_ALIGNMENT=1."
    return 0
  fi
  [[ -f "$STARTUP_SLEEP_MANIFEST" ]] || {
    log_lifecycle "ERROR: frozen Sleep manifest missing: $STARTUP_SLEEP_MANIFEST" >&2
    return 1
  }
  local remote_manifest="/tmp/${REMOTE_RUN_ID}_sleep_manifest.json"
  local remote_script="/tmp/${REMOTE_RUN_ID}_startup_sleep_alignment.py"
  local remote_module="/tmp/${REMOTE_RUN_ID}_startup_sleep_alignment_module.py"
  local remote_output="/tmp/${REMOTE_RUN_ID}_startup_sleep_alignment.json"
  local local_output="$LOG_DIR/${REMOTE_RUN_ID}_startup_sleep_alignment.json"
  log_lifecycle "Running one-shot dual-follower startup Sleep alignment on 103 at 50 Hz over 5 s."
  scp -q "$STARTUP_SLEEP_MANIFEST" "$REMOTE_HOST:$remote_manifest"
  scp -q "$ROOT_DIR/tools/run_aloha1_startup_sleep_alignment.py" "$REMOTE_HOST:$remote_script"
  scp -q "$ROOT_DIR/tools/aloha1_mapping/startup_sleep_alignment.py" "$REMOTE_HOST:$remote_module"
  remote_exec "
    set -euo pipefail
    C=\$(docker ps --format '{{.Names}}' | grep '^openpi_reward_learning_eii-aloha_ros_nodes-1$' | head -1)
    test -n \"\$C\"
    docker exec \"\$C\" mkdir -p /app/.codex /app/tools/aloha1_mapping
    docker cp '$remote_manifest' \"\$C:/app/.codex/${REMOTE_RUN_ID}_sleep_manifest.json\"
    docker cp '$remote_script' \"\$C:/tmp/${REMOTE_RUN_ID}_startup_sleep_alignment.py\"
    docker cp '$remote_module' \"\$C:/app/tools/aloha1_mapping/startup_sleep_alignment.py\"
    docker exec \"\$C\" bash -lc \
      'cd /app; source /opt/ros/noetic/setup.bash; source /root/interbotix_ws/devel/setup.bash; \
       /usr/bin/python3 /tmp/${REMOTE_RUN_ID}_startup_sleep_alignment.py \
       --manifest /app/.codex/${REMOTE_RUN_ID}_sleep_manifest.json \
       --output /tmp/${REMOTE_RUN_ID}_startup_sleep_alignment.json \
       --rate-hz 50 --move-seconds 5 --start-delay-s 2 \
       --execute-real --allow-startup-sleep-align'
    docker cp \"\$C:/tmp/${REMOTE_RUN_ID}_startup_sleep_alignment.json\" '$remote_output'
  "
  scp -q "$REMOTE_HOST:$remote_output" "$local_output"
  rm -f "$remote_manifest" "$remote_script" "$remote_module"
  .venv/bin/python - "$local_output" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("status") != "PASS_STARTUP_SLEEP_ALIGNMENT":
    raise SystemExit(f"startup Sleep alignment failed: {data.get('status')} {data.get('abort_reason')}")
if data.get("commands_published", {}).get("puppet_left") != data.get("commands_published", {}).get("puppet_right"):
    raise SystemExit("startup Sleep alignment published unequal left/right command counts")
print(json.dumps({"status": data["status"], "commands": data["commands_published"], "report": str(path.resolve())}, sort_keys=True))
PY
  log_lifecycle "Startup Sleep alignment passed; report: $local_output"
}

stop_remote_ros_started_by_launcher() {
  if [[ "$REMOTE_ROS_STARTED" != "1" ]]; then
    log_lifecycle "No ROS services were started by this launcher; no remote ROS shutdown requested."
    return 0
  fi
  local stop_failed=0
  if [[ "$REMOTE_ALOHA_PREEXISTING" != "1" ]]; then
    remote_exec "docker compose stop --timeout 20 aloha_ros_nodes" || stop_failed=1
  fi
  if [[ "$REMOTE_ROS_MASTER_PREEXISTING" != "1" ]]; then
    remote_exec "docker compose stop --timeout 20 ros_master" || stop_failed=1
  fi
  if [[ "$stop_failed" == "1" ]]; then
    log_lifecycle "ERROR: remote ROS shutdown command failed; see launcher log: $LIFECYCLE_LOG" >&2
    return 1
  fi
  local remaining
  remaining="$(remote_exec "docker ps --format '{{.Names}}' | grep -Ec '^openpi_reward_learning_eii-(ros_master|aloha_ros_nodes)-1$' || true")"
  if [[ "$remaining" != "0" ]]; then
    log_lifecycle "ERROR: remote ROS shutdown incomplete; running managed container count=$remaining" >&2
    return 1
  fi
  log_lifecycle "103 ROS services started by this launcher stopped and verified before exit."
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if ! stop_remote_ros_started_by_launcher; then
    [[ "$code" == "0" ]] && code=3
  fi
  log_lifecycle "Launcher exit code=$code"
  exit "$code"
}
trap cleanup EXIT INT TERM

start_remote_ros_if_needed
run_startup_sleep_alignment

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
