#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/eii/openpi0.5-rtc-reward-learning
BUNDLE_ROOT="$PROJECT_ROOT/remote_isaac_assets/aloha1_bottle_server/attempt1"
ISAAC_ROOT=/home/eii/Applications/isaacsim-5.1.0
APP="$ISAAC_ROOT/apps/isaacsim.exp.full.streaming.kit"
KIT="$ISAAC_ROOT/kit/kit"
STAGE="$BUNDLE_ROOT/versions/thread_release_v1/remote_stream_threaded_release_compliant_pad_test_v1.usda"
LOADER="$BUNDLE_ROOT/tools/remote_stream_cap_stage_loader.py"
REPORT="$BUNDLE_ROOT/remote_compliant_pad_test_loader_report.json"
LOG="$BUNDLE_ROOT/remote_stream_compliant_pad_test_loader.log"
EXPECTED_SHA="${ALOHA_COMPLIANT_TEST_STAGE_SHA256:?missing test Stage SHA-256}"

cd "$PROJECT_ROOT"
test -x "$KIT"
test -f "$APP"
test -f "$STAGE"
test -f "$LOADER"
test "$(sha256sum "$STAGE" | awk '{print $1}')" = "$EXPECTED_SHA"

if [ -s "$BUNDLE_ROOT/remote_stream_server.pid" ]; then
  old_pid=$(cat "$BUNDLE_ROOT/remote_stream_server.pid")
  old_cmd=$(ps -o command= -p "$old_pid" 2>/dev/null || true)
  if [[ "$old_cmd" == *"$APP"* && "$old_cmd" == *"$BUNDLE_ROOT/tools/remote_stream"* ]]; then
    kill -TERM "$old_pid"
    for _ in $(seq 1 30); do
      if ! kill -0 "$old_pid" 2>/dev/null; then break; fi
      sleep 1
    done
    if kill -0 "$old_pid" 2>/dev/null; then
      kill -KILL "$old_pid"
    fi
  fi
fi

systemctl --user stop isaac-sim-streaming.service
for _ in $(seq 1 60); do
  if ! pgrep -f "$APP" >/dev/null; then break; fi
  sleep 1
done
if pgrep -f "$APP" >/dev/null; then
  echo "old Isaac streaming process did not stop" >&2
  exit 1
fi

rm -f "$REPORT"
nohup env \
  ALOHA_REMOTE_STAGE="$STAGE" \
  ALOHA_REMOTE_STAGE_SHA256="$EXPECTED_SHA" \
  ALOHA_REMOTE_LOADER_REPORT="$REPORT" \
  ALOHA_GRIPPER_COMPLIANCE_PROFILE=accel_50ms_critical_v1 \
  "$KIT" "$APP" \
  --no-window \
  --/app/livestream/publicEndpointAddress=127.0.0.1 \
  --/app/livestream/port=49100 \
  --/app/tokens/omni_documents=/home/eii/isaac-home/Documents/Kit/shared \
  --/app/tokens/shared_documents=/home/eii/isaac-home/Documents/Kit/shared \
  --/app/tokens/app_documents=/home/eii/isaac-home/Documents/Kit/apps/IsaacSimStreaming \
  --/app/tokens/documents=/home/eii/isaac-home/Documents/Kit/apps/IsaacSimStreaming \
  --/app/captureFrame/path=/home/eii/isaac-home/Documents/Kit/shared/screenshots \
  --persistent/app/captureFrame/path=/home/eii/isaac-home/Documents/Kit/shared/screenshots \
  --ext-folder /home/eii/Applications/isaacsim-5.1.0/extsUser \
  --enable aloha.lula_base_aligned \
  --exec "$LOADER" \
  >"$LOG" 2>&1 < /dev/null &
echo $! > "$BUNDLE_ROOT/remote_stream_server.pid"
echo "started_pid=$(cat "$BUNDLE_ROOT/remote_stream_server.pid")"
