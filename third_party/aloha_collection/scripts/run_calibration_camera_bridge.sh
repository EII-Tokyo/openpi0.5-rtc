#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_SOURCE="${SCRIPT_DIR}/ros_calibration_camera_bridge.py"
CONTAINER_NAME="${ALOHA_CALIBRATION_ROS_CONTAINER:-aloha2-collect}"
BRIDGE_HOST="127.0.0.1"
BRIDGE_PORT="${ALOHA_CALIBRATION_ROS_BRIDGE_PORT:-8018}"

[[ -f "$BRIDGE_SOURCE" ]] || {
    echo "bridge source is missing: $BRIDGE_SOURCE" >&2
    exit 2
}

container_status="$(docker inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || true)"
[[ "$container_status" == "running" ]] || {
    echo "required ROS container is not running: $CONTAINER_NAME status=$container_status" >&2
    exit 3
}

if ss -ltn "( sport = :${BRIDGE_PORT} )" | tail -n +2 | grep -q .; then
    echo "bridge port is already in use: ${BRIDGE_HOST}:${BRIDGE_PORT}" >&2
    exit 4
fi

echo "starting read-only ROS camera bridge on ${BRIDGE_HOST}:${BRIDGE_PORT}" >&2
echo "container=${CONTAINER_NAME} publishers=0 robot_command_api=false" >&2

exec docker exec -i "$CONTAINER_NAME" bash -lc \
    'source /opt/ros/humble/setup.bash; source /root/interbotix_ws/install/setup.bash; exec python3 - --host "$1" --port "$2"' \
    calibration-bridge "$BRIDGE_HOST" "$BRIDGE_PORT" < "$BRIDGE_SOURCE"
