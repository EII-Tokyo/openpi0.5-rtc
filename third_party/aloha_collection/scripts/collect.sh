#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DOCKER_BIN="${COLLECT_DOCKER_BIN:-docker}"
LOCK_PATH="${COLLECT_LOCK_PATH:-/tmp/aloha-collect.lock}"
CONTAINER_NAME="aloha2-collect"
IMAGE="lyl472324464/robot:aloha-2.0"
MEMORY_BYTES="51539607552"
CONTAINER_REPO="/root/interbotix_ws/src/aloha"
ROBOT="aloha_stationary"
TASK_NAME="aloha_stationary"
TIMEOUT_SECONDS="120"
MODE="run"
EXTRA_RECORDER_ARGS=()
CREATED_CONTAINER=0

check() { printf '[CHECK] %s\n' "$*"; }
wait_log() { printf '[WAIT] %s\n' "$*"; }
ready_log() { printf '[READY] %s\n' "$*"; }
error() { printf '[ERROR] %s\n' "$*" >&2; }

usage() {
    cat <<'EOF'
Usage: scripts/collect.sh [OPTIONS] [-- RECORDER_ARGS...]

Options:
  --status             Read-only runtime status.
  --dry-run            Print resolved actions without changing state.
  --task-name NAME     Recorder task name (default: aloha_stationary).
  --robot CONFIG       Robot config name (default: aloha_stationary).
  --timeout SECONDS    Readiness timeout (default: 120).
  -h, --help           Show this help.
EOF
}

need_value() {
    if [[ $# -lt 2 || -z "$2" ]]; then
        error "$1 requires a value"
        exit 2
    fi
}

while (($#)); do
    case "$1" in
        --status)
            if [[ "$MODE" != "run" ]]; then
                error "--status conflicts with $MODE"
                exit 2
            fi
            MODE="status"
            shift
            ;;
        --dry-run)
            if [[ "$MODE" != "run" ]]; then
                error "--dry-run conflicts with $MODE"
                exit 2
            fi
            MODE="dry-run"
            shift
            ;;
        --task-name)
            need_value "$1" "${2:-}"
            TASK_NAME="$2"
            shift 2
            ;;
        --robot)
            need_value "$1" "${2:-}"
            ROBOT="$2"
            shift 2
            ;;
        --timeout)
            need_value "$1" "${2:-}"
            TIMEOUT_SECONDS="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_RECORDER_ARGS=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "unknown argument: $1"
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$MODE" == "status" && ${#EXTRA_RECORDER_ARGS[@]} -gt 0 ]]; then
    error "--status does not accept recorder arguments"
    exit 2
fi
if [[ ! "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    error "--timeout must be a positive integer"
    exit 2
fi

for arg in "${EXTRA_RECORDER_ARGS[@]}"; do
    case "$arg" in
        -t|-r|--task_name|--task-name|--robot|--start-trigger|\
        --video-encoder|--leader-hold-policy|\
        --pedal-debounce-seconds|--task_name=*|--task-name=*|\
        --robot=*|--start-trigger=*|--video-encoder=*|\
        --leader-hold-policy=*|--pedal-debounce-seconds=*)
            error "recorder argument conflicts with launcher defaults: $arg"
            exit 2
            ;;
    esac
done

if [[ ! -f "${REPO_ROOT}/config/robot/${ROBOT}.yaml" ]]; then
    error "robot config does not exist: config/robot/${ROBOT}.yaml"
    exit 2
fi

command -v flock >/dev/null 2>&1 || {
    error "required command is unavailable: flock"
    exit 2
}
if [[ "$DOCKER_BIN" == */* ]]; then
    [[ -x "$DOCKER_BIN" ]] || {
        error "Docker command is not executable: $DOCKER_BIN"
        exit 2
    }
else
    command -v "$DOCKER_BIN" >/dev/null 2>&1 || {
        error "required command is unavailable: $DOCKER_BIN"
        exit 2
    }
fi

exec {LOCK_FD}>"$LOCK_PATH"
if ! flock -n "$LOCK_FD"; then
    error "another collection launcher owns $LOCK_PATH"
    exit 3
fi

docker_cmd() {
    "$DOCKER_BIN" "$@"
}

inspect_value() {
    local template="$1"
    docker_cmd inspect --format "$template" "$CONTAINER_NAME"
}

container_exists() {
    docker_cmd inspect "$CONTAINER_NAME" >/dev/null 2>&1
}

run_in_container() {
    docker_cmd exec "$CONTAINER_NAME" \
        bash -lc '
            source /opt/ros/humble/setup.bash
            source /root/interbotix_ws/install/setup.bash
            exec "$@"
        ' collect-env "$@"
}

process_lines() {
    local pattern="$1"
    docker_cmd exec "$CONTAINER_NAME" \
        pgrep -af "$pattern" 2>/dev/null || true
}

validate_container() {
    local status image memory network privileged runtime env_lines mounts
    status="$(inspect_value '{{.State.Status}}')"
    image="$(inspect_value '{{.Config.Image}}')"
    memory="$(inspect_value '{{.HostConfig.Memory}}')"
    network="$(inspect_value '{{.HostConfig.NetworkMode}}')"
    privileged="$(inspect_value '{{.HostConfig.Privileged}}')"
    runtime="$(inspect_value '{{.HostConfig.Runtime}}')"
    env_lines="$(inspect_value \
        '{{range .Config.Env}}{{println .}}{{end}}')"
    mounts="$(inspect_value \
        '{{range .Mounts}}{{printf "%s|%s\n" .Source .Destination}}{{end}}')"

    [[ "$status" == "running" ]] || {
        error "container is not running: status=$status"
        return 1
    }
    [[ "$image" == "$IMAGE" ]] || {
        error "container image mismatch: $image"
        return 1
    }
    [[ "$memory" == "$MEMORY_BYTES" ]] || {
        error "container memory mismatch: $memory"
        return 1
    }
    [[ "$network" == "host" ]] || {
        error "container network mismatch: $network"
        return 1
    }
    [[ "$privileged" == "true" ]] || {
        error "container privileged mismatch: $privileged"
        return 1
    }
    [[ "$runtime" == "nvidia" ]] || {
        error "container runtime mismatch: $runtime"
        return 1
    }
    grep -Fxq 'NVIDIA_VISIBLE_DEVICES=all' <<<"$env_lines" || {
        error "container NVIDIA_VISIBLE_DEVICES mismatch"
        return 1
    }
    grep -Fxq \
        'NVIDIA_DRIVER_CAPABILITIES=compute,utility,video' \
        <<<"$env_lines" || {
        error "container NVIDIA_DRIVER_CAPABILITIES mismatch"
        return 1
    }
    grep -Fxq "${REPO_ROOT}|${CONTAINER_REPO}" <<<"$mounts" || {
        error "container repository mount mismatch"
        return 1
    }
    grep -Fxq '/dev|/dev' <<<"$mounts" || {
        error "container /dev mount mismatch"
        return 1
    }
}

print_docker_run() {
    printf '%q ' docker run -d \
        --name "$CONTAINER_NAME" \
        --memory=48g \
        --network=host \
        -v /dev:/dev \
        -v "${REPO_ROOT}:${CONTAINER_REPO}" \
        --privileged \
        --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
        "$IMAGE" \
        bash -lc \
        'source /root/interbotix_ws/install/setup.bash && exec ros2 launch aloha aloha_bringup.launch.py robot:='"$ROBOT"
    printf '\n'
}

create_container() {
    check "creating canonical ${CONTAINER_NAME}"
    docker_cmd image inspect "$IMAGE" >/dev/null
    docker_cmd run -d \
        --name "$CONTAINER_NAME" \
        --memory=48g \
        --network=host \
        -v /dev:/dev \
        -v "${REPO_ROOT}:${CONTAINER_REPO}" \
        --privileged \
        --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
        "$IMAGE" \
        bash -lc \
        "source /root/interbotix_ws/install/setup.bash && \
         exec ros2 launch aloha aloha_bringup.launch.py robot:=${ROBOT}"
    CREATED_CONTAINER=1
}

docker_cmd info >/dev/null

if [[ "$MODE" == "dry-run" ]]; then
    check "dry-run; no Docker or ROS mutations will run"
    if container_exists; then
        validate_container
        printf 'reuse container %q\n' "$CONTAINER_NAME"
    else
        print_docker_run
    fi
    printf '%q ' docker exec -it "$CONTAINER_NAME" \
        python3 "${CONTAINER_REPO}/scripts/record_episodes_copy.py" \
        --task_name "$TASK_NAME" \
        --robot "$ROBOT" \
        --start-trigger b \
        --video-encoder nvenc \
        --leader-hold-policy best-effort \
        --pedal-debounce-seconds 1.0 \
        --return-home-between-episodes \
        "${EXTRA_RECORDER_ARGS[@]}"
    printf '\n'
    exit 0
fi

if container_exists; then
    check "validating existing ${CONTAINER_NAME}"
    validate_container
else
    if [[ "$MODE" == "status" ]]; then
        check "container ${CONTAINER_NAME} is absent"
        exit 0
    fi
    create_container
fi

recorder_processes="$(
    process_lines 'python3 .*[r]ecord_episodes_copy.py'
)"
if [[ -n "$recorder_processes" && "$MODE" != "status" ]]; then
    error "recorder is already running:"
    printf '%s\n' "$recorder_processes" >&2
    exit 4
fi

bringup_processes="$(
    process_lines \
        'ros2 launch aloha [a]loha_bringup.launch.py'
)"
bringup_count="$(
    grep -cve '^[[:space:]]*$' <<<"$bringup_processes" || true
)"

graph_state="$(
    run_in_container \
        python3 \
        "${CONTAINER_REPO}/scripts/check_collect_ready.py" \
        --robot "$ROBOT" \
        --timeout 1 \
        --classify-graph
)"
graph_state="$(tail -n 1 <<<"$graph_state")"

if [[ "$MODE" == "status" ]]; then
    recorder_count="$(
        grep -cve '^[[:space:]]*$' <<<"$recorder_processes" || true
    )"
    check "recorder_count=${recorder_count}"
    check "bringup_count=${bringup_count} graph=${graph_state}"
    ready_log "status completed without mutations"
    exit 0
fi

if (( bringup_count > 1 )); then
    error "multiple ROS bringup processes detected"
    printf '%s\n' "$bringup_processes" >&2
    exit 5
fi

if (( bringup_count == 0 )) && (( CREATED_CONTAINER == 0 )); then
    if [[ "$graph_state" != "empty" ]]; then
        error "ROS graph is ${graph_state} without a bringup owner"
        exit 5
    fi
    check "starting ROS bringup in existing container"
    docker_cmd exec -d "$CONTAINER_NAME" \
        bash -lc "
            source /opt/ros/humble/setup.bash
            source /root/interbotix_ws/install/setup.bash
            exec ros2 launch aloha aloha_bringup.launch.py \
                robot:=${ROBOT} \
                >/tmp/aloha-bringup.log 2>&1
        "
fi

check "waiting for four arms and four cameras"
run_in_container \
    python3 \
    "${CONTAINER_REPO}/scripts/check_collect_ready.py" \
    --robot "$ROBOT" \
    --timeout "$TIMEOUT_SECONDS"

check "probing NVIDIA runtime"
run_in_container nvidia-smi >/dev/null
check "probing h264_nvenc"
run_in_container \
    ffmpeg -hide_banner -loglevel error \
    -f lavfi -i color=size=640x480:rate=50 \
    -t 1 -c:v h264_nvenc -f null - \
    >/dev/null

pedal_path="$(
    run_in_container python3 -c \
        'from aloha.local_pedal import DEFAULT_PEDAL_PATH; print(DEFAULT_PEDAL_PATH)'
)"
if ! docker_cmd exec "$CONTAINER_NAME" test -e "$pedal_path"; then
    wait_log "foot pedal is absent; keyboard b and socket remain available"
fi

if [[ "${COLLECT_TEST_ALLOW_NON_TTY:-0}" != "1" ]] \
    && { [[ ! -t 0 ]] || [[ ! -t 1 ]]; }; then
    error "interactive recorder requires a terminal"
    exit 6
fi

ready_log "launching interactive recorder"
set +e
docker_cmd exec -it "$CONTAINER_NAME" \
    bash -lc '
        source /opt/ros/humble/setup.bash
        source /root/interbotix_ws/install/setup.bash
        cd /root/interbotix_ws/src/aloha/scripts
        exec python3 record_episodes_copy.py "$@"
    ' collect-recorder \
    --task_name "$TASK_NAME" \
    --robot "$ROBOT" \
    --start-trigger b \
    --video-encoder nvenc \
    --leader-hold-policy best-effort \
    --pedal-debounce-seconds 1.0 \
    --return-home-between-episodes \
    "${EXTRA_RECORDER_ARGS[@]}"
recorder_status=$?
set -e

ready_log "recorder exited; Docker and ROS remain running"
exit "$recorder_status"
