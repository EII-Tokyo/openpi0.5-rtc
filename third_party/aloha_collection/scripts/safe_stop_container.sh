#!/usr/bin/env bash
set -eu

if [ "$#" -gt 1 ]; then
    echo "usage: $0 [container-name]" >&2
    exit 2
fi

container_name="${1:-aloha2-collect}"
if [ -z "$container_name" ]; then
    echo "container name must not be empty" >&2
    exit 2
fi

timeout_seconds="${ALOHA_SAFE_STOP_TIMEOUT_SECONDS:-180}"
poll_seconds="${ALOHA_SAFE_STOP_POLL_SECONDS:-1}"
case "$timeout_seconds" in
    ''|*[!0-9]*)
        echo "ALOHA_SAFE_STOP_TIMEOUT_SECONDS must be a positive integer" >&2
        exit 2
        ;;
esac
case "$poll_seconds" in
    ''|*[!0-9]*)
        echo "ALOHA_SAFE_STOP_POLL_SECONDS must be a non-negative integer" >&2
        exit 2
        ;;
esac
if [ "$timeout_seconds" -le 0 ]; then
    echo "ALOHA_SAFE_STOP_TIMEOUT_SECONDS must be a positive integer" >&2
    exit 2
fi
poll_advance="$poll_seconds"
if [ "$poll_advance" -eq 0 ]; then
    poll_advance=1
fi

container_ids="$(
    docker ps -a \
        --filter "name=^/${container_name}$" \
        --format '{{.ID}}'
)"
container_count="$(
    printf '%s\n' "$container_ids" | awk 'NF {count++} END {print count+0}'
)"
if [ "$container_count" -ne 1 ]; then
    echo "expected exactly one container named ${container_name}; found ${container_count}" >&2
    exit 2
fi
container_id="$container_ids"

running="$(
    docker inspect --format '{{.State.Running}}' "$container_id"
)"
if [ "$running" != "true" ]; then
    echo "container ${container_name} is not running" >&2
    exit 2
fi

recorder_pids="$(
    docker exec "$container_id" \
        pgrep -f 'python3 .*record_episodes_copy.py' || true
)"
recorder_count="$(
    printf '%s\n' "$recorder_pids" | awk 'NF {count++} END {print count+0}'
)"
if [ "$recorder_count" -ne 1 ]; then
    echo "expected exactly one recorder PID; found ${recorder_count}" >&2
    exit 2
fi
recorder_pid="$recorder_pids"

echo "requesting fail-safe recorder shutdown: container=${container_name} pid=${recorder_pid}"
docker exec "$container_id" kill -INT "$recorder_pid"

elapsed=0
expected_recovery_id="-"
validator_path="/root/interbotix_ws/src/aloha/scripts/validate_safety_state.py"
while [ "$elapsed" -lt "$timeout_seconds" ]; do
    set +e
    observation="$(
        docker exec "$container_id" \
            python3 "$validator_path" \
            /tmp/aloha_recorder_safety.json \
            "$recorder_pid" \
            "$expected_recovery_id" \
            2>&1
    )"
    validation_status=$?
    set -e
    if [ "$validation_status" -ne 0 ]; then
        echo "waiting: safety state is absent or invalid: ${observation}" >&2
        sleep "$poll_seconds"
        elapsed=$((elapsed + poll_advance))
        continue
    fi

    IFS='|' read -r state recovery_id owner_pid owner_source safe_to_stop <<EOF
$observation
EOF
    if [ "$expected_recovery_id" = "-" ] && [ "$recovery_id" != "-" ]; then
        expected_recovery_id="$recovery_id"
    fi

    case "$state" in
        SAFE_TO_STOP)
            if [ "$safe_to_stop" != "true" ] || [ "$recovery_id" = "-" ]; then
                echo "refusing to stop: incomplete SAFE_TO_STOP proof" >&2
                exit 3
            fi
            if ! docker exec "$container_id" kill -0 "$owner_pid" 2>/dev/null; then
                echo "refusing to stop: recovery owner ${owner_pid} is not live" >&2
                exit 3
            fi
            echo "all arms verified sleep; stopping container ${container_name}"
            docker stop --time 120 "$container_id"
            exit 0
            ;;
        UNSAFE_HOLD)
            echo "refusing to stop: recorder entered UNSAFE_HOLD" >&2
            echo "restore robot feedback, then request explicit retry from the recorder" >&2
            exit 3
            ;;
        RUNNING|RECOVERY_IN_PROGRESS|EXTERNAL_RECOVERY_REQUIRED|"")
            ;;
        *)
            echo "waiting: unknown safety state '${state}'" >&2
            ;;
    esac
    sleep "$poll_seconds"
    elapsed=$((elapsed + poll_advance))
done

echo "refusing to stop: no SAFE_TO_STOP state after ${timeout_seconds} seconds" >&2
exit 4
