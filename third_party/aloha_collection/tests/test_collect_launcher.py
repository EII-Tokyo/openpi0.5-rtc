from pathlib import Path
import fcntl
import os
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "collect.sh"
FAKE_DOCKER = ROOT / "tests" / "fakes" / "fake_docker.py"


def run_launcher(tmp_path, *args, **overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    log_path = tmp_path / "docker.log"
    env = os.environ.copy()
    env.update(
        {
            "COLLECT_DOCKER_BIN": str(FAKE_DOCKER),
            "COLLECT_TEST_ALLOW_NON_TTY": "1",
            "COLLECT_LOCK_PATH": str(tmp_path / "collect.lock"),
            "FAKE_DOCKER_LOG": str(log_path),
            "FAKE_REPO": str(ROOT),
        }
    )
    env.update({key: str(value) for key, value in overrides.items()})
    result = subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    calls = (
        log_path.read_text(encoding="utf-8").splitlines()
        if log_path.exists()
        else []
    )
    return result, calls


def recorder_call(calls):
    return next(
        call
        for call in calls
        if "record_episodes_copy.py" in call and "pgrep" not in call
    )


def test_healthy_runtime_is_reused_with_confirmed_defaults(tmp_path):
    result, calls = run_launcher(tmp_path)

    assert result.returncode == 0, result.stderr
    assert not any(call.startswith("run ") for call in calls)
    call = recorder_call(calls)
    assert "--task_name aloha_stationary" in call
    assert "--robot aloha_stationary" in call
    assert "--start-trigger b" in call
    assert "--video-encoder nvenc" in call
    assert "--leader-hold-policy best-effort" in call
    assert "--pedal-debounce-seconds 1.0" in call
    assert "--return-home-between-episodes" in call
    assert "--no-save-return-to-start-on-b" not in call


def test_absent_container_is_created_with_canonical_contract(tmp_path):
    result, calls = run_launcher(
        tmp_path,
        FAKE_CONTAINER="absent",
    )

    assert result.returncode == 0, result.stderr
    run_call = next(call for call in calls if call.startswith("run "))
    for fragment in (
        "--name aloha2-collect",
        "--memory=48g",
        "--network=host",
        "--privileged",
        "--runtime=nvidia",
        "NVIDIA_VISIBLE_DEVICES=all",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
        "/dev:/dev",
        "lyl472324464/robot:aloha-2.0",
        "aloha_bringup.launch.py",
        "robot:=aloha_stationary",
    ):
        assert fragment in run_call


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"FAKE_CONTAINER": "exited"}, "not running"),
        ({"FAKE_IMAGE": "wrong/image:latest"}, "image"),
        ({"FAKE_RUNTIME": "runc"}, "runtime"),
        ({"FAKE_MEMORY": "0"}, "memory"),
        ({"FAKE_NETWORK": "bridge"}, "network"),
        ({"FAKE_PRIVILEGED": "false"}, "privileged"),
        ({"FAKE_REPO": "/wrong/repository"}, "mount"),
        ({"FAKE_DEV_SOURCE": "/wrong/dev"}, "/dev"),
        ({"FAKE_VISIBLE_DEVICES": "none"}, "visible_devices"),
        (
            {"FAKE_DRIVER_CAPABILITIES": "compute,utility"},
            "driver_capabilities",
        ),
    ],
)
def test_abnormal_or_incompatible_container_fails_closed(
    tmp_path,
    overrides,
    match,
):
    result, calls = run_launcher(tmp_path, **overrides)

    assert result.returncode != 0
    assert match in (result.stdout + result.stderr).lower()
    assert not any(call.startswith("run ") for call in calls)
    assert not any(" rm " in f" {call} " for call in calls)
    assert not any(" kill " in f" {call} " for call in calls)


def test_existing_recorder_blocks_all_start_actions(tmp_path):
    result, calls = run_launcher(
        tmp_path,
        FAKE_RECORDER=(
            "412 python3 record_episodes_copy.py "
            "--task_name aloha_stationary"
        ),
    )

    assert result.returncode != 0
    assert "412" in result.stdout + result.stderr
    assert not any("check_collect_ready.py" in call for call in calls)
    assert not any(
        "record_episodes_copy.py" in call and "pgrep" not in call
        for call in calls
    )


def test_empty_graph_without_bringup_starts_one_bringup(tmp_path):
    result, calls = run_launcher(
        tmp_path,
        FAKE_BRINGUP_COUNT="0",
        FAKE_GRAPH="empty",
    )

    assert result.returncode == 0, result.stderr
    assert any(
        call.startswith("exec -d ")
        and "aloha_bringup.launch.py" in call
        for call in calls
    )


@pytest.mark.parametrize("graph_state", ["partial", "complete"])
def test_nonempty_graph_without_owner_fails_closed(
    tmp_path,
    graph_state,
):
    result, calls = run_launcher(
        tmp_path,
        FAKE_BRINGUP_COUNT="0",
        FAKE_GRAPH=graph_state,
    )

    assert result.returncode != 0
    assert graph_state in (result.stdout + result.stderr).lower()
    assert not any(call.startswith("exec -d ") for call in calls)


def test_multiple_bringups_fail_closed(tmp_path):
    result, calls = run_launcher(
        tmp_path,
        FAKE_BRINGUP_COUNT="2",
    )

    assert result.returncode != 0
    assert "multiple" in (result.stdout + result.stderr).lower()
    assert not any(
        "record_episodes_copy.py" in call and "pgrep" not in call
        for call in calls
    )


def test_readiness_or_nvenc_failure_never_starts_recorder(tmp_path):
    ready_result, ready_calls = run_launcher(
        tmp_path / "ready",
        FAKE_READY_EXIT="2",
    )
    nvenc_result, nvenc_calls = run_launcher(
        tmp_path / "nvenc",
        FAKE_NVENC_EXIT="1",
    )

    assert ready_result.returncode != 0
    assert nvenc_result.returncode != 0
    assert not any(
        "record_episodes_copy.py" in call and "pgrep" not in call
        for call in ready_calls + nvenc_calls
    )


def test_dry_run_is_mutation_free(tmp_path):
    result, calls = run_launcher(
        tmp_path,
        "--dry-run",
        FAKE_CONTAINER="absent",
    )

    assert result.returncode == 0
    assert "docker run" in result.stdout
    assert not any(call.startswith("run ") for call in calls)
    assert not any(call.startswith("exec ") for call in calls)


def test_status_is_read_only_and_reports_existing_recorder(tmp_path):
    result, calls = run_launcher(
        tmp_path,
        "--status",
        FAKE_RECORDER="412 python3 record_episodes_copy.py",
    )

    assert result.returncode == 0
    assert "recorder_count=1" in result.stdout
    assert not any(call.startswith("run ") for call in calls)
    assert not any(call.startswith("exec -d ") for call in calls)
    assert not any(
        "record_episodes_copy.py" in call and "pgrep" not in call
        for call in calls
    )


def test_conflicting_passthrough_argument_is_rejected_before_docker(
    tmp_path,
):
    result, calls = run_launcher(
        tmp_path,
        "--",
        "--video-encoder",
        "cpu",
    )

    assert result.returncode != 0
    assert "conflict" in (result.stdout + result.stderr).lower()
    assert calls == []


def test_status_and_dry_run_are_mutually_exclusive(tmp_path):
    result, calls = run_launcher(
        tmp_path,
        "--status",
        "--dry-run",
    )

    assert result.returncode == 2
    assert "conflict" in result.stderr.lower()
    assert calls == []


def test_missing_robot_config_fails_before_docker(tmp_path):
    result, calls = run_launcher(
        tmp_path,
        "--robot",
        "does_not_exist",
    )

    assert result.returncode == 2
    assert "does not exist" in result.stderr
    assert calls == []


def test_overrides_and_nonconflicting_passthrough_reach_recorder(tmp_path):
    result, calls = run_launcher(
        tmp_path,
        "--task-name",
        "wipe_table",
        "--timeout",
        "45",
        "--",
        "--random-start-positions",
    )

    assert result.returncode == 0
    call = recorder_call(calls)
    assert "--task_name wipe_table" in call
    assert "--random-start-positions" in call
    readiness_call = next(
        call for call in calls
        if "check_collect_ready.py" in call
        and "--classify-graph" not in call
    )
    assert "--timeout 45" in readiness_call


def test_missing_pedal_warns_but_keeps_keyboard_collection(tmp_path):
    result, calls = run_launcher(
        tmp_path,
        FAKE_PEDAL_EXIT="1",
    )

    assert result.returncode == 0
    assert "foot pedal is absent" in result.stdout
    assert recorder_call(calls)


def test_single_instance_lock_blocks_before_docker(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    lock_path = tmp_path / "collect.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result, calls = run_launcher(tmp_path)

    assert result.returncode == 3
    assert "another collection launcher" in result.stderr
    assert calls == []


def test_recorder_exit_status_is_preserved(tmp_path):
    result, _calls = run_launcher(
        tmp_path,
        FAKE_RECORDER_EXIT="7",
    )

    assert result.returncode == 7
