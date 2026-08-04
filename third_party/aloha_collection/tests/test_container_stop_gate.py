import pytest
from pathlib import Path
import json
import subprocess
import sys

from aloha.container_stop_gate import (
    SafetyStateRejected,
    validate_stop_observation,
)

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_safety_state.py"


def state_payload(
    *,
    state="SAFE_TO_STOP",
    safe_to_stop=True,
    recovery_id="abc",
    owner_pid=123,
    source="recorder",
    torque_off_verified=True,
):
    return {
        "schema_version": 2,
        "state": state,
        "safe_to_stop": safe_to_stop,
        "recovery_id": recovery_id,
        "owner_pid": owner_pid,
        "source": source,
        "context_ok": True,
        "robots": {
            "leader_left": {
                "status": "slept_verified",
                "phase": "complete",
                "reason": "verified",
                "max_error_rad": 0.01,
                "torque_off_verified": torque_off_verified,
            }
        },
    }


@pytest.mark.parametrize(
    "payload,match",
    [
        ({"schema_version": 1}, "schema_version"),
        (state_payload(safe_to_stop=False), "safe_to_stop"),
        (
            state_payload(torque_off_verified=False),
            "torque_off_verified",
        ),
        (
            {
                **state_payload(),
                "robots": {},
            },
            "robot results",
        ),
        (
            state_payload(owner_pid=999),
            "recorder owner_pid",
        ),
    ],
)
def test_invalid_safe_state_is_rejected(payload, match):
    with pytest.raises(SafetyStateRejected, match=match):
        validate_stop_observation(
            payload,
            recorder_pid=123,
            expected_recovery_id="abc",
        )


def test_mismatched_recovery_id_is_rejected():
    with pytest.raises(SafetyStateRejected, match="recovery_id"):
        validate_stop_observation(
            state_payload(recovery_id="other"),
            recorder_pid=123,
            expected_recovery_id="abc",
        )


def test_recorder_owned_safe_state_is_accepted():
    observation = validate_stop_observation(
        state_payload(),
        recorder_pid=123,
        expected_recovery_id="abc",
    )

    assert observation.safe_to_stop
    assert observation.owner_pid == 123
    assert observation.source == "recorder"


def test_standalone_owner_may_differ_from_recorder():
    observation = validate_stop_observation(
        state_payload(
            owner_pid=456,
            source="standalone",
        ),
        recorder_pid=123,
        expected_recovery_id="abc",
    )

    assert observation.safe_to_stop
    assert observation.owner_pid == 456


def test_running_state_does_not_latch_unowned_recovery_id():
    observation = validate_stop_observation(
        state_payload(
            state="RUNNING",
            safe_to_stop=False,
            recovery_id="unowned",
        ),
        recorder_pid=123,
        expected_recovery_id=None,
    )

    assert observation.recovery_id is None
    assert not observation.safe_to_stop


def test_validator_runs_by_absolute_path_from_an_unrelated_cwd(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(state_payload()),
        encoding="utf-8",
    )
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            str(state_path),
            "123",
            "abc",
        ],
        cwd=unrelated,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        "SAFE_TO_STOP|abc|123|recorder|true"
    )
