from types import SimpleNamespace

from aloha.safety_state import (
    RecoveryIdentity,
    publish_safety_state,
    read_safety_state,
)


def report_with(*, status, torque_off_verified):
    result = SimpleNamespace(
        status=SimpleNamespace(value=status),
        phase="verify_settle",
        reason="drift",
        max_error_rad=0.2,
        torque_off_verified=torque_off_verified,
    )
    return SimpleNamespace(
        results={"leader_left": result},
        safe_to_stop=(
            status == "slept_verified" and torque_off_verified
        ),
    )


def test_schema_v2_records_owner_recovery_and_per_arm_phase(tmp_path):
    state_path = tmp_path / "state.json"
    publish_safety_state(
        "UNSAFE_HOLD",
        report=report_with(
            status="failed",
            torque_off_verified=False,
        ),
        path=state_path,
        recovery=RecoveryIdentity(
            recovery_id="abc",
            owner_pid=123,
            source="standalone",
        ),
        context_ok=True,
        monotonic_clock=lambda: 4.5,
        wall_clock=lambda: "2026-07-30T00:00:00+00:00",
    )

    payload = read_safety_state(state_path)
    assert payload == {
        "schema_version": 2,
        "state": "UNSAFE_HOLD",
        "safe_to_stop": False,
        "recovery_id": "abc",
        "owner_pid": 123,
        "source": "standalone",
        "context_ok": True,
        "updated_wall_time": "2026-07-30T00:00:00+00:00",
        "updated_monotonic": 4.5,
        "robots": {
            "leader_left": {
                "status": "failed",
                "phase": "verify_settle",
                "reason": "drift",
                "max_error_rad": 0.2,
                "torque_off_verified": False,
            }
        },
    }


def test_safe_state_requires_report_to_be_safe(tmp_path):
    state_path = tmp_path / "state.json"
    publish_safety_state(
        "SAFE_TO_STOP",
        report=report_with(
            status="slept_verified",
            torque_off_verified=True,
        ),
        path=state_path,
        recovery=RecoveryIdentity("abc", 123, "recorder"),
    )

    assert read_safety_state(state_path)["safe_to_stop"] is True


def test_state_name_alone_cannot_claim_safe_to_stop(tmp_path):
    state_path = tmp_path / "state.json"
    publish_safety_state(
        "SAFE_TO_STOP",
        report=report_with(
            status="failed",
            torque_off_verified=False,
        ),
        path=state_path,
        recovery=RecoveryIdentity("abc", 123, "recorder"),
    )

    assert read_safety_state(state_path)["safe_to_stop"] is False
