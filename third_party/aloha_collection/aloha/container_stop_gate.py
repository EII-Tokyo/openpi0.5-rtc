"""Pure validation for the host-side container stop gate."""

from __future__ import annotations

from dataclasses import dataclass


class SafetyStateRejected(ValueError):
    """Raised when published state cannot authorize container shutdown."""


@dataclass(frozen=True)
class StopObservation:
    state: str
    recovery_id: str | None
    owner_pid: int
    source: str
    safe_to_stop: bool


def validate_stop_observation(
    payload: dict,
    *,
    recorder_pid: int,
    expected_recovery_id: str | None,
) -> StopObservation:
    if payload.get("schema_version") != 2:
        raise SafetyStateRejected("schema_version must be 2")

    state = payload.get("state")
    if not isinstance(state, str) or not state:
        raise SafetyStateRejected("state must be a non-empty string")
    owner_pid = payload.get("owner_pid")
    if not isinstance(owner_pid, int) or owner_pid <= 0:
        raise SafetyStateRejected("owner_pid must be a positive integer")
    source = payload.get("source")
    if source not in {"recorder", "standalone", "legacy"}:
        raise SafetyStateRejected("source is not recognized")

    raw_recovery_id = payload.get("recovery_id")
    recovery_id = (
        raw_recovery_id
        if isinstance(raw_recovery_id, str)
        and raw_recovery_id
        and raw_recovery_id != "unowned"
        else None
    )
    active_state = state != "RUNNING"
    if active_state and recovery_id is None:
        raise SafetyStateRejected(
            "active recovery requires a recovery_id"
        )
    if (
        expected_recovery_id is not None
        and recovery_id != expected_recovery_id
    ):
        raise SafetyStateRejected(
            "recovery_id does not match the active recovery"
        )

    safe_to_stop = payload.get("safe_to_stop") is True
    if state == "SAFE_TO_STOP":
        if not safe_to_stop:
            raise SafetyStateRejected(
                "SAFE_TO_STOP requires safe_to_stop=true"
            )
        robots = payload.get("robots")
        if not isinstance(robots, dict) or not robots:
            raise SafetyStateRejected(
                "SAFE_TO_STOP requires robot results"
            )
        for robot_name, result in robots.items():
            if not isinstance(result, dict):
                raise SafetyStateRejected(
                    f"{robot_name} result must be an object"
                )
            if result.get("status") != "slept_verified":
                raise SafetyStateRejected(
                    f"{robot_name} status is not slept_verified"
                )
            if result.get("torque_off_verified") is not True:
                raise SafetyStateRejected(
                    f"{robot_name} torque_off_verified is not true"
                )
    elif safe_to_stop:
        raise SafetyStateRejected(
            "safe_to_stop=true is only valid for SAFE_TO_STOP"
        )

    if source == "recorder" and owner_pid != recorder_pid:
        raise SafetyStateRejected(
            "recorder owner_pid does not match recorder process"
        )
    if active_state and source == "legacy":
        raise SafetyStateRejected(
            "legacy source cannot own active recovery"
        )

    return StopObservation(
        state=state,
        recovery_id=recovery_id,
        owner_pid=owner_pid,
        source=source,
        safe_to_stop=safe_to_stop,
    )
