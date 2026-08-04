"""Atomic, owner-scoped safety state shared with host tooling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Callable


DEFAULT_SAFETY_STATE_PATH = Path("/tmp/aloha_recorder_safety.json")


@dataclass(frozen=True)
class RecoveryIdentity:
    recovery_id: str
    owner_pid: int
    source: str


def _default_wall_clock() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_safety_payload(
    state: str,
    *,
    report,
    recovery: RecoveryIdentity,
    context_ok: bool,
    monotonic: float,
    wall_time: str,
) -> dict:
    robots = {}
    if report is not None:
        robots = {
            robot_name: {
                "status": result.status.value,
                "phase": getattr(result, "phase", "unknown"),
                "reason": getattr(result, "reason", ""),
                "max_error_rad": getattr(
                    result,
                    "max_error_rad",
                    None,
                ),
                "torque_off_verified": bool(
                    getattr(result, "torque_off_verified", False)
                ),
            }
            for robot_name, result in sorted(report.results.items())
        }
    safe_to_stop = bool(
        state == "SAFE_TO_STOP"
        and report is not None
        and report.safe_to_stop
    )
    return {
        "schema_version": 2,
        "state": state,
        "safe_to_stop": safe_to_stop,
        "recovery_id": recovery.recovery_id,
        "owner_pid": recovery.owner_pid,
        "source": recovery.source,
        "context_ok": bool(context_ok),
        "updated_wall_time": wall_time,
        "updated_monotonic": monotonic,
        "robots": robots,
    }


def publish_safety_state(
    state: str,
    *,
    report=None,
    path: Path | str = DEFAULT_SAFETY_STATE_PATH,
    recovery: RecoveryIdentity | None = None,
    context_ok: bool = True,
    monotonic_clock: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], str] = _default_wall_clock,
    clock: Callable[[], float] | None = None,
) -> None:
    """Atomically publish schema-v2 recovery state.

    ``clock`` remains as a migration alias for callers that previously passed
    the monotonic clock under that name.
    """

    if clock is not None:
        monotonic_clock = clock
    if recovery is None:
        recovery = RecoveryIdentity(
            recovery_id="unowned",
            owner_pid=os.getpid(),
            source="legacy",
        )
    payload = build_safety_payload(
        state,
        report=report,
        recovery=recovery,
        context_ok=context_ok,
        monotonic=monotonic_clock(),
        wall_time=wall_clock(),
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
        ) as temporary_file:
            json.dump(
                payload,
                temporary_file,
                ensure_ascii=False,
                sort_keys=True,
            )
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_name, destination)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def read_safety_state(
    path: Path | str = DEFAULT_SAFETY_STATE_PATH,
) -> dict:
    with Path(path).open(encoding="utf-8") as state_file:
        payload = json.load(state_file)
    if not isinstance(payload, dict):
        raise ValueError("safety state must be a JSON object")
    return payload
