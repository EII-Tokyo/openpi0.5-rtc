"""Pure protocol helpers for synchronized ALOHA real/simulation replay."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

WORKERS = ("isaac", "real", "cam_high")
STATES = (
    "CREATED",
    "PREPARED",
    "READY",
    "ARMED",
    "RUNNING",
    "COMPLETE",
    "ABORTED",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(name: str, value: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")


def build_run_identity(
    *,
    run_id: str,
    manifest_sha256: str,
    command_signature: str,
    command_rate_hz: int,
) -> dict[str, Any]:
    """Build the immutable identity shared by every experiment worker."""

    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    _require_sha256("manifest_sha256", manifest_sha256)
    _require_sha256("command_signature", command_signature)
    if command_rate_hz <= 0 or 1_000_000_000 % command_rate_hz:
        raise ValueError("command_rate_hz must divide one second into integer nanoseconds")
    return {
        "schema_version": 1,
        "run_id": run_id,
        "manifest_sha256": manifest_sha256,
        "command_signature": command_signature,
        "command_rate_hz": int(command_rate_hz),
        "sample_period_ns": 1_000_000_000 // int(command_rate_hz),
        "workers": list(WORKERS),
    }


def deadline_ns(
    start_monotonic_ns: int, sample_index: int, sample_period_ns: int
) -> int:
    """Return an absolute deadline without accumulating sleep error."""

    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    if sample_period_ns <= 0:
        raise ValueError("sample_period_ns must be positive")
    return int(start_monotonic_ns) + int(sample_index) * int(sample_period_ns)


def classify_start_skew(skew_ns: int, *, sample_period_ns: int) -> str:
    """Classify observed worker-start skew against one command period."""

    if sample_period_ns <= 0:
        raise ValueError("sample_period_ns must be positive")
    return (
        "SYNCHRONIZED_START_PASS"
        if abs(int(skew_ns)) <= int(sample_period_ns)
        else "POST_ALIGNED_ONLY"
    )


def validate_ready_record(
    record: Mapping[str, object], identity: Mapping[str, object]
) -> list[str]:
    """Return mismatched frozen identity fields from a worker READY record."""

    failures = [
        field
        for field in ("run_id", "manifest_sha256", "command_signature")
        if record.get(field) != identity.get(field)
    ]
    if record.get("status") != "READY":
        failures.append("status")
    if record.get("worker") not in identity.get("workers", ()):
        failures.append("worker")
    return failures
