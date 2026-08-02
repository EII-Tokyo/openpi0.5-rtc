"""Pure machine gates for ALOHA grasp initialization and finger safety."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

FINGER_NAMES = ("left_finger", "right_finger")
_SIGNATURE_EXCLUDED_KEYS = {
    "artifact_path",
    "command",
    "output_path",
    "process_id",
    "runtime_s",
    "timestamp",
}


def _finite_pair(values: list[float], *, name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    pair = (float(values[0]), float(values[1]))
    if not all(math.isfinite(value) for value in pair):
        raise ValueError(f"{name} must contain finite values")
    return pair


def _validated_limits(
    source_limits: dict[str, dict[str, float]],
) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for finger in FINGER_NAMES:
        if finger not in source_limits:
            raise ValueError(f"missing source limits for {finger}")
        lower = float(source_limits[finger]["lower"])
        upper = float(source_limits[finger]["upper"])
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError(f"source limits for {finger} must be finite")
        if lower >= upper:
            raise ValueError(f"source limits for {finger} are not ordered")
        result[finger] = (lower, upper)
    return result


def evaluate_finger_initialization(
    *,
    reset_complete: bool,
    dof_order: list[str],
    targets: list[float],
    readback: list[float],
    source_limits: dict[str, dict[str, float]],
    overlap_volume_m3: float,
) -> dict[str, object]:
    """Evaluate whether the two fingers are in an admissible reset state."""

    target_pair = _finite_pair(targets, name="targets")
    readback_pair = _finite_pair(readback, name="readback")
    limits = _validated_limits(source_limits)
    overlap = float(overlap_volume_m3)
    if not math.isfinite(overlap) or overlap < 0.0:
        raise ValueError("overlap_volume_m3 must be finite and non-negative")

    failure_codes: list[str] = []
    if not bool(reset_complete) or list(dof_order) != list(FINGER_NAMES):
        failure_codes.append("FAIL_INITIALIZATION_CONTRACT")

    margins: dict[str, dict[str, float]] = {}
    limit_violation = False
    for index, finger in enumerate(FINGER_NAMES):
        lower, upper = limits[finger]
        target = target_pair[index]
        actual = readback_pair[index]
        margins[finger] = {
            "target_lower": target - lower,
            "target_upper": upper - target,
            "readback_lower": actual - lower,
            "readback_upper": upper - actual,
        }
        if min(margins[finger].values()) < 0.0:
            limit_violation = True
    if target_pair[0] <= 0.0 or target_pair[1] >= 0.0:
        limit_violation = True
    if readback_pair[0] <= 0.0 or readback_pair[1] >= 0.0:
        limit_violation = True
    if limit_violation:
        failure_codes.append("FINGER_LIMIT_VIOLATION")
    if overlap > 0.0:
        failure_codes.append("FINGER_PAIR_OVERLAP")

    return {
        "status": "PASS" if not failure_codes else "FAIL",
        "failure_codes": failure_codes,
        "reset_complete": bool(reset_complete),
        "dof_order": list(dof_order),
        "target_m": list(target_pair),
        "readback_m": list(readback_pair),
        "source_limits_m": {
            finger: {"lower": limits[finger][0], "upper": limits[finger][1]}
            for finger in FINGER_NAMES
        },
        "limit_margins_m": margins,
        "pair_overlap_volume_m3": overlap,
    }


def _canonical_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _SIGNATURE_EXCLUDED_KEYS
        }
    if isinstance(value, list | tuple):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("initialization signature cannot contain non-finite data")
        return round(value, 12)
    return value


def canonical_initialization_signature(record: dict[str, object]) -> str:
    """Return a deterministic hash that excludes process/output identity."""

    payload = json.dumps(
        _canonical_value(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
