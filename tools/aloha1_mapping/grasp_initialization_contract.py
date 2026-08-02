"""Pure machine gates for ALOHA grasp initialization and finger safety."""

from __future__ import annotations

import hashlib
import json
import math
import struct
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


def _float32_roundtrip(value: float) -> float:
    return float(struct.unpack("f", struct.pack("f", float(value)))[0])


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
    representable_limits: dict[str, dict[str, float]] = {}
    limit_violation = False
    for index, finger in enumerate(FINGER_NAMES):
        lower, upper = limits[finger]
        target = target_pair[index]
        actual = readback_pair[index]
        readback_lower = min(lower, _float32_roundtrip(lower))
        readback_upper = max(upper, _float32_roundtrip(upper))
        representable_limits[finger] = {
            "lower": readback_lower,
            "upper": readback_upper,
        }
        margins[finger] = {
            "target_lower": target - lower,
            "target_upper": upper - target,
            "readback_lower": actual - lower,
            "readback_upper": upper - actual,
        }
        if (
            margins[finger]["target_lower"] < 0.0
            or margins[finger]["target_upper"] < 0.0
            or actual < readback_lower
            or actual > readback_upper
        ):
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
        "readback_representable_limits_m": representable_limits,
        "readback_numeric_semantics": (
            "SOURCE_DECIMAL_OR_EXACT_FLOAT32_REPRESENTATION"
        ),
        "pair_overlap_volume_m3": overlap,
    }


def _contact_paths(contact: dict[str, object]) -> tuple[str, ...]:
    return tuple(
        str(contact.get(key, ""))
        for key in (
            "actor0_path",
            "actor1_path",
            "collider0_path",
            "collider1_path",
        )
    )


def _touches_path(paths: tuple[str, ...], prim_path: str) -> bool:
    return any(path == prim_path or path.startswith(f"{prim_path}/") for path in paths)


def evaluate_finger_runtime_frame(
    *,
    frame: int,
    phase: str,
    targets: list[float],
    readback: list[float],
    source_limits: dict[str, dict[str, float]],
    pair_overlap_volume_m3: float,
    contacts: list[dict[str, object]],
    finger_paths: dict[str, str],
) -> dict[str, object]:
    """Classify a live frame without changing its commanded or measured state."""

    frame_index = int(frame)
    if frame_index < 0:
        raise ValueError("frame must be non-negative")
    if set(finger_paths) != set(FINGER_NAMES):
        raise ValueError("finger_paths must contain the two explicit finger names")

    initialization = evaluate_finger_initialization(
        reset_complete=True,
        dof_order=list(FINGER_NAMES),
        targets=targets,
        readback=readback,
        source_limits=source_limits,
        overlap_volume_m3=pair_overlap_volume_m3,
    )
    failure_codes = [
        str(code)
        for code in initialization["failure_codes"]
        if code != "FAIL_INITIALIZATION_CONTRACT"
    ]
    pair_contacts: list[dict[str, object]] = []
    environment_contacts: list[dict[str, object]] = []
    for contact in contacts:
        paths = _contact_paths(contact)
        touches_left = _touches_path(paths, finger_paths["left_finger"])
        touches_right = _touches_path(paths, finger_paths["right_finger"])
        if touches_left and touches_right:
            pair_contacts.append(dict(contact))
        elif (touches_left or touches_right) and any(
            path == "/World/environment"
            or path.startswith("/World/environment/")
            for path in paths
        ):
            environment_contacts.append(dict(contact))

    if pair_contacts and "FINGER_PAIR_UNEXPECTED_CONTACT" not in failure_codes:
        failure_codes.append("FINGER_PAIR_UNEXPECTED_CONTACT")
    active_environment_contact = any(
        (
            math.isfinite(float(contact.get("impulse_ns", math.nan)))
            and float(contact.get("impulse_ns", 0.0)) > 0.0
        )
        or (
            math.isfinite(float(contact.get("separation_m", math.nan)))
            and float(contact.get("separation_m", math.inf)) <= 0.0
        )
        for contact in environment_contacts
    )
    if (
        "FINGER_LIMIT_VIOLATION" in failure_codes
        and active_environment_contact
        and "ENVIRONMENT_CONTACT_FORCED_LIMIT_VIOLATION" not in failure_codes
    ):
        failure_codes.append("ENVIRONMENT_CONTACT_FORCED_LIMIT_VIOLATION")

    first_failure = (
        {
            "frame": frame_index,
            "phase": str(phase),
            "failure_codes": list(failure_codes),
        }
        if failure_codes
        else None
    )
    return {
        "status": "FAIL" if failure_codes else "PASS",
        "frame": frame_index,
        "phase": str(phase),
        "failure_codes": failure_codes,
        "limit_margins_m": initialization["limit_margins_m"],
        "pair_overlap_volume_m3": initialization["pair_overlap_volume_m3"],
        "finger_pair_contacts": pair_contacts,
        "finger_environment_contacts": environment_contacts,
        "first_failure": first_failure,
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
