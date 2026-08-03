"""Pure sample-key alignment for ALOHA real/simulation telemetry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

SampleKey = tuple[int, str, int]


def _key(row: Mapping[str, object]) -> SampleKey:
    return (int(row["cycle"]), str(row["segment"]), int(row["sample_index"]))


def _index_rows(
    rows: Sequence[Mapping[str, object]], *, source: str
) -> dict[SampleKey, Mapping[str, object]]:
    indexed: dict[SampleKey, Mapping[str, object]] = {}
    for row in rows:
        key = _key(row)
        if key in indexed:
            raise ValueError(f"duplicate {source} key: {key}")
        indexed[key] = row
    return indexed


def _serialize_key(key: SampleKey) -> list[object]:
    return [key[0], key[1], key[2]]


def align_rows(
    real_rows: Sequence[Mapping[str, object]],
    isaac_rows: Sequence[Mapping[str, object]],
    *,
    joint_names: Sequence[str],
) -> dict[str, Any]:
    """Align immutable raw rows by semantic command key without interpolation."""

    real = _index_rows(real_rows, source="real")
    isaac = _index_rows(isaac_rows, source="isaac")
    matched = sorted(real.keys() & isaac.keys())
    missing_real = sorted(isaac.keys() - real.keys())
    missing_isaac = sorted(real.keys() - isaac.keys())
    errors: dict[str, list[float]] = {str(name): [] for name in joint_names}
    for key in matched:
        real_q = [float(value) for value in real[key]["q"]]  # type: ignore[index]
        isaac_q = [float(value) for value in isaac[key]["q"]]  # type: ignore[index]
        if len(real_q) != len(joint_names) or len(isaac_q) != len(joint_names):
            raise ValueError(f"joint vector length mismatch at key {key}")
        if not all(math.isfinite(value) for value in (*real_q, *isaac_q)):
            raise ValueError(f"non-finite joint vector at key {key}")
        for index, name in enumerate(joint_names):
            errors[str(name)].append(real_q[index] - isaac_q[index])
    per_joint = {}
    for name, values in errors.items():
        per_joint[name] = {
            "sample_count": len(values),
            "signed_real_minus_isaac_mean_rad": (
                sum(values) / len(values) if values else None
            ),
            "mean_abs_error_rad": (
                sum(abs(value) for value in values) / len(values) if values else None
            ),
            "rmse_rad": (
                math.sqrt(sum(value * value for value in values) / len(values))
                if values
                else None
            ),
            "max_abs_error_rad": max((abs(value) for value in values), default=None),
        }
    return {
        "schema_version": 1,
        "matched_keys": [_serialize_key(key) for key in matched],
        "matched_sample_count": len(matched),
        "missing_real_keys": [_serialize_key(key) for key in missing_real],
        "missing_isaac_keys": [_serialize_key(key) for key in missing_isaac],
        "derived_interpolation_performed": False,
        "error_definition": "real_q_rad - isaac_q_rad",
        "per_joint": per_joint,
    }


def classify_correspondence(
    metrics: Mapping[str, object], *, thresholds: Mapping[str, object] | None
) -> dict[str, object]:
    """Classify independent correspondence layers without inventing thresholds."""

    command = metrics.get("command_identity") is True
    semantics = metrics.get("joint_semantics") is True
    endpoints = metrics.get("kinematic_endpoints") is True
    start = metrics.get("start_classification") == "SYNCHRONIZED_START_PASS"
    if thresholds is None:
        dynamic = "CALIBRATION_PENDING"
    else:
        dynamic = (
            "PASS"
            if metrics.get("dynamic_within_frozen_thresholds") is True
            else "FAIL"
        )
    layers = {
        "COMMAND_IDENTITY": "PASS" if command else "FAIL",
        "JOINT_SEMANTICS": "PASS" if semantics else "FAIL",
        "KINEMATIC_ENDPOINT_CORRESPONDENCE": "PASS" if endpoints else "FAIL",
        "DYNAMIC_TRAJECTORY_CORRESPONDENCE": dynamic,
        "START_SYNCHRONIZATION": "PASS" if start else "POST_ALIGNED_ONLY",
    }
    if not command or not semantics:
        status = "SIGNAL_MAPPING_FAILURE"
    elif not endpoints:
        status = "KINEMATIC_ENDPOINT_MISMATCH"
    elif not start:
        status = "POST_ALIGNED_ONLY"
    elif dynamic == "FAIL":
        status = "DYNAMIC_RESPONSE_MISMATCH"
    elif dynamic == "PASS":
        status = "SYNCHRONIZED_KINEMATIC_AND_DYNAMIC_CORRESPONDENCE_PASS"
    else:
        status = (
            "KINEMATIC_AND_SIGNAL_DIGITAL_TWIN_PASS_DYNAMIC_CALIBRATION_PENDING"
        )
    return {"status": status, "layers": layers}
