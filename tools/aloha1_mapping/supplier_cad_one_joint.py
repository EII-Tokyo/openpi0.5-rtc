"""Pure planning and evaluation helpers for supplier-CAD one-joint tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any


def build_bidirectional_targets(
    *,
    start: Sequence[float],
    joint_index: int,
    lower: float,
    upper: float,
    requested_delta: float,
) -> list[list[float]]:
    """Build negative/positive targets with a ten-percent limit margin."""

    if requested_delta <= 0.0 or not math.isfinite(requested_delta):
        raise ValueError("requested_delta must be finite and positive")
    margin = requested_delta * 0.1
    negative = [float(value) for value in start]
    positive = [float(value) for value in start]
    negative[joint_index] = max(
        float(start[joint_index]) - requested_delta,
        lower + margin if math.isfinite(lower) else -math.inf,
    )
    positive[joint_index] = min(
        float(start[joint_index]) + requested_delta,
        upper - margin if math.isfinite(upper) else math.inf,
    )
    return [negative, positive]


def evaluate_one_joint_run(
    *,
    dof_names: Sequence[str],
    commanded_indices: Sequence[int],
    commanded_delta: Sequence[float],
    start: Sequence[float],
    end: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    readback_minimum: float,
    target_tolerance: float,
    unexpected_tolerance: float,
    symmetric_pair: tuple[int, int] | None = None,
    symmetric_tolerance: float | None = None,
) -> dict[str, Any]:
    """Evaluate direction, target, range, isolation, and optional symmetry."""

    commanded = set(commanded_indices)
    actual_delta = [
        float(after) - float(before)
        for before, after in zip(start, end, strict=True)
    ]
    targets = [
        float(start[index]) + float(delta)
        for index, delta in zip(
            commanded_indices,
            commanded_delta,
            strict=True,
        )
    ]
    direction_ok = all(
        actual_delta[index] * float(delta) > 0.0
        for index, delta in zip(
            commanded_indices,
            commanded_delta,
            strict=True,
        )
    )
    readback_ok = all(
        abs(actual_delta[index]) >= readback_minimum
        for index in commanded_indices
    )
    target_error = max(
        (
            abs(float(end[index]) - target)
            for index, target in zip(
                commanded_indices,
                targets,
                strict=True,
            )
        ),
        default=math.inf,
    )
    target_ok = target_error <= target_tolerance
    range_ok = all(
        (
            (not math.isfinite(float(lo)) or float(value) >= float(lo) - 1e-6)
            and (
                not math.isfinite(float(hi))
                or float(value) <= float(hi) + 1e-6
            )
        )
        for value, lo, hi in zip(end, lower, upper, strict=True)
    )
    max_unexpected = max(
        (
            abs(delta)
            for index, delta in enumerate(actual_delta)
            if index not in commanded
        ),
        default=0.0,
    )
    isolation_ok = max_unexpected <= unexpected_tolerance

    symmetric_residual = None
    symmetry_ok = True
    if symmetric_pair is not None:
        left, right = symmetric_pair
        symmetric_residual = abs(float(end[left]) + float(end[right]))
        if symmetric_tolerance is None:
            raise ValueError(
                "symmetric_tolerance is required with symmetric_pair"
            )
        symmetry_ok = symmetric_residual <= symmetric_tolerance

    passed = all(
        (
            direction_ok,
            readback_ok,
            target_ok,
            range_ok,
            isolation_ok,
            symmetry_ok,
        )
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "dof_names": list(dof_names),
        "commanded_indices": list(commanded_indices),
        "commanded_names": [
            dof_names[index] for index in commanded_indices
        ],
        "commanded_delta": list(commanded_delta),
        "actual_delta": actual_delta,
        "direction_ok": direction_ok,
        "readback_ok": readback_ok,
        "target_error": target_error,
        "target_ok": target_ok,
        "range_ok": range_ok,
        "max_unexpected_delta": max_unexpected,
        "isolation_ok": isolation_ok,
        "symmetric_residual": symmetric_residual,
        "symmetry_ok": symmetry_ok,
        "start": list(start),
        "end": list(end),
    }


def summarize_robots(
    robots: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return literal PASS/FAIL/PARTIAL across left and right robots."""

    statuses = [str(record["status"]) for record in robots.values()]
    if "FAIL" in statuses:
        status = "FAIL"
    elif any(item in {"PARTIAL", "NOT_RUN"} for item in statuses):
        status = "PARTIAL"
    else:
        status = "PASS"
    return {
        "status": status,
        "robot_count": len(robots),
        "pass_count": statuses.count("PASS"),
        "fail_count": statuses.count("FAIL"),
        "not_run_count": statuses.count("NOT_RUN"),
        "partial_count": statuses.count("PARTIAL"),
    }
