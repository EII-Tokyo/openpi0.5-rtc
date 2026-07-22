"""Pure helpers shared by the A20 articulation metadata gates."""

import math
from typing import Any

_DOF_FIELDS = (
    "path",
    "name",
    "joint_type",
    "axis",
    "lower_limit",
    "upper_limit",
    "body0",
    "body1",
    "index",
)

_PROHIBITED_SAFETY_FLAGS = (
    "physics_stepped",
    "actions_applied",
    "targets_written",
    "stage_saved",
)


def compare_dof_records(
    expected: list[dict[str, Any]], observed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare ordered DOF records exactly and return stable mismatches."""
    mismatches: list[dict[str, Any]] = []
    if len(expected) != len(observed):
        mismatches.append(
            {
                "field": "count",
                "index": None,
                "expected": len(expected),
                "observed": len(observed),
            }
        )

    mismatches.extend(
        {
            "field": field,
            "index": index,
            "expected": expected_record.get(field),
            "observed": observed_record.get(field),
        }
        for index, (expected_record, observed_record) in enumerate(
            zip(expected, observed, strict=False)
        )
        for field in _DOF_FIELDS
        if expected_record.get(field) != observed_record.get(field)
    )

    expected_paths = [record.get("path") for record in expected]
    observed_paths = [record.get("path") for record in observed]
    mismatches.extend(
        {
            "field": "missing",
            "index": None,
            "expected": path,
            "observed": None,
        }
        for path in expected_paths
        if path not in observed_paths
    )
    mismatches.extend(
        {
            "field": "unexpected",
            "index": None,
            "expected": None,
            "observed": path,
        }
        for path in observed_paths
        if path not in expected_paths
    )

    return {
        "ok": not mismatches,
        "expected_count": len(expected),
        "observed_count": len(observed),
        "mismatches": mismatches,
    }


def validate_dof_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate DOF identity uniqueness and finite, increasing limits."""
    errors: list[dict[str, Any]] = []

    for field in ("path", "name"):
        indices_by_value: dict[Any, list[int]] = {}
        for index, record in enumerate(records):
            indices_by_value.setdefault(record.get(field), []).append(index)
        for value, indices in indices_by_value.items():
            if len(indices) > 1:
                errors.append(
                    {
                        "code": f"duplicate_{field}",
                        "field": field,
                        "value": value,
                        "indices": indices,
                    }
                )

    for index, record in enumerate(records):
        lower = record.get("lower_limit")
        upper = record.get("upper_limit")
        limits_are_finite = True
        for field, value in (("lower_limit", lower), ("upper_limit", upper)):
            if not isinstance(value, int | float) or not math.isfinite(value):
                limits_are_finite = False
                errors.append(
                    {
                        "code": "non_finite_limit",
                        "field": field,
                        "index": index,
                        "value": value,
                    }
                )
        if limits_are_finite and lower >= upper:
            errors.append(
                {
                    "code": "invalid_limit_order",
                    "index": index,
                    "lower_limit": lower,
                    "upper_limit": upper,
                }
            )

    return {"ok": not errors, "errors": errors}


def validate_safety_flags(payload: dict[str, Any]) -> dict[str, Any]:
    """Reject evidence that reports any prohibited runtime or write action."""
    errors = [
        {
            "code": "prohibited_safety_flag",
            "field": field,
            "observed": True,
        }
        for field in _PROHIBITED_SAFETY_FLAGS
        if payload.get(field) is True
    ]
    return {"ok": not errors, "errors": errors}
