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
    validation_errors = [
        {"side": side, **error}
        for side, records in (("expected", expected), ("observed", observed))
        for error in validate_dof_records(records)["errors"]
    ]
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

    result = {
        "ok": not mismatches and not validation_errors,
        "expected_count": len(expected),
        "observed_count": len(observed),
        "mismatches": mismatches,
    }
    if validation_errors:
        result["validation_errors"] = validation_errors
    return result


def validate_dof_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate DOF identity uniqueness and finite, increasing limits."""
    errors: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        errors.extend(
            {"code": "missing_field", "index": index, "field": field}
            for field in _DOF_FIELDS
            if field not in record
        )
        errors.extend(
            {
                "code": "invalid_field_type",
                "index": index,
                "field": field,
                "expected": "non-empty string",
                "observed_type": type(record[field]).__name__,
            }
            for field in ("path", "name")
            if field in record
            and (
                not isinstance(record[field], str) or not record[field].strip()
            )
        )

    for field in ("path", "name"):
        indices_by_value: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            value = record.get(field)
            if isinstance(value, str) and value.strip():
                indices_by_value.setdefault(value, []).append(index)
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
            if field not in record:
                limits_are_finite = False
            elif isinstance(value, bool) or not isinstance(value, int | float):
                limits_are_finite = False
                errors.append(
                    {
                        "code": "invalid_field_type",
                        "index": index,
                        "field": field,
                        "expected": "finite int or float",
                        "observed_type": type(value).__name__,
                    }
                )
            elif not math.isfinite(value):
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
    errors: list[dict[str, Any]] = []
    for field in _PROHIBITED_SAFETY_FLAGS:
        if field not in payload:
            errors.append({"code": "missing_field", "field": field})
        elif not isinstance(payload[field], bool):
            errors.append(
                {
                    "code": "invalid_field_type",
                    "field": field,
                    "expected": "bool",
                    "observed_type": type(payload[field]).__name__,
                }
            )
        elif payload[field]:
            errors.append(
                {
                    "code": "prohibited_safety_flag",
                    "field": field,
                    "observed": True,
                }
            )
    return {"ok": not errors, "errors": errors}
