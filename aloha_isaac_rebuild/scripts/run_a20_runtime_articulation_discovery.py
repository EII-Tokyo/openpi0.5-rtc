"""Pure aggregation contract for A20 runtime articulation discovery evidence."""

from __future__ import annotations

from typing import Any

from aloha_isaac_rebuild.scripts.a20_articulation_gate_common import compare_dof_records
from aloha_isaac_rebuild.scripts.a20_articulation_gate_common import validate_dof_records
from aloha_isaac_rebuild.scripts.a20_articulation_gate_common import validate_safety_flags

_PASS = "PASS_A20_RUNTIME_ARTICULATION_DISCOVERY_NO_STEP"
_FAIL = "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"
_BLOCKED = "BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION"
_RUN_PASS = "PASS_RUNTIME_PROBE"
_REQUIRED_RUN_FIELDS = (
    "status",
    "process_status",
    "returncode",
    "timed_out",
    "articulation_root",
    "articulation_count",
    "dof_count",
    "valid_handle",
    "records",
    "requires_unapproved_initialization",
)


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _layer1_errors(layer1: object) -> list[dict[str, Any]]:
    details: list[str] = []
    if not isinstance(layer1, dict):
        return [{"code": "invalid_layer1_evidence", "details": ["not_a_dict"]}]

    if layer1.get("status") != "PASS_A20_USD_DOF_METADATA":
        details.append("status")
    if layer1.get("ok") is not True:
        details.append("ok")
    if layer1.get("mismatches") != []:
        details.append("mismatches")
    if layer1.get("errors") != []:
        details.append("errors")

    expected = layer1.get("expected")
    observed = layer1.get("observed")
    if not (
        isinstance(expected, list)
        and isinstance(observed, list)
        and all(isinstance(record, dict) for record in expected + observed)
    ):
        details.append("records_shape")
    else:
        if len(expected) != 16 or not validate_dof_records(expected)["ok"]:
            details.append("expected_records")
        comparison = compare_dof_records(expected, observed)
        if not comparison["ok"]:
            details.append("expected_observed_consistency")

    inputs = layer1.get("inputs")
    if not isinstance(inputs, dict):
        details.append("inputs")
    else:
        stage = inputs.get("stage")
        if not isinstance(stage, dict):
            details.append("stage_input")
        elif not (
            isinstance(stage.get("path"), str)
            and bool(stage["path"])
            and _valid_sha256(stage.get("pre_sha256"))
            and stage.get("pre_sha256") == stage.get("post_sha256")
            and stage.get("consistent_during_audit") is True
        ):
            details.append("stage_hash_consistency")
        for name in ("mapping", "config"):
            item = inputs.get(name)
            if not (
                isinstance(item, dict)
                and isinstance(item.get("path"), str)
                and bool(item["path"])
                and _valid_sha256(item.get("sha256"))
            ):
                details.append(f"{name}_input")

    safety = validate_safety_flags(layer1)
    if not safety["ok"]:
        details.append("safety_flags")

    return (
        [{"code": "invalid_layer1_evidence", "details": details}]
        if details
        else []
    )


def _run_errors(
    run: object, run_index: int, expected: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    errors: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    if not isinstance(run, dict):
        return ([{"code": "invalid_run_shape", "run_index": run_index}], [], False)

    missing = [field for field in _REQUIRED_RUN_FIELDS if field not in run]
    errors.extend(
        {"code": "missing_field", "run_index": run_index, "field": field}
        for field in missing
    )

    errors.extend(
        {
            "code": "invalid_field_type",
            "run_index": run_index,
            "field": field,
            "expected": "bool",
            "observed_type": type(run[field]).__name__,
        }
        for field in ("timed_out", "valid_handle", "requires_unapproved_initialization")
        if field in run and not isinstance(run[field], bool)
    )
    errors.extend(
        {
            "code": "invalid_field_type",
            "run_index": run_index,
            "field": field,
            "expected": "int",
            "observed_type": type(run[field]).__name__,
        }
        for field in ("returncode", "articulation_count", "dof_count")
        if field in run and type(run[field]) is not int
    )

    safety = validate_safety_flags(run)
    errors.extend({"run_index": run_index, **error} for error in safety["errors"])

    process_ok = (
        run.get("process_status") == "completed"
        and type(run.get("returncode")) is int
        and run["returncode"] == 0
        and run.get("timed_out") is False
    )
    if not process_ok:
        errors.append({"code": "subprocess_failure", "run_index": run_index})

    blocked = (
        run.get("status") == _BLOCKED
        and run.get("requires_unapproved_initialization") is True
        and run.get("valid_handle") is False
    )
    if run.get("status") not in (_RUN_PASS, _BLOCKED):
        errors.append({"code": "invalid_run_status", "run_index": run_index})
    elif run.get("status") == _BLOCKED and not blocked:
        errors.append({"code": "invalid_blocked_evidence", "run_index": run_index})
    elif run.get("status") == _RUN_PASS and run.get("requires_unapproved_initialization") is not False:
        errors.append({"code": "unexpected_initialization_requirement", "run_index": run_index})

    if run.get("articulation_root") != "/aloha/root_joint":
        errors.append({"code": "invalid_articulation_root", "run_index": run_index})
    if type(run.get("articulation_count")) is int and run["articulation_count"] != 1:
        errors.append({"code": "invalid_articulation_count", "run_index": run_index})
    if type(run.get("dof_count")) is int and run["dof_count"] != 16:
        errors.append({"code": "invalid_dof_count", "run_index": run_index})
    if not blocked and run.get("valid_handle") is not True:
        errors.append({"code": "invalid_handle", "run_index": run_index})

    records = run.get("records")
    if not (
        isinstance(records, list) and all(isinstance(record, dict) for record in records)
    ):
        errors.append({"code": "invalid_records_shape", "run_index": run_index})
    else:
        comparison = compare_dof_records(expected, records)
        if not comparison["ok"]:
            errors.append({"code": "runtime_records_mismatch", "run_index": run_index})
            mismatches.extend(
                {"run_index": run_index, **mismatch}
                for mismatch in comparison["mismatches"]
            )
            mismatches.extend(
                {"run_index": run_index, "validation_error": validation_error}
                for validation_error in comparison.get("validation_errors", [])
            )

    return errors, mismatches, blocked


def aggregate_runtime_runs(
    layer1: object, runs: object
) -> dict[str, Any]:
    """Aggregate exactly three saved runtime run dictionaries, fail-closed."""
    errors = _layer1_errors(layer1)
    mismatches: list[dict[str, Any]] = []
    run_count = len(runs) if isinstance(runs, list) else None
    if run_count != 3:
        errors.append(
            {"code": "invalid_run_count", "expected": 3, "observed": run_count}
        )

    blocked_runs: list[int] = []
    if not errors and isinstance(layer1, dict) and isinstance(runs, list):
        expected = layer1["expected"]
        for run_index, run in enumerate(runs):
            run_errors, run_mismatches, blocked = _run_errors(
                run, run_index, expected
            )
            errors.extend(run_errors)
            mismatches.extend(run_mismatches)
            if blocked:
                blocked_runs.append(run_index)

    if errors:
        status = _FAIL
    elif blocked_runs:
        status = _BLOCKED
    else:
        status = _PASS
    return {
        "status": status,
        "ok": status == _PASS,
        "run_count": run_count,
        "blocked_run_indices": blocked_runs,
        "errors": errors,
        "mismatches": mismatches,
    }
