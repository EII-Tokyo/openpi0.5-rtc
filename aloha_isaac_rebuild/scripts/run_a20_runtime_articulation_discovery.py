"""Pure aggregation contract for A20 runtime articulation discovery evidence."""

from __future__ import annotations

import argparse
import ast
from contextlib import suppress
from datetime import datetime
import hashlib
import html
import inspect
from itertools import pairwise
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any
import uuid

import yaml

from aloha_isaac_rebuild.scripts.a20_articulation_gate_common import compare_dof_records
from aloha_isaac_rebuild.scripts.a20_articulation_gate_common import validate_dof_records
from aloha_isaac_rebuild.scripts.a20_articulation_gate_common import validate_safety_flags
from aloha_isaac_rebuild.scripts.a20_policy_runtime_order_adapter import build_order_adapter
from aloha_isaac_rebuild.scripts.a20_policy_runtime_order_adapter import round_trip_check

_PASS = "PASS_A20_RUNTIME_ARTICULATION_DISCOVERY_NO_STEP"
_FAIL = "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"
_BLOCKED = "BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION"
_RUN_PASS = "PASS_RUNTIME_PROBE"
MAX_REPORT_ISSUES = 20
MAX_REPORT_OPERATIONS = 20
MAX_REPORT_BYTES = 32_768
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
    "invocation_id",
    "pid",
    "isaac_sim_version",
    "started_at",
    "finished_at",
    "inputs",
    "initialization_operations",
    "handle_validity_method",
)
_RUNTIME_FIELD_SOURCES = {
    "path": "runtime", "name": "runtime", "joint_type": "runtime",
    "lower_limit": "runtime", "upper_limit": "runtime", "index": "runtime",
    "axis": "layer1", "body0": "layer1", "body1": "layer1",
}
_FLOAT32_REVOLUTE_DEGREES_ABS_TOL = 1e-5
_FLOAT32_PRISMATIC_ABS_TOL = 1e-7


def _compare_runtime_records(expected, observed):
    expected_paths = [record.get("path") for record in expected]
    observed_paths = [record.get("path") for record in observed]
    expected_path_set = set(expected_paths)
    observed_path_set = set(observed_paths)
    validation_errors = [
        {"side": side, **error}
        for side, records in (("expected", expected), ("runtime", observed))
        for error in validate_dof_records(records)["errors"]
    ]
    missing = sorted(expected_path_set - observed_path_set, key=repr)
    unexpected = sorted(observed_path_set - expected_path_set, key=repr)
    if missing or unexpected or validation_errors:
        mismatches = [
            {
                "field": "runtime_path_inventory",
                "index": None,
                "expected": expected_paths,
                "observed": observed_paths,
            }
        ]
        return {
            "ok": False,
            "mismatches": mismatches,
            "validation_errors": validation_errors,
            "failure_code": "runtime_inventory_mismatch",
        }

    runtime_by_path = {record["path"]: record for record in observed}
    normalized = []
    source_errors = []
    for canonical_index, authored in enumerate(expected):
        raw_runtime = runtime_by_path[authored["path"]]
        runtime = dict(raw_runtime)
        runtime["index"] = canonical_index
        normalized.append(runtime)
        if runtime.get("field_sources") != _RUNTIME_FIELD_SOURCES:
            source_errors.append(
                {
                    "code": "invalid_field_sources",
                    "runtime_index": raw_runtime.get("index"),
                    "path": raw_runtime.get("path"),
                }
            )
        for field in ("lower_limit", "upper_limit"):
            left, right = authored.get(field), runtime.get(field)
            tolerance = (
                _FLOAT32_REVOLUTE_DEGREES_ABS_TOL
                if authored.get("joint_type") == "PhysicsRevoluteJoint"
                else _FLOAT32_PRISMATIC_ABS_TOL
            )
            if (
                isinstance(left, int | float)
                and not isinstance(left, bool)
                and isinstance(right, int | float)
                and not isinstance(right, bool)
                and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
            ):
                runtime[field] = left
    comparison = compare_dof_records(expected, normalized)
    if source_errors:
        comparison["ok"] = False
        comparison.setdefault("validation_errors", []).extend(source_errors)
    if not comparison["ok"]:
        comparison["failure_code"] = "runtime_semantic_metadata_mismatch"
    return comparison


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None


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
        try:
            adapter = build_order_adapter(layer1.get("policy_contract"), expected)
        except (KeyError, TypeError, ValueError):
            details.append("policy_contract")
        else:
            if adapter.get("canonical_order") != [record.get("path") for record in expected]:
                details.append("policy_contract_paths")
            if round_trip_check(adapter).get("status") != "PASS":
                details.append("policy_contract_round_trip")

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

    return [{"code": "invalid_layer1_evidence", "details": details}] if details else []


def _run_errors(
    run: object, run_index: int, expected: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    errors: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    if not isinstance(run, dict):
        return ([{"code": "invalid_run_shape", "run_index": run_index}], [], False)

    missing = [field for field in _REQUIRED_RUN_FIELDS if field not in run]
    errors.extend({"code": "missing_field", "run_index": run_index, "field": field} for field in missing)

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

    pid = run.get("pid")
    if "pid" in run and (type(pid) is not int or pid <= 0):
        errors.append({"code": "invalid_pid", "run_index": run_index})
    version = run.get("isaac_sim_version")
    if "isaac_sim_version" in run and not (isinstance(version, str) and bool(version.strip())):
        errors.append({"code": "invalid_isaac_sim_version", "run_index": run_index})
    invocation = run.get("invocation_id")
    if "invocation_id" in run and not (isinstance(invocation, str) and bool(invocation.strip())):
        errors.append({"code": "invalid_invocation_id", "run_index": run_index})
    inputs = run.get("inputs")
    if "inputs" in run:
        if not isinstance(inputs, dict):
            errors.append({"code": "invalid_inputs", "run_index": run_index})
        else:
            for name in ("stage", "mapping", "config"):
                item = inputs.get(name)
                if not isinstance(item, dict) or not _valid_sha256(item.get("sha256")):
                    errors.append({"code": "invalid_input_hash", "run_index": run_index, "input": name})
    parsed_times = {field: _timestamp(run.get(field)) for field in ("started_at", "finished_at")}
    for field, parsed in parsed_times.items():
        if field in run and parsed is None:
            errors.append({"code": "invalid_timestamp", "run_index": run_index, "field": field})
    if all(parsed_times.values()) and parsed_times["finished_at"] < parsed_times["started_at"]:
        errors.append({"code": "reversed_timestamps", "run_index": run_index})

    operations = run.get("initialization_operations")
    operations_valid = isinstance(operations, list) and all(
        isinstance(item, str) and bool(item.strip()) for item in operations
    )
    if "initialization_operations" in run and not operations_valid:
        errors.append({"code": "invalid_initialization_operations", "run_index": run_index})
    elif operations_valid:
        if run.get("requires_unapproved_initialization") is True and not operations:
            errors.append({"code": "missing_required_initialization_operations", "run_index": run_index})
        if run.get("requires_unapproved_initialization") is False and operations:
            errors.append({"code": "unexpected_initialization_operations", "run_index": run_index})

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
    reported_failures = run.get("errors")
    runtime_api_blocked = (
        blocked
        and isinstance(reported_failures, list)
        and len(reported_failures) == 1
        and isinstance(reported_failures[0], dict)
        and reported_failures[0].get("code") == "forbidden_initialization_required"
        and reported_failures[0].get("required_operation") in {
            "timeline.play", "physics.step", "physics.update_simulation"
        }
        and isinstance(reported_failures[0].get("source_api"), str)
        and bool(reported_failures[0]["source_api"].strip())
        and operations == [reported_failures[0]["required_operation"]]
    )
    if run.get("status") not in (_RUN_PASS, _BLOCKED):
        errors.append({"code": "invalid_run_status", "run_index": run_index})
    elif run.get("status") == _BLOCKED and not blocked:
        errors.append({"code": "invalid_blocked_evidence", "run_index": run_index})
    elif run.get("status") == _RUN_PASS and run.get("requires_unapproved_initialization") is not False:
        errors.append({"code": "unexpected_initialization_requirement", "run_index": run_index})

    if not runtime_api_blocked and run.get("articulation_root") != "/aloha/root_joint":
        errors.append({"code": "invalid_articulation_root", "run_index": run_index})
    if not runtime_api_blocked and type(run.get("articulation_count")) is int and run["articulation_count"] != 1:
        errors.append({"code": "invalid_articulation_count", "run_index": run_index})
    if not runtime_api_blocked and type(run.get("dof_count")) is int and run["dof_count"] != 16:
        errors.append({"code": "invalid_dof_count", "run_index": run_index})
    if not blocked and run.get("valid_handle") is not True:
        errors.append({"code": "invalid_handle", "run_index": run_index})
    if not blocked and run.get("handle_validity_method") != "tensor_view_structural_proof_v1":
        errors.append({"code": "invalid_handle_validity_method", "run_index": run_index})

    records = run.get("records")
    if not (isinstance(records, list) and all(isinstance(record, dict) for record in records)):
        errors.append({"code": "invalid_records_shape", "run_index": run_index})
    elif not runtime_api_blocked:
        comparison = _compare_runtime_records(expected, records)
        if not comparison["ok"]:
            errors.append(
                {
                    "code": comparison.get(
                        "failure_code", "runtime_semantic_metadata_mismatch"
                    ),
                    "run_index": run_index,
                }
            )
            mismatches.extend({"run_index": run_index, **mismatch} for mismatch in comparison["mismatches"])
            mismatches.extend(
                {"run_index": run_index, "validation_error": validation_error}
                for validation_error in comparison.get("validation_errors", [])
            )

    return errors, mismatches, blocked


def aggregate_runtime_runs(layer1: object, runs: object) -> dict[str, Any]:
    """Aggregate exactly three saved runtime run dictionaries, fail-closed."""
    errors = _layer1_errors(layer1)
    mismatches: list[dict[str, Any]] = []
    run_count = len(runs) if isinstance(runs, list) else None
    if run_count != 3:
        errors.append({"code": "invalid_run_count", "expected": 3, "observed": run_count})

    blocked_runs: list[int] = []
    order_adapter: dict[str, Any] | None = None
    raw_order_matches_canonical: bool | None = None
    if not errors and isinstance(layer1, dict) and isinstance(runs, list):
        expected = layer1["expected"]
        for run_index, run in enumerate(runs):
            run_errors, run_mismatches, blocked = _run_errors(run, run_index, expected)
            errors.extend(run_errors)
            mismatches.extend(run_mismatches)
            if blocked and not run_errors:
                blocked_runs.append(run_index)

        valid_runs = [run for run in runs if isinstance(run, dict)]
        invocations = [run.get("invocation_id") for run in valid_runs]
        versions = [run.get("isaac_sim_version") for run in valid_runs]
        if all(isinstance(value, str) for value in invocations) and len(set(invocations)) != len(invocations):
            errors.append({"code": "duplicate_invocation"})
        if all(isinstance(value, str) for value in versions) and len(set(versions)) != 1:
            errors.append({"code": "isaac_version_mismatch"})
        fingerprints = [json.dumps(run.get("inputs"), sort_keys=True, default=repr) for run in valid_runs]
        if len(set(fingerprints)) != 1:
            errors.append({"code": "input_hash_mismatch"})
        runtime_order_fingerprints = [
            json.dumps(
                [record.get("path") for record in run.get("records", [])],
                default=repr,
            )
            if isinstance(run.get("records"), list)
            else "invalid"
            for run in valid_runs
        ]
        if len(set(runtime_order_fingerprints)) != 1:
            errors.append({"code": "runtime_order_nondeterministic"})
        if valid_runs and isinstance(valid_runs[0].get("records"), list):
            try:
                order_adapter = build_order_adapter(
                    layer1["policy_contract"], valid_runs[0]["records"]
                )
                order_adapter["round_trip_check"] = round_trip_check(order_adapter)
                if order_adapter["round_trip_check"].get("status") != "PASS":
                    errors.append({"code": "order_adapter_round_trip_failed"})
                raw_order_matches_canonical = order_adapter.get(
                    "runtime_order"
                ) == [record.get("path") for record in expected]
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(
                    {"code": "order_adapter_validation_failed", "message": str(exc)}
                )
        layer_stage = layer1["inputs"]["stage"]["post_sha256"]
        if any(
            not isinstance(run.get("inputs"), dict)
            or not isinstance(run["inputs"].get("stage"), dict)
            or run["inputs"]["stage"].get("sha256") != layer_stage
            for run in valid_runs
        ):
            errors.append({"code": "layer1_stage_hash_mismatch"})
        for previous, current in pairwise(valid_runs):
            previous_start = _timestamp(previous.get("started_at"))
            previous_finish = _timestamp(previous.get("finished_at"))
            current_start = _timestamp(current.get("started_at"))
            if previous_start and current_start and current_start < previous_start:
                errors.append({"code": "nonmonotonic_start_timestamp"})
            if previous_finish and current_start and current_start < previous_finish:
                errors.append({"code": "run_timestamp_overlap"})

    if errors:
        blocked_runs = []
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
        "expected": layer1.get("expected") if isinstance(layer1, dict) else None,
        "order_adapter": order_adapter,
        "raw_order_matches_canonical": raw_order_matches_canonical,
    }


def is_exact_runtime_pass(payload: object, trusted_layer1: object = None) -> bool:
    """Allow automation to continue only on the complete, no-step A20 pass contract."""
    if not (
        not _layer1_errors(trusted_layer1)
        and isinstance(trusted_layer1, dict)
        and isinstance(payload, dict)
        and payload.get("status") == _PASS
        and payload.get("ok") is True
        and payload.get("errors") == []
        and payload.get("mismatches") == []
        and payload.get("run_count") == 3
        and payload.get("blocked_run_indices") == []
        and isinstance(payload.get("runs"), list)
        and len(payload["runs"]) == 3
        and isinstance(payload.get("expected"), list)
        and len(payload["expected"]) == 16
        and validate_dof_records(payload["expected"])["ok"]
        and all(payload.get(flag) is False for flag in ("physics_stepped", "actions_applied", "targets_written", "stage_saved"))
    ):
        return False
    _, live_errors = _trusted_layer1_inputs(trusted_layer1)
    if live_errors:
        return False
    trusted_expected = trusted_layer1["expected"]
    if not compare_dof_records(trusted_expected, payload["expected"])["ok"]:
        return False
    reaggregated = aggregate_runtime_runs(trusted_layer1, payload["runs"])
    if reaggregated["status"] != _PASS or reaggregated["errors"] or reaggregated["mismatches"]:
        return False
    if (
        payload.get("order_adapter") != reaggregated.get("order_adapter")
        or payload.get("raw_order_matches_canonical")
        is not reaggregated.get("raw_order_matches_canonical")
    ):
        return False
    trusted_inputs = trusted_layer1["inputs"]
    trusted_hashes = {
        "stage": trusted_inputs["stage"]["post_sha256"],
        "mapping": trusted_inputs["mapping"]["sha256"],
        "config": trusted_inputs["config"]["sha256"],
    }
    for index, run in enumerate(payload["runs"]):
        if not isinstance(run, dict):
            return False
        run_errors, run_mismatches, blocked = _run_errors(run, index, trusted_expected)
        provenance = run.get("provenance")
        if (
            run_errors
            or run_mismatches
            or blocked
            or run.get("cleanup_verified") is not True
            or not isinstance(provenance, dict)
            or provenance.get("schema_version") != SCHEMA_VERSION
            or not isinstance(provenance.get("safety_checker"), dict)
            or provenance["safety_checker"].get("ok") is not True
            or any(run.get("inputs", {}).get(name, {}).get("sha256") != digest for name, digest in trusted_hashes.items())
        ):
            return False
    return True


def _artifact_status(payload: object) -> str:
    if not isinstance(payload, dict) or not isinstance(payload.get("status"), str) or not isinstance(payload.get("ok"), bool):
        return "MALFORMED_OR_MISSING"
    return payload["status"]


def _count(payload: object, field: str) -> str:
    value = payload.get(field) if isinstance(payload, dict) else None
    return str(len(value)) if isinstance(value, list) else "unknown"


def _short(value: object) -> str:
    return value[:12] if isinstance(value, str) and value else "unknown"


def _bounded_field(value: object, limit: int) -> str:
    single_line = " ".join(str(value if value is not None else "unknown").split())
    single_line = re.sub(r"javascript:", "javascript&#58;", single_line, flags=re.IGNORECASE)
    single_line = html.escape(single_line, quote=True)
    single_line = re.sub(r"([`\[\]()])", r"\\\1", single_line)
    if len(single_line) <= limit:
        return single_line
    return f"{single_line[:limit]} [truncated]"


def _valid_asset_validator(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    status = payload.get("status")
    if status == "PASS_A20_ASSET_VALIDATOR_READ_ONLY_NO_BLOCKING_ISSUES":
        return payload.get("ok") is True and payload.get("blocking_issue_count") == 0 and payload.get("issues") == []
    if status == "FAIL_A20_ASSET_VALIDATOR_BLOCKING_ISSUES":
        issues = payload.get("issues")
        return payload.get("ok") is False and isinstance(issues, list) and bool(issues)
    return False


def _asset_validator_clean(payload: object) -> bool:
    return bool(
        _valid_asset_validator(payload)
        and isinstance(payload, dict)
        and payload.get("status") == "PASS_A20_ASSET_VALIDATOR_READ_ONLY_NO_BLOCKING_ISSUES"
        and payload.get("ok") is True
        and payload.get("blocking_issue_count") == 0
        and payload.get("issues") == []
    )


def _render_bool(payload: object, field: str) -> str:
    value = payload.get(field) if isinstance(payload, dict) else None
    if isinstance(value, bool):
        return str(value).lower()
    runs = payload.get("runs") if isinstance(payload, dict) else None
    if isinstance(runs, list) and len(runs) == 3 and all(isinstance(run, dict) and isinstance(run.get(field), bool) for run in runs):
        return str(any(run[field] for run in runs)).lower()
    return "unknown"


def _report_runtime_semantics(
    layer1: object, layer2: object
) -> tuple[str, str, str, str]:
    status = layer2.get("status") if isinstance(layer2, dict) else None
    if status == _BLOCKED:
        return "BLOCKED", "BLOCKED", "BLOCKED", "BLOCKED"
    runs = layer2.get("runs") if isinstance(layer2, dict) else None
    expected = layer1.get("expected") if isinstance(layer1, dict) else None
    if not (
        isinstance(runs, list)
        and len(runs) == 3
        and all(isinstance(run, dict) for run in runs)
        and isinstance(expected, list)
    ):
        return "FAIL", "FAIL", "FAIL", "FAIL"

    runtime_fields = (
        "status", "process_status", "returncode", "probe_returncode", "timed_out", "cleanup_verified",
        "requires_unapproved_initialization", "initialization_operations",
        "isaac_sim_version", "inputs", "articulation_root", "articulation_count",
        "dof_count", "valid_handle", "handle_validity_method", "records",
        "physics_stepped", "actions_applied", "targets_written", "stage_saved",
    )
    provenance_fields = (
        "schema_version", "probe_sha256", "coordinator_sha256", "safety_checker_sha256",
        "git_head", "git_dirty",
    )
    fingerprints = []
    for run in runs:
        provenance = run.get("provenance") if isinstance(run.get("provenance"), dict) else {}
        facts = {field: run.get(field) for field in runtime_fields}
        facts["provenance"] = {field: provenance.get(field) for field in provenance_fields}
        fingerprints.append(json.dumps(facts, sort_keys=True, default=repr))
    determinism = "PASS" if len(set(fingerprints)) == 1 else "FAIL"
    records_semantically_match = all(
        _compare_runtime_records(expected, run["records"])["ok"]
        for run in runs
        if isinstance(run.get("records"), list)
    ) and all(isinstance(run.get("records"), list) for run in runs)
    adapter = layer2.get("order_adapter") if isinstance(layer2, dict) else None
    mapping_status = (
        "PASS"
        if isinstance(adapter, dict) and adapter.get("mapping_complete") is True
        else "FAIL"
    )
    round_trip_status = (
        "PASS"
        if isinstance(adapter, dict)
        and isinstance(adapter.get("round_trip_check"), dict)
        and adapter["round_trip_check"].get("status") == "PASS"
        else "FAIL"
    )
    semantic_status = "PASS" if records_semantically_match else "FAIL"
    return determinism, semantic_status, mapping_status, round_trip_status


def format_two_layer_report(asset_validator: object, layer1: object, layer2: object) -> str:
    """Render bounded, fail-closed A20 evidence without embedding Isaac logs."""
    asset_status = _artifact_status(asset_validator)
    layer1_status = _artifact_status(layer1)
    layer2_status = _artifact_status(layer2)
    overall = "READY" if _asset_validator_clean(asset_validator) and not _layer1_errors(layer1) and is_exact_runtime_pass(layer2, layer1) else "NOT_READY"

    issues = asset_validator.get("issues") if isinstance(asset_validator, dict) else None
    bounded_issues = issues[:MAX_REPORT_ISSUES] if isinstance(issues, list) else []
    issue_lines = [
        f"- Blocking issue: [{_bounded_field(issue.get('severity'), 40)}] "
        f"{_bounded_field(issue.get('rule'), 80)} at {_bounded_field(issue.get('at'), 160)}: "
        f"{_bounded_field(issue.get('message'), 280)} "
        f"(suggestion: {_bounded_field(issue.get('suggestion'), 200)})"
        for issue in bounded_issues
        if isinstance(issue, dict)
    ]
    if isinstance(issues, list) and len(issues) > MAX_REPORT_ISSUES:
        issue_lines.append(f"- [truncated] {len(issues) - MAX_REPORT_ISSUES} additional issues omitted.")
    if not issue_lines:
        issue_lines.append("- Blocking issue: none recorded (artifact is malformed/missing unless status is clean).")

    expected = layer1.get("expected") if isinstance(layer1, dict) else None
    observed = layer1.get("observed") if isinstance(layer1, dict) else None
    layer1_stage = layer1.get("inputs", {}).get("stage", {}) if isinstance(layer1, dict) and isinstance(layer1.get("inputs"), dict) else {}
    layer2_runs = layer2.get("runs") if isinstance(layer2, dict) else None
    all_operations = sorted(
        {
            operation
            for run in layer2_runs or []
            if isinstance(run, dict)
            for operation in run.get("initialization_operations", [])
            if isinstance(operation, str)
        }
    ) if isinstance(layer2_runs, list) else []
    operations = all_operations[:MAX_REPORT_OPERATIONS]
    provenance = layer2.get("provenance", {}) if isinstance(layer2, dict) and isinstance(layer2.get("provenance"), dict) else {}
    blocked = layer2_status == _BLOCKED
    determinism, semantic_match, mapping_status, round_trip_status = (
        _report_runtime_semantics(layer1, layer2)
    )
    raw_order_matches_canonical = (
        layer2.get("raw_order_matches_canonical")
        if isinstance(layer2, dict) and not blocked
        else None
    )
    raw_order_summary = (
        "yes"
        if raw_order_matches_canonical is True
        else "no"
        if raw_order_matches_canonical is False
        else "unknown"
    )
    next_action = (
        "The runtime handle requires timeline Play and a physics simulation step; these operations were not approved."
        if blocked
        else "No blocked-runtime action is authorized by this report."
    )
    if operations:
        rendered_operations = ", ".join(_bounded_field(operation, 160) for operation in operations)
        omitted = len(all_operations) - len(operations)
        omitted_note = f" [truncated] {omitted} additional operations omitted;" if omitted else ""
        next_action = f"Required operations: {rendered_operations};{omitted_note} these operations were not approved."

    report = "\n".join(
        [
            "# A20 two-layer articulation discovery gate",
            "",
            f"Overall: {_bounded_field(overall, 40)}",
            "",
            "## Asset Validator",
            "",
            f"- Status: {_bounded_field(asset_status, 120)}",
            f"- Blocking issue count: {_bounded_field(asset_validator.get('blocking_issue_count', 'unknown') if isinstance(asset_validator, dict) else 'unknown', 40)}",
            *issue_lines,
            "- Independence: A two-layer PASS does not mean Asset Validator is clean; this gate remains separate.",
            "",
            "## Layer 1",
            "",
            f"- Status: {_bounded_field(layer1_status, 120)}",
            f"- Expected DOFs: {len(expected) if isinstance(expected, list) else 'unknown'}",
            f"- Observed DOFs: {len(observed) if isinstance(observed, list) else 'unknown'}",
            f"- Mismatches: {_count(layer1, 'mismatches')}",
            f"- Stage: {_bounded_field(layer1_stage.get('path', 'unknown') if isinstance(layer1_stage, dict) else 'unknown', 240)}",
            f"- Stage SHA-256: {_bounded_field(_short(layer1_stage.get('post_sha256') if isinstance(layer1_stage, dict) else None), 20)}",
            "",
            "## Layer 2",
            "",
            f"- Status: {_bounded_field(layer2_status, 120)}",
            f"- Runs: {layer2.get('run_count', 'unknown') if isinstance(layer2, dict) else 'unknown'}",
            f"- Three-run raw runtime determinism: {determinism}",
            f"- Runtime joint semantic match: {semantic_match}",
            f"- Policy-to-runtime mapping: {mapping_status}",
            f"- Policy/runtime round trip: {round_trip_status}",
            f"- Raw order equals canonical order: {raw_order_summary} (informational)",
            f"- Errors: {_count(layer2, 'errors')}",
            f"- Mismatches: {_count(layer2, 'mismatches')}",
            "- Exit contract: BLOCKED=2, PASS=0, FAIL=1",
            f"- Git revision: {_bounded_field(_short(provenance.get('git_head')), 20)}",
            f"- Probe SHA-256: {_bounded_field(_short(provenance.get('probe_sha256')), 20)}",
            f"- Coordinator SHA-256: {_bounded_field(_short(provenance.get('coordinator_sha256')), 20)}",
            f"- Report generation ID: {_bounded_field(layer2.get('report_generation_id') if isinstance(layer2, dict) else None, 80)}",
            f"- Runtime evidence SHA-256: {_short(hashlib.sha256(json.dumps(layer2, sort_keys=True, default=repr).encode()).hexdigest())}",
            f"- Next action: {next_action}",
            "",
            "## Safety and readiness",
            "",
            f"- Physics stepped: {_render_bool(layer2, 'physics_stepped')}",
            f"- Actions applied: {_render_bool(layer2, 'actions_applied')}",
            f"- Targets written: {_render_bool(layer2, 'targets_written')}",
            f"- Stage saved: {_render_bool(layer2, 'stage_saved')}",
            "- Collision ready: false",
            "- Control ready: false",
            "- Replay ready: false",
            "- Contact ready: false",
            "- Training ready: false",
            "",
            "This report is a bounded summary. Consult the local JSON artifacts for complete structured evidence.",
            "",
        ]
    )
    if len(report.encode("utf-8")) > MAX_REPORT_BYTES:
        return "# A20 two-layer articulation discovery gate\n\nOverall: NOT_READY\n\n- Status: FAIL_REPORT_SIZE_LIMIT\n- [truncated] Report exceeded bounded UTF-8 size.\n"
    return report


MARKER = "A20_RUNTIME_DISCOVERY_JSON="
DEFAULT_CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml")
SCHEMA_VERSION = "a20-runtime-discovery-v2"
DEFAULT_OUTPUT_CAP = 1024 * 1024
DEFAULT_MARKER_CAP = 256 * 1024


def _digest(path: Path) -> str:
    value = __import__("hashlib").sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def check_probe_source(source: str) -> dict[str, Any]:
    """Fail-closed static boundary for the exact probe bytes executed by the parent."""
    errors: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"ok": False, "errors": [f"syntax_error:{exc.msg}"]}
    forbidden_calls = {
        "play", "step", "reset", "initialize", "initialize_async", "set_joint_positions", "set_joint_velocities",
        "set_joint_efforts", "apply_action", "save", "Save", "Export", "Flatten",
        "getattr", "setattr", "__import__", "exec", "eval", "import_module", "open", "update", "update_simulation",
    }
    reviewed_calls = {
        "ArgumentParser", "Path", "RuntimeDiscoveryError", "SimulationApp", "SystemExit",
        "__init__", "_as_list", "_call_runtime_api", "_digest", "_discover_runtime_records",
        "_emit_marker", "_now", "_safe_version", "add_argument", "append", "bool", "close",
        "create_articulation_view", "create_simulation_view", "cwd", "degrees", "dumps",
        "enumerate", "expected_dof_records", "file_digest", "float", "force_load_physics_from_usd",
        "get", "get_context", "get_dof_limits", "get_physx_interface", "get_stage_id", "getpid",
        "hasattr", "hexdigest", "int", "isinstance", "isoformat", "len", "list", "loads", "main",
        "now", "open", "open_stage", "operation", "parse_args", "printer", "read_text", "replace",
        "resolve", "safe_load", "serializer", "set", "set_subspace_roots", "sorted", "start_simulation", "str", "strftime", "super",
        "tolist", "version", "zip",
    }

    def is_forbidden(name: str) -> bool:
        return name in forbidden_calls or name.startswith("set_joint")

    reviewed_imports = {
        ("importfrom", "__future__", "annotations", None),
        ("import", "argparse", None, None),
        ("importfrom", "datetime", "UTC", None),
        ("importfrom", "datetime", "datetime", None),
        ("import", "hashlib", None, None),
        ("import", "importlib.metadata", None, None),
        ("import", "json", None, None),
        ("import", "math", None, None),
        ("import", "os", None, None),
        ("importfrom", "pathlib", "Path", None),
        ("import", "yaml", None, None),
        ("importfrom", "isaacsim", "SimulationApp", None),
        ("importfrom", "omni.physics", "tensors", None),
        ("importfrom", "omni.physx", "get_physx_interface", None),
        ("import", "omni.usd", None, None),
        ("importfrom", "aloha_isaac_rebuild.scripts.audit_a20_usd_dof_metadata", "expected_dof_records", None),
    }
    aliases: dict[str, str] = {}

    def qualified_name(node: ast.AST) -> str | None:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
            return ".".join(reversed(parts))
        return None

    reviewed_runtime_callables = {
        "usd_context.open_stage",
        "physics_interface.force_load_physics_from_usd",
        "physics_interface.start_simulation",
        "usd_context.get_stage_id",
        "tensors_module.create_simulation_view",
        "simulation_view.set_subspace_roots",
        "simulation_view.create_articulation_view",
        "articulation_view.get_dof_limits",
    }
    reviewed_attribute_calls = {
        "argparse.ArgumentParser", "parser.add_argument", "parser.parse_args", "value.tolist",
        "records.append", "path.open", "importlib.metadata.version", "os.getpid", "yaml.safe_load",
        "json.loads", "json.dumps", "config_path.read_text", "mapping_path.read_text",
        "omni.usd.get_context", "math.degrees", "hashlib.file_digest", "Path.cwd", "datetime.now",
        "app.close",
        "context.open_stage", "interface.force_load_physics_from_usd", "interface.start_simulation",
        "tensors.create_simulation_view", "context.get_stage_id", "view.set_subspace_roots",
        "view.create_articulation_view", "articulation.get_dof_limits",
    }
    allowed_app_store_nodes: set[int] = set()
    for candidate in ast.walk(tree):
        if not (
            isinstance(candidate, ast.Assign)
            and len(candidate.targets) == 1
            and isinstance(candidate.targets[0], ast.Name)
            and candidate.targets[0].id == "app"
        ):
            continue
        none_assignment = isinstance(candidate.value, ast.Constant) and candidate.value.value is None
        simulation_assignment = (
            isinstance(candidate.value, ast.Call)
            and isinstance(candidate.value.func, ast.Name)
            and candidate.value.func.id == "SimulationApp"
            and len(candidate.value.args) == 1
            and not candidate.value.keywords
            and isinstance(candidate.value.args[0], ast.Dict)
            and len(candidate.value.args[0].keys) == 1
            and isinstance(candidate.value.args[0].keys[0], ast.Constant)
            and candidate.value.args[0].keys[0].value == "headless"
            and isinstance(candidate.value.args[0].values[0], ast.Constant)
            and candidate.value.args[0].values[0].value is True
        )
        if none_assignment or simulation_assignment:
            allowed_app_store_nodes.add(id(candidate.targets[0]))

    for candidate in ast.walk(tree):
        if (
            isinstance(candidate, ast.Name)
            and candidate.id == "app"
            and isinstance(candidate.ctx, ast.Store)
            and id(candidate) not in allowed_app_store_nodes
        ):
            errors.append("app_binding_not_allowed:name_store")
        elif isinstance(candidate, ast.arg) and candidate.arg == "app":
            errors.append("app_binding_not_allowed:argument")
        elif isinstance(candidate, ast.ExceptHandler) and candidate.name == "app":
            errors.append("app_binding_not_allowed:except")
        elif isinstance(candidate, ast.MatchAs) and candidate.name == "app":
            errors.append("app_binding_not_allowed:match_as")
        elif isinstance(candidate, ast.MatchStar) and candidate.name == "app":
            errors.append("app_binding_not_allowed:match_star")
        elif isinstance(candidate, ast.MatchMapping) and candidate.rest == "app":
            errors.append("app_binding_not_allowed:match_mapping_rest")
        elif isinstance(candidate, ast.Global | ast.Nonlocal) and "app" in candidate.names:
            errors.append("app_binding_not_allowed:scope_declaration")
        elif isinstance(candidate, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and candidate.name == "app":
            errors.append("app_binding_not_allowed:definition")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if ("import", alias.name, None, alias.asname) not in reviewed_imports:
                    errors.append(f"import_not_allowed:{alias.name}")
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if ("importfrom", module, alias.name, alias.asname) not in reviewed_imports:
                    errors.append(f"import_not_allowed:{module}.{alias.name}")
                aliases[alias.asname or alias.name] = f"{module}.{alias.name}"
        elif isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else "dynamic"
            reviewed_path_open = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "open"
                and qualified_name(node.func.value) == "path"
                and bool(node.args)
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "rb"
            )
            if (is_forbidden(name) and not reviewed_path_open) or name == "dynamic":
                errors.append(f"call_not_allowed:{name}")
            elif name not in reviewed_calls:
                errors.append(f"unreviewed_call:{name}")
            if isinstance(node.func, ast.Name) and node.func.id == "_call_runtime_api":
                callable_name = qualified_name(node.args[1]) if len(node.args) >= 2 else None
                if callable_name not in reviewed_runtime_callables:
                    errors.append(f"runtime_callable_not_allowed:{callable_name}")
            if isinstance(node.func, ast.Attribute) and node.func.attr == "open":
                receiver = qualified_name(node.func.value)
                mode = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else None
                if receiver != "path" or mode != "rb":
                    errors.append(f"open_not_allowed:{receiver}:{mode}")
            if isinstance(node.func, ast.Attribute):
                target = qualified_name(node.func)
                receiver_node = node.func.value
                repo_path_resolve = (
                    node.func.attr == "resolve"
                    and isinstance(receiver_node, ast.BinOp)
                    and isinstance(receiver_node.op, ast.Div)
                    and isinstance(receiver_node.left, ast.Name)
                    and receiver_node.left.id == "repo"
                    and (
                        (
                            isinstance(receiver_node.right, ast.Name)
                            and receiver_node.right.id == "CONFIG"
                        )
                        or (
                            isinstance(receiver_node.right, ast.Subscript)
                            and isinstance(receiver_node.right.value, ast.Name)
                            and receiver_node.right.value.id == "outputs"
                        )
                    )
                )
                nested_ok = (
                    node.func.attr == "strftime"
                    and isinstance(receiver_node, ast.Call)
                    and qualified_name(receiver_node.func) == "datetime.now"
                ) or (
                    node.func.attr == "__init__"
                    and isinstance(receiver_node, ast.Call)
                    and isinstance(receiver_node.func, ast.Name)
                    and receiver_node.func.id == "super"
                ) or (
                    repo_path_resolve
                    and (
                        isinstance(receiver_node, ast.BinOp)
                    )
                ) or (
                    node.func.attr == "resolve"
                    and isinstance(receiver_node, ast.Call)
                    and qualified_name(receiver_node.func) == "Path.cwd"
                ) or (
                    node.func.attr == "hexdigest"
                    and isinstance(receiver_node, ast.Call)
                    and qualified_name(receiver_node.func) == "hashlib.file_digest"
                )
                if target not in reviewed_attribute_calls and not nested_ok:
                    errors.append(f"attribute_call_not_allowed:{target or node.func.attr}")
                if target == "app.close" and (node.args or node.keywords):
                    errors.append("app_close_arguments_not_allowed")
            if isinstance(node.func, ast.Name) and node.func.id in aliases:
                target = aliases[node.func.id].rsplit(".", 1)[-1]
                if is_forbidden(target):
                    errors.append(f"aliased_call_not_allowed:{target}")
        elif (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Attribute)
            and is_forbidden(node.value.attr)
        ):
            errors.append(f"attribute_alias_not_allowed:{node.value.attr}")
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and is_forbidden(node.value.id):
            errors.append(f"name_alias_not_allowed:{node.value.id}")
        elif (
            isinstance(node, ast.AnnAssign | ast.NamedExpr)
            and isinstance(node.value, ast.Attribute)
            and is_forbidden(node.value.attr)
        ):
            errors.append(f"attribute_alias_not_allowed:{node.value.attr}")
        elif (
            isinstance(node, ast.AnnAssign | ast.NamedExpr)
            and isinstance(node.value, ast.Name)
            and is_forbidden(node.value.id)
        ):
            errors.append(f"name_alias_not_allowed:{node.value.id}")
    return {"ok": not errors, "errors": sorted(set(errors))}


def _code_provenance(repo_root: Path, probe_path: Path, coordinator_path: Path) -> dict[str, Any]:
    probe_bytes = probe_path.read_bytes()
    checker = check_probe_source(probe_bytes.decode("utf-8"))
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--", str(probe_path), str(coordinator_path)],
                cwd=repo_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        head, dirty = "unknown", True
    return {
        "schema_version": SCHEMA_VERSION,
        "probe_sha256": hashlib.sha256(probe_bytes).hexdigest(),
        "coordinator_sha256": hashlib.sha256(coordinator_path.read_bytes()).hexdigest(),
        "git_head": head,
        "git_dirty": dirty,
        "safety_checker": checker,
        "safety_checker_sha256": hashlib.sha256(inspect.getsource(check_probe_source).encode()).hexdigest(),
    }


def _trusted_layer1_inputs(layer1: dict[str, Any]) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    trusted: dict[str, dict[str, str]] = {}
    errors: list[dict[str, Any]] = []
    inputs = layer1.get("inputs") if isinstance(layer1, dict) else None
    if not isinstance(inputs, dict):
        return {}, [{"code": "invalid_layer1_inputs"}]
    for name in ("config", "mapping", "stage"):
        item = inputs.get(name)
        raw_path = item.get("path") if isinstance(item, dict) else None
        expected_hash = item.get("post_sha256") if name == "stage" and isinstance(item, dict) else item.get("sha256") if isinstance(item, dict) else None
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            errors.append({"code": "layer1_path_not_absolute", "input": name})
            continue
        path = Path(raw_path).resolve()
        if str(path) != raw_path:
            errors.append({"code": "layer1_path_not_canonical", "input": name})
            continue
        try:
            actual_hash = _digest(path)
        except OSError as exc:
            errors.append({"code": "layer1_input_unreadable", "input": name, "message": str(exc)})
            continue
        if actual_hash != expected_hash:
            errors.append({"code": "layer1_live_hash_mismatch", "input": name})
        trusted[name] = {"path": str(path), "sha256": actual_hash}
    return trusted, errors


def _terminate_process_group(process: subprocess.Popen[bytes], grace: float = 1.0) -> bool:
    def group_active() -> bool:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().split()
                if int(fields[4]) == process.pid and fields[2] != "Z":
                    return True
            except (FileNotFoundError, PermissionError, IndexError, ValueError):
                continue
        return False

    if group_active():
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    if process.poll() is None:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=grace)
    deadline = time.monotonic() + grace
    while group_active() and time.monotonic() < deadline:
        time.sleep(0.02)
    if group_active():
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        if process.poll() is None:
            try:
                process.wait(timeout=max(grace, 1.0))
            except subprocess.TimeoutExpired:
                return False
        deadline = time.monotonic() + max(grace, 1.0)
        while group_active() and time.monotonic() < deadline:
            time.sleep(0.02)
    return not group_active()


def _execute_probe(
    argv: list[str], cwd: Path, timeout_seconds: float, output_cap: int = DEFAULT_OUTPUT_CAP,
    marker_cap: int = DEFAULT_MARKER_CAP,
) -> dict[str, Any]:
    """Execute one probe in a new process group with bounded streaming capture."""
    started_wall = datetime.now().astimezone().isoformat()
    started_mono = time.monotonic()
    process = subprocess.Popen(
        argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    selector = selectors.DefaultSelector()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
        assert stream is not None
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, name)
    timed_out = False
    exceeded = False
    try:
        while selector.get_map():
            remaining = timeout_seconds - (time.monotonic() - started_mono)
            if remaining <= 0:
                timed_out = True
                break
            for key, _ in selector.select(min(0.1, remaining)):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                target = buffers[key.data]
                allowed = max(0, output_cap - len(target))
                target.extend(chunk[:allowed])
                if len(chunk) > allowed:
                    exceeded = True
                    break
            if exceeded:
                break
        if not timed_out and not exceeded:
            process.wait(timeout=max(0.1, timeout_seconds - (time.monotonic() - started_mono)))
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        selector.close()
    cleanup_verified = _terminate_process_group(process)
    stdout = buffers["stdout"].decode("utf-8", "replace")
    stderr = buffers["stderr"].decode("utf-8", "replace")
    marker_bytes = sum(len(line.encode()) for line in stdout.splitlines() if line.startswith(MARKER))
    if marker_bytes > marker_cap:
        exceeded = True
    return {
        "process_status": "timeout" if timed_out else "output_limit_exceeded" if exceeded else "completed" if process.returncode == 0 else "failed",
        "returncode": process.returncode if process.returncode is not None else -1,
        "timed_out": timed_out,
        "output_limit_exceeded": exceeded,
        "stdout": stdout,
        "stderr": stderr,
        "observed_pid": process.pid,
        "parent_started_at": started_wall,
        "parent_finished_at": datetime.now().astimezone().isoformat(),
        "parent_monotonic_started": started_mono,
        "parent_monotonic_finished": time.monotonic(),
        "cleanup_verified": cleanup_verified,
    }


def _exit_code(status: str) -> int:
    return 0 if status == _PASS else 2 if status == _BLOCKED else 1


def _summary(value: object, limit: int = 4000) -> str:
    text = "" if value is None else str(value)
    return text[-limit:]


def run_three_probes(
    layer1: dict[str, Any],
    repo_root: Path,
    interpreter: Path,
    probe_path: Path,
    timeout_seconds: float,
    run_command=None,
    invocation_ids: list[str] | None = None,
) -> dict[str, Any]:
    ids = invocation_ids or [str(uuid.uuid4()) for _ in range(3)]
    errors: list[dict[str, Any]] = []
    if len(ids) != 3 or len(set(ids)) != 3:
        return {
            "status": _FAIL,
            "ok": False,
            "runs": [],
            "errors": [{"code": "invalid_invocation_ids"}],
            "mismatches": [],
        }
    runs: list[dict[str, Any]] = []
    provenance = _code_provenance(repo_root, probe_path, Path(__file__).resolve())
    if not provenance["safety_checker"]["ok"]:
        return {"status": _FAIL, "ok": False, "runs": [], "errors": [{"code": "unsafe_probe_source", "details": provenance["safety_checker"]["errors"]}], "mismatches": [], "provenance": provenance}
    trusted_inputs: dict[str, dict[str, str]] | None = None
    if run_command is None:
        trusted_inputs, trust_errors = _trusted_layer1_inputs(layer1)
        if trust_errors:
            return {"status": _FAIL, "ok": False, "runs": [], "errors": trust_errors, "mismatches": [], "provenance": provenance}
    for index, invocation_id in enumerate(ids):
        argv = [str(interpreter), "-u", str(probe_path), "--invocation-id", invocation_id]
        try:
            if run_command is None:
                execution = _execute_probe(argv, repo_root, timeout_seconds)
                stdout, stderr = execution["stdout"], execution["stderr"]
                returncode = execution["returncode"]
            else:
                completed = run_command(argv, cwd=repo_root, timeout=timeout_seconds, check=False)
                stdout, stderr = completed.stdout or "", completed.stderr or ""
                returncode = completed.returncode
                execution = {
                    "process_status": "completed" if returncode == 0 else "failed",
                    "timed_out": False,
                    "observed_pid": None,
                    "cleanup_verified": True,
                    "parent_started_at": None,
                    "parent_finished_at": None,
                    "parent_monotonic_started": None,
                    "parent_monotonic_finished": None,
                }
            markers = [line[len(MARKER) :] for line in stdout.splitlines() if line.startswith(MARKER)]
            if len(markers) != 1:
                raise ValueError(f"marker_count:{len(markers)}")
            payload = json.loads(markers[0])
            if not isinstance(payload, dict):
                raise ValueError("payload_not_object")
            payload.update(
                process_status=execution["process_status"],
                returncode=returncode,
                timed_out=execution["timed_out"],
                stdout_summary=_summary(stdout),
                stderr_summary=_summary(stderr),
                observed_pid=execution["observed_pid"] if execution["observed_pid"] is not None else payload.get("pid"),
                parent_started_at=execution["parent_started_at"],
                parent_finished_at=execution["parent_finished_at"],
                parent_monotonic_started=execution["parent_monotonic_started"],
                parent_monotonic_finished=execution["parent_monotonic_finished"],
                cleanup_verified=execution["cleanup_verified"],
                provenance=provenance,
            )
            if payload.get("invocation_id") != invocation_id:
                errors.append({"code": "invocation_mismatch", "run_index": index})
            if execution["observed_pid"] is not None and payload.get("pid") != execution["observed_pid"]:
                errors.append({"code": "observed_pid_mismatch", "run_index": index})
            if trusted_inputs is not None and payload.get("inputs") != trusted_inputs:
                errors.append({"code": "layer1_input_path_or_hash_mismatch", "run_index": index})
            if execution["parent_started_at"] is not None:
                child_start, child_finish = _timestamp(payload.get("started_at")), _timestamp(payload.get("finished_at"))
                parent_start, parent_finish = _timestamp(execution["parent_started_at"]), _timestamp(execution["parent_finished_at"])
                if not child_start or not child_finish or child_start < parent_start or child_finish > parent_finish:
                    errors.append({"code": "child_timestamp_outside_parent_bounds", "run_index": index})
            runs.append(payload)
        except subprocess.TimeoutExpired as exc:
            runs.append(
                {
                    "status": _FAIL,
                    "process_status": "timeout",
                    "returncode": -1,
                    "timed_out": True,
                    "invocation_id": invocation_id,
                    "stdout_summary": _summary(exc.output),
                    "stderr_summary": _summary(exc.stderr),
                }
            )
            errors.append({"code": "subprocess_timeout", "run_index": index})
        except Exception as exc:
            runs.append(
                {
                    "status": _FAIL,
                    "process_status": "protocol_error",
                    "returncode": -1,
                    "timed_out": False,
                    "invocation_id": invocation_id,
                }
            )
            errors.append({"code": "probe_protocol_error", "run_index": index, "message": str(exc)})
    valid = [run for run in runs if "pid" in run]
    if len({run.get("invocation_id") for run in valid}) != len(valid):
        errors.append({"code": "duplicate_invocation"})
    if valid:
        versions = {run.get("isaac_sim_version") for run in valid}
        fingerprints = {json.dumps(run.get("inputs"), sort_keys=True) for run in valid}
        if len(versions) != 1:
            errors.append({"code": "isaac_version_mismatch"})
        if len(fingerprints) != 1:
            errors.append({"code": "input_hash_mismatch"})
        layer_stage = layer1.get("inputs", {}).get("stage", {}).get("post_sha256")
        if any(run.get("inputs", {}).get("stage", {}).get("sha256") != layer_stage for run in valid):
            errors.append({"code": "layer1_stage_hash_mismatch"})
        for previous, current in pairwise(valid):
            if str(previous.get("finished_at")) > str(current.get("started_at")):
                errors.append({"code": "run_timestamp_overlap"})
    aggregate = aggregate_runtime_runs(layer1, runs)
    aggregate["runs"] = runs
    aggregate["invocation_ids"] = ids
    aggregate["schema_version"] = SCHEMA_VERSION
    aggregate["provenance"] = provenance
    if errors:
        aggregate["status"], aggregate["ok"] = _FAIL, False
        aggregate["errors"] = errors + aggregate["errors"]
    return aggregate


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def _load_json_fail_closed(path: Path) -> object:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"status": "MALFORMED_OR_MISSING", "ok": False, "artifact_path": str(path)}
    if not isinstance(payload, dict):
        return {"status": "MALFORMED_OR_MISSING", "ok": False, "artifact_path": str(path)}
    return payload


def _fail_result(result: object, code: str) -> dict[str, Any]:
    payload = dict(result) if isinstance(result, dict) else {}
    payload["status"], payload["ok"] = _FAIL, False
    errors = payload.get("errors") if isinstance(payload.get("errors"), list) else []
    errors.append({"code": code})
    payload["errors"] = errors
    payload["mismatches"] = payload.get("mismatches") if isinstance(payload.get("mismatches"), list) else []
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--interpreter", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--report-from-existing", action="store_true")
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    outputs = config["outputs"]
    layer1_path = repo / outputs["a20_usd_dof_metadata_json"]
    asset_path = repo / outputs["a20_asset_validator_json"]
    report_path = (repo / outputs["a20_two_layer_articulation_discovery_md"]).resolve()
    sentinel = "# A20 two-layer articulation discovery gate\n\nOverall: NOT_READY\n\n- Status: REPORT_GENERATION_IN_PROGRESS\n"
    try:
        _atomic_write_text(report_path, sentinel)
    except Exception:
        report_path.unlink(missing_ok=True)
    layer1 = _load_json_fail_closed(layer1_path)
    output = (repo / outputs["a20_runtime_articulation_discovery_json"]).resolve()
    if args.report_from_existing:
        result = _load_json_fail_closed(output)
    else:
        result = run_three_probes(
            layer1,
            repo,
            args.interpreter,
            repo / "aloha_isaac_rebuild/scripts/probe_a20_runtime_articulation_once.py",
            args.timeout,
        )
        for flag in ("physics_stepped", "actions_applied", "targets_written", "stage_saved"):
            runs = result.get("runs", [])
            result[flag] = not (len(runs) == 3 and all(run.get(flag) is False for run in runs))
        _atomic_write(output, result)
    live_inputs_before, live_errors_before = _trusted_layer1_inputs(layer1)
    if live_errors_before or (result.get("status") == _PASS and not is_exact_runtime_pass(result, layer1)):
        result = _fail_result(result, "trusted_layer1_live_validation_failed" if live_errors_before else "exact_runtime_pass_contract_failed")
        _atomic_write(output, result)
    asset_validator = _load_json_fail_closed(asset_path)
    if any(payload.get("status") == "MALFORMED_OR_MISSING" for payload in (layer1, result, asset_validator)):
        result = _fail_result(result, "malformed_or_missing_artifact")
    result["report_generation_id"] = str(uuid.uuid4())
    report = format_two_layer_report(asset_validator, layer1, result)
    live_inputs_after, live_errors_after = _trusted_layer1_inputs(layer1)
    if live_errors_after or live_inputs_after != live_inputs_before:
        result = _fail_result(result, "trusted_layer1_changed_during_report")
        report = format_two_layer_report(asset_validator, layer1, result)
    _atomic_write(output, result)
    try:
        _atomic_write_text(report_path, report)
        live_inputs_final, live_errors_final = _trusted_layer1_inputs(layer1)
        if live_errors_final or live_inputs_final != live_inputs_before:
            result = _fail_result(result, "trusted_layer1_changed_after_report")
            _atomic_write(output, result)
            _atomic_write_text(report_path, format_two_layer_report(asset_validator, layer1, result))
    except Exception as exc:
        report_path.unlink(missing_ok=True)
        if not isinstance(result, dict):
            result = {}
        result["status"], result["ok"] = _FAIL, False
        errors = result.get("errors") if isinstance(result.get("errors"), list) else []
        errors.append({"code": "report_write_failed", "message": str(exc)})
        result["errors"] = errors
        _atomic_write(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return _exit_code(result.get("status"))


if __name__ == "__main__":
    raise SystemExit(main())
