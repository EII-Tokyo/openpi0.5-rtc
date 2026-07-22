from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import pytest

from aloha_isaac_rebuild.scripts.run_a20_runtime_articulation_discovery import aggregate_runtime_runs

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "aloha_isaac_rebuild/scripts/run_a20_runtime_articulation_discovery.py"


def _record(index: int) -> dict[str, object]:
    name = f"joint_{index:02d}"
    return {
        "path": f"/aloha/joints/{name}",
        "name": name,
        "joint_type": "PhysicsRevoluteJoint",
        "axis": "X",
        "lower_limit": -1.0,
        "upper_limit": 1.0,
        "body0": [f"/aloha/link_{index:02d}"],
        "body1": [f"/aloha/link_{index + 1:02d}"],
        "index": index,
    }


def _layer1() -> dict[str, object]:
    records = [_record(index) for index in range(16)]
    return {
        "status": "PASS_A20_USD_DOF_METADATA",
        "ok": True,
        "expected": records,
        "observed": deepcopy(records),
        "mismatches": [],
        "errors": [],
        "inputs": {
            "stage": {
                "path": "/workspace/a19_clean_articulation_candidate.usda",
                "pre_sha256": "a" * 64,
                "post_sha256": "a" * 64,
                "consistent_during_audit": True,
            },
            "mapping": {"path": "/workspace/a17.json", "sha256": "b" * 64},
            "config": {"path": "/workspace/config.yaml", "sha256": "c" * 64},
        },
        "physics_stepped": False,
        "actions_applied": False,
        "targets_written": False,
        "stage_saved": False,
    }


def _run() -> dict[str, object]:
    return {
        "status": "PASS_RUNTIME_PROBE",
        "process_status": "completed",
        "returncode": 0,
        "timed_out": False,
        "articulation_root": "/aloha/root_joint",
        "articulation_count": 1,
        "dof_count": 16,
        "valid_handle": True,
        "records": [_record(index) for index in range(16)],
        "requires_unapproved_initialization": False,
        "physics_stepped": False,
        "actions_applied": False,
        "targets_written": False,
        "stage_saved": False,
    }


def _runs() -> list[dict[str, object]]:
    return [deepcopy(_run()) for _ in range(3)]


def test_three_exact_saved_runs_pass() -> None:
    result = aggregate_runtime_runs(_layer1(), _runs())

    assert result["status"] == "PASS_A20_RUNTIME_ARTICULATION_DISCOVERY_NO_STEP"
    assert result["ok"] is True
    assert result["errors"] == []
    assert result["mismatches"] == []
    assert result["run_count"] == 3


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        (lambda runs: runs[1]["records"].reverse(), "runtime_records_mismatch"),
        (lambda runs: runs[1].update(valid_handle=False), "invalid_handle"),
        (lambda runs: runs[1].update(articulation_count=2), "invalid_articulation_count"),
        (lambda runs: runs[1].update(dof_count=15), "invalid_dof_count"),
        (lambda runs: runs[1].update(physics_stepped=True), "prohibited_safety_flag"),
        (lambda runs: runs[1].update(process_status="failed", returncode=1), "subprocess_failure"),
        (lambda runs: runs[1].update(process_status="timeout", timed_out=True), "subprocess_failure"),
    ],
)
def test_runtime_mismatch_or_unsafe_run_fails(mutation, error_code: str) -> None:
    runs = _runs()
    mutation(runs)

    result = aggregate_runtime_runs(_layer1(), runs)

    assert result["status"] == "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"
    assert result["ok"] is False
    assert any(error["code"] == error_code for error in result["errors"])


def test_structurally_valid_blocked_run_has_blocked_status() -> None:
    runs = _runs()
    runs[1].update(
        status="BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION",
        valid_handle=False,
        requires_unapproved_initialization=True,
    )

    result = aggregate_runtime_runs(_layer1(), runs)

    assert result["status"] == "BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION"
    assert result["ok"] is False
    assert result["errors"] == []


def test_malformed_blocked_run_fails_instead_of_masking_error() -> None:
    runs = _runs()
    runs[0].update(
        status="BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION",
        valid_handle=False,
        requires_unapproved_initialization=True,
        physics_stepped=True,
    )

    result = aggregate_runtime_runs(_layer1(), runs)

    assert result["status"] == "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"
    assert any(error["code"] == "prohibited_safety_flag" for error in result["errors"])


@pytest.mark.parametrize("run_count", [0, 1, 2, 4])
def test_requires_exactly_three_runs(run_count: int) -> None:
    runs = _runs()[:run_count] if run_count < 3 else [*_runs(), _run()]

    result = aggregate_runtime_runs(_layer1(), runs)

    assert result["status"] == "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"
    assert result["errors"][0] == {
        "code": "invalid_run_count",
        "expected": 3,
        "observed": run_count,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda layer1: layer1.update(status="FAIL_A20_USD_DOF_METADATA"),
        lambda layer1: layer1.update(ok=False),
        lambda layer1: layer1["expected"].pop(),
        lambda layer1: layer1["observed"].reverse(),
        lambda layer1: layer1["mismatches"].append({"field": "path"}),
        lambda layer1: layer1["errors"].append({"code": "bad_input"}),
        lambda layer1: layer1["inputs"]["stage"].update(post_sha256="d" * 64),
        lambda layer1: layer1.update(physics_stepped=True),
    ],
)
def test_invalid_layer1_evidence_fails_closed(mutation) -> None:
    layer1 = _layer1()
    mutation(layer1)

    result = aggregate_runtime_runs(layer1, _runs())

    assert result["status"] == "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"
    assert any(error["code"] == "invalid_layer1_evidence" for error in result["errors"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda run: run.pop("records"),
        lambda run: run.pop("valid_handle"),
        lambda run: run.update(valid_handle=1),
        lambda run: run.pop("physics_stepped"),
        lambda run: run.update(actions_applied=0),
        lambda run: run.update(timed_out="false"),
        lambda run: run.update(requires_unapproved_initialization="false"),
    ],
)
def test_missing_fields_and_non_bool_values_fail_closed(mutation) -> None:
    runs = _runs()
    mutation(runs[2])

    result = aggregate_runtime_runs(_layer1(), runs)

    assert result["status"] == "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"
    assert result["errors"]


def test_blocked_status_requires_explicit_initialization_marker() -> None:
    runs = _runs()
    runs[2].update(
        status="BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION",
        valid_handle=False,
    )

    result = aggregate_runtime_runs(_layer1(), runs)

    assert result["status"] == "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"


def test_module_is_pure_and_has_no_runtime_or_subprocess_imports() -> None:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    prohibited = ("isaacsim", "omni", "pxr", "subprocess")
    assert not any(name == prefix or name.startswith(f"{prefix}.") for name in imports for prefix in prohibited)
