from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from aloha_isaac_rebuild.scripts.run_a20_runtime_articulation_discovery import aggregate_runtime_runs
from aloha_isaac_rebuild.scripts.run_a20_runtime_articulation_discovery import run_three_probes

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "aloha_isaac_rebuild/scripts/run_a20_runtime_articulation_discovery.py"
PROBE = ROOT / "aloha_isaac_rebuild/scripts/probe_a20_runtime_articulation_once.py"
MARKER = "A20_RUNTIME_DISCOVERY_JSON="


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
        "invocation_id": "placeholder",
        "pid": 1,
        "isaac_sim_version": "5.1.0.0",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "inputs": {"stage": {"sha256": "a" * 64}, "mapping": {"sha256": "b" * 64}, "config": {"sha256": "c" * 64}},
        "initialization_operations": [],
    }


def _runs() -> list[dict[str, object]]:
    runs = [deepcopy(_run()) for _ in range(3)]
    for index, run in enumerate(runs):
        run.update(
            invocation_id=f"run-{index}",
            pid=index + 1,
            started_at=f"2026-01-01T00:00:0{index * 2}+00:00",
            finished_at=f"2026-01-01T00:00:0{index * 2 + 1}+00:00",
        )
    return runs


def _set_layer1_hash(layer1: dict[str, object], location: str, invalid_hash: str) -> None:
    inputs = layer1["inputs"]
    if location == "stage":
        inputs["stage"]["pre_sha256"] = invalid_hash
        inputs["stage"]["post_sha256"] = invalid_hash
    else:
        inputs[location]["sha256"] = invalid_hash


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("returncode", False),
        ("returncode", 0.0),
        ("returncode", "0"),
        ("returncode", None),
        ("articulation_count", True),
        ("articulation_count", 1.0),
        ("articulation_count", "1"),
        ("articulation_count", None),
        ("dof_count", True),
        ("dof_count", 16.0),
        ("dof_count", "16"),
        ("dof_count", None),
    ],
)
def test_runtime_integer_fields_reject_bool_float_string_and_none(field: str, value: object) -> None:
    runs = _runs()
    runs[1][field] = value

    result = aggregate_runtime_runs(_layer1(), runs)

    assert result["status"] == "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"
    assert {
        "code": "invalid_field_type",
        "run_index": 1,
        "field": field,
        "expected": "int",
        "observed_type": type(value).__name__,
    } in result["errors"]


def test_structurally_valid_blocked_run_has_blocked_status() -> None:
    runs = _runs()
    runs[1].update(
        status="BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION",
        valid_handle=False,
        requires_unapproved_initialization=True,
        initialization_operations=["timeline Play"],
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
        initialization_operations=["timeline Play"],
        physics_stepped=True,
    )

    result = aggregate_runtime_runs(_layer1(), runs)

    assert result["status"] == "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"
    assert any(error["code"] == "prohibited_safety_flag" for error in result["errors"])
    assert result["blocked_run_indices"] == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda run: run.pop("records"),
        lambda run: run.update(process_status="failed", returncode=1),
        lambda run: run.update(actions_applied=True),
        lambda run: run["records"].reverse(),
    ],
)
def test_invalid_blocked_run_is_not_reported_as_blocked(mutation) -> None:
    runs = _runs()
    runs[1].update(
        status="BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION",
        valid_handle=False,
        requires_unapproved_initialization=True,
        initialization_operations=["timeline Play"],
    )
    mutation(runs[1])

    result = aggregate_runtime_runs(_layer1(), runs)

    assert result["status"] == "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"
    assert result["blocked_run_indices"] == []


@pytest.mark.parametrize("location", ["config", "mapping", "stage"])
@pytest.mark.parametrize(
    "invalid_hash",
    [
        "+" + "a" * 63,
        "-" + "a" * 63,
        " " + "a" * 63,
        "A" * 64,
        "g" * 64,
        "a" * 63,
    ],
    ids=["plus", "minus", "whitespace", "uppercase", "nonhex", "wrong_length"],
)
def test_layer1_hashes_require_exact_lowercase_sha256(location: str, invalid_hash: str) -> None:
    layer1 = _layer1()
    _set_layer1_hash(layer1, location, invalid_hash)

    result = aggregate_runtime_runs(layer1, _runs())

    assert result["status"] == "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"
    assert any(error["code"] == "invalid_layer1_evidence" for error in result["errors"])


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


def test_module_has_no_isaac_runtime_imports() -> None:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names} | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }

    prohibited = ("isaacsim", "omni", "pxr")
    assert not any(name == prefix or name.startswith(f"{prefix}.") for name in imports for prefix in prohibited)


def test_probe_source_has_static_safety_boundary_and_four_flags() -> None:
    tree = ast.parse(PROBE.read_text(encoding="utf-8"))
    forbidden = {
        "play",
        "step",
        "reset",
        "initialize_simulation_context_async",
        "set_joint_positions",
        "set_joint_velocities",
        "set_joint_efforts",
        "apply_action",
        "save",
        "Save",
        "Export",
        "Flatten",
    }
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    imports = "\n".join(
        [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        + [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    ).lower()
    assert not (forbidden & (attrs | calls))
    assert "controller" not in imports
    assert "action" not in imports
    source = PROBE.read_text(encoding="utf-8")
    for flag in ("physics_stepped", "actions_applied", "targets_written", "stage_saved"):
        assert flag in source


def _probe_payload(invocation: str, pid: int, start: str, end: str) -> dict[str, object]:
    run = _run()
    run.update(
        invocation_id=invocation,
        pid=pid,
        started_at=start,
        finished_at=end,
        isaac_sim_version="5.1.0",
        inputs={"stage": {"sha256": "a" * 64}, "mapping": {"sha256": "b" * 64}, "config": {"sha256": "c" * 64}},
    )
    return run


def test_coordinator_runs_three_fresh_sequential_processes_with_strict_argv() -> None:
    calls = []
    payloads = [
        _probe_payload("i0", 101, "2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z"),
        _probe_payload("i1", 102, "2026-01-01T00:00:02Z", "2026-01-01T00:00:03Z"),
        _probe_payload("i2", 103, "2026-01-01T00:00:04Z", "2026-01-01T00:00:05Z"),
    ]

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        invocation = argv[argv.index("--invocation-id") + 1]
        payload = payloads[len(calls) - 1]
        payload["invocation_id"] = invocation
        return subprocess.CompletedProcess(argv, 0, MARKER + json.dumps(payload) + "\n", "")

    result = run_three_probes(
        layer1=_layer1(),
        repo_root=ROOT,
        interpreter=Path("/isaac/python"),
        probe_path=PROBE,
        timeout_seconds=9,
        run_command=fake_run,
        invocation_ids=["i0", "i1", "i2"],
    )
    assert result["status"] == "PASS_A20_RUNTIME_ARTICULATION_DISCOVERY_NO_STEP"
    assert len(calls) == 3
    for argv, kwargs in calls:
        assert isinstance(argv, list)
        assert argv[0] == "/isaac/python"
        assert kwargs == {"cwd": ROOT, "timeout": 9, "capture_output": True, "text": True, "check": False}
    assert [run["pid"] for run in result["runs"]] == [101, 102, 103]


@pytest.mark.parametrize("mode", ["timeout", "nonzero", "missing", "multiple", "malformed", "mismatch"])
def test_coordinator_protocol_failures_are_structured(mode: str) -> None:
    def fake_run(argv, **kwargs):
        invocation = argv[argv.index("--invocation-id") + 1]
        payload = _probe_payload(invocation, 100 + len(invocation), "2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z")
        if mode == "timeout":
            raise subprocess.TimeoutExpired(argv, 9, output="partial", stderr="late")
        if mode == "nonzero":
            return subprocess.CompletedProcess(argv, 7, MARKER + json.dumps(payload), "bad")
        if mode == "missing":
            return subprocess.CompletedProcess(argv, 0, "none", "")
        if mode == "multiple":
            return subprocess.CompletedProcess(argv, 0, (MARKER + json.dumps(payload) + "\n") * 2, "")
        if mode == "malformed":
            return subprocess.CompletedProcess(argv, 0, MARKER + "{", "")
        payload["invocation_id"] = "wrong"
        return subprocess.CompletedProcess(argv, 0, MARKER + json.dumps(payload), "")

    result = run_three_probes(_layer1(), ROOT, Path("/isaac/python"), PROBE, 9, fake_run, ["a", "bb", "ccc"])
    assert result["status"] == "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"
    assert result["errors"]


@pytest.mark.parametrize("mutation", ["pid", "version", "hash", "time"])
def test_cross_run_identity_version_hash_and_time_must_match(mutation: str) -> None:
    count = 0

    def fake_run(argv, **kwargs):
        nonlocal count
        invocation = argv[argv.index("--invocation-id") + 1]
        payload = _probe_payload(
            invocation, 200 + count, f"2026-01-01T00:00:0{count * 2}Z", f"2026-01-01T00:00:0{count * 2 + 1}Z"
        )
        if count == 1 and mutation == "pid":
            payload["pid"] = 200
        if count == 1 and mutation == "version":
            payload["isaac_sim_version"] = "5.0.0"
        if count == 1 and mutation == "hash":
            payload["inputs"]["stage"]["sha256"] = "d" * 64
        if count == 1 and mutation == "time":
            payload["started_at"] = "2025-01-01T00:00:00Z"
        count += 1
        return subprocess.CompletedProcess(argv, 0, MARKER + json.dumps(payload), "")

    result = run_three_probes(_layer1(), ROOT, Path("/isaac/python"), PROBE, 9, fake_run, ["a", "b", "c"])
    assert result["status"] == "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"


def test_all_runs_missing_process_provenance_fields_fail_with_missing_fields() -> None:
    runs = _runs()
    for run in runs:
        for field in ("pid", "isaac_sim_version", "started_at", "finished_at"):
            run.pop(field, None)
    result = aggregate_runtime_runs(_layer1(), runs)
    assert result["status"] == "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"
    missing = {
        (error.get("run_index"), error.get("field")) for error in result["errors"] if error["code"] == "missing_field"
    }
    assert missing == {
        (index, field) for index in range(3) for field in ("pid", "isaac_sim_version", "started_at", "finished_at")
    }


@pytest.mark.parametrize("field", ["pid", "isaac_sim_version", "started_at", "finished_at"])
def test_each_process_provenance_field_is_required(field: str) -> None:
    runs = _runs_with_provenance()
    runs[1].pop(field)
    result = aggregate_runtime_runs(_layer1(), runs)
    assert {"code": "missing_field", "run_index": 1, "field": field} in result["errors"]


def _runs_with_provenance() -> list[dict[str, object]]:
    runs = _runs()
    for index, run in enumerate(runs):
        run.update(
            invocation_id=f"run-{index}",
            pid=300 + index,
            isaac_sim_version="5.1.0.0",
            started_at=f"2026-01-01T00:00:0{index * 2}+00:00",
            finished_at=f"2026-01-01T00:00:0{index * 2 + 1}+00:00",
            inputs={"stage": {"sha256": "a" * 64}, "mapping": {"sha256": "b" * 64}, "config": {"sha256": "c" * 64}},
        )
    return runs


@pytest.mark.parametrize("pid", [True, 1.0, "1", None, 0, -1])
def test_pid_requires_exact_positive_integer(pid: object) -> None:
    runs = _runs_with_provenance()
    runs[1]["pid"] = pid
    result = aggregate_runtime_runs(_layer1(), runs)
    assert result["status"] == "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"
    assert any(error["code"] == "invalid_pid" for error in result["errors"])


@pytest.mark.parametrize("version", [None, 5.1, "", "   "])
def test_version_requires_nonempty_string(version: object) -> None:
    runs = _runs_with_provenance()
    runs[1]["isaac_sim_version"] = version
    result = aggregate_runtime_runs(_layer1(), runs)
    assert any(error["code"] == "invalid_isaac_sim_version" for error in result["errors"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("started_at", "not-a-time"),
        ("finished_at", ""),
        ("started_at", "2026-01-01T00:00:00"),
        ("finished_at", 123),
    ],
)
def test_timestamps_must_be_parseable_timezone_aware_strings(field: str, value: object) -> None:
    runs = _runs_with_provenance()
    runs[1][field] = value
    result = aggregate_runtime_runs(_layer1(), runs)
    assert any(error["code"] == "invalid_timestamp" for error in result["errors"])


def test_finished_timestamp_cannot_precede_started_timestamp() -> None:
    runs = _runs_with_provenance()
    runs[1]["finished_at"] = "2025-01-01T00:00:00+00:00"
    result = aggregate_runtime_runs(_layer1(), runs)
    assert any(error["code"] == "reversed_timestamps" for error in result["errors"])


@pytest.mark.parametrize(
    "mutation", ["duplicate_pid", "duplicate_invocation", "version_mismatch", "started_nonmonotonic", "overlap"]
)
def test_three_run_process_provenance_is_cross_validated(mutation: str) -> None:
    runs = _runs_with_provenance()
    if mutation == "duplicate_pid":
        runs[1]["pid"] = runs[0]["pid"]
    elif mutation == "duplicate_invocation":
        runs[1]["invocation_id"] = runs[0]["invocation_id"]
    elif mutation == "version_mismatch":
        runs[1]["isaac_sim_version"] = "5.0.0"
    elif mutation == "started_nonmonotonic":
        runs[1]["started_at"] = "2025-01-01T00:00:00+00:00"
    else:
        runs[1]["started_at"] = runs[0]["started_at"]
    result = aggregate_runtime_runs(_layer1(), runs)
    assert result["status"] == "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"


def test_all_runs_missing_initialization_operations_fail_as_missing_fields() -> None:
    runs = _runs()
    for run in runs:
        run.pop("initialization_operations")
    result = aggregate_runtime_runs(_layer1(), runs)
    assert [error for error in result["errors"] if error["code"] == "missing_field"] == [
        {"code": "missing_field", "run_index": index, "field": "initialization_operations"} for index in range(3)
    ]


def test_single_missing_initialization_operations_fails() -> None:
    runs = _runs()
    runs[1].pop("initialization_operations")
    result = aggregate_runtime_runs(_layer1(), runs)
    assert {"code": "missing_field", "run_index": 1, "field": "initialization_operations"} in result["errors"]


@pytest.mark.parametrize("value", [None, "play", 1, {}, [""], ["  "], [1], ["play", None]])
def test_initialization_operations_requires_exact_list_of_nonempty_strings(value: object) -> None:
    runs = _runs()
    runs[1]["initialization_operations"] = value
    result = aggregate_runtime_runs(_layer1(), runs)
    assert any(error["code"] == "invalid_initialization_operations" for error in result["errors"])


def test_blocked_requires_nonempty_initialization_operations() -> None:
    runs = _runs()
    runs[1].update(
        status="BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION",
        valid_handle=False,
        requires_unapproved_initialization=True,
        initialization_operations=[],
    )
    result = aggregate_runtime_runs(_layer1(), runs)
    assert any(error["code"] == "missing_required_initialization_operations" for error in result["errors"])


def test_pass_requires_empty_initialization_operations() -> None:
    runs = _runs()
    runs[1]["initialization_operations"] = ["timeline Play"]
    result = aggregate_runtime_runs(_layer1(), runs)
    assert any(error["code"] == "unexpected_initialization_operations" for error in result["errors"])
