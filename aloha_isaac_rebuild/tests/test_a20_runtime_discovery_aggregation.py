from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import types

import pytest
import yaml

from aloha_isaac_rebuild.scripts import run_a20_runtime_articulation_discovery as coordinator
from aloha_isaac_rebuild.scripts.run_a20_runtime_articulation_discovery import _atomic_write
from aloha_isaac_rebuild.scripts.run_a20_runtime_articulation_discovery import _code_provenance
from aloha_isaac_rebuild.scripts.run_a20_runtime_articulation_discovery import _execute_probe
from aloha_isaac_rebuild.scripts.run_a20_runtime_articulation_discovery import _exit_code
from aloha_isaac_rebuild.scripts.run_a20_runtime_articulation_discovery import _trusted_layer1_inputs
from aloha_isaac_rebuild.scripts.run_a20_runtime_articulation_discovery import aggregate_runtime_runs
from aloha_isaac_rebuild.scripts.run_a20_runtime_articulation_discovery import check_probe_source
from aloha_isaac_rebuild.scripts.run_a20_runtime_articulation_discovery import format_two_layer_report
from aloha_isaac_rebuild.scripts.run_a20_runtime_articulation_discovery import is_exact_runtime_pass
from aloha_isaac_rebuild.scripts.run_a20_runtime_articulation_discovery import run_three_probes

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "aloha_isaac_rebuild/scripts/run_a20_runtime_articulation_discovery.py"
PROBE = ROOT / "aloha_isaac_rebuild/scripts/probe_a20_runtime_articulation_once.py"
MARKER = "A20_RUNTIME_DISCOVERY_JSON="
LIVE_STAGE_HASH = hashlib.sha256(MODULE.read_bytes()).hexdigest()
LIVE_MAPPING_HASH = hashlib.sha256(PROBE.read_bytes()).hexdigest()


def test_strict_probe_checker_rejects_dynamic_and_alias_bypasses() -> None:
    assert check_probe_source("import json\njson.dumps({})\n")["ok"] is True
    for source in (
        "getattr(stage, 'Save')()\n",
        "__import__('omni.timeline')\n",
        "from importlib import import_module\nimport_module('omni')\n",
        "runner = app.update\nrunner()\n",
        "app.update()\n",
        "runner: object = app.update\nrunner()\n",
        "(runner := app.update)()\n",
        "runner = eval\nrunner('1 + 1')\n",
        "from external_helper import inspect_stage\ninspect_stage()\n",
    ):
        result = check_probe_source(source)
        assert result["ok"] is False, source


@pytest.mark.parametrize(
    "call",
    [
        "play",
        "step",
        "reset",
        "initialize",
        "initialize_async",
        "set_joint_positions",
        "set_joint_custom_target",
        "set_joint_efforts",
        "apply_action",
        "save",
        "Export",
        "Flatten",
        "exec",
        "eval",
    ],
)
def test_strict_probe_checker_rejects_every_direct_forbidden_call(call: str) -> None:
    result = check_probe_source(f"runtime.{call}()\n")
    assert result["ok"] is False


def test_probe_checker_allows_only_the_reviewed_no_step_runtime_discovery_calls() -> None:
    source = (
        "import omni.usd\n"
        "from omni.physics import tensors\n"
        "from omni.physx import get_physx_interface\n"
        "context = omni.usd.get_context()\n"
        "context.open_stage('candidate.usda')\n"
        "interface = get_physx_interface()\n"
        "interface.force_load_physics_from_usd()\n"
        "interface.start_simulation()\n"
        "view = tensors.create_simulation_view('numpy', stage_id=context.get_stage_id())\n"
        "view.set_subspace_roots('/')\n"
        "articulation = view.create_articulation_view(['/aloha/root_joint'])\n"
        "articulation.get_dof_limits()\n"
    )
    assert check_probe_source(source)["ok"] is True
    assert check_probe_source(source + "interface.update_simulation(0.0, 0.0)\n")["ok"] is False


def test_runtime_discovery_builds_records_from_real_tensor_view() -> None:
    namespace = runpy.run_path(str(PROBE), run_name="probe_runtime_records_test")
    calls: list[object] = []

    class FakeMetadata:
        def __init__(self):
            self.dof_names = ["joint_00", "joint_01"]
            self.dof_types = [types.SimpleNamespace(name="Rotation"), types.SimpleNamespace(name="Translation")]

    class FakeArticulationView:
        def __init__(self):
            self.count = 1
            self.max_dofs = 2
            self.prim_paths = ["/aloha/root_joint"]
            self.dof_paths = [["/aloha/joints/joint_00", "/aloha/joints/joint_01"]]
            self.shared_metatype = FakeMetadata()

        def get_dof_limits(self):
            calls.append("get_dof_limits")
            return [[[-1.25, 1.5], [0.01, 0.04]]]

    class FakeSimulationView:
        def set_subspace_roots(self, root):
            calls.append(("set_subspace_roots", root))

        def create_articulation_view(self, paths):
            calls.append(("create_articulation_view", paths))
            return FakeArticulationView()

    class FakeTensors:
        @staticmethod
        def create_simulation_view(backend, *, stage_id):
            calls.append(("create_simulation_view", backend, stage_id))
            return FakeSimulationView()

    class FakePhysics:
        def force_load_physics_from_usd(self):
            calls.append("force_load_physics_from_usd")

        def start_simulation(self):
            calls.append("start_simulation")

    class FakeContext:
        def open_stage(self, path):
            calls.append(("open_stage", path))
            return True

        def get_stage_id(self):
            return 42

    expected = [_record(0), _record(1)]
    records, facts = namespace["_discover_runtime_records"](
        "/tmp/candidate.usda", expected, FakeContext(), FakePhysics(), FakeTensors
    )

    assert calls == [
        ("open_stage", "/tmp/candidate.usda"),
        "force_load_physics_from_usd",
        "start_simulation",
        ("create_simulation_view", "numpy", 42),
        ("set_subspace_roots", "/"),
        ("create_articulation_view", ["/aloha/root_joint"]),
        "get_dof_limits",
    ]
    assert facts == {"articulation_root": "/aloha/root_joint", "articulation_count": 1, "dof_count": 2}
    assert [record["name"] for record in records] == ["joint_00", "joint_01"]
    assert [record["path"] for record in records] == ["/aloha/joints/joint_00", "/aloha/joints/joint_01"]
    assert [record["joint_type"] for record in records] == ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint"]
    assert [(record["lower_limit"], record["upper_limit"]) for record in records] == [
        (math.degrees(-1.25), math.degrees(1.5)),
        (0.01, 0.04),
    ]


def test_runtime_discovery_failure_names_the_exact_failed_api() -> None:
    namespace = runpy.run_path(str(PROBE), run_name="probe_runtime_failure_test")

    class BrokenContext:
        def open_stage(self, path):
            raise RuntimeError("USD load refused")

    with pytest.raises(namespace["RuntimeDiscoveryError"]) as raised:
        namespace["_discover_runtime_records"]("candidate.usda", [], BrokenContext(), object(), object())
    assert raised.value.api == "omni.usd.get_context().open_stage"
    assert "USD load refused" in str(raised.value)


def test_execute_probe_caps_output_with_structured_failure(tmp_path: Path) -> None:
    helper = tmp_path / "noisy.py"
    helper.write_text("import sys\nsys.stdout.write('x' * 10000)\n", encoding="utf-8")
    result = _execute_probe([sys.executable, str(helper)], tmp_path, 5, 512, 256)
    assert result["process_status"] == "output_limit_exceeded"
    assert result["output_limit_exceeded"] is True
    assert len(result["stdout"]) <= 512
    assert result["cleanup_verified"] is True


def test_atomic_write_fsyncs_and_preserves_previous_on_replace_failure(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "evidence.json"
    target.write_text('{"old": true}\n', encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("replace denied")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace denied"):
        _atomic_write(target, {"new": True})
    assert target.read_text(encoding="utf-8") == '{"old": true}\n'
    assert list(tmp_path.glob(".evidence.json.*")) == []


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("PASS_A20_RUNTIME_ARTICULATION_DISCOVERY_NO_STEP", 0),
        ("FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY", 1),
        ("BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION", 2),
    ],
)
def test_exit_status_contract(status: str, expected: int) -> None:
    assert _exit_code(status) == expected


def test_code_provenance_binds_exact_probe_and_coordinator_bytes() -> None:
    provenance = _code_provenance(ROOT, PROBE, MODULE)
    assert provenance["schema_version"] == "a20-runtime-discovery-v2"
    assert provenance["probe_sha256"] == hashlib.sha256(PROBE.read_bytes()).hexdigest()
    assert provenance["coordinator_sha256"] == hashlib.sha256(MODULE.read_bytes()).hexdigest()
    assert isinstance(provenance["git_head"], str)
    assert provenance["git_head"]
    assert isinstance(provenance["git_dirty"], bool)
    assert provenance["safety_checker"]["ok"] is True


def test_execute_probe_records_parent_observed_pid_and_bounds(tmp_path: Path) -> None:
    helper = tmp_path / "pid.py"
    helper.write_text("import os\nprint(os.getpid())\n", encoding="utf-8")
    result = _execute_probe([sys.executable, str(helper)], tmp_path, 5)
    assert int(result["stdout"].strip()) == result["observed_pid"]
    assert result["parent_monotonic_started"] <= result["parent_monotonic_finished"]
    assert result["cleanup_verified"] is True


def test_probe_helpers_fail_closed_and_close_even_if_marker_serialization_fails(monkeypatch) -> None:
    fake_isaac = types.ModuleType("isaacsim")
    fake_isaac.SimulationApp = object
    monkeypatch.setitem(sys.modules, "isaacsim", fake_isaac)
    namespace = runpy.run_path(str(PROBE), run_name="probe_test")
    assert namespace["_safe_version"]("definitely-missing-distribution") == "unknown"
    emitted = []

    def broken_serializer(*args, **kwargs):
        raise TypeError("no json")

    namespace["_emit_marker"]({"status": "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"}, emitted.append, broken_serializer)
    assert emitted
    assert emitted[0].startswith(MARKER)


def test_probe_main_emits_fail_payload_when_app_close_raises(monkeypatch, tmp_path: Path) -> None:
    closed: list[bool] = []

    class BrokenCloseApp:
        def __init__(self, options):
            assert options == {"headless": True}

        def close(self):
            closed.append(True)
            raise RuntimeError("close failed")

    fake_isaac = types.ModuleType("isaacsim")
    fake_isaac.SimulationApp = BrokenCloseApp
    fake_pxr = types.ModuleType("pxr")
    fake_pxr.Usd = types.SimpleNamespace()
    fake_pxr.UsdPhysics = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "isaacsim", fake_isaac)
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)
    namespace = runpy.run_path(str(PROBE), run_name="probe_close_test")
    emitted: list[dict[str, object]] = []
    namespace["main"].__globals__["_emit_marker"] = emitted.append
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(PROBE), "--invocation-id", "close-test"])

    assert namespace["main"]() == 1
    assert closed == [True]
    assert emitted[0]["status"] == "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"


def test_coordinator_production_source_never_uses_subprocess_run_capture_output() -> None:
    source = MODULE.read_text(encoding="utf-8")
    assert "run_command=subprocess.run" not in source
    assert "capture_output=True" not in source


def test_timeout_kills_spawned_descendant_process_group(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    helper = tmp_path / "tree.py"
    helper.write_text(
        "import pathlib, subprocess, sys, time\n"
        "child=subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])\n"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    result = _execute_probe([sys.executable, str(helper), str(pid_file)], tmp_path, 0.3)
    assert result["timed_out"] is True
    assert result["cleanup_verified"] is True
    child_pid = int(pid_file.read_text())
    stat_path = Path(f"/proc/{child_pid}/stat")
    assert not stat_path.exists() or stat_path.read_text().split()[2] == "Z"


def test_normal_parent_exit_cleans_detached_stdio_descendant_in_same_group(tmp_path: Path) -> None:
    pid_file = tmp_path / "descendant.pid"
    helper = tmp_path / "normal_parent_with_descendant.py"
    helper.write_text(
        "import os, pathlib, subprocess, sys\n"
        "with open(os.devnull, 'wb') as sink:\n"
        " child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], stdout=sink, stderr=sink, stdin=subprocess.DEVNULL)\n"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))\n",
        encoding="utf-8",
    )
    result = _execute_probe([sys.executable, str(helper), str(pid_file)], tmp_path, 5)
    assert result["returncode"] == 0
    assert result["cleanup_verified"] is True
    descendant_pid = int(pid_file.read_text())
    stat_path = Path(f"/proc/{descendant_pid}/stat")
    assert not stat_path.exists() or stat_path.read_text().split()[2] == "Z"


def test_trusted_layer1_inputs_require_canonical_paths_and_live_hashes(tmp_path: Path) -> None:
    paths = {}
    for name in ("config", "mapping", "stage"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        paths[name] = path
    layer1 = _layer1()
    layer1["inputs"] = {
        "config": {"path": str(paths["config"]), "sha256": hashlib.sha256(b"config").hexdigest()},
        "mapping": {"path": str(paths["mapping"]), "sha256": hashlib.sha256(b"mapping").hexdigest()},
        "stage": {
            "path": str(paths["stage"]),
            "pre_sha256": hashlib.sha256(b"stage").hexdigest(),
            "post_sha256": hashlib.sha256(b"stage").hexdigest(),
            "consistent_during_audit": True,
        },
    }
    trusted, errors = _trusted_layer1_inputs(layer1)
    assert errors == []
    assert {name: trusted[name]["path"] for name in trusted} == {name: str(path.resolve()) for name, path in paths.items()}
    layer1["inputs"]["mapping"]["path"] = str(paths["mapping"].parent / "." / paths["mapping"].name)
    layer1["inputs"]["mapping"]["sha256"] = "0" * 64
    _, errors = _trusted_layer1_inputs(layer1)
    assert {error["code"] for error in errors} == {"layer1_live_hash_mismatch"}


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
                "path": str(MODULE),
                "pre_sha256": LIVE_STAGE_HASH,
                "post_sha256": LIVE_STAGE_HASH,
                "consistent_during_audit": True,
            },
            "mapping": {"path": str(PROBE), "sha256": LIVE_MAPPING_HASH},
            "config": {"path": str(PROBE), "sha256": LIVE_MAPPING_HASH},
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
        "inputs": {"stage": {"sha256": LIVE_STAGE_HASH}, "mapping": {"sha256": LIVE_MAPPING_HASH}, "config": {"sha256": LIVE_MAPPING_HASH}},
        "initialization_operations": [],
        "cleanup_verified": True,
        "provenance": {
            "schema_version": "a20-runtime-discovery-v2",
            "safety_checker": {"ok": True},
        },
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


def test_runtime_api_failure_can_block_without_claiming_discovered_records() -> None:
    runs = _runs()
    runs[1].update(
        status="BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION",
        valid_handle=False,
        requires_unapproved_initialization=True,
        initialization_operations=["omni.physx.IPhysx.start_simulation"],
        articulation_root=None,
        articulation_count=0,
        dof_count=0,
        records=[],
        errors=[
            {
                "code": "runtime_api_failure",
                "api": "omni.physx.IPhysx.start_simulation",
                "message": "start_simulation: unavailable",
            }
        ],
    )

    result = aggregate_runtime_runs(_layer1(), runs)

    assert result["status"] == "BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION"
    assert result["errors"] == []
    assert result["blocked_run_indices"] == [1]


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
        inputs={"stage": {"sha256": LIVE_STAGE_HASH}, "mapping": {"sha256": LIVE_MAPPING_HASH}, "config": {"sha256": LIVE_MAPPING_HASH}},
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
        assert kwargs == {"cwd": ROOT, "timeout": 9, "check": False}
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


@pytest.mark.parametrize("mutation", ["version", "hash", "time"])
def test_cross_run_identity_version_hash_and_time_must_match(mutation: str) -> None:
    count = 0

    def fake_run(argv, **kwargs):
        nonlocal count
        invocation = argv[argv.index("--invocation-id") + 1]
        payload = _probe_payload(
            invocation, 200 + count, f"2026-01-01T00:00:0{count * 2}Z", f"2026-01-01T00:00:0{count * 2 + 1}Z"
        )
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
            inputs={"stage": {"sha256": LIVE_STAGE_HASH}, "mapping": {"sha256": LIVE_MAPPING_HASH}, "config": {"sha256": LIVE_MAPPING_HASH}},
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
    "mutation", ["duplicate_invocation", "version_mismatch", "started_nonmonotonic", "overlap"]
)
def test_three_run_process_provenance_is_cross_validated(mutation: str) -> None:
    runs = _runs_with_provenance()
    if mutation == "duplicate_invocation":
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


def _asset_validator() -> dict[str, object]:
    return {
        "status": "FAIL_A20_ASSET_VALIDATOR_BLOCKING_ISSUES",
        "ok": False,
        "blocking_issue_count": 1,
        "issues": [
            {
                "rule": "JointStateChecker",
                "severity": "FAILURE",
                "message": 'Joint State for "/aloha/root_joint" is not coherent with transforms',
                "suggestion": "Change XForms to match Joint State",
            }
        ],
        "stage_path": "/workspace/a19_clean_articulation_candidate.usda",
        "collision_ready": False,
        "control_ready": False,
        "replay_ready": False,
        "training_eligible": False,
        "physics_stepped": False,
    }


def _blocked_layer2() -> dict[str, object]:
    payload = aggregate_runtime_runs(_layer1(), _runs())
    payload.update(
        status="BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION",
        ok=False,
        blocked_run_indices=[0, 1, 2],
        runs=[
            {
                **run,
                "status": "BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION",
                "valid_handle": False,
                "requires_unapproved_initialization": True,
                "initialization_operations": ["timeline Play", "physics simulation step"],
            }
            for run in _runs()
        ],
    )
    return payload


def test_two_layer_report_keeps_independent_gates_and_readiness_false() -> None:
    report = format_two_layer_report(_asset_validator(), _layer1(), _blocked_layer2())

    for heading in ("## Asset Validator", "## Layer 1", "## Layer 2", "## Safety and readiness"):
        assert heading in report
    assert "Overall: NOT_READY" in report
    assert "FAIL_A20_ASSET_VALIDATOR_BLOCKING_ISSUES" in report
    assert "JointStateChecker" in report
    assert "Joint State for &quot;/aloha/root_joint&quot; is not coherent with transforms" in report
    assert "PASS_A20_USD_DOF_METADATA" in report
    assert "Expected DOFs: 16" in report
    assert "Observed DOFs: 16" in report
    assert "Mismatches: 0" in report
    assert "Three-run determinism: BLOCKED" in report
    assert "Exit contract: BLOCKED=2, PASS=0, FAIL=1" in report
    for statement in (
        "Physics stepped: false",
        "Actions applied: false",
        "Targets written: false",
        "Stage saved: false",
        "Collision ready: false",
        "Control ready: false",
        "Replay ready: false",
        "Contact ready: false",
        "Training ready: false",
    ):
        assert statement in report
    assert "A two-layer PASS does not mean Asset Validator is clean" in report
    assert "timeline Play" in report
    assert "physics simulation step" in report
    assert "not approved" in report


@pytest.mark.parametrize("bad", [None, [], "bad", {}, {"status": "PASS_A20_USD_DOF_METADATA"}])
def test_two_layer_report_fails_closed_for_missing_or_malformed_artifacts(bad: object) -> None:
    report = format_two_layer_report(bad, bad, bad)
    assert "Overall: NOT_READY" in report
    assert "MALFORMED_OR_MISSING" in report


def test_two_layer_report_summarizes_provenance_without_embedding_logs() -> None:
    layer2 = _blocked_layer2()
    layer2["provenance"] = {
        "git_head": "a" * 40,
        "probe_sha256": "b" * 64,
        "coordinator_sha256": "c" * 64,
    }
    layer2["runs"][0]["stdout_summary"] = "FULL ISAAC LOG MUST NOT APPEAR"
    report = format_two_layer_report(_asset_validator(), _layer1(), layer2)
    assert "aaaaaaaaaaaa" in report
    assert "bbbbbbbbbbbb" in report
    assert str(MODULE) in report
    assert "FULL ISAAC LOG MUST NOT APPEAR" not in report


def test_is_exact_runtime_pass_accepts_only_complete_safe_pass() -> None:
    exact = aggregate_runtime_runs(_layer1(), _runs())
    exact.update(
        runs=_runs(), physics_stepped=False, actions_applied=False, targets_written=False, stage_saved=False
    )
    assert is_exact_runtime_pass(exact, _layer1()) is True
    mutations = [
        lambda value: value.update(status="BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION"),
        lambda value: value.update(ok=False),
        lambda value: value["errors"].append({"code": "x"}),
        lambda value: value["mismatches"].append({"field": "x"}),
        lambda value: value.update(physics_stepped=True),
        lambda value: value.pop("actions_applied"),
    ]
    for mutation in mutations:
        candidate = deepcopy(exact)
        mutation(candidate)
        assert is_exact_runtime_pass(candidate, _layer1()) is False
    assert is_exact_runtime_pass(None) is False


def test_minimal_fake_pass_artifacts_cannot_render_ready_or_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = {
        "status": "PASS_A20_RUNTIME_ARTICULATION_DISCOVERY_NO_STEP",
        "ok": True,
        "errors": [],
        "mismatches": [],
        "physics_stepped": False,
        "actions_applied": False,
        "targets_written": False,
        "stage_saved": False,
    }
    assert is_exact_runtime_pass(fake) is False
    assert "Overall: NOT_READY" in format_two_layer_report(
        {"status": "PASS_FAKE", "ok": True}, {"status": "PASS_FAKE", "ok": True}, fake
    )


def test_report_never_hides_true_or_unknown_safety_evidence() -> None:
    layer2 = _blocked_layer2()
    layer2["physics_stepped"] = True
    report = format_two_layer_report(_asset_validator(), _layer1(), layer2)
    assert "Physics stepped: true" in report
    assert "Overall: NOT_READY" in report
    report = format_two_layer_report(_asset_validator(), _layer1(), {})
    assert "Physics stepped: unknown" in report


def test_report_write_failure_removes_old_report_and_forces_fail_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    artifacts.mkdir()
    reports.mkdir()
    layer1_path = artifacts / "layer1.json"
    runtime_path = artifacts / "runtime.json"
    asset_path = artifacts / "asset.json"
    report_path = reports / "report.md"
    config_path = tmp_path / "config.yaml"
    layer1_path.write_text(json.dumps(_layer1()), encoding="utf-8")
    runtime = aggregate_runtime_runs(_layer1(), _runs())
    runtime.update(physics_stepped=False, actions_applied=False, targets_written=False, stage_saved=False)
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    asset_path.write_text(json.dumps(_asset_validator()), encoding="utf-8")
    report_path.write_text("STALE PASS", encoding="utf-8")
    config_path.write_text(
        yaml.safe_dump(
            {
                "outputs": {
                    "a20_usd_dof_metadata_json": str(layer1_path.relative_to(tmp_path)),
                    "a20_runtime_articulation_discovery_json": str(runtime_path.relative_to(tmp_path)),
                    "a20_asset_validator_json": str(asset_path.relative_to(tmp_path)),
                    "a20_two_layer_articulation_discovery_md": str(report_path.relative_to(tmp_path)),
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(MODULE), "--config", str(config_path), "--report-from-existing"])
    monkeypatch.setattr(coordinator, "_atomic_write_text", lambda *_: (_ for _ in ()).throw(OSError("disk full")))

    assert coordinator.main() == 1
    assert not report_path.exists()
    rewritten = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert rewritten["status"] == "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"
    assert rewritten["errors"][-1]["code"] == "report_write_failed"


def test_online_main_writes_false_safety_flags_and_exits_zero_only_for_exact_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    artifacts.mkdir()
    reports.mkdir()
    layer1_path = artifacts / "layer1.json"
    runtime_path = artifacts / "runtime.json"
    asset_path = artifacts / "asset.json"
    report_path = reports / "report.md"
    config_path = tmp_path / "config.yaml"
    layer1_path.write_text(json.dumps(_layer1()), encoding="utf-8")
    asset = _asset_validator()
    asset.update(
        status="PASS_A20_ASSET_VALIDATOR_READ_ONLY_NO_BLOCKING_ISSUES",
        ok=True,
        blocking_issue_count=0,
        issues=[],
    )
    asset_path.write_text(json.dumps(asset), encoding="utf-8")
    config_path.write_text(
        yaml.safe_dump(
            {
                "outputs": {
                    "a20_usd_dof_metadata_json": str(layer1_path.relative_to(tmp_path)),
                    "a20_runtime_articulation_discovery_json": str(runtime_path.relative_to(tmp_path)),
                    "a20_asset_validator_json": str(asset_path.relative_to(tmp_path)),
                    "a20_two_layer_articulation_discovery_md": str(report_path.relative_to(tmp_path)),
                }
            }
        ),
        encoding="utf-8",
    )
    result = aggregate_runtime_runs(_layer1(), _runs())
    result["runs"] = _runs()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(MODULE), "--config", str(config_path)])
    monkeypatch.setattr(coordinator, "run_three_probes", lambda *_: deepcopy(result))

    assert coordinator.main() == 0
    written = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert is_exact_runtime_pass(written, _layer1()) is True
    for flag in ("physics_stepped", "actions_applied", "targets_written", "stage_saved"):
        assert written[flag] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(runs=[{}, {}, {}]),
        lambda payload: payload["runs"][1].update(physics_stepped=True),
        lambda payload: payload["runs"][1].update(status="BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION"),
        lambda payload: payload["runs"][1].update(valid_handle=False),
        lambda payload: payload["runs"][1].update(requires_unapproved_initialization=True),
        lambda payload: payload["runs"][1].update(initialization_operations=["timeline Play"]),
        lambda payload: payload["runs"][1].pop("cleanup_verified"),
        lambda payload: payload["runs"][1].pop("provenance"),
        lambda payload: payload["runs"][1].pop("records"),
        lambda payload: payload["runs"][1]["records"].pop(),
        lambda payload: payload["errors"].append({"code": "bad"}),
        lambda payload: payload["mismatches"].append({"field": "path"}),
    ],
)
def test_exact_runtime_pass_revalidates_every_saved_run_fail_closed(mutation) -> None:
    payload = aggregate_runtime_runs(_layer1(), _runs())
    payload.update(
        runs=_runs(),
        physics_stepped=False,
        actions_applied=False,
        targets_written=False,
        stage_saved=False,
    )
    for run in payload["runs"]:
        run["cleanup_verified"] = True
        run["provenance"] = {"schema_version": "a20-runtime-discovery-v2", "safety_checker": {"ok": True}}
    mutation(payload)
    assert is_exact_runtime_pass(payload, _layer1()) is False


def test_exact_runtime_pass_rejects_top_level_run_contradiction() -> None:
    payload = aggregate_runtime_runs(_layer1(), _runs())
    payload.update(
        runs=_runs(),
        physics_stepped=False,
        actions_applied=False,
        targets_written=False,
        stage_saved=False,
    )
    for run in payload["runs"]:
        run["cleanup_verified"] = True
        run["provenance"] = {"schema_version": "a20-runtime-discovery-v2", "safety_checker": {"ok": True}}
    payload["runs"][0]["records"][0]["axis"] = "Z"
    assert is_exact_runtime_pass(payload, _layer1()) is False


def test_report_bounds_and_single_lines_untrusted_asset_validator_issues() -> None:
    asset = _asset_validator()
    asset["blocking_issue_count"] = 100
    asset["issues"] = [
        {
            "rule": f"rule-{index}-" + "r" * 1000,
            "severity": "FAILURE\nforged heading",
            "at": "/aloha/root_joint\n# injected" + "a" * 1000,
            "message": "message\nnext line " + "m" * 5000,
            "suggestion": "suggestion\r\nnext line " + "s" * 5000,
        }
        for index in range(100)
    ]
    report = format_two_layer_report(asset, _layer1(), _blocked_layer2())
    assert len(report) < 20_000
    assert report.count("- Blocking issue: [") == 20
    assert "[truncated]" in report
    assert "80 additional issues omitted" in report
    assert "\n# injected" not in report
    assert "\nnext line" not in report


def test_exact_runtime_pass_binds_records_to_independent_trusted_layer1() -> None:
    trusted = _layer1()
    payload = aggregate_runtime_runs(trusted, _runs())
    payload.update(
        runs=_runs(),
        physics_stepped=False,
        actions_applied=False,
        targets_written=False,
        stage_saved=False,
    )
    assert is_exact_runtime_pass(payload, trusted) is True
    forged = deepcopy(payload["expected"])
    forged[0].update(axis="Z", name="forged_name", path="/aloha/joints/forged", lower_limit=-9.0)
    payload["expected"] = forged
    for run in payload["runs"]:
        run["records"] = deepcopy(forged)
    assert is_exact_runtime_pass(payload, trusted) is False


def test_exact_runtime_pass_requires_valid_trusted_layer1() -> None:
    trusted = _layer1()
    payload = aggregate_runtime_runs(trusted, _runs())
    payload.update(
        runs=_runs(),
        physics_stepped=False,
        actions_applied=False,
        targets_written=False,
        stage_saved=False,
    )
    assert is_exact_runtime_pass(payload) is False
    assert is_exact_runtime_pass(payload, None) is False
    assert is_exact_runtime_pass(payload, {"status": "PASS_FAKE", "ok": True}) is False
    malformed = _layer1()
    malformed["inputs"]["stage"]["post_sha256"] = "0" * 64
    assert is_exact_runtime_pass(payload, malformed) is False


def test_exact_runtime_pass_binds_all_run_input_hashes_to_trusted_layer1() -> None:
    trusted = _layer1()
    payload = aggregate_runtime_runs(trusted, _runs())
    payload.update(
        runs=_runs(),
        physics_stepped=False,
        actions_applied=False,
        targets_written=False,
        stage_saved=False,
    )
    assert is_exact_runtime_pass(payload, trusted) is True
    for run in payload["runs"]:
        run["inputs"]["mapping"]["sha256"] = "d" * 64
    assert is_exact_runtime_pass(payload, trusted) is False


def test_asset_validator_failure_keeps_overall_not_ready_with_exact_layers() -> None:
    layer1 = _layer1()
    layer2 = aggregate_runtime_runs(layer1, _runs())
    layer2.update(
        runs=_runs(), physics_stepped=False, actions_applied=False, targets_written=False, stage_saved=False
    )
    report = format_two_layer_report(_asset_validator(), layer1, layer2)
    assert "Overall: NOT_READY" in report


@pytest.mark.parametrize("scalar", [None, [], "READY", 7, True])
def test_offline_scalar_artifact_removes_stale_ready_and_returns_structured_fail(
    scalar: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    artifacts.mkdir()
    reports.mkdir()
    layer1_path = artifacts / "layer1.json"
    runtime_path = artifacts / "runtime.json"
    asset_path = artifacts / "asset.json"
    report_path = reports / "report.md"
    config_path = tmp_path / "config.yaml"
    layer1_path.write_text(json.dumps(scalar), encoding="utf-8")
    runtime_path.write_text(json.dumps(scalar), encoding="utf-8")
    asset_path.write_text(json.dumps(scalar), encoding="utf-8")
    report_path.write_text("Overall: READY\nSTALE READY", encoding="utf-8")
    config_path.write_text(
        yaml.safe_dump(
            {
                "outputs": {
                    "a20_usd_dof_metadata_json": str(layer1_path.relative_to(tmp_path)),
                    "a20_runtime_articulation_discovery_json": str(runtime_path.relative_to(tmp_path)),
                    "a20_asset_validator_json": str(asset_path.relative_to(tmp_path)),
                    "a20_two_layer_articulation_discovery_md": str(report_path.relative_to(tmp_path)),
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(MODULE), "--config", str(config_path), "--report-from-existing"])
    assert coordinator.main() == 1
    assert "Overall: READY" not in report_path.read_text(encoding="utf-8")
    assert "Overall: NOT_READY" in report_path.read_text(encoding="utf-8")
    rewritten = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert rewritten["status"] == "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"
    assert rewritten["errors"]


def test_report_globally_bounds_and_escapes_all_untrusted_fields() -> None:
    asset = _asset_validator()
    asset["status"] = "FAIL\n# forged <script>alert(1)</script> [x](javascript:x) `tick`"
    asset["blocking_issue_count"] = "1\n# forged"
    asset["issues"] = [
        {
            "rule": "<img src=x onerror=alert(1)> [rule](javascript:x) `r`",
            "severity": "FAILURE",
            "at": "/stage\n# heading",
            "message": "<script>alert(1)</script>\n# heading",
            "suggestion": "[click](javascript:x) `code`",
        }
    ]
    layer1 = _layer1()
    layer1["inputs"]["stage"]["path"] = "/stage\n# injected <img src=x> [x](y) `z`" + "p" * 5000
    layer2 = _blocked_layer2()
    layer2["runs"][0]["initialization_operations"] = [
        f"op-{index}\n# injected <script>x</script> [x](y) `z`" + "q" * 1000 for index in range(1000)
    ]
    report = format_two_layer_report(asset, layer1, layer2)
    assert len(report.encode("utf-8")) <= 32_768
    assert "<script>" not in report
    assert "<img" not in report
    assert "javascript:" not in report
    assert "\n# injected" not in report
    assert "[truncated]" in report
    assert "additional operations omitted" in report


def test_offline_self_consistent_forged_three_jsons_cannot_exit_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layer1 = _layer1()
    runtime = aggregate_runtime_runs(layer1, _runs())
    runtime.update(
        runs=_runs(), physics_stepped=False, actions_applied=False, targets_written=False, stage_saved=False
    )
    forged_hashes = {"stage": "a" * 64, "mapping": "b" * 64, "config": "c" * 64}
    for name, digest in forged_hashes.items():
        forged_path = str((tmp_path / f"nonexistent-{name}").resolve())
        if name == "stage":
            layer1["inputs"][name].update(path=forged_path, pre_sha256=digest, post_sha256=digest)
        else:
            layer1["inputs"][name].update(path=forged_path, sha256=digest)
        for run in runtime["runs"]:
            run["inputs"][name]["sha256"] = digest
    asset = _asset_validator()
    asset.update(
        status="PASS_A20_ASSET_VALIDATOR_READ_ONLY_NO_BLOCKING_ISSUES",
        ok=True,
        blocking_issue_count=0,
        issues=[],
    )
    layer1_path = tmp_path / "layer1.json"
    runtime_path = tmp_path / "runtime.json"
    asset_path = tmp_path / "asset.json"
    report_path = tmp_path / "report.md"
    config_path = tmp_path / "config.yaml"
    layer1_path.write_text(json.dumps(layer1), encoding="utf-8")
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    asset_path.write_text(json.dumps(asset), encoding="utf-8")
    report_path.write_text("Overall: READY\nSTALE", encoding="utf-8")
    config_path.write_text(
        yaml.safe_dump(
            {
                "outputs": {
                    "a20_usd_dof_metadata_json": layer1_path.name,
                    "a20_runtime_articulation_discovery_json": runtime_path.name,
                    "a20_asset_validator_json": asset_path.name,
                    "a20_two_layer_articulation_discovery_md": report_path.name,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(MODULE), "--config", str(config_path), "--report-from-existing"])
    assert coordinator.main() == 1
    assert "Overall: READY" not in report_path.read_text(encoding="utf-8")
    assert json.loads(runtime_path.read_text(encoding="utf-8"))["status"] == "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"
