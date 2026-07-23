#!/usr/bin/env python3
"""Fail-closed coordinator for the two A21 target-readback batches.

This coordinator deliberately has no Isaac imports.  The only Isaac execution is
the reviewed Task5 probe, launched once for the left target slots and, only after
an exact left success, once again in a fresh process for the right target slots.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import selectors
import stat
import subprocess
import sys
import time
import uuid

import yaml

from aloha_isaac_rebuild.scripts import audit_a21_policy_target_limit_preflight as a21a
from aloha_isaac_rebuild.scripts import probe_a21_runtime_target_readback_once as probe_contract
from aloha_isaac_rebuild.scripts.run_a20_runtime_articulation_discovery import _atomic_write_text
from aloha_isaac_rebuild.scripts.run_a20_runtime_articulation_discovery import _load_json_fail_closed
from aloha_isaac_rebuild.scripts.run_a20_runtime_articulation_discovery import _terminate_process_group
from aloha_isaac_rebuild.scripts.run_a20_runtime_articulation_discovery import is_exact_runtime_pass

MARKER = probe_contract.MARKER
PASS_STATUS = "PASS_A21_RUNTIME_TARGET_READBACK_RESTORED_NO_STEP"
FAIL_STATUS = "FAIL_A21_RUNTIME_TARGET_READBACK"
DEFAULT_CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml")
DEFAULT_OUTPUT_CAP = 1024 * 1024
DEFAULT_MARKER_CAP = 256 * 1024
LEFT_INDICES = [0, 2, 4, 6, 8, 10, 12, 13]
RIGHT_INDICES = [1, 3, 5, 7, 9, 11, 14, 15]
_SHA256 = set("0123456789abcdef")


def _error(code: str, **fields: object) -> dict[str, object]:
    return {"code": code, **fields}


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256


def _code_provenance(repo_root: Path, probe_path: Path, coordinator_path: Path) -> dict[str, object]:
    """Bind source hashes to one clean git HEAD before any subprocess is started."""
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
        "git_head": head,
        "git_dirty": dirty,
        "probe_sha256": hashlib.sha256(probe_bytes).hexdigest(),
        "coordinator_sha256": _digest(coordinator_path),
        "safety_checker": checker,
        "safety_checker_sha256": hashlib.sha256(inspect.getsource(check_probe_source).encode("utf-8")).hexdigest(),
    }


def _qualified_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def check_probe_source(source: str) -> dict[str, object]:
    """AST allowlist for the exact, target-only Task5 probe source.

    The checker rejects every import alias, dynamic call target, unsafe call or
    unsafe callable alias.  The two target-buffer APIs are allowed only in their
    reviewed Task5 forms: a local ``getter``/``setter`` binding and direct
    restoration calls on ``view``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"ok": False, "errors": [f"syntax_error:{exc.msg}"]}
    errors: list[str] = []
    imports = {
        ("from", "__future__", "annotations", None),
        ("import", "argparse", None, None),
        ("from", "contextlib", "suppress", None),
        ("from", "copy", "deepcopy", None),
        ("from", "datetime", "UTC", None),
        ("from", "datetime", "datetime", None),
        ("import", "hashlib", None, None),
        ("import", "json", None, None),
        ("import", "math", None, None),
        ("import", "os", None, None),
        ("from", "pathlib", "Path", None),
        ("import", "numpy", None, "np"),
        ("import", "yaml", None, None),
        (
            "from",
            "aloha_isaac_rebuild.scripts.a20_policy_runtime_order_adapter",
            "SCHEMA_VERSION",
            "A20_SCHEMA_VERSION",
        ),
        ("from", "aloha_isaac_rebuild.scripts.run_a20_runtime_articulation_discovery", "is_exact_runtime_pass", None),
        ("from", "isaacsim", "SimulationApp", None),
        ("from", "omni.physics", "tensors", None),
        ("from", "omni.physx", "get_physx_interface", None),
        ("import", "omni.usd", None, None),
    }
    forbidden = {
        "play",
        "step",
        "reset",
        "update",
        "update_simulation",
        "simulate",
        "set_dof_positions",
        "set_dof_velocities",
        "set_dof_efforts",
        "set_dof_velocity_targets",
        "set_dof_effort_targets",
        "set_dof_stiffnesses",
        "set_dof_dampings",
        "apply_action",
        "save",
        "Save",
        "Export",
        "Flatten",
        "exec",
        "eval",
        "getattr",
        "setattr",
        "__import__",
        "import_module",
    }
    direct_calls = {
        "Path",
        "ValueError",
        "RuntimeError",
        "SystemExit",
        "SimulationApp",
        "_digest",
        "_error",
        "_finite_float",
        "_strict_index",
        "_runtime_bounds",
        "_validate_vector",
        "_validate_target_contract",
        "_target_array",
        "_safety",
        "_configured_path",
        "_bound_input_path",
        "_require_exact_a20_evidence",
        "_require_current_layer1_inputs",
        "_emit_marker",
        "_now",
        "batch_policy_indices",
        "choose_interior_delta",
        "exercise_target_batch",
        "deepcopy",
        "float",
        "int",
        "str",
        "list",
        "set",
        "tuple",
        "range",
        "len",
        "enumerate",
        "sorted",
        "zip",
        "isinstance",
        "all",
        "any",
        "print",
        "get_physx_interface",
        "is_exact_runtime_pass",
        "suppress",
    }
    allowed_attributes = {
        "ArgumentParser",
        "add_argument",
        "append",
        "array_equal",
        "allclose",
        "cwd",
        "copy",
        "dumps",
        "file_digest",
        "force_load_physics_from_usd",
        "get",
        "get_context",
        "get_dof_position_targets",
        "get_stage_id",
        "getpid",
        "hexdigest",
        "is_absolute",
        "is_file",
        "isfinite",
        "issubdtype",
        "loads",
        "now",
        "all",
        "open",
        "open_stage",
        "parse_args",
        "radians",
        "read_text",
        "resolve",
        "safe_load",
        "set_dof_position_targets",
        "set_subspace_roots",
        "start_simulation",
        "strftime",
        "tolist",
        "create_simulation_view",
        "create_articulation_view",
        "close",
        "items",
        "extend",
        "add",
    }
    allowed_aliases = {"getter", "setter"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                item = ("import", alias.name, None, alias.asname)
                if item not in imports:
                    errors.append(f"import_not_allowed:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            errors.extend(
                f"import_not_allowed:{module}.{alias.name}"
                for alias in node.names
                if ("from", module, alias.name, alias.asname) not in imports
            )
        elif isinstance(node, ast.Assign | ast.AnnAssign | ast.NamedExpr):
            value = node.value
            if isinstance(value, ast.Attribute) and value.attr in forbidden:
                errors.append(f"attribute_alias_not_allowed:{value.attr}")
            if isinstance(value, ast.Name) and value.id in forbidden:
                errors.append(f"name_alias_not_allowed:{value.id}")
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in allowed_aliases:
                    expected = {
                        "getter": "view.get_dof_position_targets",
                        "setter": "view.set_dof_position_targets",
                    }[target.id]
                    if _qualified_name(value) != expected:
                        errors.append(f"target_api_alias_not_allowed:{target.id}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name in forbidden:
                    errors.append(f"call_not_allowed:{name}")
                elif name not in direct_calls | allowed_aliases | {"main"}:
                    errors.append(f"unreviewed_call:{name}")
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
                if name in forbidden or (name.startswith("set_dof_") and name != "set_dof_position_targets"):
                    errors.append(f"call_not_allowed:{name}")
                elif name not in allowed_attributes:
                    errors.append(f"attribute_call_not_allowed:{_qualified_name(node.func) or name}")
                elif (
                    name == "set_dof_position_targets" and _qualified_name(node.func) != "view.set_dof_position_targets"
                ):
                    errors.append("target_setter_receiver_not_allowed")
                elif (
                    name == "get_dof_position_targets" and _qualified_name(node.func) != "view.get_dof_position_targets"
                ):
                    errors.append("target_getter_receiver_not_allowed")
            else:
                errors.append("dynamic_call_not_allowed")
    return {"ok": not errors, "errors": sorted(set(errors))}


def _exact_preflight(payload: object) -> bool:
    return bool(a21a._exact_pass(payload))  # noqa: SLF001 - exact reviewed A21a contract.


def _safety(run: dict[str, object]) -> dict[str, object] | None:
    value = run.get("safety")
    if isinstance(value, dict):
        return value
    result = run.get("result")
    return result.get("safety") if isinstance(result, dict) and isinstance(result.get("safety"), dict) else None


def _input_binding(inputs: object, name: str) -> tuple[str, str] | None:
    if not isinstance(inputs, dict) or not isinstance(inputs.get(name), dict):
        return None
    binding = inputs[name]
    path, digest = binding.get("path"), binding.get("sha256")
    return (path, digest) if isinstance(path, str) and path.startswith("/") and _valid_digest(digest) else None


def aggregate_batches(preflight: object, runs: object) -> dict[str, object]:
    """Aggregate the exact left-then-right Task5 pass contract without mutation."""
    safe_preflight = deepcopy(preflight)
    safe_runs = deepcopy(runs)
    errors: list[dict[str, object]] = []
    if not _exact_preflight(preflight):
        errors.append(_error("a21a_preflight_not_exact_pass"))
    if not isinstance(runs, list) or len(runs) != 2:
        errors.append(_error("invalid_run_count", expected=2, observed=len(runs) if isinstance(runs, list) else None))
        typed_runs: list[dict[str, object]] = []
    else:
        typed_runs = [run for run in runs if isinstance(run, dict)]
        if len(typed_runs) != 2:
            errors.append(_error("invalid_run_type"))
    expected_sides = (("left", LEFT_INDICES), ("right", RIGHT_INDICES))
    invocations: list[object] = []
    pids: list[object] = []
    input_fingerprints: list[dict[str, tuple[str, str]]] = []
    provenance_fingerprints: list[dict[str, str]] = []
    all_indices: list[int] = []
    prohibited = ("physics_stepped", "positions_written", "velocities_written", "efforts_written")
    for index, (side, indices) in enumerate(expected_sides):
        if index >= len(typed_runs):
            continue
        run = typed_runs[index]
        if run.get("status") != probe_contract.PASS_STATUS:
            errors.append(_error("single_run_status_not_exact_pass", run_index=index))
        if run.get("batch") != side:
            errors.append(_error("batch_order_or_label_mismatch", run_index=index, expected=side))
        actual_indices = run.get("runtime_indices")
        result = run.get("result")
        if actual_indices != indices or not isinstance(result, dict) or result.get("runtime_indices") != indices:
            errors.append(_error("runtime_indices_mismatch", run_index=index, expected=indices))
        if run.get("marker_count") != 1:
            errors.append(_error("marker_count_not_exactly_one", run_index=index))
        if run.get("returncode") != 0 or run.get("timed_out") is not False or run.get("cleanup_verified") is not True:
            errors.append(_error("process_completion_contract_failed", run_index=index))
        if run.get("output_limit_exceeded") is not False:
            errors.append(_error("output_limit_contract_failed", run_index=index))
        if result is None or not isinstance(result, dict) or result.get("ok") is not True:
            errors.append(_error("target_result_not_exact_success", run_index=index))
        safety = _safety(run)
        if not isinstance(safety, dict):
            errors.append(_error("missing_safety", run_index=index))
        else:
            errors.extend(
                _error("prohibited_safety_flag", run_index=index, field=flag)
                for flag in prohibited
                if safety.get(flag) is not False
            )
            if safety.get("target_only_no_step") is not True:
                errors.append(_error("target_only_declaration_missing", run_index=index))
            if safety.get("targets_written") is not True:
                errors.append(_error("targets_written_not_true", run_index=index))
            if safety.get("targets_restored") is not True:
                errors.append(_error("targets_restored_not_true", run_index=index))
        errors.extend(
            _error("prohibited_safety_flag", run_index=index, field=flag)
            for flag in (
                "physics_stepped",
                "actions_applied",
                "positions_written",
                "velocities_written",
                "efforts_written",
                "stage_saved",
            )
            if run.get(flag) is not False
        )
        invocations.append(run.get("invocation_id"))
        pids.append(run.get("pid"))
        all_indices.extend(actual_indices if isinstance(actual_indices, list) else [])
        bindings: dict[str, tuple[str, str]] = {}
        for name in ("config", "stage", "mapping", "a20_evidence", "a20_layer1"):
            binding = _input_binding(run.get("inputs"), name)
            if binding is None:
                errors.append(_error("invalid_input_binding", run_index=index, input=name))
            else:
                bindings[name] = binding
        input_fingerprints.append(bindings)
        provenance = run.get("provenance")
        if not isinstance(provenance, dict) or not all(
            _valid_digest(provenance.get(name)) for name in ("probe_sha256", "coordinator_sha256")
        ):
            errors.append(_error("invalid_provenance", run_index=index))
        elif (
            not isinstance(provenance.get("git_head"), str)
            or len(provenance["git_head"]) != 40
            or set(provenance["git_head"]) - _SHA256
            or provenance.get("git_dirty") is not False
        ):
            errors.append(_error("stale_or_dirty_git_provenance", run_index=index))
        elif (
            not isinstance(provenance.get("safety_checker"), dict) or provenance["safety_checker"].get("ok") is not True
        ):
            errors.append(_error("unsafe_probe_source", run_index=index))
        else:
            provenance_fingerprints.append(
                {
                    "probe_sha256": provenance["probe_sha256"],
                    "coordinator_sha256": provenance["coordinator_sha256"],
                    "git_head": provenance["git_head"],
                }
            )
    if len(invocations) == 2 and (
        not all(isinstance(value, str) and value for value in invocations) or len(set(invocations)) != 2
    ):
        errors.append(_error("duplicate_or_invalid_invocation_id"))
    if len(pids) == 2 and (
        not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in pids)
        or len(set(pids)) != 2
    ):
        errors.append(_error("duplicate_or_invalid_pid"))
    if len(input_fingerprints) == 2 and input_fingerprints[0] != input_fingerprints[1]:
        errors.append(_error("probe_input_hash_mismatch"))
    if len(provenance_fingerprints) == 2 and provenance_fingerprints[0] != provenance_fingerprints[1]:
        errors.append(_error("stale_or_dirty_git_provenance"))
    if sorted(all_indices) != list(range(16)) or len(set(all_indices)) != 16:
        errors.append(_error("runtime_index_inventory_not_exact"))
    preflight_inputs = preflight.get("inputs") if isinstance(preflight, dict) else None
    if len(input_fingerprints) == 2 and isinstance(preflight_inputs, dict):
        first = input_fingerprints[0]
        expected_from_preflight = {
            "config": _input_binding(preflight_inputs, "config"),
            "a20_layer1": _input_binding(preflight_inputs, "layer1"),
            "a20_evidence": _input_binding(preflight_inputs, "layer2"),
        }
        if any(expected is None or first.get(name) != expected for name, expected in expected_from_preflight.items()):
            errors.append(_error("preflight_a20_input_hash_mismatch"))
    status = PASS_STATUS if not errors else FAIL_STATUS
    return {
        "schema_version": "a21-runtime-target-readback-v1",
        "status": status,
        "ok": status == PASS_STATUS,
        "preflight": safe_preflight,
        "runs": safe_runs,
        "run_count": len(runs) if isinstance(runs, list) else None,
        "runtime_indices": sorted(all_indices),
        "targets_written": status == PASS_STATUS,
        "targets_restored": status == PASS_STATUS,
        "physics_stepped": False,
        "actions_applied": False,
        "stage_saved": False,
        "errors": errors,
    }


def _execute_probe(argv: list[str], cwd: Path, timeout_seconds: float) -> dict[str, object]:
    """Task4-style bounded capture in a dedicated process group."""
    process = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    selector = selectors.DefaultSelector()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    started = time.monotonic()
    timed_out = exceeded = False
    try:
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            assert stream is not None
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        while selector.get_map():
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                timed_out = True
                break
            for key, _ in selector.select(min(0.1, remaining)):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                target = buffers[key.data]
                permitted = max(0, DEFAULT_OUTPUT_CAP - len(target))
                target.extend(chunk[:permitted])
                if len(chunk) > permitted:
                    exceeded = True
                    break
            if exceeded:
                break
        if not timed_out and not exceeded:
            process.wait(timeout=max(0.1, timeout_seconds - (time.monotonic() - started)))
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        selector.close()
    cleanup_verified = _terminate_process_group(process)
    stdout = buffers["stdout"].decode("utf-8", "replace")
    stderr = buffers["stderr"].decode("utf-8", "replace")
    marker_bytes = sum(len(line.encode("utf-8")) for line in stdout.splitlines() if line.startswith(MARKER))
    exceeded = exceeded or marker_bytes > DEFAULT_MARKER_CAP
    return {
        "process_status": "timeout"
        if timed_out
        else "output_limit_exceeded"
        if exceeded
        else "completed"
        if process.returncode == 0
        else "failed",
        "returncode": process.returncode if process.returncode is not None else -1,
        "timed_out": timed_out,
        "output_limit_exceeded": exceeded,
        "cleanup_verified": cleanup_verified,
        "observed_pid": process.pid,
        "stdout": stdout,
        "stderr": stderr,
    }


def _batch_payload(execution: dict[str, object], invocation_id: str, side: str) -> dict[str, object]:
    stdout = execution.get("stdout")
    markers = [line[len(MARKER) :] for line in str(stdout or "").splitlines() if line.startswith(MARKER)]
    payload: dict[str, object]
    try:
        if execution.get("output_limit_exceeded") is True:
            raise ValueError("output_limit_exceeded")
        if len(markers) != 1:
            raise ValueError(f"marker_count:{len(markers)}")
        if len(markers[0].encode("utf-8")) > DEFAULT_MARKER_CAP:
            raise ValueError("marker_oversize")
        decoded = json.loads(markers[0])
        if not isinstance(decoded, dict):
            raise ValueError("marker_not_object")
        payload = decoded
    except (ValueError, json.JSONDecodeError) as exc:
        payload = {"status": probe_contract.FAIL_STATUS, "errors": [_error("marker_protocol_error", message=str(exc))]}
    payload = deepcopy(payload)
    payload.update(
        batch=side,
        marker_count=len(markers),
        process_status=execution.get("process_status"),
        returncode=execution.get("returncode"),
        timed_out=execution.get("timed_out"),
        output_limit_exceeded=execution.get("output_limit_exceeded"),
        cleanup_verified=execution.get("cleanup_verified"),
        observed_pid=execution.get("observed_pid"),
    )
    for flag in (
        "physics_stepped",
        "actions_applied",
        "positions_written",
        "velocities_written",
        "efforts_written",
        "stage_saved",
    ):
        payload.setdefault(flag, False)
    if payload.get("invocation_id") != invocation_id:
        payload["status"] = probe_contract.FAIL_STATUS
        payload.setdefault("errors", []).append(_error("invocation_id_mismatch"))
    if payload.get("pid") != execution.get("observed_pid"):
        payload["status"] = probe_contract.FAIL_STATUS
        payload.setdefault("errors", []).append(_error("pid_mismatch"))
    return payload


def _single_batch_pass(run: object, *, side: str) -> bool:
    """Small no-preflight gate used solely to decide whether R may start."""
    if not isinstance(run, dict) or run.get("status") != probe_contract.PASS_STATUS:
        return False
    if run.get("batch") != side or run.get("runtime_indices") != (LEFT_INDICES if side == "left" else RIGHT_INDICES):
        return False
    if (
        any(
            run.get(field) is not expected
            for field, expected in (
                ("marker_count", 1),
                ("timed_out", False),
                ("output_limit_exceeded", False),
                ("cleanup_verified", True),
            )
        )
        or run.get("returncode") != 0
    ):
        return False
    safety = _safety(run)
    if (
        not isinstance(safety, dict)
        or safety.get("targets_written") is not True
        or safety.get("targets_restored") is not True
    ):
        return False
    if any(
        safety.get(flag) is not False
        for flag in ("physics_stepped", "positions_written", "velocities_written", "efforts_written")
    ):
        return False
    return isinstance(run.get("result"), dict) and run["result"].get("ok") is True


def run_two_batches(
    repo_root: Path,
    interpreter: Path,
    probe_path: Path,
    timeout_seconds: float,
    *,
    execute: Callable[[list[str], Path, float], dict[str, object]] = _execute_probe,
    invocation_ids: tuple[str, str] | None = None,
    extra_args: list[str] | None = None,
    provenance: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Run left, then only run right after the left protocol itself is exact."""
    ids = invocation_ids or (str(uuid.uuid4()), str(uuid.uuid4()))
    if len(ids) != 2 or len(set(ids)) != 2:
        return [{"status": probe_contract.FAIL_STATUS, "errors": [_error("invalid_invocation_ids")]}]
    runs: list[dict[str, object]] = []
    for side, invocation_id in zip(("left", "right"), ids, strict=True):
        argv = [str(interpreter), "-u", str(probe_path), "--invocation-id", invocation_id, "--batch", side]
        if extra_args:
            argv.extend(extra_args)
        run = _batch_payload(execute(argv, repo_root, timeout_seconds), invocation_id, side)
        run["provenance"] = (
            deepcopy(provenance)
            if isinstance(provenance, dict)
            else {
                "probe_sha256": _digest(probe_path),
                "coordinator_sha256": _digest(Path(__file__).resolve()),
                "safety_checker": check_probe_source(probe_path.read_text(encoding="utf-8")),
            }
        )
        runs.append(run)
        if side == "left" and not _single_batch_pass(run, side="left"):
            break
    return runs


def format_report(result: object) -> str:
    """Render only the immutable aggregate JSON contract into the A21 report."""
    payload = result if isinstance(result, dict) else {}
    ready = payload.get("status") == PASS_STATUS and payload.get("ok") is True
    runs = payload.get("runs") if isinstance(payload.get("runs"), list) else []
    preflight = payload.get("preflight") if isinstance(payload.get("preflight"), dict) else {}

    def line(index: int, label: str) -> str:
        run = runs[index] if index < len(runs) and isinstance(runs[index], dict) else {}
        return f"- Batch {label}: {run.get('status', 'NOT_RUN')}; indices: {run.get('runtime_indices', 'unknown')}"

    return "\n".join(
        (
            "# A21 target readback gate",
            "",
            f"Overall: {'READY' if ready else 'NOT_READY'}",
            "",
            f"- A21a status: {preflight.get('status', 'MALFORMED_OR_MISSING')}",
            line(0, "L"),
            line(1, "R"),
            f"- Targets restored: {payload.get('targets_restored')}",
            f"- Physics stepped: {payload.get('physics_stepped', 'unknown')}",
            f"- Actions applied: {payload.get('actions_applied', 'unknown')}",
            f"- Stage saved: {payload.get('stage_saved', 'unknown')}",
            "- Motion ready: false",
            "- Hold ready: false",
            "- Collision ready: false",
            "- Contact ready: false",
            "- Replay ready: false",
            "- Training ready: false",
            "- Next gate: A22 reviewed drive gains and micro-motion",
            "",
        )
    )


def _canonical_file(repo: Path, value: Path, *, label: str) -> Path:
    candidate = value if value.is_absolute() else repo / value
    resolved = candidate.resolve()
    if candidate != resolved or not resolved.is_file():
        raise ValueError(f"{label} must be canonical, existing, and regular")
    return resolved


def _lexical_executable(value: Path) -> Path:
    """Validate an absolute launcher without resolving its symlink leaf."""
    candidate = value.expanduser()
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("interpreter must be an absolute path without traversal")
    launcher = Path(os.path.abspath(candidate))
    try:
        mode = launcher.stat().st_mode
    except OSError as exc:
        raise ValueError("interpreter must exist") from exc
    if not stat.S_ISREG(mode) or not os.access(launcher, os.X_OK):
        raise ValueError("interpreter must be an executable regular file")
    return launcher


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--probe", type=Path, default=Path("aloha_isaac_rebuild/scripts/probe_a21_runtime_target_readback_once.py")
    )
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--layer1", type=Path)
    parser.add_argument("--layer2", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--interpreter", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    output: Path | None = None
    report: Path | None = None
    result: dict[str, object]
    try:
        config_path = _canonical_file(repo, args.config, label="config")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not isinstance(config.get("outputs"), dict):
            raise ValueError("config outputs must be an object")
        outputs = config["outputs"]
        stage = _canonical_file(repo, args.stage or Path(outputs["a19_clean_articulation_candidate"]), label="stage")
        mapping = _canonical_file(
            repo, args.mapping or Path(outputs["a17_clean_articulation_mapping_plan_json"]), label="mapping"
        )
        layer1_path = _canonical_file(repo, args.layer1 or Path(outputs["a20_usd_dof_metadata_json"]), label="layer1")
        layer2_path = _canonical_file(
            repo, args.layer2 or Path(outputs["a20_runtime_articulation_discovery_json"]), label="layer2"
        )
        preflight_path = _canonical_file(
            repo, args.preflight or Path(outputs["a21_policy_target_limit_preflight_json"]), label="preflight"
        )
        probe_path = _canonical_file(repo, args.probe, label="probe")
        interpreter = _lexical_executable(args.interpreter)
        output = (args.output or repo / outputs["a21_runtime_target_readback_json"]).resolve()
        report = (args.report or repo / outputs["a21_target_limit_and_readback_md"]).resolve()
        provenance = _code_provenance(repo, probe_path, Path(__file__).resolve())
        layer1, layer2, preflight = (
            _load_json_fail_closed(layer1_path),
            _load_json_fail_closed(layer2_path),
            _load_json_fail_closed(preflight_path),
        )
        if (
            provenance["git_dirty"] is not False
            or provenance["safety_checker"].get("ok") is not True
            or not is_exact_runtime_pass(layer2, layer1)
            or not _exact_preflight(preflight)
        ):
            result = aggregate_batches(preflight, [])
            result["errors"].append(_error("prerequisite_validation_failed", provenance=provenance))
        else:
            extra = [
                "--config",
                str(config_path),
                "--stage",
                str(stage),
                "--mapping",
                str(mapping),
                "--a20-evidence",
                str(layer2_path),
            ]
            result = aggregate_batches(
                preflight,
                run_two_batches(repo, interpreter, probe_path, args.timeout, extra_args=extra, provenance=provenance),
            )
        result["inputs"] = {
            "config": {"path": str(config_path), "sha256": _digest(config_path)},
            "stage_before": {"path": str(stage), "sha256": _digest(stage)},
            "mapping": {"path": str(mapping), "sha256": _digest(mapping)},
            "layer1": {"path": str(layer1_path), "sha256": _digest(layer1_path)},
            "layer2": {"path": str(layer2_path), "sha256": _digest(layer2_path)},
            "preflight": {"path": str(preflight_path), "sha256": _digest(preflight_path)},
            "probe": {"path": str(probe_path), "sha256": _digest(probe_path)},
        }
        after = _digest(stage)
        result["inputs"]["stage_after"] = {"path": str(stage), "sha256": after}
        if after != result["inputs"]["stage_before"]["sha256"]:
            result["status"], result["ok"] = FAIL_STATUS, False
            result["errors"].append(_error("stage_hash_changed_during_coordinate"))
    except Exception as exc:
        result = {
            "schema_version": "a21-runtime-target-readback-v1",
            "status": FAIL_STATUS,
            "ok": False,
            "runs": [],
            "errors": [_error("cli_input_error", message=str(exc))],
        }
    if output is not None:
        a21a._atomic_write(output, result)  # noqa: SLF001 - Task4 reviewed hardened atomic writer.
    if report is not None:
        try:
            _atomic_write_text(report, format_report(result))
        except Exception:
            report.unlink(missing_ok=True)
            result["status"], result["ok"] = FAIL_STATUS, False
            result.setdefault("errors", []).append(_error("report_write_failed"))
            if output is not None:
                a21a._atomic_write(output, result)  # noqa: SLF001 - Task4 reviewed hardened atomic writer.
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == PASS_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
