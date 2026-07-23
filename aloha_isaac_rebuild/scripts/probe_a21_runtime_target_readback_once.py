#!/usr/bin/env python3
"""One-shot A21 target-buffer readback probe; it never advances Isaac physics."""

from __future__ import annotations

import argparse
from contextlib import suppress
from copy import deepcopy
from datetime import UTC
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import yaml

from aloha_isaac_rebuild.scripts.a20_policy_runtime_order_adapter import SCHEMA_VERSION as A20_SCHEMA_VERSION
from aloha_isaac_rebuild.scripts.run_a20_runtime_articulation_discovery import is_exact_runtime_pass

MARKER = "A21_RUNTIME_TARGET_READBACK_JSON="
PASS_STATUS = "PASS_A21_RUNTIME_TARGET_READBACK_ONCE"
FAIL_STATUS = "FAIL_A21_RUNTIME_TARGET_READBACK_ONCE"
DEFAULT_CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml")
ARM_DELTA_RAD = math.radians(0.25)
FINGER_DELTA_M = 0.00025
READBACK_ATOL = 1e-7
_RUNTIME_DIMENSION = 16
_POLICY_DIMENSION = 14
_EXPECTED_RUNTIME_INDICES = {
    "left": [0, 2, 4, 6, 8, 10, 12, 13],
    "right": [1, 3, 5, 7, 9, 11, 14, 15],
}


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _error(code: str, message: object, **fields: object) -> dict[str, object]:
    return {"code": code, "message": str(message), **fields}


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _strict_index(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def batch_policy_indices(side: str) -> list[int]:
    """Return the exact seven policy slots that form one ALOHA arm batch."""
    if side == "left":
        return list(range(7))
    if side == "right":
        return list(range(7, 14))
    raise ValueError(f"side must be 'left' or 'right', got {side!r}")


def choose_interior_delta(
    runtime_index: int,
    baseline: float,
    lower_limit: float,
    upper_limit: float,
    magnitude: float,
) -> float:
    """Pick the approved parity delta, reversing it only to stay strictly in bounds."""
    index = _strict_index(runtime_index, field="runtime_index")
    base = _finite_float(baseline, field="baseline")
    lower = _finite_float(lower_limit, field="lower_limit")
    upper = _finite_float(upper_limit, field="upper_limit")
    size = _finite_float(magnitude, field="magnitude")
    if not lower < upper:
        raise ValueError("lower_limit must be less than upper_limit")
    if not lower <= base <= upper:
        raise ValueError("baseline must be within live limits")
    if size <= 0.0:
        raise ValueError("magnitude must be positive")
    preferred = size if index % 2 == 0 else -size
    for candidate in (preferred, -preferred):
        if lower < base + candidate < upper:
            return candidate
    raise ValueError("no strictly interior target is available for requested delta")


def _runtime_bounds(record: dict[str, object]) -> tuple[float, float, str]:
    lower = _finite_float(record.get("lower_limit"), field="lower_limit")
    upper = _finite_float(record.get("upper_limit"), field="upper_limit")
    if lower >= upper:
        raise ValueError("runtime lower_limit must be less than upper_limit")
    joint_type = record.get("joint_type")
    if joint_type == "PhysicsRevoluteJoint":
        return math.radians(lower), math.radians(upper), joint_type
    if joint_type == "PhysicsPrismaticJoint":
        return lower, upper, joint_type
    raise ValueError(f"unsupported runtime joint_type: {joint_type!r}")


def _validate_vector(values: object, *, field: str) -> list[int]:
    if not isinstance(values, list) or len(values) != _RUNTIME_DIMENSION:
        raise ValueError(f"{field} must contain exactly 16 indices")
    indices = [_strict_index(value, field=field) for value in values]
    if sorted(indices) != list(range(_RUNTIME_DIMENSION)):
        raise ValueError(f"{field} inventory must be exactly 0..15")
    return indices


def _validate_target_contract(
    adapter: object, runtime_records: object, side: str
) -> tuple[list[int], dict[int, dict[str, object]]]:
    """Parse the effective A20 adapter, then bind its selected entries to raw records."""
    policy_indices = batch_policy_indices(side)
    if not isinstance(adapter, dict):
        raise ValueError("adapter must be an object")
    if adapter.get("schema_version") != A20_SCHEMA_VERSION:
        raise ValueError("invalid A20 adapter schema_version")
    if adapter.get("policy_dimension") != _POLICY_DIMENSION:
        raise ValueError("invalid A20 adapter policy_dimension")
    if adapter.get("runtime_dimension") != _RUNTIME_DIMENSION:
        raise ValueError("invalid A20 adapter runtime_dimension")
    if adapter.get("mapping_complete") is not True:
        raise ValueError("A20 adapter mapping_complete must be exactly true")
    canonical_order = adapter.get("canonical_order")
    runtime_order = adapter.get("runtime_order")
    if (
        not isinstance(canonical_order, list)
        or not isinstance(runtime_order, list)
        or len(canonical_order) != _RUNTIME_DIMENSION
        or len(runtime_order) != _RUNTIME_DIMENSION
        or any(not isinstance(path, str) or not path for path in canonical_order + runtime_order)
        or len(set(canonical_order)) != _RUNTIME_DIMENSION
        or len(set(runtime_order)) != _RUNTIME_DIMENSION
        or set(canonical_order) != set(runtime_order)
    ):
        raise ValueError("invalid A20 adapter canonical/runtime path inventories")
    canonical_to_runtime = _validate_vector(
        adapter.get("canonical_to_runtime_indices"), field="canonical_to_runtime_indices"
    )
    runtime_to_canonical = _validate_vector(
        adapter.get("runtime_to_canonical_indices"), field="runtime_to_canonical_indices"
    )
    if any(
        runtime_to_canonical[runtime_index] != canonical_index
        for canonical_index, runtime_index in enumerate(canonical_to_runtime)
    ):
        raise ValueError("A20 adapter index vectors must be mutual inverses")
    if any(
        runtime_order[runtime_index] != canonical_order[canonical_index]
        for canonical_index, runtime_index in enumerate(canonical_to_runtime)
    ):
        raise ValueError("A20 adapter paths do not match index vectors")

    if not isinstance(runtime_records, list) or len(runtime_records) != _RUNTIME_DIMENSION:
        raise ValueError("runtime_records must contain exactly 16 objects")
    by_index: dict[int, dict[str, object]] = {}
    paths: set[str] = set()
    for record in runtime_records:
        if not isinstance(record, dict):
            raise ValueError("runtime_records must contain objects")
        index = _strict_index(record.get("index"), field="runtime index")
        if index in by_index:
            raise ValueError(f"duplicate runtime index: {index}")
        path = record.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"invalid runtime path at index {index}")
        if path in paths:
            raise ValueError(f"duplicate runtime path: {path}")
        _runtime_bounds(record)
        by_index[index] = record
        paths.add(path)
    if sorted(by_index) != list(range(_RUNTIME_DIMENSION)) or paths != set(runtime_order):
        raise ValueError("runtime record inventory must exactly match A20 adapter")
    if any(by_index[index]["path"] != runtime_order[index] for index in range(_RUNTIME_DIMENSION)):
        raise ValueError("runtime record paths do not match A20 runtime order")

    entries = adapter.get("policy_to_runtime")
    if (
        not isinstance(entries, list)
        or len(entries) != _POLICY_DIMENSION
        or not all(isinstance(entry, dict) for entry in entries)
    ):
        raise ValueError("invalid A20 policy_to_runtime entries")
    expanded_by_policy: dict[int, list[int]] = {}
    seen_runtime: list[int] = []
    for expected_policy, entry in enumerate(entries):
        if _strict_index(entry.get("openpi_index"), field="openpi_index") != expected_policy:
            raise ValueError("A20 policy entries must be in exact policy order")
        indices = entry.get("runtime_indices")
        transforms = entry.get("transforms")
        if not isinstance(indices, list) or not isinstance(transforms, list) or len(indices) != len(transforms):
            raise ValueError(f"invalid A20 transforms for policy index {expected_policy}")
        if len(indices) != (2 if expected_policy in {6, 13} else 1):
            raise ValueError(f"missing paired finger or invalid arm mapping at policy index {expected_policy}")
        parsed: list[int] = []
        for raw_index, transform in zip(indices, transforms, strict=True):
            runtime_index = _strict_index(raw_index, field="adapter runtime index")
            if runtime_index not in by_index:
                raise ValueError(f"adapter runtime index outside inventory: {runtime_index}")
            if not isinstance(transform, dict) or transform.get("path") != by_index[runtime_index]["path"]:
                raise ValueError("adapter transform path must match raw runtime record")
            for field in ("sign", "offset", "scale"):
                value = _finite_float(transform.get(field), field=f"transform {field}")
                if field == "scale" and value == 0.0:
                    raise ValueError("adapter transform scale must not be zero")
            parsed.append(runtime_index)
        if len(set(parsed)) != len(parsed):
            raise ValueError("duplicate runtime index within A20 policy entry")
        expanded_by_policy[expected_policy] = parsed
        seen_runtime.extend(parsed)
    if sorted(seen_runtime) != list(range(_RUNTIME_DIMENSION)):
        raise ValueError("A20 policy transforms must cover raw runtime inventory exactly once")
    for finger_policy, prefix in ((6, "/aloha/joints/left_"), (13, "/aloha/joints/right_")):
        finger_paths = [by_index[index]["path"] for index in expanded_by_policy[finger_policy]]
        if set(finger_paths) != {prefix + "left_finger", prefix + "right_finger"}:
            raise ValueError(f"missing complete paired fingers for policy index {finger_policy}")
    selected = [runtime_index for policy_index in policy_indices for runtime_index in expanded_by_policy[policy_index]]
    if selected != _EXPECTED_RUNTIME_INDICES[side] or len(set(selected)) != 8:
        raise ValueError(f"A20 adapter does not resolve expected {side} raw runtime target slots")
    return selected, by_index


def _target_array(value: object, *, field: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != (1, _RUNTIME_DIMENSION):
        raise ValueError(f"{field} must be a numeric ndarray with shape (1, 16)")
    if not np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.bool_):
        raise ValueError(f"{field} must be a numeric ndarray with shape (1, 16)")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{field} must contain only finite values")
    return value


def _safety(*, targets_written: bool, targets_restored: bool) -> dict[str, bool]:
    return {
        "physics_stepped": False,
        "positions_written": False,
        "velocities_written": False,
        "efforts_written": False,
        "targets_written": targets_written,
        "targets_restored": targets_restored,
        "target_only_no_step": True,
    }


def exercise_target_batch(view: object, adapter: object, runtime_records: object, *, side: str) -> dict[str, object]:
    """Write eight target slots once, prove readback, and always attempt full restoration."""
    result: dict[str, object] = {
        "ok": False,
        "side": side,
        "runtime_indices": [],
        "deltas": [],
        "baseline": None,
        "modified": None,
        "readback": None,
        "restoration": {"attempted": False, "readback": None, "ok": False},
        "errors": [],
        "safety": _safety(targets_written=False, targets_restored=False),
    }
    errors: list[dict[str, object]] = result["errors"]  # type: ignore[assignment]
    baseline: np.ndarray | None = None
    write_phase_started = False
    targets_written = False
    targets_restored = False
    try:
        indices, by_index = _validate_target_contract(adapter, runtime_records, side)
        result["runtime_indices"] = indices
        getter = view.get_dof_position_targets
        baseline = _target_array(getter(), field="baseline").copy()
        result["baseline"] = baseline.tolist()
        for runtime_index in range(_RUNTIME_DIMENSION):
            lower, upper, _ = _runtime_bounds(by_index[runtime_index])
            value = float(baseline[0, runtime_index])
            if not lower <= value <= upper:
                raise ValueError(f"baseline outside live limits at runtime index {runtime_index}")
        modified = baseline.copy()
        deltas: list[dict[str, object]] = []
        for runtime_index in indices:
            lower, upper, joint_type = _runtime_bounds(by_index[runtime_index])
            magnitude = ARM_DELTA_RAD if joint_type == "PhysicsRevoluteJoint" else FINGER_DELTA_M
            delta = choose_interior_delta(runtime_index, float(baseline[0, runtime_index]), lower, upper, magnitude)
            modified[0, runtime_index] = baseline[0, runtime_index] + delta
            deltas.append(
                {
                    "runtime_index": runtime_index,
                    "path": by_index[runtime_index]["path"],
                    "joint_type": joint_type,
                    "lower_limit": lower,
                    "upper_limit": upper,
                    "baseline": float(baseline[0, runtime_index]),
                    "delta": delta,
                    "target": float(modified[0, runtime_index]),
                }
            )
        result["deltas"] = deltas
        result["modified"] = modified.tolist()
        setter = view.set_dof_position_targets
        write_phase_started = True
        setter(modified, [0])
        targets_written = True
        readback = _target_array(getter(), field="modified readback")
        result["readback"] = readback.tolist()
        if not np.allclose(readback[:, indices], modified[:, indices], rtol=0.0, atol=READBACK_ATOL):
            raise ValueError("modified target readback does not match intended target values")
        untouched = sorted(set(range(_RUNTIME_DIMENSION)) - set(indices))
        if not np.array_equal(readback[:, untouched], baseline[:, untouched]):
            raise ValueError("setter changed a target slot outside the selected batch")
    except Exception as exc:
        errors.append(_error("target_exercise_error", exc))
    finally:
        if baseline is not None and write_phase_started:
            result["restoration"]["attempted"] = True  # type: ignore[index]
            try:
                view.set_dof_position_targets(baseline.copy(), [0])
                restored_readback = _target_array(view.get_dof_position_targets(), field="restoration readback")
                result["restoration"]["readback"] = restored_readback.tolist()  # type: ignore[index]
                if not np.array_equal(restored_readback, baseline):
                    raise ValueError("restoration readback does not exactly match baseline")
                result["restoration"]["ok"] = True  # type: ignore[index]
                targets_restored = True
            except Exception as exc:
                errors.append(_error("target_restoration_error", exc))
    result["safety"] = _safety(targets_written=targets_written, targets_restored=targets_restored)
    result["ok"] = not errors and targets_written and targets_restored
    return result


def _configured_path(repo: Path, configured: object, *, field: str) -> Path:
    if not isinstance(configured, str) or not configured:
        raise ValueError(f"config outputs.{field} must be a non-empty path")
    candidate = repo / configured
    resolved = candidate.resolve()
    if candidate.resolve(strict=False) != resolved or not resolved.is_file():
        raise ValueError(f"configured {field} must be an existing regular file")
    return resolved


def _bound_input_path(repo: Path, supplied: Path | None, configured: Path, *, field: str) -> Path:
    if supplied is None:
        return configured
    candidate = supplied if supplied.is_absolute() else repo / supplied
    resolved = candidate.resolve()
    if resolved != configured:
        raise ValueError(f"--{field} must exactly match the configured current input")
    return resolved


def _emit_marker(payload: dict[str, object]) -> None:
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except Exception as exc:
        encoded = json.dumps(
            {"status": FAIL_STATUS, "errors": [_error("marker_serialization_error", exc)]},
            sort_keys=True,
            separators=(",", ":"),
        )
    print(MARKER + encoded, flush=True)


def main() -> int:
    """Run one target-only readback batch after A20 evidence is bound to current inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invocation-id", required=True)
    parser.add_argument("--batch", required=True, choices=("left", "right"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--a20-evidence", type=Path)
    args = parser.parse_args()
    started = _now()
    payload: dict[str, object] = {
        "status": FAIL_STATUS,
        "invocation_id": args.invocation_id,
        "batch": args.batch,
        "pid": os.getpid(),
        "started_at": started,
        "finished_at": started,
        "inputs": {},
        "adapter_provenance": None,
        "record_provenance": None,
        "result": None,
        "errors": [],
        "safety": _safety(targets_written=False, targets_restored=False),
        "declaration": "target-only/no-step; no position, velocity, or effort writes",
    }
    app = None
    try:
        repo = Path.cwd().resolve()
        requested_config = args.config if args.config.is_absolute() else repo / args.config
        config_path = requested_config.resolve()
        if requested_config != config_path or not config_path.is_file():
            raise ValueError("config path must be canonical, existing, and regular")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not isinstance(config.get("outputs"), dict):
            raise ValueError("config outputs must be an object")
        outputs = config["outputs"]
        stage_path = _bound_input_path(
            repo,
            args.stage,
            _configured_path(
                repo, outputs.get("a19_clean_articulation_candidate"), field="a19_clean_articulation_candidate"
            ),
            field="stage",
        )
        mapping_path = _bound_input_path(
            repo,
            args.mapping,
            _configured_path(
                repo,
                outputs.get("a17_clean_articulation_mapping_plan_json"),
                field="a17_clean_articulation_mapping_plan_json",
            ),
            field="mapping",
        )
        evidence_path = _bound_input_path(
            repo,
            args.a20_evidence,
            _configured_path(
                repo,
                outputs.get("a20_runtime_articulation_discovery_json"),
                field="a20_runtime_articulation_discovery_json",
            ),
            field="a20-evidence",
        )
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        if not isinstance(mapping, dict):
            raise ValueError("mapping must be an object")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if not is_exact_runtime_pass(evidence):
            raise ValueError("A20 runtime evidence is not an exact no-step pass")
        adapter = evidence.get("order_adapter")
        runs = evidence.get("runs")
        if not isinstance(runs, list) or not runs or not isinstance(runs[0], dict):
            raise ValueError("A20 runtime evidence has no bound runtime records")
        records = runs[0].get("records")
        inputs = {
            "config": {"path": str(config_path), "sha256": _digest(config_path)},
            "stage": {"path": str(stage_path), "sha256": _digest(stage_path)},
            "mapping": {"path": str(mapping_path), "sha256": _digest(mapping_path)},
            "a20_evidence": {"path": str(evidence_path), "sha256": _digest(evidence_path)},
        }
        payload["inputs"] = inputs
        payload["adapter_provenance"] = {
            "source": "a20_runtime_articulation_discovery.order_adapter",
            "schema_version": adapter.get("schema_version") if isinstance(adapter, dict) else None,
        }
        payload["record_provenance"] = {
            "source": "a20_runtime_articulation_discovery.runs[0].records",
            "run_invocation_id": runs[0].get("invocation_id"),
        }

        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True})
        from omni.physics import tensors
        from omni.physx import get_physx_interface
        import omni.usd

        opened = omni.usd.get_context().open_stage(str(stage_path))
        if opened is False:
            raise RuntimeError("omni.usd.get_context().open_stage returned false")
        interface = get_physx_interface()
        if interface.force_load_physics_from_usd() is False:
            raise RuntimeError("force_load_physics_from_usd returned false")
        if interface.start_simulation() is False:
            raise RuntimeError("start_simulation returned false")
        stage_id = omni.usd.get_context().get_stage_id()
        simulation_view = tensors.create_simulation_view("numpy", stage_id=stage_id)
        simulation_view.set_subspace_roots("/")
        articulation_view = simulation_view.create_articulation_view(["/aloha/root_joint"])
        if articulation_view is None:
            raise RuntimeError("create_articulation_view returned none")
        result = exercise_target_batch(articulation_view, deepcopy(adapter), deepcopy(records), side=args.batch)
        payload["result"] = result
        payload["safety"] = result["safety"]
        if result["ok"] is True:
            payload["status"] = PASS_STATUS
        else:
            payload["errors"] = result["errors"]
    except Exception as exc:
        payload["errors"] = [_error("probe_error", exc)]
    finally:
        payload["finished_at"] = _now()
        _emit_marker(payload)
        if app is not None:
            with suppress(Exception):
                app.close()
    return 0 if payload["status"] == PASS_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
