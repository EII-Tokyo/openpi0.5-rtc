#!/usr/bin/env python3
"""One-shot, no-step A20 Isaac runtime articulation discovery probe."""

from __future__ import annotations

import argparse
from datetime import UTC
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

import yaml

MARKER = "A20_RUNTIME_DISCOVERY_JSON="
CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml")
SAFETY = {
    "physics_stepped": False,
    "actions_applied": False,
    "targets_written": False,
    "stage_saved": False,
}


class RuntimeDiscoveryError(RuntimeError):
    """Identify the exact reviewed runtime API that failed."""

    def __init__(self, api: str, cause: object):
        self.api = api
        super().__init__(f"{api}: {cause}")


def _call_runtime_api(api: str, operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except Exception as exc:
        raise RuntimeDiscoveryError(api, exc) from exc


def _as_list(value):
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _discover_runtime_records(
    stage_path: str,
    expected: list[dict[str, object]],
    usd_context,
    physics_interface,
    tensors_module,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Create and inspect a PhysX tensor articulation view without advancing time."""
    opened = _call_runtime_api(
        "omni.usd.get_context().open_stage", usd_context.open_stage, stage_path
    )
    if opened is False:
        raise RuntimeDiscoveryError("omni.usd.get_context().open_stage", "returned false")
    _call_runtime_api(
        "omni.physx.IPhysx.force_load_physics_from_usd",
        physics_interface.force_load_physics_from_usd,
    )
    _call_runtime_api(
        "omni.physx.IPhysx.start_simulation", physics_interface.start_simulation
    )
    stage_id = _call_runtime_api(
        "omni.usd.get_context().get_stage_id", usd_context.get_stage_id
    )
    simulation_view = _call_runtime_api(
        "omni.physics.tensors.create_simulation_view",
        tensors_module.create_simulation_view,
        "numpy",
        stage_id=stage_id,
    )
    _call_runtime_api(
        "omni.physics.tensors.SimulationView.set_subspace_roots",
        simulation_view.set_subspace_roots,
        "/",
    )
    articulation_view = _call_runtime_api(
        "omni.physics.tensors.SimulationView.create_articulation_view",
        simulation_view.create_articulation_view,
        ["/aloha/root_joint"],
    )
    if articulation_view is None:
        raise RuntimeDiscoveryError(
            "omni.physics.tensors.SimulationView.create_articulation_view", "returned none"
        )

    metadata = articulation_view.shared_metatype
    names = list(metadata.dof_names)
    types = list(metadata.dof_types)
    raw_paths = _as_list(articulation_view.dof_paths)
    paths = list(raw_paths[0]) if raw_paths and isinstance(raw_paths[0], list | tuple) else raw_paths
    raw_limits = _call_runtime_api(
        "omni.physics.tensors.ArticulationView.get_dof_limits",
        articulation_view.get_dof_limits,
    )
    limit_sets = _as_list(raw_limits)
    limits = limit_sets[0] if limit_sets and len(limit_sets) == 1 else limit_sets
    dof_count = int(articulation_view.max_dofs)
    articulation_count = int(articulation_view.count)
    prim_paths = list(articulation_view.prim_paths)
    root = str(prim_paths[0]) if articulation_count == 1 and len(prim_paths) == 1 else None
    if not (len(expected) == len(names) == len(types) == len(paths) == len(limits) == dof_count):
        raise RuntimeDiscoveryError(
            "omni.physics.tensors.ArticulationView.metadata",
            f"inconsistent lengths expected={len(expected)} names={len(names)} types={len(types)} "
            f"paths={len(paths)} limits={len(limits)} max_dofs={dof_count}",
        )

    records: list[dict[str, object]] = []
    for index, (template, name, path, dof_type, limit) in enumerate(
        zip(expected, names, paths, types, limits, strict=True)
    ):
        type_name = dof_type.name
        joint_type = {
            "Rotation": "PhysicsRevoluteJoint",
            "Translation": "PhysicsPrismaticJoint",
        }.get(type_name)
        if joint_type is None:
            raise RuntimeDiscoveryError(
                "omni.physics.tensors.ArticulationView.shared_metatype.dof_types",
                f"unsupported DOF type at index {index}: {dof_type}",
            )
        bounds = _as_list(limit)
        if len(bounds) != 2:
            raise RuntimeDiscoveryError(
                "omni.physics.tensors.ArticulationView.get_dof_limits",
                f"invalid bounds at index {index}: {bounds}",
            )
        record = {
            **template,
            "path": str(path),
            "name": str(name),
            "joint_type": joint_type,
            "lower_limit": math.degrees(float(bounds[0]))
            if type_name == "Rotation"
            else float(bounds[0]),
            "upper_limit": math.degrees(float(bounds[1]))
            if type_name == "Rotation"
            else float(bounds[1]),
            "index": index,
        }
        records.append(record)
    return records, {
        "articulation_root": root,
        "articulation_count": articulation_count,
        "dof_count": dof_count,
    }


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except Exception:
        return "unknown"


def _emit_marker(payload: dict[str, object], printer=print, serializer=json.dumps) -> None:
    try:
        encoded = serializer(payload, sort_keys=True, separators=(",", ":"))
    except Exception as exc:
        encoded = json.dumps(
            {
                "status": "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY",
                "errors": [{"code": "marker_serialization_error", "message": str(exc)}],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    try:
        printer(MARKER + encoded, flush=True)
    except TypeError:
        printer(MARKER + encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invocation-id", required=True)
    args = parser.parse_args()
    started = _now()
    payload: dict[str, object] = {
        "status": "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY",
        "probe_returncode": 1,
        "invocation_id": args.invocation_id,
        "pid": os.getpid(),
        "started_at": started,
        "finished_at": started,
        "isaac_sim_version": _safe_version("isaacsim"),
        "inputs": {},
        "articulation_root": None,
        "articulation_count": 0,
        "dof_count": 0,
        "valid_handle": False,
        "records": [],
        "requires_unapproved_initialization": False,
        "initialization_operations": [],
        **SAFETY,
    }
    app = None
    try:
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True})
        from omni.physics import tensors
        from omni.physx import get_physx_interface
        import omni.usd

        from aloha_isaac_rebuild.scripts.audit_a20_usd_dof_metadata import expected_dof_records

        repo = Path.cwd().resolve()
        config_path = (repo / CONFIG).resolve()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        outputs = config["outputs"]
        stage_path = (repo / outputs["a19_clean_articulation_candidate"]).resolve()
        mapping_path = (repo / outputs["a17_clean_articulation_mapping_plan_json"]).resolve()
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        records = expected_dof_records(mapping)
        records, runtime = _discover_runtime_records(
            str(stage_path),
            records,
            omni.usd.get_context(),
            get_physx_interface(),
            tensors,
        )
        payload = {
            "status": "PASS_RUNTIME_PROBE",
            "probe_returncode": 0,
            "invocation_id": args.invocation_id,
            "pid": os.getpid(),
            "started_at": started,
            "finished_at": _now(),
            "isaac_sim_version": _safe_version("isaacsim"),
            "inputs": {
                "config": {"path": str(config_path), "sha256": _digest(config_path)},
                "mapping": {"path": str(mapping_path), "sha256": _digest(mapping_path)},
                "stage": {"path": str(stage_path), "sha256": _digest(stage_path)},
            },
            **runtime,
            "valid_handle": True,
            "records": records,
            "requires_unapproved_initialization": False,
            "initialization_operations": [],
            **SAFETY,
        }
        if runtime != {
            "articulation_root": "/aloha/root_joint",
            "articulation_count": 1,
            "dof_count": 16,
        }:
            payload["status"] = "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"
            payload["valid_handle"] = False
    except RuntimeDiscoveryError as exc:
        payload = {
            **payload,
            "status": "BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION",
            "probe_returncode": 0,
            "finished_at": _now(),
            "requires_unapproved_initialization": True,
            "initialization_operations": [exc.api],
            "errors": [{"code": "runtime_api_failure", "api": exc.api, "message": str(exc)}],
        }
    except Exception as exc:
        payload = {
            "status": "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY",
            "probe_returncode": 1,
            "invocation_id": args.invocation_id,
            "pid": os.getpid(),
            "started_at": started,
            "finished_at": _now(),
            "isaac_sim_version": _safe_version("isaacsim"),
            "inputs": {},
            "articulation_root": None,
            "articulation_count": 0,
            "dof_count": 0,
            "valid_handle": False,
            "records": [],
            "requires_unapproved_initialization": False,
            "initialization_operations": [],
            "errors": [{"code": "probe_error", "message": str(exc)}],
            **SAFETY,
        }
    finally:
        payload["finished_at"] = _now()
        try:
            _emit_marker(payload)
        finally:
            if app is not None:
                try:  # noqa: SIM105 -- close is best-effort after marker emission
                    app.close()
                except Exception:
                    pass
    return 0 if payload["status"] in {
        "PASS_RUNTIME_PROBE",
        "BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
