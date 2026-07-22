#!/usr/bin/env python3
"""Fail-closed A20 Layer 1 audit of authored USD articulation metadata."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import yaml

from aloha_isaac_rebuild.scripts.a20_articulation_gate_common import compare_dof_records
from aloha_isaac_rebuild.scripts.a20_articulation_gate_common import validate_dof_records
from aloha_isaac_rebuild.scripts.a20_articulation_gate_common import validate_safety_flags


def _bootstrap_bundled_openusd() -> None:
    """Expose Isaac's bundled OpenUSD bindings without importing Isaac or Kit."""
    try:
        __import__("pxr")
        return
    except ModuleNotFoundError:
        pass
    site_packages = Path(sys.prefix) / "lib/python3.11/site-packages"
    candidates = sorted(
        site_packages.glob("isaacsim/extscache/omni.usd.libs-*.cp311")
    )
    if len(candidates) != 1:
        raise ModuleNotFoundError(
            f"expected one bundled OpenUSD package under {site_packages}, found {candidates}"
        )
    python_library = Path(sys.executable).resolve().parents[1] / "lib/libpython3.11.so.1.0"
    if not python_library.is_file():
        raise ModuleNotFoundError(f"missing Python shared library: {python_library}")
    ctypes.CDLL(str(python_library), mode=ctypes.RTLD_GLOBAL)
    for library in ("libusd_tf.so", "libusd_usd.so", "libusd_usdPhysics.so"):
        ctypes.CDLL(str(candidates[0] / "bin" / library), mode=ctypes.RTLD_GLOBAL)
    sys.path.insert(0, str(candidates[0]))


_bootstrap_bundled_openusd()

Usd = importlib.import_module("pxr.Usd")
UsdPhysics = importlib.import_module("pxr.UsdPhysics")


DEFAULT_CONFIG = Path(
    "aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml"
)
_DOF_TYPES = {
    "PhysicsRevoluteJoint": UsdPhysics.RevoluteJoint,
    "PhysicsPrismaticJoint": UsdPhysics.PrismaticJoint,
}
_SAFETY_FLAGS = {
    "physics_stepped": False,
    "actions_applied": False,
    "targets_written": False,
    "stage_saved": False,
}


def _absolute_existing_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is not an existing file: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_record(path: Path) -> dict[str, str]:
    digest = _sha256(path)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"invalid SHA-256 for {path}: {digest!r}")
    return {"path": str(path), "sha256": digest}


def _single_target(joint: UsdPhysics.Joint, relationship_name: str) -> list[str]:
    relationship = (
        joint.GetBody0Rel() if relationship_name == "body0" else joint.GetBody1Rel()
    )
    return [str(target) for target in relationship.GetTargets()]


def expected_dof_records(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    """Build canonical expected records solely from the A17 mapping artifact."""
    order = mapping.get("proposed_canonical_dof_order")
    joint_records = mapping.get("joint_records")
    if not isinstance(order, list) or not isinstance(joint_records, list):
        raise ValueError("A17 mapping lacks canonical order or joint records")
    records_by_path = {
        record.get("proposed_clean_joint_path"): record
        for record in joint_records
        if isinstance(record, dict) and record.get("is_dof_joint") is True
    }
    expected: list[dict[str, Any]] = []
    for index, order_record in enumerate(order):
        path = order_record.get("clean_joint_path")
        source = records_by_path.get(path)
        if source is None:
            raise ValueError(f"A17 canonical path has no DOF joint record: {path}")
        expected.append(
            {
                "index": index,
                "path": path,
                "name": source.get("source_joint_name"),
                "joint_type": source.get("joint_type"),
                "axis": source.get("axis"),
                "lower_limit": float(source["lower_limit"]),
                "upper_limit": float(source["upper_limit"]),
                "body0": source.get("clean_body0"),
                "body1": source.get("clean_body1"),
            }
        )
    validation = validate_dof_records(expected)
    if not validation["ok"]:
        raise ValueError(f"invalid A17 expected records: {validation['errors']}")
    return expected


def _observed_record(stage: Usd.Stage, path: str, index: int) -> dict[str, Any]:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise ValueError(f"missing expected DOF prim: {path}")
    prim_type = prim.GetTypeName()
    schema_type = _DOF_TYPES.get(prim_type)
    if schema_type is None:
        raise ValueError(f"unsupported DOF joint type at {path}: {prim_type}")
    joint = schema_type(prim)
    return {
        "index": index,
        "path": path,
        "name": prim.GetName(),
        "joint_type": prim_type,
        "axis": str(joint.GetAxisAttr().Get()),
        "lower_limit": float(joint.GetLowerLimitAttr().Get()),
        "upper_limit": float(joint.GetUpperLimitAttr().Get()),
        "body0": _single_target(joint, "body0"),
        "body1": _single_target(joint, "body1"),
    }


def evaluate_metadata(
    default_prim: str | None,
    articulation_root_paths: list[str],
    expected: list[dict[str, Any]],
    observed: list[dict[str, Any]],
    *,
    observed_dof_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate collected metadata with shared fail-closed validators."""
    comparison = compare_dof_records(expected, observed)
    mismatches = list(comparison["mismatches"])
    errors = list(comparison.get("validation_errors", []))
    if default_prim != "/aloha":
        mismatches.append(
            {"field": "default_prim", "expected": "/aloha", "observed": default_prim}
        )
    if articulation_root_paths != ["/aloha/root_joint"]:
        mismatches.append(
            {
                "field": "articulation_root_paths",
                "expected": ["/aloha/root_joint"],
                "observed": articulation_root_paths,
            }
        )
    expected_paths = [record.get("path") for record in expected]
    actual_paths = (
        observed_dof_paths
        if observed_dof_paths is not None
        else [record.get("path") for record in observed]
    )
    if actual_paths != expected_paths:
        mismatches.append(
            {"field": "dof_paths", "expected": expected_paths, "observed": actual_paths}
        )
    safety = validate_safety_flags(_SAFETY_FLAGS)
    errors.extend(safety["errors"])
    return {"ok": not mismatches and not errors, "mismatches": mismatches, "errors": errors}


def _collect(config_path: Path) -> dict[str, Any]:
    config_path = _absolute_existing_file(config_path, "config_path")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outputs = config.get("outputs", {})
    stage_path = _absolute_existing_file(
        Path(outputs["a19_clean_articulation_candidate"]), "stage_path"
    )
    mapping_path = _absolute_existing_file(
        Path(outputs["a17_clean_articulation_mapping_plan_json"]), "mapping_path"
    )
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    expected = expected_dof_records(mapping)

    stage = Usd.Stage.Open(str(stage_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise ValueError(f"could not open USD stage: {stage_path}")
    default_prim = stage.GetDefaultPrim()
    default_prim_path = str(default_prim.GetPath()) if default_prim else None
    articulation_root_paths = sorted(
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    )
    observed_dof_paths = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith("/aloha/joints/")
        and prim.GetTypeName() in _DOF_TYPES
    ]
    expected_paths = [record["path"] for record in expected]
    observed_by_path = set(observed_dof_paths)
    observed = [
        _observed_record(stage, path, index)
        for index, path in enumerate(expected_paths)
        if path in observed_by_path
    ]
    evaluation = evaluate_metadata(
        default_prim_path,
        articulation_root_paths,
        expected,
        observed,
        observed_dof_paths=observed_dof_paths,
    )
    ok = evaluation["ok"]
    return {
        "status": "PASS_A20_USD_DOF_METADATA" if ok else "FAIL_A20_USD_DOF_METADATA",
        "ok": ok,
        "inputs": {
            "config": _input_record(config_path),
            "mapping": _input_record(mapping_path),
            "stage": _input_record(stage_path),
        },
        "default_prim": default_prim_path,
        "articulation_root_paths": articulation_root_paths,
        "expected": expected,
        "observed": observed,
        "mismatches": evaluation["mismatches"],
        "errors": evaluation["errors"],
        **_SAFETY_FLAGS,
    }


def collect_usd_dof_metadata(config_path: Path | str) -> dict[str, Any]:
    """Collect Layer 1 metadata and convert all input failures to gate failures."""
    try:
        return _collect(Path(config_path))
    except Exception as exc:  # fail closed at the public/CLI boundary
        return {
            "status": "FAIL_A20_USD_DOF_METADATA",
            "ok": False,
            "inputs": {"config": {"path": str(Path(config_path).resolve())}},
            "default_prim": None,
            "articulation_root_paths": [],
            "expected": [],
            "observed": [],
            "mismatches": [],
            "errors": [{"code": "collection_error", "message": str(exc)}],
            **_SAFETY_FLAGS,
        }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = collect_usd_dof_metadata(args.config)
    output = args.json_output
    if output is None:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        output = Path(config["outputs"]["a20_usd_dof_metadata_json"])
    _atomic_write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS_A20_USD_DOF_METADATA" else 1


if __name__ == "__main__":
    raise SystemExit(main())
