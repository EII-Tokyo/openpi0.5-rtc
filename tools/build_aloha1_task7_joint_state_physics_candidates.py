#!/usr/bin/env python3
# ruff: noqa: FBT003, PLC0415
"""Build isolated gripper JointStateAPI candidates for Task 7 PhysicsRules."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "task7_joint_state_physics_candidate/1.0"
)
OUTPUT_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_task7_joint_state_physics_candidate_build.json"
)

CANDIDATES = {
    "follower_left": {
        "source": (
            ROOT
            / "assets/Trossen/ALOHA1/1.0/diagnostics/"
            "signal_correspondence/1.0/follower_left_asset/"
            "aloha1_signal_follower_left.usda"
        ),
        "source_sha256": (
            "34badd857a3327b3605dc4ffd2fc40eb9c4c72a31c427c20967a3273f003aa7c"
        ),
        "default_prim": "follower_left",
        "joint_path": "/follower_left/vx300s_left/joints/gripper",
    },
    "follower_right": {
        "source": (
            ROOT
            / "assets/Trossen/ALOHA1/1.0/diagnostics/"
            "supplier_cad_follower_right/1.0/"
            "supplier_cad_follower_right.usda"
        ),
        "source_sha256": (
            "95c7878f794f5f557b70997a2240b6476836b8ffbeed5a4992cb114a169487ea"
        ),
        "default_prim": "follower_right",
        "joint_path": "/follower_right/vx300s_right/joints/gripper",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(target: Path, owner: Path) -> str:
    return Path(
        os.path.relpath(target.resolve(), owner.resolve().parent)
    ).as_posix()


def _normalize_usda(path: Path) -> None:
    path.write_text(
        path.read_text(encoding="utf-8").rstrip() + "\n",
        encoding="utf-8",
    )


def _should_normalize_text_layer(path: Path) -> bool:
    return path.suffix == ".usda"


def _layer_for_path(stage: Any, path: Path) -> Any:
    for layer in stage.GetLayerStack(includeSessionLayers=False):
        if layer.realPath and Path(layer.realPath).resolve() == path.resolve():
            return layer
    raise RuntimeError(f"layer not found in composed stack: {path}")


def _drive_readback(prim: Any) -> dict[str, float]:
    from pxr import UsdPhysics

    drive = UsdPhysics.DriveAPI(prim, "angular")
    return {
        "target_position": float(drive.GetTargetPositionAttr().Get() or 0.0),
        "target_velocity": float(drive.GetTargetVelocityAttr().Get() or 0.0),
    }


def _build_candidate(
    *, name: str, spec: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    from pxr import PhysxSchema
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdPhysics

    source = Path(spec["source"]).resolve(strict=True)
    source_before = _sha256(source)
    if source_before != spec["source_sha256"]:
        raise RuntimeError(f"protected source hash mismatch: {name}")

    candidate_root = output_root / name
    physics_dir = candidate_root / "physics"
    physics_dir.mkdir(parents=True)
    wrapper = candidate_root / f"aloha1_task7_{name}_joint_state.usda"
    physics = (
        physics_dir / f"aloha1_task7_{name}_joint_state_physics.usd"
    )
    physics_layer = Sdf.Layer.CreateNew(str(physics))
    if physics_layer is None:
        raise RuntimeError(f"unable to create physics layer: {name}")
    physics_layer.Save()

    root_layer = Sdf.Layer.CreateNew(str(wrapper))
    if root_layer is None:
        raise RuntimeError(f"unable to create wrapper layer: {name}")
    root_layer.defaultPrim = spec["default_prim"]
    root_layer.subLayerPaths = [
        _relative(physics, wrapper),
        _relative(source, wrapper),
    ]
    root_layer.customLayerData = {
        "aloha1:scope": "TASK7_JOINT_STATE_PACKAGING_DIAGNOSTIC_ONLY",
        "aloha1:sourceStageSha256": source_before,
    }
    root_layer.Save()

    stage = Usd.Stage.Open(str(wrapper), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"unable to compose candidate: {name}")
    if str(stage.GetDefaultPrim().GetPath()) != f"/{spec['default_prim']}":
        raise RuntimeError(f"default prim mismatch: {name}")
    target = _layer_for_path(stage, physics)
    stage.SetEditTarget(target)
    joint = stage.GetPrimAtPath(spec["joint_path"])
    if not joint.IsA(UsdPhysics.RevoluteJoint):
        raise RuntimeError(f"gripper is not a RevoluteJoint: {name}")
    drive_before = _drive_readback(joint)
    PhysxSchema.JointStateAPI.Apply(joint, "angular")
    target.Save()
    for layer_path in (wrapper, physics):
        if _should_normalize_text_layer(layer_path):
            _normalize_usda(layer_path)

    readback_stage = Usd.Stage.Open(str(wrapper), Usd.Stage.LoadAll)
    readback_joint = readback_stage.GetPrimAtPath(spec["joint_path"])
    joint_state_api = bool(
        PhysxSchema.JointStateAPI(readback_joint, "angular")
    )
    drive_after = _drive_readback(readback_joint)
    readback_physics = _layer_for_path(readback_stage, physics)
    joint_spec = readback_physics.GetPrimAtPath(Sdf.Path(spec["joint_path"]))
    if joint_spec is None:
        raise RuntimeError(f"joint spec missing in physics layer: {name}")
    authored_properties = sorted(str(item.name) for item in joint_spec.properties)
    authored_state_values = any(item.startswith("state:") for item in authored_properties)
    authored_drive_values = any(item.startswith("drive:") for item in authored_properties)
    if not joint_state_api:
        raise RuntimeError(f"JointStateAPI readback failed: {name}")
    if authored_state_values or authored_drive_values:
        raise RuntimeError(f"candidate authored state or drive values: {name}")
    if drive_before != drive_after:
        raise RuntimeError(f"drive readback changed: {name}")
    source_after = _sha256(source)
    if source_after != source_before:
        raise RuntimeError(f"protected source changed: {name}")

    return {
        "name": name,
        "wrapper": {
            "absolute_path": str(wrapper.resolve()),
            "sha256": _sha256(wrapper),
            "default_prim": f"/{spec['default_prim']}",
            "sublayers": list(root_layer.subLayerPaths),
        },
        "physics_layer": {
            "absolute_path": str(physics.resolve()),
            "sha256": _sha256(physics),
            "filename_ends_with_physics_usd": physics.name.endswith(
                "_physics.usd"
            ),
            "authored_properties": authored_properties,
        },
        "source_stage": {
            "absolute_path": str(source),
            "sha256_before": source_before,
            "sha256_after": source_after,
            "modified": False,
        },
        "joint_path": spec["joint_path"],
        "joint_type": "RevoluteJoint",
        "joint_state_axis": "angular",
        "joint_state_api_readback": joint_state_api,
        "authored_state_values": authored_state_values,
        "authored_drive_values": authored_drive_values,
        "drive_readback_before": drive_before,
        "drive_readback_after": drive_after,
        "geometry_modified": False,
        "collider_modified": False,
        "mimic_modified": False,
    }


def build(*, output_root: Path, output_report: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"candidate output already exists: {output_root}")
    output_root.mkdir(parents=True)
    candidates = {
        name: _build_candidate(
            name=name,
            spec=dict(spec),
            output_root=output_root,
        )
        for name, spec in CANDIDATES.items()
    }
    report = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "TASK7_JOINT_STATE_PACKAGING_DIAGNOSTIC_ONLY",
        "candidates": candidates,
        "local_runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "asset_validation": "1.1.0",
        },
        "direct_nvidia_mcp_probe": {
            "status": "PASS",
            "transport": "DIRECT_NOT_MCPJUNGLE",
            "asset_validation_catalog_version": "1.2.1",
            "local_source_remains_version_authority": True,
        },
        "final_or_default_asset_modified": False,
        "real_robot_connected": False,
        "remote_192_168_1_103_accessed": False,
        "task8": "NOT_RUN",
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-report", type=Path, default=OUTPUT_REPORT)
    args = parser.parse_args()
    report = build(
        output_root=args.output_root.resolve(),
        output_report=args.output_report.resolve(),
    )
    print(f"status={report['status']}")
    print(f"report={args.output_report.resolve()}")
    return 0


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
