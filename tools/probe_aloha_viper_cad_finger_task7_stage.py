#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Read-only Isaac Sim 5.1 probe for Task 7 robot-scope decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_finger_task5_bottle/"
    "aloha_viperx_supplier_cad_bottle_task5.usda"
)
DEFAULT_OUTPUT = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "task7_robot_scope/stage_probe.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collider_paths(prim: Any) -> list[str]:
    from pxr import Usd
    from pxr import UsdPhysics

    return [
        str(candidate.GetPath())
        for candidate in Usd.PrimRange(
            prim,
            Usd.TraverseInstanceProxies(),
        )
        if candidate.HasAPI(UsdPhysics.CollisionAPI)
    ]


def _quaternion_record(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    imaginary = value.GetImaginary()
    return {
        "real": float(value.GetReal()),
        "imaginary": [float(component) for component in imaginary],
    }


def probe(stage_path: Path) -> dict[str, Any]:
    from pxr import PhysxSchema
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    stage_path = stage_path.resolve(strict=True)
    stage = Usd.Stage.Open(str(stage_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"unable to open Stage: {stage_path}")

    joint_body_targets: set[str] = set()
    joints = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = [str(path) for path in joint.GetBody0Rel().GetTargets()]
        body1 = [str(path) for path in joint.GetBody1Rel().GetTargets()]
        joint_body_targets.update(body0)
        joint_body_targets.update(body1)
        axis = None
        if prim.IsA(UsdPhysics.RevoluteJoint):
            axis = "angular"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            axis = "linear"
        state = PhysxSchema.JointStateAPI(prim, axis) if axis else None
        joints.append(
            {
                "path": str(prim.GetPath()),
                "body0": body0,
                "body1": body1,
                "axis": axis,
                "joint_axis": (
                    str(prim.GetAttribute("physics:axis").Get())
                    if prim.HasAttribute("physics:axis")
                    else None
                ),
                "local_pos0": list(joint.GetLocalPos0Attr().Get()),
                "local_pos1": list(joint.GetLocalPos1Attr().Get()),
                "local_rot0": _quaternion_record(
                    joint.GetLocalRot0Attr().Get()
                ),
                "local_rot1": _quaternion_record(
                    joint.GetLocalRot1Attr().Get()
                ),
                "applied_schemas": list(prim.GetAppliedSchemas()),
                "has_joint_state_api": bool(state),
            }
        )

    rigid_bodies = []
    xform_cache = UsdGeom.XformCache()
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rigid = UsdPhysics.RigidBodyAPI(prim)
        mass = UsdPhysics.MassAPI(prim)
        colliders = _collider_paths(prim)
        world = xform_cache.GetLocalToWorldTransform(prim)
        rigid_bodies.append(
            {
                "path": str(prim.GetPath()),
                "rigid_body_enabled": rigid.GetRigidBodyEnabledAttr().Get(),
                "joint_body_target": str(prim.GetPath()) in joint_body_targets,
                "collider_count": len(colliders),
                "collider_paths": colliders,
                "world_transform_row_major": [
                    float(world[row][column])
                    for row in range(4)
                    for column in range(4)
                ],
                "mass": mass.GetMassAttr().Get(),
                "center_of_mass": (
                    list(mass.GetCenterOfMassAttr().Get())
                    if mass.GetCenterOfMassAttr().Get() is not None
                    else None
                ),
                "diagonal_inertia": (
                    list(mass.GetDiagonalInertiaAttr().Get())
                    if mass.GetDiagonalInertiaAttr().Get() is not None
                    else None
                ),
                "principal_axes": _quaternion_record(
                    mass.GetPrincipalAxesAttr().Get()
                ),
                "applied_schemas": list(prim.GetAppliedSchemas()),
            }
        )

    default_prim = stage.GetDefaultPrim()
    return {
        "schema_version": 1,
        "status": "PASS",
        "read_only": True,
        "stage": {
            "absolute_path": str(stage_path),
            "sha256": _sha256(stage_path),
            "default_prim": (
                str(default_prim.GetPath()) if default_prim else None
            ),
        },
        "root_children": [
            str(child.GetPath())
            for child in default_prim.GetChildren()
        ],
        "joint_body_targets": sorted(joint_body_targets),
        "joints": joints,
        "rigid_bodies": rigid_bodies,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = probe(args.stage)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"status={report['status']}")
    print(f"output={output}")
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
