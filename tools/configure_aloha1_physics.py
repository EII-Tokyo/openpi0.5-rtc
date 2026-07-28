#!/usr/bin/env python3
"""Create ALOHA 1 debug/force physics-profile USD wrappers in Isaac Sim 5.1."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml

from tools.aloha1_mapping.physics_config import build_missing_dynamics_report
from tools.aloha1_mapping.physics_config import build_physics_plan


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_profiles(
    plan: dict[str, Any],
    *,
    report_path: Path,
) -> dict[str, Any]:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from pxr import PhysxSchema
        from pxr import Sdf
        from pxr import Usd
        from pxr import UsdGeom
        from pxr import UsdPhysics
        from pxr import UsdShade

        outputs = []
        for robot in plan["robots"]:
            base_usd = Path(robot["base_usd"])
            profile_dir = Path(robot["profile_dir"])
            profile_dir.mkdir(parents=True, exist_ok=True)
            for profile_name, profile in plan["profiles"].items():
                output = profile_dir / f"{robot['name']}_{profile_name}.usd"
                if output.exists():
                    layer = Sdf.Layer.FindOrOpen(str(output))
                    if layer is None:
                        raise RuntimeError(
                            f"unable to reopen generated profile: {output}"
                        )
                    layer.Clear()
                    stage = Usd.Stage.Open(layer)
                else:
                    stage = Usd.Stage.CreateNew(str(output))
                if stage is None:
                    raise RuntimeError(
                        f"unable to create generated profile: {output}"
                    )
                root = UsdGeom.Xform.Define(
                    stage, Sdf.Path(f"/{robot['name']}")
                ).GetPrim()
                stage.SetDefaultPrim(root)
                relative_base = os.path.relpath(base_usd, output.parent)
                if not root.GetReferences().AddReference(
                    relative_base, f"/{robot['name']}"
                ):
                    raise RuntimeError(
                        f"failed to reference {base_usd} from {output}"
                    )
                authored_dofs = []
                for dof in robot["dofs"]:
                    prim = stage.GetPrimAtPath(
                        f"/{robot['name']}/joints/{dof['name']}"
                    )
                    if not prim:
                        raise RuntimeError(
                            f"joint prim missing in profile composition: "
                            f"{robot['name']} {dof['name']}"
                        )
                    axis = (
                        "linear"
                        if dof["joint_type"] == "prismatic"
                        else "angular"
                    )
                    home_authored = (
                        dof["home_si"]
                        if axis == "linear"
                        else math.degrees(dof["home_si"])
                    )
                    if prim.HasAPI(PhysxSchema.JointStateAPI, axis):
                        state_api = PhysxSchema.JointStateAPI.Get(prim, axis)
                    else:
                        state_api = PhysxSchema.JointStateAPI.Apply(prim, axis)
                    if not state_api:
                        raise RuntimeError(
                            f"unable to apply JointStateAPI: {prim.GetPath()}"
                        )
                    if state_api.GetPositionAttr():
                        state_api.GetPositionAttr().Set(home_authored)
                    else:
                        state_api.CreatePositionAttr(home_authored)
                    if state_api.GetVelocityAttr():
                        state_api.GetVelocityAttr().Set(0.0)
                    else:
                        state_api.CreateVelocityAttr(0.0)
                    physx_joint = PhysxSchema.PhysxJointAPI(prim)
                    velocity_authored = (
                        dof["velocity_limit_si"]
                        if axis == "linear"
                        else math.degrees(dof["velocity_limit_si"])
                    )
                    physx_joint.GetMaxJointVelocityAttr().Set(
                        velocity_authored
                    )
                    drive_values = None
                    if dof["author_drive"]:
                        drive = UsdPhysics.DriveAPI(prim, axis)
                        stiffness = drive.GetStiffnessAttr().Get()
                        damping = drive.GetDampingAttr().Get()
                        drive.GetTypeAttr().Set(profile["drive_type"])
                        drive.GetTargetPositionAttr().Set(home_authored)
                        drive.GetTargetVelocityAttr().Set(0.0)
                        drive.GetMaxForceAttr().Set(dof["max_force"])
                        drive.GetStiffnessAttr().Set(stiffness)
                        drive.GetDampingAttr().Set(damping)
                        drive_values = {
                            "type": profile["drive_type"],
                            "stiffness_authored": stiffness,
                            "damping_authored": damping,
                            "max_force": dof["max_force"],
                            "max_velocity_authored": velocity_authored,
                            "target_position_authored": home_authored,
                        }
                    authored_dofs.append(
                        {
                            "name": dof["name"],
                            "axis": axis,
                            "mimic": dof["mimic"],
                            "drive": drive_values,
                            "initial_position_authored": home_authored,
                        }
                    )

                material = UsdShade.Material.Define(
                    stage,
                    f"/{robot['name']}/PhysicsMaterials/"
                    "temporary_fingertip",
                )
                material_api = UsdPhysics.MaterialAPI.Apply(
                    material.GetPrim()
                )
                material_api.CreateStaticFrictionAttr().Set(
                    plan["fingertip_material"]["static_friction"]
                )
                material_api.CreateDynamicFrictionAttr().Set(
                    plan["fingertip_material"]["dynamic_friction"]
                )
                material_api.CreateRestitutionAttr().Set(
                    plan["fingertip_material"]["restitution"]
                )
                bound_paths = []
                for side in ("left", "right"):
                    collision_path = (
                        f"/{robot['name']}/{robot['name']}_{side}_finger_link/"
                        "collisions"
                    )
                    collision_prim = stage.GetPrimAtPath(collision_path)
                    if not collision_prim:
                        raise RuntimeError(
                            f"finger collision prim missing: {collision_path}"
                        )
                    binding = UsdShade.MaterialBindingAPI.Apply(
                        collision_prim
                    )
                    binding.Bind(
                        material,
                        UsdShade.Tokens.weakerThanDescendants,
                        "physics",
                    )
                    bound_paths.append(collision_path)
                stage.GetRootLayer().Save()
                outputs.append(
                    {
                        "robot": robot["name"],
                        "profile": profile_name,
                        "status": profile["status"],
                        "usd": str(output.resolve()),
                        "dofs": authored_dofs,
                        "fingertip_material_bound_to": bound_paths,
                    }
                )
        report = {
            "schema_version": 1,
            "status": "PARTIAL",
            "default_profile": plan["default_profile"],
            "outputs": outputs,
            "hard_blockers": [
                plan["profiles"]["sim2real_force_drive"]["hard_blocker"],
                plan["fingertip_material"]["hard_blocker"],
            ],
        }
        _write_json(report_path, report)
    except Exception as error:
        _write_json(
            report_path,
            {
                "schema_version": 1,
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        app.close()
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    root = arguments.project_root.resolve(strict=True)
    plan = build_physics_plan(root)
    config_path = root / "configs/aloha1_physics_profiles.yaml"
    config_path.write_text(
        yaml.safe_dump(plan, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    _write_json(
        root / "reports/aloha1_mapping/missing_dynamics.json",
        build_missing_dynamics_report(root),
    )
    write_profiles(
        plan,
        report_path=root / "reports/aloha1_mapping/physics_profiles.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
