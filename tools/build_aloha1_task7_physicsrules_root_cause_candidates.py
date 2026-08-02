#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Build isolated one-variable Task 7 PhysicsRules candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FROZEN_STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0"
    / "aloha1_cad_derived_full_body_collider_gripper_decomposition_"
    "tabletop_zero_z_up_meters_diagnostic.usda"
)
FROZEN_SHA256 = (
    "327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9"
)
OUTPUT_ROOT = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "task7_physicsrules_root_cause_candidates/1.0"
)
OUTPUT_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_task7_physicsrules_root_cause_candidates.json"
)
SOURCES = {
    "follower_left": {
        "path": (
            ROOT
            / "assets/Trossen/ALOHA1/1.0/diagnostics/"
            "cad_derived_task7_rule_candidates/1.0/Trossen/vx300s_left/1.0/"
            "vx300s_left.usda"
        ),
        "root": "/vx300s_left",
    },
    "follower_right": {
        "path": (
            ROOT
            / "assets/Trossen/ALOHA1/1.0/diagnostics/"
            "cad_derived_task7_rule_candidates/1.0/Trossen/vx300s_right/1.0/"
            "vx300s_right.usda"
        ),
        "root": "/vx300s_right",
    },
}
PROFILES = (
    "joint_state_zero",
    "virtual_helpers_without_rigid_body",
    "baseline_gripper_fixed_group_split",
)
JOINT_NAMES = (
    "elbow",
    "left_finger",
    "right_finger",
    "shoulder",
    "wrist_angle",
)
VIRTUAL_HELPER_SUFFIXES = (
    "ee_arm_link",
    "ee_gripper_link",
    "fingers_link",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(target: Path, owner: Path) -> str:
    return Path(os.path.relpath(target.resolve(), owner.resolve().parent)).as_posix()


def _layer_for_path(stage: Any, path: Path) -> Any:
    resolved = path.resolve()
    for layer in stage.GetLayerStack(includeSessionLayers=False):
        if layer.realPath and Path(layer.realPath).resolve() == resolved:
            return layer
    raise RuntimeError(f"layer not found in stack: {resolved}")


def _collision_descendants(stage: Any, prim_path: str) -> list[str]:
    from pxr import Usd
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.IsActive():
        return []
    return sorted(
        str(item.GetPath())
        for item in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
        if item.IsActive() and item.HasAPI(UsdPhysics.CollisionAPI)
    )


def _create_wrapper(
    *,
    profile: str,
    robot: str,
    source: Path,
    source_root: str,
    output_root: Path,
) -> tuple[Path, Path, Any]:
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom

    destination = output_root / profile / robot
    destination.mkdir(parents=True)
    wrapper = destination / f"{robot}_{profile}.usda"
    override = destination / f"{robot}_{profile}_override.usda"
    override_layer = Sdf.Layer.CreateNew(str(override))
    if override_layer is None:
        raise RuntimeError(f"cannot create override layer: {override}")
    override_layer.Save()
    stage = Usd.Stage.CreateNew(str(wrapper))
    root = UsdGeom.Xform.Define(stage, source_root).GetPrim()
    root.GetReferences().AddReference(_relative(source, wrapper), source_root)
    stage.GetRootLayer().subLayerPaths = [_relative(override, wrapper)]
    stage.SetDefaultPrim(root)
    stage.GetRootLayer().customLayerData = {
        "aloha1:scope": "DIAGNOSTIC_ONLY_NOT_FINAL",
        "aloha1:profile": profile,
        "aloha1:sourceCandidateSha256": _sha256(source),
        "aloha1:frozenStageSha256": FROZEN_SHA256,
    }
    stage.GetRootLayer().Save()
    composed = Usd.Stage.Open(str(wrapper), Usd.Stage.LoadAll)
    if composed is None:
        raise RuntimeError(f"cannot compose wrapper: {wrapper}")
    composed.SetEditTarget(_layer_for_path(composed, override))
    return wrapper, override, composed


def _apply_joint_state_zero(stage: Any, root_path: str, _robot: str) -> dict[str, Any]:
    from pxr import PhysxSchema
    from pxr import UsdPhysics

    records = []
    for name in JOINT_NAMES:
        path = f"{root_path}/joints/{name}"
        prim = stage.GetPrimAtPath(path)
        axis = "angular" if prim.IsA(UsdPhysics.RevoluteJoint) else "linear"
        api = PhysxSchema.JointStateAPI(prim, axis)
        before = float(api.GetPositionAttr().Get())
        api.GetPositionAttr().Set(0.0)
        records.append(
            {
                "prim_path": path,
                "axis": axis,
                "position_before": before,
                "position_after": float(api.GetPositionAttr().Get()),
            }
        )
    return {
        "changed_variable": "joint_state_position",
        "records": records,
        "joint_or_drive_target_modified": False,
        "body_transform_modified": False,
    }


def _apply_virtual_helper_removal(
    stage: Any,
    root_path: str,
    robot: str,
) -> dict[str, Any]:
    from pxr import UsdPhysics

    records = []
    for suffix in VIRTUAL_HELPER_SUFFIXES:
        path = f"{root_path}/{robot}_{suffix}"
        prim = stage.GetPrimAtPath(path)
        before = {
            "rigid_body": prim.HasAPI(UsdPhysics.RigidBodyAPI),
            "mass": prim.HasAPI(UsdPhysics.MassAPI),
        }
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        prim.RemoveAPI(UsdPhysics.MassAPI)
        records.append(
            {
                "prim_path": path,
                "before": before,
                "after": {
                    "rigid_body": prim.HasAPI(UsdPhysics.RigidBodyAPI),
                    "mass": prim.HasAPI(UsdPhysics.MassAPI),
                },
            }
        )
    return {
        "changed_variable": "virtual_helper_body_semantics",
        "records": records,
        "joint_targets_modified": False,
        "collider_modified": False,
    }


def _apply_gripper_group_split(
    stage: Any,
    root_path: str,
    robot: str,
) -> dict[str, Any]:
    gripper = f"{root_path}/{robot}_gripper_link"
    bar = f"{root_path}/{robot}_gripper_bar_link"
    source_gripper = f"{gripper}/collisions"
    source_bar = f"{bar}/collisions"
    cad_group = f"{gripper}/cad_derived_collisions/cad_derived_gripper_link"
    paths = {
        "source_gripper": source_gripper,
        "source_bar": source_bar,
        "cad_group": cad_group,
    }
    before = {
        name: {
            "active": stage.GetPrimAtPath(path).IsActive(),
            "collision_descendants": _collision_descendants(stage, path),
        }
        for name, path in paths.items()
    }
    stage.GetPrimAtPath(source_gripper).SetActive(True)  # noqa: FBT003
    stage.GetPrimAtPath(source_bar).SetActive(True)  # noqa: FBT003
    stage.GetPrimAtPath(cad_group).SetActive(False)  # noqa: FBT003
    after = {
        name: {
            "active": stage.GetPrimAtPath(path).IsActive(),
            "collision_descendants": _collision_descendants(stage, path),
        }
        for name, path in paths.items()
    }
    return {
        "changed_variable": "gripper_fixed_group_collider_representation",
        "paths": paths,
        "before": before,
        "after": after,
        "finger_colliders_modified": False,
        "collision_geometry_source": "PINNED_ALOHA_VX300S_URDF_MESHES",
    }


APPLIERS = {
    "joint_state_zero": _apply_joint_state_zero,
    "virtual_helpers_without_rigid_body": _apply_virtual_helper_removal,
    "baseline_gripper_fixed_group_split": _apply_gripper_group_split,
}


def _build_one(
    *,
    profile: str,
    robot: str,
    spec: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    from pxr import Usd

    source = Path(spec["path"]).resolve(strict=True)
    source_before = _sha256(source)
    wrapper, override, stage = _create_wrapper(
        profile=profile,
        robot=robot,
        source=source,
        source_root=str(spec["root"]),
        output_root=output_root,
    )
    change = APPLIERS[profile](stage, str(spec["root"]), robot)
    stage.GetEditTarget().GetLayer().Save()
    stage.GetRootLayer().Save()
    readback = Usd.Stage.Open(str(wrapper), Usd.Stage.LoadAll)
    if readback is None:
        raise RuntimeError(f"cannot reopen candidate: {wrapper}")
    source_after = _sha256(source)
    if source_before != source_after:
        raise RuntimeError(f"source candidate changed: {source}")
    return {
        "profile": profile,
        "follower": robot,
        "scope": "DIAGNOSTIC_ONLY_NOT_FINAL",
        "changed_variable_count": 1,
        "change": change,
        "wrapper": {
            "absolute_path": str(wrapper.resolve()),
            "sha256": _sha256(wrapper),
            "default_prim": str(readback.GetDefaultPrim().GetPath()),
            "sublayers": list(readback.GetRootLayer().subLayerPaths),
        },
        "override_layer": {
            "absolute_path": str(override.resolve()),
            "sha256": _sha256(override),
        },
        "source_candidate": {
            "absolute_path": str(source),
            "sha256_before": source_before,
            "sha256_after": source_after,
            "modified": False,
        },
    }


def build(output_root: Path, output_report: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"candidate output exists: {output_root}")
    frozen = FROZEN_STAGE.resolve(strict=True)
    frozen_before = _sha256(frozen)
    if frozen_before != FROZEN_SHA256:
        raise RuntimeError("frozen Stage hash mismatch")
    profiles = {
        profile: [
            _build_one(
                profile=profile,
                robot=robot,
                spec=dict(spec),
                output_root=output_root,
            )
            for robot, spec in SOURCES.items()
        ]
        for profile in PROFILES
    }
    frozen_after = _sha256(frozen)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "frozen_stage": {
            "absolute_path": str(frozen),
            "sha256_before": frozen_before,
            "sha256_after": frozen_after,
            "modified": frozen_before != frozen_after,
        },
        "profiles": profiles,
        "mimic_modified": False,
        "friction_mass_inertia_timestep_solver_modified": False,
        "final_or_default_asset_modified": False,
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
    report = build(args.output_root.resolve(), args.output_report.resolve())
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output_report.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "create_new_stage": False,
            "disable_viewport_updates": True,
        }
    )
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
