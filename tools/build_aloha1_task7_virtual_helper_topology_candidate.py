#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Build an isolated, frame-preserving collapse of empty fixed helper bodies."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SHA256 = "327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9"
SOURCE_ROOT = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "task7_physicsrules_root_cause_candidates/1.0/"
    "baseline_gripper_fixed_group_split"
)
OUTPUT_ROOT = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "task7_physicsrules_root_cause_candidates/1.0/"
    "virtual_helper_topology_collapse"
)
OUTPUT_REPORT = (
    ROOT
    / "reports/aloha1_mapping/aloha1_task7_virtual_helper_topology_candidate.json"
)
FOLLOWERS = ("follower_left", "follower_right")
HELPER_SUFFIXES = ("ee_arm_link", "fingers_link", "ee_gripper_link")
DISABLED_FIXED_JOINTS = ("ee_arm", "ee_bar", "ee_gripper")
REPARENT_BODY0 = {
    "gripper": "gripper_link",
    "gripper_bar": "gripper_link",
    "left_finger": "gripper_bar_link",
    "right_finger": "gripper_bar_link",
}


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


def _joint_local_frame(joint: Any, body_index: int) -> Any:
    from pxr import Gf

    position = (
        joint.GetLocalPos0Attr().Get()
        if body_index == 0
        else joint.GetLocalPos1Attr().Get()
    )
    rotation = (
        joint.GetLocalRot0Attr().Get()
        if body_index == 0
        else joint.GetLocalRot1Attr().Get()
    )
    matrix = Gf.Matrix4d()
    matrix.SetTranslate(Gf.Vec3d(position))
    matrix.SetRotateOnly(
        Gf.Quatd(rotation.GetReal(), *rotation.GetImaginary()).GetNormalized()
    )
    return matrix


def _matrix_residual(left: Any, right: Any) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(left, dtype=np.float64)
                - np.asarray(right, dtype=np.float64)
            )
        )
    )


def _reparent_body0(stage: Any, joint_path: str, new_body0_path: str) -> dict[str, Any]:
    from pxr import Gf
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(joint_path)
    joint = UsdPhysics.Joint(prim)
    old_targets = joint.GetBody0Rel().GetTargets()
    if len(old_targets) != 1:
        raise RuntimeError(f"joint must have one body0 target: {joint_path}: {old_targets}")
    old_body0_path = str(old_targets[0])
    old_body = stage.GetPrimAtPath(old_body0_path)
    new_body = stage.GetPrimAtPath(new_body0_path)
    if not old_body.IsValid() or not new_body.IsValid():
        raise RuntimeError(f"missing old/new body for {joint_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    old_world = cache.GetLocalToWorldTransform(old_body)
    new_world = cache.GetLocalToWorldTransform(new_body)
    old_local = _joint_local_frame(joint, 0)
    old_joint_world = old_local * old_world
    # NVIDIA Robot Schema masking operation convention (USD row vectors):
    # new_local = old_local * old_body_world * inverse(new_body_world).
    new_local = old_joint_world * new_world.GetInverse()
    transform = Gf.Transform(new_local)
    position = transform.GetTranslation()
    quaternion = transform.GetRotation().GetQuat().GetNormalized()
    joint.GetBody0Rel().SetTargets([Sdf.Path(new_body0_path)])
    joint.GetLocalPos0Attr().Set(Gf.Vec3f(position))
    joint.GetLocalRot0Attr().Set(
        Gf.Quatf(
            float(quaternion.GetReal()),
            Gf.Vec3f(quaternion.GetImaginary()),
        )
    )
    readback_joint_world = _joint_local_frame(joint, 0) * new_world
    return {
        "joint_path": joint_path,
        "old_body0_path": old_body0_path,
        "new_body0_path": new_body0_path,
        "new_local_pos0": [float(value) for value in joint.GetLocalPos0Attr().Get()],
        "new_local_rot0_wxyz": [
            float(joint.GetLocalRot0Attr().Get().GetReal()),
            *[float(value) for value in joint.GetLocalRot0Attr().Get().GetImaginary()],
        ],
        "joint_world_frame_residual": _matrix_residual(
            old_joint_world, readback_joint_world
        ),
    }


def _build_one(follower: str) -> dict[str, Any]:
    from pxr import PhysxSchema
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    root_name = "vx300s_left" if follower == "follower_left" else "vx300s_right"
    root_path = f"/{root_name}"
    source = SOURCE_ROOT / follower / f"{follower}_baseline_gripper_fixed_group_split.usda"
    source = source.resolve(strict=True)
    source_before = _sha256(source)
    destination = OUTPUT_ROOT / follower
    destination.mkdir(parents=True)
    wrapper = destination / f"{follower}_virtual_helper_topology_collapse.usda"
    override = destination / f"{follower}_virtual_helper_topology_collapse_override.usda"
    override_layer = Sdf.Layer.CreateNew(str(override))
    if override_layer is None:
        raise RuntimeError(f"cannot create override layer: {override}")
    override_layer.Save()
    wrapper_stage = Usd.Stage.CreateNew(str(wrapper))
    root = UsdGeom.Xform.Define(wrapper_stage, root_path).GetPrim()
    root.GetReferences().AddReference(_relative(source, wrapper), root_path)
    wrapper_stage.GetRootLayer().subLayerPaths = [_relative(override, wrapper)]
    wrapper_stage.SetDefaultPrim(root)
    wrapper_stage.GetRootLayer().customLayerData = {
        "aloha1:scope": "DIAGNOSTIC_ONLY_NOT_FINAL",
        "aloha1:profile": "virtual_helper_topology_collapse",
        "aloha1:inputProfile": "baseline_gripper_fixed_group_split",
        "aloha1:frozenStageSha256": FROZEN_SHA256,
    }
    wrapper_stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(wrapper), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"cannot open wrapper: {wrapper}")
    stage.SetEditTarget(_layer_for_path(stage, override))

    helper_records = []
    for suffix in HELPER_SUFFIXES:
        path = f"{root_path}/{follower}_{suffix}"
        prim = stage.GetPrimAtPath(path)
        before = list(prim.GetAppliedSchemas())
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        prim.RemoveAPI(UsdPhysics.MassAPI)
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
        helper_records.append(
            {
                "prim_path": path,
                "applied_schemas_before": before,
                "applied_schemas_after": list(prim.GetAppliedSchemas()),
                "has_rigid_body_after": prim.HasAPI(UsdPhysics.RigidBodyAPI),
                "has_mass_after": prim.HasAPI(UsdPhysics.MassAPI),
            }
        )

    disabled_records = []
    for name in DISABLED_FIXED_JOINTS:
        path = f"{root_path}/joints/{name}"
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(path))
        before = joint.GetJointEnabledAttr().Get()
        joint.GetJointEnabledAttr().Set(False)  # noqa: FBT003
        disabled_records.append(
            {
                "joint_path": path,
                "enabled_before": bool(before),
                "enabled_after": bool(joint.GetJointEnabledAttr().Get()),
            }
        )

    reparented = []
    for joint_name, new_suffix in REPARENT_BODY0.items():
        reparented.append(
            _reparent_body0(
                stage,
                f"{root_path}/joints/{joint_name}",
                f"{root_path}/{follower}_{new_suffix}",
            )
        )
    stage.GetEditTarget().GetLayer().Save()
    stage.GetRootLayer().Save()
    readback = Usd.Stage.Open(str(wrapper), Usd.Stage.LoadAll)
    if readback is None:
        raise RuntimeError(f"cannot reopen candidate: {wrapper}")
    source_after = _sha256(source)
    residual = max(item["joint_world_frame_residual"] for item in reparented)
    return {
        "follower": follower,
        "root_prim": root_path,
        "source": {"absolute_path": str(source), "sha256": source_before},
        "source_modified": source_before != source_after,
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
        "helper_body_count": len(helper_records),
        "disabled_fixed_joint_count": len(disabled_records),
        "reparented_joint_count": len(reparented),
        "maximum_joint_world_frame_residual": residual,
        "helpers": helper_records,
        "disabled_fixed_joints": disabled_records,
        "reparented_joints": reparented,
    }


def main() -> int:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"candidate output exists: {OUTPUT_ROOT}")
    candidates = [_build_one(follower) for follower in FOLLOWERS]
    report = {
        "schema_version": 1,
        "geometry_topology_status": (
            "PASS"
            if all(
                not item["source_modified"]
                and item["maximum_joint_world_frame_residual"] < 1.0e-9
                for item in candidates
            )
            else "FAIL"
        ),
        # Removing the fixed helper rigid bodies also removes their authored
        # MassAPI opinions.  Frame preservation alone is therefore not enough
        # to call the candidate physically equivalent or promotable.
        "status": "PARTIAL",
        "scope": "DIAGNOSTIC_ONLY_NOT_FINAL",
        "input_profile": "baseline_gripper_fixed_group_split",
        "frozen_stage_sha256": FROZEN_SHA256,
        "official_formula_source": (
            "DIRECT_NVIDIA_ISAAC_MCP_ROBOT_SCHEMA_MASKING_OPS_REPARENT_BODY0"
        ),
        "local_runtime_formula_probe_required": True,
        "candidates": candidates,
        "joint_state_modified": False,
        "mimic_modified": False,
        "mass_semantics_modified": True,
        "physics_equivalence": "PARTIAL_HELPER_MASS_NOT_CONSERVED",
        "other_physics_parameters_modified": False,
        "promotion_status": "BLOCKED_HELPER_MASS_INERTIA_SEMANTICS",
        "final_or_default_asset_modified": False,
        "task8": "NOT_RUN",
    }
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "geometry_topology_status": report["geometry_topology_status"],
                "candidates": len(candidates),
            }
        )
    )
    return 0 if report["geometry_topology_status"] == "PASS" else 2


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "create_new_stage": False})
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
