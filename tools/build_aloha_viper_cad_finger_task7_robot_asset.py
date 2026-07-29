#!/usr/bin/env python3
# ruff: noqa: FBT003, PLC0415
"""Build an isolated robot-scoped Task 7 asset in Isaac Sim 5.1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_finger_task5_bottle/"
    "aloha_viperx_supplier_cad_bottle_task5.usda"
)
APPROVED_SOURCE = (
    ROOT / "local_eval_assets/aloha_isaac_assets/aloha_viperx.usd"
)
IMPORTED_FOLLOWER_ASSET = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/follower_vx300s/"
    "follower_left/follower_left.usd"
)
OUTPUT_ROOT = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "supplier_cad_follower_left/1.3"
)
OUTPUT_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task7_robot_asset_v1_3.json"
)
EXPECTED_APPROVED_SOURCE_HASH = (
    "b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e"
)
EXPECTED_TASK5_STAGE_HASH = (
    "62697e4b25a7ec82234cc9ebd79d4a6d530a6ead0165519cbd275c0fa3f32178"
)
EXPECTED_IMPORTED_FOLLOWER_HASH = (
    "a1e6beb663f70e4c1d6493f5be80da2894e075c931e0fa5211e8c91d59a8205d"
)
ROBOT_NAME = "supplier_cad_follower_left"
ROBOT_PATH = f"/{ROBOT_NAME}"
SOURCE_ROBOT_PREFIX = "/workcell/vx300s_left/"
ROBOT_PREFIX = f"{ROBOT_PATH}/vx300s_left/"
JOINT_PREFIX = f"{ROBOT_PATH}/joints/"

DEACTIVATED_ROOT_CHILDREN = (
    "table",
    "midair",
    "table_frame_T",
    "placeholder_pipe",
    "worldBody",
)
DEACTIVATED_ROBOT_HELPERS = (
    f"{ROBOT_PREFIX}vx300s_left_camera_focus",
    f"{JOINT_PREFIX}vx300s_left_camera_focus",
    f"{JOINT_PREFIX}rootJoint_table",
    f"{JOINT_PREFIX}rootJoint_midair",
    f"{JOINT_PREFIX}rootJoint_table_frame_T",
    f"{JOINT_PREFIX}rootJoint_placeholder_pipe",
)
LINK_NAMES = (
    "vx300s_left",
    "vx300s_left_shoulder_link",
    "vx300s_left_upper_arm_link",
    "vx300s_left_upper_forearm_link",
    "vx300s_left_lower_forearm_link",
    "vx300s_left_wrist_link",
    "vx300s_left_gripper_link",
    "vx300s_left_gripper_prop_link",
    "vx300s_left_left_finger_link",
    "vx300s_left_right_finger_link",
)
JOINT_NAMES = (
    "rootJoint_vx300s_left",
    "vx300s_left_waist",
    "vx300s_left_shoulder",
    "vx300s_left_elbow",
    "vx300s_left_forearm_roll",
    "vx300s_left_wrist_angle",
    "vx300s_left_wrist_rotate",
    "vx300s_left_gripper_prop_link",
    "vx300s_left_left_finger",
    "vx300s_left_right_finger",
)
DOF_NAMES = (
    "vx300s_left_waist",
    "vx300s_left_shoulder",
    "vx300s_left_elbow",
    "vx300s_left_forearm_roll",
    "vx300s_left_wrist_angle",
    "vx300s_left_wrist_rotate",
    "vx300s_left_left_finger",
    "vx300s_left_right_finger",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_asset_path(target: Path, owner: Path) -> str:
    return Path(
        Path(target).resolve().relative_to(ROOT.resolve())
    ).as_posix() if owner.resolve().parent == ROOT.resolve() else (
        Path(
            __import__("os").path.relpath(
                target.resolve(),
                owner.resolve().parent,
            )
        ).as_posix()
    )


def _require_hash(path: Path, expected: str) -> str:
    actual = _sha256(path.resolve(strict=True))
    if actual != expected:
        raise RuntimeError(
            f"protected hash mismatch: {path}: {actual} != {expected}"
        )
    return actual


def _create_layers(
    *,
    output_root: Path,
) -> tuple[Path, Path, Path]:
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom

    if output_root.exists():
        raise FileExistsError(
            f"diagnostic output already exists: {output_root}"
        )
    configuration_dir = output_root / "configuration"
    physics_dir = output_root / "physics"
    configuration_dir.mkdir(parents=True)
    physics_dir.mkdir()

    wrapper_path = output_root / f"{ROBOT_NAME}.usda"
    configuration_path = (
        configuration_dir / f"{ROBOT_NAME}_configuration.usda"
    )
    physics_path = physics_dir / f"{ROBOT_NAME}_physics.usd"

    configuration_layer = Sdf.Layer.CreateNew(str(configuration_path))
    physics_layer = Sdf.Layer.CreateNew(str(physics_path))
    if configuration_layer is None or physics_layer is None:
        raise RuntimeError("unable to create Task 7 diagnostic layers")
    configuration_layer.Save()
    physics_layer.Save()

    stage = Usd.Stage.CreateNew(str(wrapper_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, ROBOT_PATH).GetPrim()
    if not root.GetReferences().AddReference(
        _relative_asset_path(SOURCE_STAGE, wrapper_path),
        Sdf.Path("/workcell"),
    ):
        raise RuntimeError("unable to reference Task 5 diagnostic workcell")
    stage.GetRootLayer().subLayerPaths = [
        _relative_asset_path(physics_path, wrapper_path),
        _relative_asset_path(configuration_path, wrapper_path),
    ]
    stage.SetDefaultPrim(root)
    stage.GetRootLayer().Save()
    return wrapper_path, configuration_path, physics_path


def _layer_for_path(stage: Any, path: Path) -> Any:
    for layer in stage.GetLayerStack(includeSessionLayers=False):
        if layer.realPath and Path(layer.realPath).resolve() == path.resolve():
            return layer
    raise RuntimeError(f"layer not found in stack: {path}")


def _author_configuration(
    *,
    wrapper_path: Path,
    configuration_path: Path,
) -> dict[str, Any]:
    from pxr import Sdf
    from pxr import Usd
    from usd.schema.isaac import robot_schema

    stage = Usd.Stage.Open(str(wrapper_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"unable to compose wrapper: {wrapper_path}")
    target = _layer_for_path(stage, configuration_path)
    stage.SetEditTarget(target)

    deactivated = []
    for child in DEACTIVATED_ROOT_CHILDREN:
        path = f"{ROBOT_PATH}/{child}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"expected workcell helper missing: {path}")
        stage.OverridePrim(path).SetActive(False)
        deactivated.append(path)
    for path in DEACTIVATED_ROBOT_HELPERS:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"expected robot helper missing: {path}")
        stage.OverridePrim(path).SetActive(False)
        deactivated.append(path)

    root = stage.GetPrimAtPath(ROBOT_PATH)
    robot_schema.ApplyRobotAPI(root)

    link_paths = [f"{ROBOT_PREFIX}{name}" for name in LINK_NAMES]
    joint_paths = [f"{JOINT_PREFIX}{name}" for name in JOINT_NAMES]
    for path in link_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"robot link missing: {path}")
        robot_schema.ApplyLinkAPI(prim)
    for path in joint_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"robot joint missing: {path}")
        robot_schema.ApplyJointAPI(prim)

    relationships = {}
    for relation, paths in (
        (robot_schema.Relations.ROBOT_LINKS, link_paths),
        (robot_schema.Relations.ROBOT_JOINTS, joint_paths),
    ):
        rel = root.GetRelationship(relation.name)
        rel.ClearTargets(True)
        for path in paths:
            rel.AddTarget(
                Sdf.Path(path),
                Usd.ListPositionBackOfPrependList,
            )
        relationships[relation.name] = [
            str(path) for path in rel.GetTargets()
        ]
    target.Save()
    return {
        "deactivated": deactivated,
        "link_paths": link_paths,
        "joint_paths": joint_paths,
        "relationships": relationships,
    }


def _author_joint_states(
    *,
    wrapper_path: Path,
    physics_path: Path,
) -> list[dict[str, Any]]:
    from pxr import PhysxSchema
    from pxr import Usd
    from pxr import UsdPhysics

    stage = Usd.Stage.Open(str(wrapper_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"unable to compose wrapper: {wrapper_path}")
    target = _layer_for_path(stage, physics_path)
    stage.SetEditTarget(target)

    records = []
    for name in DOF_NAMES:
        path = f"{JOINT_PREFIX}{name}"
        prim = stage.GetPrimAtPath(path)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            axis = "angular"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            axis = "linear"
        else:
            raise RuntimeError(f"unsupported Task 7 joint type: {path}")
        drive = UsdPhysics.DriveAPI(prim, axis)
        target_position = drive.GetTargetPositionAttr().Get()
        target_velocity = drive.GetTargetVelocityAttr().Get()
        position = float(target_position or 0.0)
        velocity = float(target_velocity or 0.0)
        if abs(position) > 1.0e-12 or abs(velocity) > 1.0e-12:
            raise RuntimeError(
                "JointState schema fallback is zero but drive target is "
                f"nonzero: {path}: position={position}, velocity={velocity}"
            )
        PhysxSchema.JointStateAPI.Apply(prim, axis)
        records.append(
            {
                "path": path,
                "axis": axis,
                "position": position,
                "velocity": velocity,
                "authored_value_policy": (
                    "SCHEMA_FALLBACK_ZERO_MATCHES_DRIVE_TARGET"
                ),
            }
        )
    target.Save()
    return records


def _author_base_evidence(
    *,
    wrapper_path: Path,
    physics_path: Path,
) -> dict[str, Any]:
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    imported = Usd.Stage.Open(
        str(IMPORTED_FOLLOWER_ASSET),
        Usd.Stage.LoadAll,
    )
    if imported is None:
        raise RuntimeError("unable to open imported follower asset")
    source_base_path = "/follower_left/follower_left_base_link"
    source_base = imported.GetPrimAtPath(source_base_path)
    source_mass = UsdPhysics.MassAPI(source_base)
    source_collisions = f"{source_base_path}/collisions"
    if not imported.GetPrimAtPath(source_collisions).IsValid():
        raise RuntimeError("imported follower base collisions missing")

    stage = Usd.Stage.Open(str(wrapper_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"unable to compose wrapper: {wrapper_path}")
    target = _layer_for_path(stage, physics_path)
    stage.SetEditTarget(target)
    destination_base_path = f"{ROBOT_PREFIX}vx300s_left"
    destination_base = stage.GetPrimAtPath(destination_base_path)
    destination_mass = UsdPhysics.MassAPI.Apply(destination_base)
    destination_mass.CreateMassAttr().Set(
        source_mass.GetMassAttr().Get()
    )
    destination_mass.CreateCenterOfMassAttr().Set(
        source_mass.GetCenterOfMassAttr().Get()
    )
    destination_mass.CreateDiagonalInertiaAttr().Set(
        source_mass.GetDiagonalInertiaAttr().Get()
    )
    destination_mass.CreatePrincipalAxesAttr().Set(
        source_mass.GetPrincipalAxesAttr().Get()
    )

    destination_collisions = (
        f"{destination_base_path}/collisions"
    )
    collision_root = UsdGeom.Xform.Define(
        stage,
        destination_collisions,
    ).GetPrim()
    if not collision_root.GetReferences().AddReference(
        _relative_asset_path(IMPORTED_FOLLOWER_ASSET, physics_path),
        Sdf.Path(source_collisions),
    ):
        raise RuntimeError("unable to reference imported base collider")
    target.Save()
    return {
        "status": "PASS",
        "source_asset": str(IMPORTED_FOLLOWER_ASSET.resolve()),
        "source_asset_sha256": _sha256(IMPORTED_FOLLOWER_ASSET),
        "source_prim": source_base_path,
        "destination_prim": destination_base_path,
        "collision_source": source_collisions,
        "collision_destination": destination_collisions,
        "mass": float(source_mass.GetMassAttr().Get()),
        "center_of_mass": [
            float(value)
            for value in source_mass.GetCenterOfMassAttr().Get()
        ],
        "diagonal_inertia": [
            float(value)
            for value in source_mass.GetDiagonalInertiaAttr().Get()
        ],
        "principal_axes": {
            "real": float(
                source_mass.GetPrincipalAxesAttr().Get().GetReal()
            ),
            "imaginary": [
                float(value)
                for value in (
                    source_mass.GetPrincipalAxesAttr()
                    .Get()
                    .GetImaginary()
                )
            ],
        },
        "frame_evidence": {
            "base_frame_rotation": "IDENTITY_IN_BOTH_ASSETS",
            "waist_local_position_m": [0.0, 0.0, 0.07900000363588333],
            "approved_joint_axis_expression": (
                "X_WITH_MINUS_90_DEGREE_Y_JOINT_FRAME"
            ),
            "imported_joint_axis_expression": "Z_WITH_IDENTITY_JOINT_FRAME",
            "classification": "EQUIVALENT_LOCAL_BASE_FRAME",
        },
    }


def _readback(
    *,
    wrapper_path: Path,
    configuration: dict[str, Any],
    joint_states: list[dict[str, Any]],
) -> dict[str, Any]:
    from pxr import PhysxSchema
    from pxr import Usd
    from pxr import UsdPhysics
    from usd.schema.isaac import robot_schema

    stage = Usd.Stage.Open(str(wrapper_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"unable to reopen wrapper: {wrapper_path}")
    root = stage.GetDefaultPrim()
    if str(root.GetPath()) != ROBOT_PATH:
        raise RuntimeError("unexpected robot diagnostic default prim")
    if not root.HasAPI(robot_schema.Classes.ROBOT_API.value):
        raise RuntimeError("RobotAPI readback failed")

    state_readback = []
    for record in joint_states:
        prim = stage.GetPrimAtPath(record["path"])
        state = PhysxSchema.JointStateAPI(prim, record["axis"])
        state_readback.append(
            {
                **record,
                "applied": bool(state),
                "position_readback": state.GetPositionAttr().Get(),
                "velocity_readback": state.GetVelocityAttr().Get(),
            }
        )
    destination_base = stage.GetPrimAtPath(
        f"{ROBOT_PREFIX}vx300s_left"
    )
    base_mass = UsdPhysics.MassAPI(destination_base)
    base_collider_count = sum(
        int(candidate.HasAPI(UsdPhysics.CollisionAPI))
        for candidate in Usd.PrimRange(
            destination_base,
            Usd.TraverseInstanceProxies(),
        )
    )
    return {
        "default_prim": str(root.GetPath()),
        "robot_api": True,
        "robot_links": [
            str(path)
            for path in root.GetRelationship(
                robot_schema.Relations.ROBOT_LINKS.name
            ).GetTargets()
        ],
        "robot_joints": [
            str(path)
            for path in root.GetRelationship(
                robot_schema.Relations.ROBOT_JOINTS.name
            ).GetTargets()
        ],
        "joint_states": state_readback,
        "base": {
            "mass": base_mass.GetMassAttr().Get(),
            "diagonal_inertia": list(
                base_mass.GetDiagonalInertiaAttr().Get()
            ),
            "collider_count": base_collider_count,
        },
        "deactivated": configuration["deactivated"],
    }


def build(
    *,
    output_root: Path,
    output_report: Path,
) -> dict[str, Any]:
    source_hash = _require_hash(
        APPROVED_SOURCE,
        EXPECTED_APPROVED_SOURCE_HASH,
    )
    task5_hash = _require_hash(
        SOURCE_STAGE,
        EXPECTED_TASK5_STAGE_HASH,
    )
    imported_follower_hash = _require_hash(
        IMPORTED_FOLLOWER_ASSET,
        EXPECTED_IMPORTED_FOLLOWER_HASH,
    )
    wrapper_path, configuration_path, physics_path = _create_layers(
        output_root=output_root,
    )
    configuration = _author_configuration(
        wrapper_path=wrapper_path,
        configuration_path=configuration_path,
    )
    joint_states = _author_joint_states(
        wrapper_path=wrapper_path,
        physics_path=physics_path,
    )
    base_evidence = _author_base_evidence(
        wrapper_path=wrapper_path,
        physics_path=physics_path,
    )
    readback = _readback(
        wrapper_path=wrapper_path,
        configuration=configuration,
        joint_states=joint_states,
    )
    if _sha256(APPROVED_SOURCE) != source_hash:
        raise RuntimeError("approved source changed during build")
    if _sha256(SOURCE_STAGE) != task5_hash:
        raise RuntimeError("Task 5 Stage changed during build")

    files = {
        name: {
            "absolute_path": str(path.resolve()),
            "sha256": _sha256(path),
        }
        for name, path in (
            ("wrapper", wrapper_path),
            ("configuration", configuration_path),
            ("physics", physics_path),
        )
    }
    report = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "DIAGNOSTIC_ONLY_NOT_FINAL",
        "approved_source": {
            "absolute_path": str(APPROVED_SOURCE.resolve()),
            "sha256": source_hash,
            "modified": False,
        },
        "task5_stage": {
            "absolute_path": str(SOURCE_STAGE.resolve()),
            "sha256": task5_hash,
            "modified": False,
        },
        "imported_follower_asset": {
            "absolute_path": str(IMPORTED_FOLLOWER_ASSET.resolve()),
            "sha256": imported_follower_hash,
            "modified": False,
        },
        "files": files,
        "configuration": configuration,
        "joint_states": joint_states,
        "base_evidence": base_evidence,
        "readback": readback,
        "final_default_collider_modified": False,
        "task8": "NOT_RUN",
    }
    output_report = output_report.resolve()
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
        output_report=args.output_report,
    )
    print(f"status={report['status']}")
    print(f"wrapper={report['files']['wrapper']['absolute_path']}")
    print(f"report={args.output_report.resolve()}")
    return 0


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    exit_code = 1
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        for extension_id in (
            "isaacsim.robot.schema",
            "isaacsim.asset.validation",
        ):
            if not manager.is_extension_enabled(extension_id):
                manager.set_extension_enabled_immediate(
                    extension_id,
                    True,
                )
            if not manager.is_extension_enabled(extension_id):
                raise RuntimeError(
                    f"required extension disabled: {extension_id}"
                )
        exit_code = main()
    except Exception:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
