from __future__ import annotations

import argparse
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from examples.aloha_real.constants import (
    PUPPET_GRIPPER_JOINT_CLOSE,
    PUPPET_GRIPPER_JOINT_OPEN,
    PUPPET_GRIPPER_POSITION_CLOSE,
    PUPPET_GRIPPER_POSITION_OPEN,
    START_ARM_POSE,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_USD = (
    REPO_ROOT
    / "local_eval_assets/aloha_isaac_menagerie_deep_black/aloha2_menagerie_scene_deep_black.usd"
)
DEFAULT_OUTPUT_USD = (
    REPO_ROOT
    / "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/"
    / "aloha2_menagerie_scene_deep_black_real_start_pose.usd"
)

REAL_START_ARM_POSE = tuple(float(value) for value in START_ARM_POSE)
REAL_RUNTIME_RESET_QPOS14 = (
    0.0,
    -0.96,
    1.16,
    1.57,
    0.0,
    -1.57,
    PUPPET_GRIPPER_JOINT_OPEN,
    0.0,
    -0.96,
    1.16,
    0.0,
    0.0,
    0.0,
    PUPPET_GRIPPER_JOINT_OPEN,
)

ALOHA_USD_JOINTS = (
    ("/scene/joints/left_waist", "angular"),
    ("/scene/joints/left_shoulder", "angular"),
    ("/scene/joints/left_elbow", "angular"),
    ("/scene/joints/left_forearm_roll", "angular"),
    ("/scene/joints/left_wrist_angle", "angular"),
    ("/scene/joints/left_wrist_rotate", "angular"),
    ("/scene/joints/left_left_finger", "linear"),
    ("/scene/joints/left_right_finger", "linear"),
    ("/scene/joints/right_waist", "angular"),
    ("/scene/joints/right_shoulder", "angular"),
    ("/scene/joints/right_elbow", "angular"),
    ("/scene/joints/right_forearm_roll", "angular"),
    ("/scene/joints/right_wrist_angle", "angular"),
    ("/scene/joints/right_wrist_rotate", "angular"),
    ("/scene/joints/right_left_finger", "linear"),
    ("/scene/joints/right_right_finger", "linear"),
)

ALOHA_ROOT_JOINTS = (
    "/scene/joints/rootJoint_left_base_link",
    "/scene/joints/rootJoint_right_base_link",
)
ALOHA_FINGER_JOINTS = (
    "/scene/joints/left_left_finger",
    "/scene/joints/left_right_finger",
    "/scene/joints/right_left_finger",
    "/scene/joints/right_right_finger",
)
FINGER_PRISMATIC_UPPER_LIMIT = float(PUPPET_GRIPPER_POSITION_OPEN)
FINGER_PRISMATIC_LOWER_LIMIT = float(PUPPET_GRIPPER_POSITION_CLOSE)


@dataclass(frozen=True)
class PoseRecord:
    joint_path: str
    drive_type: str
    position: float


@dataclass(frozen=True)
class RootJointAnchorRecord:
    joint_path: str
    local_pos0: tuple[float, float, float]
    local_rot0: tuple[float, float, float, float]


def _prepare_output_bundle(source_usd: Path, output_usd: Path) -> Path:
    source_usd = source_usd.resolve()
    output_usd = output_usd.resolve()
    if source_usd == output_usd:
        return output_usd

    source_dir = source_usd.parent
    output_dir = output_usd.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if source_dir != output_dir:
        shutil.copytree(source_dir, output_dir, dirs_exist_ok=True)

    copied_source = output_dir / source_usd.name
    if copied_source != output_usd:
        shutil.copy2(copied_source, output_usd)
    return output_usd


def _start_isaac_headless():
    from isaacsim import SimulationApp

    return SimulationApp({"headless": True})


def pose_to_usd_joint_positions(pose: tuple[float, ...]) -> tuple[float, ...]:
    """Convert robot pose units into USD joint units: degrees for hinges, meters for sliders."""
    if len(pose) != len(ALOHA_USD_JOINTS):
        raise ValueError(f"pose has {len(pose)} values; expected {len(ALOHA_USD_JOINTS)}")

    pose = tuple(_to_isaac_finger_position(value, i) for i, value in enumerate(pose))
    converted: list[float] = []
    for value, (_, drive_type) in zip(pose, ALOHA_USD_JOINTS, strict=True):
        if drive_type == "angular":
            converted.append(math.degrees(value))
        elif drive_type == "linear":
            converted.append(value)
        else:
            raise ValueError(f"unsupported drive type: {drive_type}")
    return tuple(converted)


def puppet_gripper_joint_to_isaac_finger_position(gripper_joint: float) -> float:
    """Map the real robot puppet gripper joint angle qpos[6] to Isaac finger travel."""
    normalized = (float(gripper_joint) - PUPPET_GRIPPER_JOINT_CLOSE) / (
        PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE
    )
    finger_position = normalized * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
    finger_position += PUPPET_GRIPPER_POSITION_CLOSE
    return min(max(finger_position, FINGER_PRISMATIC_LOWER_LIMIT), FINGER_PRISMATIC_UPPER_LIMIT)


def qpos14_to_isaac_joint_pose(qpos: tuple[float, ...]) -> tuple[float, ...]:
    """Convert runtime qpos layout into the imported Isaac/MJCF articulation layout.

    Runtime ALOHA uses [left arm 6, left gripper joint, right arm 6, right gripper joint].
    The imported Isaac articulation exposes each gripper as two positive prismatic finger joints.
    """
    if len(qpos) != 14:
        raise ValueError(f"qpos has {len(qpos)} values; expected 14")

    left_finger = puppet_gripper_joint_to_isaac_finger_position(qpos[6])
    right_finger = puppet_gripper_joint_to_isaac_finger_position(qpos[13])
    return (
        *tuple(float(value) for value in qpos[:6]),
        left_finger,
        left_finger,
        *tuple(float(value) for value in qpos[7:13]),
        right_finger,
        right_finger,
    )


REAL_RUNTIME_RESET_POSE = qpos14_to_isaac_joint_pose(REAL_RUNTIME_RESET_QPOS14)


def _to_isaac_finger_position(value: float, index: int) -> float:
    # ACT-style START_ARM_POSE stores the paired finger as +/- opening.
    # The imported MJCF/Isaac prismatic finger joints both use positive [0, opening] coordinates.
    if index in (6, 7, 14, 15):
        return abs(value)
    return value


def split_real_start_pose_for_isaac_articulations(
    pose: tuple[float, ...] = REAL_RUNTIME_RESET_POSE,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(pose) != len(ALOHA_USD_JOINTS):
        raise ValueError(f"pose has {len(pose)} values; expected {len(ALOHA_USD_JOINTS)}")
    isaac_pose = tuple(_to_isaac_finger_position(value, i) for i, value in enumerate(pose))
    return isaac_pose[:8], isaac_pose[8:]


def root_joint_world_anchor_from_body_translation(
    body_translation: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return root-joint anchors for a joint from world to a translated body."""
    return tuple(float(value) for value in body_translation), (0.0, 0.0, 0.0)


def build_pose_records(pose: tuple[float, ...]) -> tuple[PoseRecord, ...]:
    positions = pose_to_usd_joint_positions(pose)
    return tuple(
        PoseRecord(joint_path=joint_path, drive_type=drive_type, position=position)
        for (joint_path, drive_type), position in zip(ALOHA_USD_JOINTS, positions, strict=True)
    )


def _write_pose_records_to_stage(stage, records: tuple[PoseRecord, ...]) -> int:
    from pxr import PhysxSchema, UsdPhysics

    applied = 0
    for record in records:
        prim = stage.GetPrimAtPath(record.joint_path)
        if not prim.IsValid():
            raise RuntimeError(f"stage is missing expected ALOHA joint: {record.joint_path}")

        drive = UsdPhysics.DriveAPI.Apply(prim, record.drive_type)
        joint_state = PhysxSchema.JointStateAPI.Apply(prim, record.drive_type)
        drive.CreateTargetPositionAttr().Set(record.position)
        drive.CreateTargetVelocityAttr().Set(0.0)
        joint_state.CreatePositionAttr().Set(record.position)
        joint_state.CreateVelocityAttr().Set(0.0)
        applied += 1
    return applied


def _write_finger_joint_limits_to_stage(stage) -> int:
    from pxr import UsdPhysics

    applied = 0
    for joint_path in ALOHA_FINGER_JOINTS:
        prim = stage.GetPrimAtPath(joint_path)
        if not prim.IsValid():
            continue
        joint = UsdPhysics.PrismaticJoint(prim)
        joint.CreateLowerLimitAttr().Set(FINGER_PRISMATIC_LOWER_LIMIT)
        joint.CreateUpperLimitAttr().Set(FINGER_PRISMATIC_UPPER_LIMIT)
        applied += 1
    return applied


def _write_finger_joint_limits_to_authored_layers(stage) -> int:
    from pxr import Usd

    layers = []
    seen = set()
    for joint_path in ALOHA_FINGER_JOINTS:
        prim = stage.GetPrimAtPath(joint_path)
        if not prim.IsValid():
            continue
        for spec in prim.GetPrimStack():
            layer = spec.layer
            identifier = layer.identifier
            if identifier in seen:
                continue
            seen.add(identifier)
            layers.append(layer)

    applied = 0
    for layer in layers:
        layer_path = layer.realPath or layer.identifier
        if not layer_path:
            continue
        layer_stage = Usd.Stage.Open(layer_path)
        if layer_stage is None:
            continue
        applied += _write_finger_joint_limits_to_stage(layer_stage)
        layer_stage.GetRootLayer().Save()
    return applied


def _compute_root_joint_world_anchor_records(stage) -> tuple[RootJointAnchorRecord, ...]:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    records: list[RootJointAnchorRecord] = []
    for joint_path in ALOHA_ROOT_JOINTS:
        prim = stage.GetPrimAtPath(joint_path)
        if not prim.IsValid():
            continue

        joint = UsdPhysics.Joint(prim)
        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        if body0_targets or not body1_targets:
            continue

        body1_prim = stage.GetPrimAtPath(body1_targets[0])
        if not body1_prim.IsValid():
            continue

        local_pos1 = joint.GetLocalPos1Attr().Get() or Gf.Vec3f(0.0, 0.0, 0.0)
        local_rot1 = joint.GetLocalRot1Attr().Get() or Gf.Quatf(1.0)

        local1 = Gf.Matrix4d(1.0)
        local1.SetRotate(
            Gf.Rotation(
                Gf.Quatd(
                    float(local_rot1.GetReal()),
                    Gf.Vec3d(*(float(value) for value in local_rot1.GetImaginary())),
                )
            )
        )
        local1.SetTranslate(Gf.Vec3d(*(float(value) for value in local_pos1)))
        joint_world = local1 * cache.GetLocalToWorldTransform(body1_prim)

        translation = joint_world.ExtractTranslation()
        rotation = joint_world.ExtractRotationQuat()
        records.append(
            RootJointAnchorRecord(
                joint_path=joint_path,
                local_pos0=root_joint_world_anchor_from_body_translation(tuple(translation))[0],
                local_rot0=(
                    float(rotation.GetReal()),
                    *(float(value) for value in rotation.GetImaginary()),
                ),
            )
        )
    return tuple(records)


def _write_root_joint_anchor_records_to_stage(stage, records: tuple[RootJointAnchorRecord, ...]) -> int:
    from pxr import Gf, UsdPhysics

    applied = 0
    for record in records:
        prim = stage.GetPrimAtPath(record.joint_path)
        if not prim.IsValid():
            continue
        joint = UsdPhysics.Joint(prim)
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*record.local_pos0))
        joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(
                record.local_rot0[0],
                Gf.Vec3f(*record.local_rot0[1:]),
            )
        )
        applied += 1
    return applied


def _write_root_joint_world_anchors_to_stage(stage) -> int:
    records = _compute_root_joint_world_anchor_records(stage)
    return _write_root_joint_anchor_records_to_stage(stage, records)


def _write_root_joint_world_anchors_to_authored_layers(stage) -> int:
    from pxr import Usd

    records = _compute_root_joint_world_anchor_records(stage)
    if not records:
        return 0

    layers = []
    seen = set()
    for joint_path in ALOHA_ROOT_JOINTS:
        prim = stage.GetPrimAtPath(joint_path)
        if not prim.IsValid():
            continue
        for spec in prim.GetPrimStack():
            layer = spec.layer
            identifier = layer.identifier
            if identifier in seen:
                continue
            seen.add(identifier)
            layers.append(layer)

    applied = 0
    for layer in layers:
        layer_path = layer.realPath or layer.identifier
        if not layer_path:
            continue
        layer_stage = Usd.Stage.Open(layer_path)
        if layer_stage is None:
            continue
        applied += _write_root_joint_anchor_records_to_stage(layer_stage, records)
        layer_stage.GetRootLayer().Save()
    return applied


def apply_initial_pose(source_usd: Path, output_usd: Path, pose: tuple[float, ...] = REAL_RUNTIME_RESET_POSE) -> int:
    from pxr import Sdf, Usd

    if not source_usd.exists():
        raise FileNotFoundError(f"source USD does not exist: {source_usd}")

    output_usd = _prepare_output_bundle(source_usd, output_usd)
    records = build_pose_records(pose)
    stage = Usd.Stage.Open(str(output_usd))
    if stage is None:
        raise RuntimeError(f"failed to open USD stage: {output_usd}")

    applied = _write_pose_records_to_stage(stage, records)
    applied += _write_finger_joint_limits_to_stage(stage)
    applied += _write_finger_joint_limits_to_authored_layers(stage)
    applied += _write_root_joint_world_anchors_to_stage(stage)
    applied += _write_root_joint_world_anchors_to_authored_layers(stage)
    stage.GetRootLayer().Save()

    sublayers = [Path(path) for path in stage.GetRootLayer().subLayerPaths]
    for sublayer in sublayers:
        sublayer_path = output_usd.parent / sublayer
        if not sublayer_path.exists():
            continue
        sub_stage = Usd.Stage.Open(str(sublayer_path))
        if sub_stage is None:
            continue
        try:
            applied += _write_pose_records_to_stage(sub_stage, records)
            applied += _write_finger_joint_limits_to_stage(sub_stage)
            applied += _write_root_joint_world_anchors_to_stage(sub_stage)
        except RuntimeError:
            continue
        sub_stage.GetRootLayer().Save()

    Sdf.Layer.FindOrOpen(str(output_usd)).Save()
    return applied


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a real ALOHA start pose to an Isaac Sim USD stage.")
    parser.add_argument("--source-usd", type=Path, default=DEFAULT_SOURCE_USD)
    parser.add_argument("--output-usd", type=Path, default=DEFAULT_OUTPUT_USD)
    args = parser.parse_args()

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    app = _start_isaac_headless()
    try:
        applied = apply_initial_pose(args.source_usd.resolve(), args.output_usd.resolve())
        print(f"output_usd={args.output_usd.resolve()}")
        print(f"pose_records_written={applied}")
        if applied == 0:
            raise RuntimeError("no ALOHA joints were updated")
    finally:
        app.close()


if __name__ == "__main__":
    main()
