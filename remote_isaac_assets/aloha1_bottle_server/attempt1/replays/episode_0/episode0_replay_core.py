"""Shared Isaac Sim runtime for deterministic ALOHA episode-0 replay.

This module deliberately teleports recorded articulation state and kinematically
tracks the bottle/cap.  It is a visual/data-alignment replay, not a physics test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


CLASSIFICATION = "KINEMATIC_VISUAL_REPLAY_NOT_PHYSICS_ACCEPTANCE"
EXPECTED_DOF_NAMES = [
    "waist", "shoulder", "elbow", "forearm_roll", "wrist_angle",
    "wrist_rotate", "gripper", "left_finger", "right_finger",
]
ACTIVE_DOF_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 7], dtype=np.int64)
LEFT_EE = "/World/follower_left/vx300s_left/follower_left_ee_gripper_link"
RIGHT_EE = "/World/follower_right/vx300s_right/follower_right_ee_gripper_link"
BOTTLE = "/World/ALOHA1RemoteBottleSession/Bottle500"
CAP = "/World/ALOHA1RemoteBottleSession/BottleCap"
STATUS = "/World/Episode0Replay"
ATTACH_BOTTLE = 174
RELEASE_OBJECTS = 768
RIGHT_GRASP_RUNS = ((217, 290), (391, 469), (583, 755))
FINGER_CLOSED_M = 0.0440
FINGER_OPEN_M = 0.0579


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle(bundle_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = json.loads((bundle_dir / "episode_0_replay_manifest.json").read_text())
    payload_path = bundle_dir / manifest["payload"]["path"]
    actual_hash = sha256(payload_path)
    if actual_hash != manifest["payload"]["sha256"]:
        raise ValueError(f"replay payload SHA mismatch: {actual_hash}")
    archive = np.load(payload_path, allow_pickle=False)
    payload = {name: np.asarray(archive[name]) for name in archive.files}
    if payload["action"].shape != (918, 14) or float(payload["frequency_hz"]) != 50.0:
        raise ValueError("episode-0 replay payload contract drift")
    return manifest, payload


def normalized_gripper_to_m(value: float) -> float:
    return FINGER_CLOSED_M + float(np.clip(value, 0.0, 1.0)) * (FINGER_OPEN_M - FINGER_CLOSED_M)


def usd_world_matrix(stage: Any, path: str) -> np.ndarray:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise ValueError(f"missing required prim: {path}")
    return np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64).T


def matrix_pose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation

    position = np.asarray(matrix[:3, 3], dtype=np.float64)
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat(scalar_first=True)
    return position, quaternion


def right_holds_cap(frame: int) -> bool:
    return any(start <= frame <= end for start, end in RIGHT_GRASP_RUNS)


def prepare_kinematic_objects(stage: Any) -> None:
    """Configure replay objects before the first PhysX Play/view creation."""
    from pxr import UsdGeom, UsdPhysics

    for path in (BOTTLE, CAP):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise ValueError(f"missing required prim: {path}")
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr().Set(True)
        UsdGeom.Imageable(prim).MakeVisible()


class Episode0Replay:
    """Stateful frame applicator; caller owns simulation/update pacing."""

    def __init__(self, stage: Any, left: Any, right: Any, bottle: Any, cap: Any, payload: dict[str, np.ndarray]):
        from pxr import Sdf

        self.stage = stage
        self.left = left
        self.right = right
        self.bottle = bottle
        self.cap = cap
        self.payload = payload
        if list(left.dof_names) != EXPECTED_DOF_NAMES or list(right.dof_names) != EXPECTED_DOF_NAMES:
            raise ValueError(f"DOF order drift: left={left.dof_names}, right={right.dof_names}")

        self.initial_bottle = usd_world_matrix(stage, BOTTLE)
        self.initial_cap = usd_world_matrix(stage, CAP)
        self.bottle_to_cap = np.linalg.inv(self.initial_bottle) @ self.initial_cap
        self.left_to_bottle: np.ndarray | None = None
        self.right_to_cap: np.ndarray | None = None
        self.last_right_holds = False
        self.last_bottle = self.initial_bottle.copy()
        self.last_cap = self.initial_cap.copy()
        self.max_attach_jump_m = 0.0
        self.frames_applied = 0

        prepare_kinematic_objects(stage)
        status = stage.DefinePrim(STATUS, "Scope")
        status.CreateAttribute("replay:classification", Sdf.ValueTypeNames.String, custom=True).Set(CLASSIFICATION)
        status.CreateAttribute("replay:savesStage", Sdf.ValueTypeNames.Bool, custom=True).Set(False)
        status.CreateAttribute("replay:usesRos", Sdf.ValueTypeNames.Bool, custom=True).Set(False)
        self.frame_attr = status.CreateAttribute("replay:frame", Sdf.ValueTypeNames.Int, custom=True)
        self.state_attr = status.CreateAttribute("replay:state", Sdf.ValueTypeNames.String, custom=True)
        self.label_attr = status.CreateAttribute("replay:label", Sdf.ValueTypeNames.String, custom=True)

    def reset(self) -> None:
        from pxr import UsdGeom

        self.left_to_bottle = None
        self.right_to_cap = None
        self.last_right_holds = False
        self.last_bottle = self.initial_bottle.copy()
        self.last_cap = self.initial_cap.copy()
        self.max_attach_jump_m = 0.0
        self.frames_applied = 0
        UsdGeom.Imageable(self.stage.GetPrimAtPath(BOTTLE)).MakeVisible()
        UsdGeom.Imageable(self.stage.GetPrimAtPath(CAP)).MakeVisible()
        self._set_object_pose(self.bottle, self.initial_bottle)
        self._set_object_pose(self.cap, self.initial_cap)

    @staticmethod
    def _set_object_pose(prim: Any, matrix: np.ndarray) -> None:
        position, orientation = matrix_pose(matrix)
        prim.set_world_pose(position=position, orientation=orientation)

    def apply_robot_frame(self, frame: int) -> None:
        command = np.asarray(self.payload["action"][frame], dtype=np.float64)
        left_values = np.concatenate((command[:6], [normalized_gripper_to_m(command[6])]))
        right_values = np.concatenate((command[7:13], [normalized_gripper_to_m(command[13])]))
        self.left.set_joint_positions(left_values, joint_indices=ACTIVE_DOF_INDICES)
        self.right.set_joint_positions(right_values, joint_indices=ACTIVE_DOF_INDICES)
        self.left.set_joint_velocities(np.zeros(7), joint_indices=ACTIVE_DOF_INDICES)
        self.right.set_joint_velocities(np.zeros(7), joint_indices=ACTIVE_DOF_INDICES)

    def apply_objects_and_metadata(self, frame: int) -> None:
        from pxr import UsdGeom

        left_ee = usd_world_matrix(self.stage, LEFT_EE)
        right_ee = usd_world_matrix(self.stage, RIGHT_EE)

        if frame < ATTACH_BOTTLE:
            bottle_world = self.initial_bottle
        else:
            if self.left_to_bottle is None:
                self.left_to_bottle = np.linalg.inv(left_ee) @ self.initial_bottle
            bottle_world = left_ee @ self.left_to_bottle

        holds = right_holds_cap(frame)
        if frame < RIGHT_GRASP_RUNS[0][0]:
            cap_world = bottle_world @ self.bottle_to_cap
        elif holds:
            if not self.last_right_holds or self.right_to_cap is None:
                before = self.last_cap.copy()
                self.right_to_cap = np.linalg.inv(right_ee) @ before
                candidate = right_ee @ self.right_to_cap
                self.max_attach_jump_m = max(self.max_attach_jump_m, float(np.linalg.norm(candidate[:3, 3] - before[:3, 3])))
            cap_world = right_ee @ self.right_to_cap
        else:
            if self.last_right_holds:
                self.bottle_to_cap = np.linalg.inv(bottle_world) @ self.last_cap
            cap_world = bottle_world @ self.bottle_to_cap

        self._set_object_pose(self.bottle, bottle_world)
        self._set_object_pose(self.cap, cap_world)
        self.last_bottle = bottle_world.copy()
        self.last_cap = cap_world.copy()
        self.last_right_holds = holds

        if frame >= RELEASE_OBJECTS:
            UsdGeom.Imageable(self.stage.GetPrimAtPath(BOTTLE)).MakeInvisible()
            UsdGeom.Imageable(self.stage.GetPrimAtPath(CAP)).MakeInvisible()

        self.frame_attr.Set(int(frame))
        self.state_attr.Set(str(self.payload["state"][frame]))
        self.label_attr.Set(str(self.payload["label"][frame]))
        self.frames_applied += 1

    def commanded_active_positions(self, frame: int, side: str) -> np.ndarray:
        command = np.asarray(self.payload["action"][frame], dtype=np.float64)
        offset = 0 if side == "left" else 7
        return np.concatenate((command[offset : offset + 6], [normalized_gripper_to_m(command[offset + 6])]))
