"""Explicit signal semantics for the Stationary ALOHA 1 followers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

ARM_JOINTS = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]
ACTIVE_ONE_JOINT_TESTS = [
    *ARM_JOINTS,
    "gripper",
    "left_finger",
]
ROS_JOINT_STATE_ORDER = [
    *ARM_JOINTS,
    "gripper",
    "left_finger",
    "right_finger",
]
DATASET_14D_ORDER = [
    *(f"left_{name}" for name in ARM_JOINTS),
    "left_gripper_normalized",
    *(f"right_{name}" for name in ARM_JOINTS),
    "right_gripper_normalized",
]
HOME_ARM = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
HOME_LEFT_FINGER_M = 0.02239
HOME_RIGHT_FINGER_M = -0.02239

RUNTIME_SPECS = {
    "follower_left": {
        "prefix": "",
        "articulation_path": ("/World/follower_left/vx300s_left/root_joint"),
        "base_link_path": ("/World/follower_left/vx300s_left/follower_left_base_link"),
        "end_effector_path": ("/World/follower_left/vx300s_left/follower_left_gripper_link"),
        "runtime_expected_order": [
            "waist",
            "shoulder",
            "elbow",
            "forearm_roll",
            "wrist_angle",
            "wrist_rotate",
            "gripper",
            "left_finger",
            "right_finger",
        ],
    },
    "follower_right": {
        "prefix": "",
        "articulation_path": ("/World/follower_right/vx300s_right/root_joint"),
        "base_link_path": ("/World/follower_right/vx300s_right/follower_right_base_link"),
        "end_effector_path": ("/World/follower_right/vx300s_right/follower_right_gripper_link"),
        "runtime_expected_order": [
            "waist",
            "shoulder",
            "elbow",
            "forearm_roll",
            "wrist_angle",
            "wrist_rotate",
            "gripper",
            "left_finger",
            "right_finger",
        ],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_dof_name(robot: str, runtime_name: str) -> str:
    prefix = RUNTIME_SPECS[robot]["prefix"]
    if prefix and runtime_name.startswith(prefix):
        return runtime_name[len(prefix) :]
    legacy_prefix = "vx300s_left_" if robot == "follower_left" else ""
    if legacy_prefix and runtime_name.startswith(legacy_prefix):
        return runtime_name[len(legacy_prefix) :]
    return runtime_name


def _urdf_order(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    return [joint.attrib["name"] for joint in root.findall("joint") if joint.attrib["type"] != "fixed"]


def _dataset_indices(side: str, name: str) -> tuple[int | None, int | None]:
    base = 0 if side == "left" else 7
    if name in ARM_JOINTS:
        index = base + ARM_JOINTS.index(name)
        return index, index
    if name in {"gripper", "left_finger", "right_finger"}:
        return base + 6, base + 6
    return None, None


def _mapping_rows(robot: str) -> list[dict[str, Any]]:
    side = "left" if robot.endswith("_left") else "right"
    order = RUNTIME_SPECS[robot]["runtime_expected_order"]
    rows = []
    for runtime_index, runtime_name in enumerate(order):
        name = canonical_dof_name(robot, runtime_name)
        state_index, action_index = _dataset_indices(side, name)
        if name in ARM_JOINTS:
            unit = "rad"
            sign = 1.0
            offset = 0.0
            role = "ARM_SIGNAL"
        elif name == "left_finger":
            unit = "m"
            sign = 1.0
            offset = 0.021
            role = "GRIPPER_STATE_READBACK_AND_DIAGNOSTIC_TARGET"
        elif name == "right_finger":
            unit = "m"
            sign = -1.0
            offset = -0.021
            role = "GRIPPER_MIMIC_READBACK"
        else:
            unit = "rad"
            sign = 1.0
            offset = -0.6213
            role = "AUXILIARY_GRIPPER_COMMAND_JOINT"
        rows.append(
            {
                "canonical_name": name,
                "runtime_name": runtime_name,
                "isaac_index": runtime_index,
                "ros_index": ROS_JOINT_STATE_ORDER.index(name),
                "dataset_state_index": state_index,
                "dataset_action_index": action_index,
                "unit": unit,
                "sign": sign,
                "offset": offset,
                "role": role,
            }
        )
    return rows


def build_signal_mapping_plan(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    robots: dict[str, Any] = {}
    for robot, runtime in RUNTIME_SPECS.items():
        urdf = root / "generated/urdf" / f"{robot}.urdf"
        robots[robot] = {
            **runtime,
            "model": "aloha_vx300s",
            "urdf": str(urdf.resolve(strict=True)),
            "urdf_sha256": _sha256(urdf),
            "urdf_nonfixed_order": _urdf_order(urdf),
            "ros_joint_state_order": list(ROS_JOINT_STATE_ORDER),
            "runtime_expected_order": list(runtime["runtime_expected_order"]),
            "runtime_canonical_order": [canonical_dof_name(robot, name) for name in runtime["runtime_expected_order"]],
            "mapping": _mapping_rows(robot),
        }
    return {
        "schema_version": 2,
        "status": "PENDING_RUNTIME_READBACK",
        "scope": "KINEMATIC_AND_SIGNAL_CORRESPONDENCE_BASELINE",
        "order_policy": "EXPLICIT_SOURCE_ORDER_NEVER_ALPHABETICAL",
        "dataset_14d_order": list(DATASET_14D_ORDER),
        "robots": robots,
        "known_structural_difference": {
            "follower_left_runtime_dof_count": 9,
            "follower_right_runtime_dof_count": 9,
            "difference": "NONE_IN_SIGNAL_BASELINE",
            "hidden": False,
            "task7a_gate": (
                "all six arm signals and normalized gripper semantics must remain explicit and runtime-verified"
            ),
        },
    }


def build_small_up_down_targets() -> dict[str, Any]:
    home = list(HOME_ARM)
    small_up = list(home)
    small_up[1] -= 0.08
    return {
        "joint": "shoulder",
        "unit": "rad",
        "home": home,
        "small_up": small_up,
        "return_home": list(home),
        "maximum_absolute_delta_rad": 0.08,
        "trajectory": "cubic_smoothstep",
        "direction_status": "PENDING_RUNTIME_END_EFFECTOR_Z_READBACK",
    }


def build_fixed_oblique_camera_spec(
    robot_world_points: np.ndarray,
    robot: str,
) -> dict[str, Any]:
    """Derive one fixed full-arm camera from the composed robot geometry."""
    points = np.asarray(robot_world_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("robot_world_points must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("robot_world_points must be finite")
    if robot not in RUNTIME_SPECS:
        raise ValueError(f"unknown robot: {robot}")

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    target = (minimum + maximum) / 2.0
    span = float(np.linalg.norm(maximum - minimum))
    if span <= 0.0:
        raise ValueError("robot geometry span must be positive")

    x_sign = -1.0 if robot == "follower_left" else 1.0
    direction = np.asarray(
        [x_sign, -0.85, 0.72],
        dtype=np.float64,
    )
    direction /= np.linalg.norm(direction)
    position = target + direction * (2.2 * span + 0.35)
    return {
        "position_world_m": position.tolist(),
        "target_world_m": target.tolist(),
        "robot_aabb_min_world_m": minimum.tolist(),
        "robot_aabb_max_world_m": maximum.tolist(),
        "robot_geometry_span_m": span,
        "view": "geometry_derived_full_arm_oblique",
        "reason": (
            "fixed oblique view is derived from composed full-arm visual "
            "mesh bounds to expose the driven joint, end effector, and "
            "vertical motion without changing the tested trajectory"
        ),
        "fixed_for_robot_phase_group": True,
    }
