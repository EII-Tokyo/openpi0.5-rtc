from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

OFFICIAL_JOINT_ORDER = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
)

OFFICIAL_TROSSEN_M = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.536494],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.42705],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

OFFICIAL_TROSSEN_SLIST = np.asarray(
    [
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.12705, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.42705, 0.0, 0.05955],
        [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
        [0.0, 1.0, 0.0, -0.42705, 0.0, 0.35955],
        [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
    ],
    dtype=np.float64,
).T


def skew(vector: Any) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def twist_to_se3(twist: Any) -> np.ndarray:
    vector = np.asarray(twist, dtype=np.float64)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError("twist must be a finite six-vector [omega, v]")
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[:3, :3] = skew(vector[:3])
    matrix[:3, 3] = vector[3:]
    return matrix


def adjoint(transform: Any) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if not is_rigid_transform(matrix):
        raise ValueError("adjoint requires a proper rigid transform")
    rotation = matrix[:3, :3]
    position = matrix[:3, 3]
    result = np.zeros((6, 6), dtype=np.float64)
    result[:3, :3] = rotation
    result[3:, 3:] = rotation
    result[3:, :3] = skew(position) @ rotation
    return result


def is_rigid_transform(transform: Any, *, atol: float = 1e-12) -> bool:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return False
    rotation = matrix[:3, :3]
    return bool(
        np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=atol)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=atol)
        and math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=atol)
    )


def quaternion_wxyz_to_matrix(quaternion: Any) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("quaternion must be finite and nonzero")
    return Rotation.from_quat([x / norm, y / norm, z / norm, w / norm]).as_matrix()


def poe_fk(joint_positions: Any) -> np.ndarray:
    q = np.asarray(joint_positions, dtype=np.float64)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("joint_positions must be a finite six-vector")
    transform = np.eye(4, dtype=np.float64)
    for index, position in enumerate(q):
        transform = transform @ expm(twist_to_se3(OFFICIAL_TROSSEN_SLIST[:, index]) * position)
    return transform @ OFFICIAL_TROSSEN_M


def poe_space_jacobian(joint_positions: Any) -> np.ndarray:
    q = np.asarray(joint_positions, dtype=np.float64)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("joint_positions must be a finite six-vector")
    jacobian = np.zeros((6, 6), dtype=np.float64)
    jacobian[:, 0] = OFFICIAL_TROSSEN_SLIST[:, 0]
    transform = np.eye(4, dtype=np.float64)
    for index in range(1, 6):
        transform = transform @ expm(twist_to_se3(OFFICIAL_TROSSEN_SLIST[:, index - 1]) * q[index - 1])
        jacobian[:, index] = adjoint(transform) @ OFFICIAL_TROSSEN_SLIST[:, index]
    return jacobian


def _origin_transform(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
    rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def load_urdf_chain(urdf_path: Path) -> dict[str, object]:
    root = ET.parse(urdf_path).getroot()
    joints = list(root.findall("joint"))
    child_to_joint = {joint.find("child").attrib["link"]: joint for joint in joints}
    target = next(
        link.attrib["name"] for link in root.findall("link") if link.attrib["name"].endswith("_ee_gripper_link")
    )
    chain: list[ET.Element] = []
    current = target
    while current in child_to_joint:
        joint = child_to_joint[current]
        chain.append(joint)
        current = joint.find("parent").attrib["link"]
    chain.reverse()
    arm_joints = {joint.attrib["name"]: joint for joint in joints if joint.attrib["name"] in OFFICIAL_JOINT_ORDER}
    if tuple(name for name in OFFICIAL_JOINT_ORDER if name in arm_joints) != OFFICIAL_JOINT_ORDER:
        raise ValueError(f"URDF does not contain the explicit official joint order: {urdf_path}")
    limits = {
        name: {
            "lower": float(arm_joints[name].find("limit").attrib["lower"]),
            "upper": float(arm_joints[name].find("limit").attrib["upper"]),
        }
        for name in OFFICIAL_JOINT_ORDER
    }
    return {
        "path": str(urdf_path.resolve()),
        "chain": chain,
        "limits": limits,
        "root_link": current,
        "target_link": target,
    }


def urdf_fk(chain_record: dict[str, object], joint_positions: Any) -> np.ndarray:
    q = np.asarray(joint_positions, dtype=np.float64)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("joint_positions must be a finite six-vector")
    positions = dict(zip(OFFICIAL_JOINT_ORDER, q, strict=True))
    transform = np.eye(4, dtype=np.float64)
    for joint in chain_record["chain"]:
        transform = transform @ _origin_transform(joint)
        name = joint.attrib["name"]
        if name in positions:
            axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
            motion = np.eye(4, dtype=np.float64)
            motion[:3, :3] = Rotation.from_rotvec(axis * positions[name]).as_matrix()
            transform = transform @ motion
    return transform


def numerical_space_jacobian(fk_function: Any, joint_positions: Any, *, step: float) -> np.ndarray:
    q = np.asarray(joint_positions, dtype=np.float64)
    transform = fk_function(q)
    inverse = np.linalg.inv(transform)
    jacobian = np.zeros((6, 6), dtype=np.float64)
    for index in range(6):
        delta = np.zeros(6)
        delta[index] = step
        derivative = (fk_function(q + delta) - fk_function(q - delta)) / (2.0 * step)
        spatial = derivative @ inverse
        jacobian[:3, index] = [spatial[2, 1], spatial[0, 2], spatial[1, 0]]
        jacobian[3:, index] = spatial[:3, 3]
    return jacobian


def rotation_distance_rad(first: Any, second: Any) -> float:
    relative = np.asarray(first)[:3, :3].T @ np.asarray(second)[:3, :3]
    return float(Rotation.from_matrix(relative).magnitude())
