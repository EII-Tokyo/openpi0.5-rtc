from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

CANONICAL_GRIPPER_LINK = "follower_left_ee_gripper_link"
CAD_CONTACT_HELPER_SUFFIX = "/follower_left_ee_gripper_link/aloha1_supplier_cad_clearance_grasp_frame"


@dataclass(frozen=True)
class ClosureError:
    translation_m: float
    rotation_rad: float


def validate_rigid_transform(
    matrix: Sequence[Sequence[float]],
    *,
    atol: float = 1e-10,
) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError("rigid transform must be a 4x4 matrix")
    if not np.isfinite(value).all():
        raise ValueError("rigid transform must contain only finite values")
    if not np.allclose(
        value[3],
        [0.0, 0.0, 0.0, 1.0],
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError("invalid homogeneous row")
    rotation = value[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError("rotation is not orthogonal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=atol, rtol=0.0):
        raise ValueError(f"rotation determinant must be +1, got {determinant}")
    return value


def rigid_transform(
    rotation: Sequence[Sequence[float]],
    translation: Sequence[float],
) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = np.asarray(rotation, dtype=np.float64)
    value[:3, 3] = np.asarray(translation, dtype=np.float64)
    return validate_rigid_transform(value)


def closure_error(
    expected: Sequence[Sequence[float]],
    observed: Sequence[Sequence[float]],
) -> ClosureError:
    expected_value = validate_rigid_transform(expected)
    observed_value = validate_rigid_transform(observed)
    delta = np.linalg.inv(expected_value) @ observed_value
    cosine = float(np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    return ClosureError(
        translation_m=float(np.linalg.norm(delta[:3, 3])),
        rotation_rad=float(np.arccos(cosine)),
    )


def convert_contact_pose_to_gripper_pose(
    *,
    object_from_contact: Sequence[Sequence[float]],
    gripper_from_contact: Sequence[Sequence[float]],
) -> np.ndarray:
    object_from_contact_value = validate_rigid_transform(object_from_contact)
    gripper_from_contact_value = validate_rigid_transform(gripper_from_contact)
    return validate_rigid_transform(object_from_contact_value @ np.linalg.inv(gripper_from_contact_value))


def _rotation_from_urdf_rpy(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation_x = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    rotation_y = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rotation_z = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rotation_z @ rotation_y @ rotation_x


def _vector_attribute(
    element: ET.Element | None,
    name: str,
) -> np.ndarray:
    if element is None or name not in element.attrib:
        return np.zeros(3, dtype=np.float64)
    values = np.fromstring(element.attrib[name], sep=" ", dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"invalid URDF {name}: {element.attrib[name]!r}")
    return values


def derive_urdf_fixed_transform(
    urdf_path: Path | str,
    *,
    source_link: str,
    target_link: str,
) -> np.ndarray:
    root = ET.parse(Path(urdf_path)).getroot()
    children: dict[str, list[tuple[str, np.ndarray, str]]] = {}
    for joint in root.findall("joint"):
        parent_element = joint.find("parent")
        child_element = joint.find("child")
        if parent_element is None or child_element is None:
            raise ValueError("URDF joint is missing parent or child")
        parent = parent_element.attrib["link"]
        child = child_element.attrib["link"]
        origin = joint.find("origin")
        parent_from_child = rigid_transform(
            _rotation_from_urdf_rpy(_vector_attribute(origin, "rpy")),
            _vector_attribute(origin, "xyz"),
        )
        children.setdefault(parent, []).append((child, parent_from_child, joint.attrib.get("type", "")))

    pending: list[tuple[str, np.ndarray]] = [(source_link, np.eye(4, dtype=np.float64))]
    visited: set[str] = set()
    while pending:
        link, source_from_link = pending.pop()
        if link == target_link:
            return validate_rigid_transform(source_from_link)
        if link in visited:
            continue
        visited.add(link)
        for child, link_from_child, joint_type in children.get(link, []):
            if joint_type != "fixed":
                continue
            pending.append((child, source_from_link @ link_from_child))
    raise ValueError(f"no fixed-joint path from {source_link!r} to {target_link!r}")


def validate_native_gripper_dofs(
    *,
    cspace_position: Mapping[str, float],
    pregrasp_cspace_position: Mapping[str, float],
    active_joint: str,
    mimic_joint: str,
) -> dict[str, object]:
    closed_keys = list(cspace_position)
    open_keys = list(pregrasp_cspace_position)
    if mimic_joint in closed_keys or mimic_joint in open_keys:
        raise ValueError(f"mimic joint {mimic_joint!r} must not be an active Grasp Editor DOF")
    if closed_keys != [active_joint] or open_keys != [active_joint]:
        raise ValueError(f"native Grasp Editor YAML must contain exactly the active joint {active_joint!r}")
    values = [
        float(cspace_position[active_joint]),
        float(pregrasp_cspace_position[active_joint]),
    ]
    if not np.isfinite(values).all():
        raise ValueError("gripper positions must be finite")
    return {
        "active_joint": active_joint,
        "mimic_joint": mimic_joint,
        "active_keys": [active_joint],
        "status": "PASS",
    }
