"""Evidence-bound URDF audit for Task 7A geometry-free helper links."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

HELPER_SUFFIXES = (
    "ee_arm_link",
    "fingers_link",
    "ee_gripper_link",
)

ALLOWED_SEMANTIC_CLASSES = {
    "VIRTUAL_KINEMATIC_HELPER",
    "MASS_BEARING_SOURCE_LINK_WITHOUT_COLLIDER",
    "FIXED_FRAME_ALIAS",
    "INCONCLUSIVE",
}


def classify_helper_semantics(
    *,
    visual_count: int,
    collision_count: int,
    inertial_count: int,
    parent_joint_types: list[str],
    child_joint_types: list[str],
) -> str:
    """Classify a helper link from explicit URDF structure only."""
    if visual_count or collision_count:
        return "INCONCLUSIVE"
    if len(parent_joint_types) != 1:
        return "INCONCLUSIVE"
    if parent_joint_types[0] != "fixed":
        return "MASS_BEARING_SOURCE_LINK_WITHOUT_COLLIDER"
    if not child_joint_types:
        return "FIXED_FRAME_ALIAS"
    if inertial_count == 1:
        return "VIRTUAL_KINEMATIC_HELPER"
    return "INCONCLUSIVE"


def _joint_inventory(
    root: ET.Element,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
    parent_joints: dict[str, list[dict[str, str]]] = defaultdict(list)
    child_joints: dict[str, list[dict[str, str]]] = defaultdict(list)
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        record = {
            "name": str(joint.get("name")),
            "type": str(joint.get("type")),
            "parent": str(parent.get("link")),
            "child": str(child.get("link")),
        }
        parent_joints[record["child"]].append(record)
        child_joints[record["parent"]].append(record)
    return parent_joints, child_joints


def audit_urdf_helper_links(
    urdf_path: Path,
    robot_name: str,
) -> dict[str, dict[str, Any]]:
    """Return deterministic source semantics for one follower URDF."""
    resolved = urdf_path.resolve(strict=True)
    root = ET.parse(resolved).getroot()
    links = {str(link.get("name")): link for link in root.findall("link")}
    parent_joints, child_joints = _joint_inventory(root)
    records: dict[str, dict[str, Any]] = {}
    for suffix in HELPER_SUFFIXES:
        name = f"{robot_name}_{suffix}"
        if name not in links:
            raise ValueError(f"missing helper link {name} in {resolved}")
        link = links[name]
        inertials = link.findall("inertial")
        mass_values = [float(mass.get("value")) for inertial in inertials for mass in inertial.findall("mass")]
        incoming = parent_joints.get(name, [])
        outgoing = child_joints.get(name, [])
        visual_count = len(link.findall("visual"))
        collision_count = len(link.findall("collision"))
        inertial_count = len(inertials)
        semantic_class = classify_helper_semantics(
            visual_count=visual_count,
            collision_count=collision_count,
            inertial_count=inertial_count,
            parent_joint_types=[item["type"] for item in incoming],
            child_joint_types=[item["type"] for item in outgoing],
        )
        if semantic_class not in ALLOWED_SEMANTIC_CLASSES:
            raise AssertionError(f"unexpected semantic class {semantic_class}")
        records[name] = {
            "urdf_path": str(resolved),
            "link_name": name,
            "visual_count": visual_count,
            "collision_count": collision_count,
            "inertial_count": inertial_count,
            "mass_kg": mass_values[0] if len(mass_values) == 1 else None,
            "parent_joints": incoming,
            "child_joints": outgoing,
            "semantic_class": semantic_class,
            "invent_collider_allowed": False,
            "remove_rigid_body_api_allowed": False,
            "decision_basis": (
                "Source URDF/Xacro has no visual or collision geometry for "
                "this frame. Placeholder inertial values do not define a "
                "physical shape, and changing body semantics requires a "
                "separate source-backed articulation regression."
            ),
        }
    return records
