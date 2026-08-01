"""Classify ALOHA follower links for CAD-derived collision work."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from tools.aloha1_mapping.task7a_helper_link_audit import audit_urdf_helper_links

ALLOWED_CLASSIFICATIONS = {
    "PHYSICAL_CAD_DERIVABLE",
    "VIRTUAL_FRAME_NO_COLLIDER",
    "PHYSICAL_EXISTING_VALIDATED_COLLIDER",
    "HARD_BLOCKER_CAD_TO_LINK_IDENTITY",
}

CAD_OBJECT_BY_LINK_SUFFIX = {
    "base_link": "Part__Feature",
    "shoulder_link": "Part__Feature001",
    "upper_arm_link": "Part__Feature002",
    "upper_forearm_link": "Part__Feature003",
    "lower_forearm_link": "Part__Feature004",
    "wrist_link": "Part__Feature005",
    "gripper_link": "Part__Feature006",
    "left_finger_link": "Part__Feature007",
    "right_finger_link": "Part__Feature008",
}

ACCEPTED_FINGER_SUFFIXES = {
    "left_finger_link",
    "right_finger_link",
}


def classify_link(
    *,
    helper_semantic: str | None,
    accepted_cad_finger: bool,
    cad_object_name: str | None,
) -> str:
    """Return the evidence-bounded collision-semantics class."""
    if helper_semantic in {
        "VIRTUAL_KINEMATIC_HELPER",
        "FIXED_FRAME_ALIAS",
    }:
        return "VIRTUAL_FRAME_NO_COLLIDER"
    if accepted_cad_finger:
        return "PHYSICAL_EXISTING_VALIDATED_COLLIDER"
    if cad_object_name is not None:
        return "PHYSICAL_CAD_DERIVABLE"
    return "HARD_BLOCKER_CAD_TO_LINK_IDENTITY"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _link_suffix(link_name: str, robot_name: str) -> str:
    prefix = f"{robot_name}_"
    if not link_name.startswith(prefix):
        raise ValueError(f"link {link_name} does not use prefix {prefix}")
    return link_name[len(prefix) :]


def _determinant3(matrix: list[list[float]]) -> float:
    a, b, c = matrix[0][:3]
    d, e, f = matrix[1][:3]
    g, h, i = matrix[2][:3]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (
        d * h - e * g
    )


def _simple_viper_objects(
    cad_assembly_report: Path,
) -> dict[str, dict[str, Any]]:
    report = _load_json(cad_assembly_report)
    sources = [
        source
        for source in report["sources"]
        if source["source_label"] == "simple_viper"
    ]
    if len(sources) != 1:
        raise ValueError("expected exactly one simple_viper CAD source")
    return {item["name"]: item for item in sources[0]["objects"]}


def _finger_matrix(
    gripper_mapping: dict[str, Any],
    suffix: str,
) -> list[list[float]]:
    key = "left_matrix" if suffix == "left_finger_link" else "right_matrix"
    return gripper_mapping["cad_to_finger_link_mapping"][key]


def audit_follower_links(
    *,
    urdf_path: Path,
    robot_name: str,
    cad_assembly_report: Path,
    helper_report: Path,
    gripper_mapping_report: Path,
) -> list[dict[str, Any]]:
    """Audit every URDF link without authoring collision geometry."""
    resolved_urdf = urdf_path.resolve(strict=True)
    gripper_mapping = _load_json(gripper_mapping_report)
    cad_objects = _simple_viper_objects(cad_assembly_report)
    helper_records = audit_urdf_helper_links(resolved_urdf, robot_name)
    helper_by_suffix = {
        _link_suffix(name, robot_name): record
        for name, record in helper_records.items()
    }
    root = ET.parse(resolved_urdf).getroot()
    side = robot_name.removeprefix("follower_")
    evidence_paths = [
        str(resolved_urdf),
        str(cad_assembly_report.resolve(strict=True)),
        str(helper_report.resolve(strict=True)),
        str(gripper_mapping_report.resolve(strict=True)),
    ]
    links: list[dict[str, Any]] = []
    for link in root.findall("link"):
        link_name = str(link.get("name"))
        suffix = _link_suffix(link_name, robot_name)
        helper = helper_by_suffix.get(suffix)
        helper_semantic = helper["semantic_class"] if helper else None
        cad_object_name = CAD_OBJECT_BY_LINK_SUFFIX.get(suffix)
        accepted_finger = suffix in ACCEPTED_FINGER_SUFFIXES
        classification = classify_link(
            helper_semantic=helper_semantic,
            accepted_cad_finger=accepted_finger,
            cad_object_name=cad_object_name,
        )
        if classification not in ALLOWED_CLASSIFICATIONS:
            raise AssertionError(f"unexpected classification {classification}")

        visual_meshes = [
            str(mesh.get("filename"))
            for mesh in link.findall("./visual/geometry/mesh")
        ]
        collision_meshes = [
            str(mesh.get("filename"))
            for mesh in link.findall("./collision/geometry/mesh")
        ]
        cad_object = (
            cad_objects.get(cad_object_name)
            if cad_object_name is not None
            else None
        )
        if cad_object_name is not None and cad_object is None:
            raise ValueError(f"missing CAD object {cad_object_name}")
        source_placement_matrix = (
            cad_object["global_placement"]["matrix"]
            if cad_object is not None
            else None
        )
        cad_to_link_matrix = (
            _finger_matrix(gripper_mapping, suffix)
            if accepted_finger
            else None
        )
        if classification == "PHYSICAL_EXISTING_VALIDATED_COLLIDER":
            registration_status = "VERIFIED_EXISTING_SUPPLIER_CAD_REGISTRATION"
        elif classification == "PHYSICAL_CAD_DERIVABLE":
            registration_status = "PENDING_PHASE3_NUMERICAL_REGISTRATION"
        elif classification == "VIRTUAL_FRAME_NO_COLLIDER":
            registration_status = "NOT_APPLICABLE_VIRTUAL_FRAME"
        else:
            registration_status = "HARD_BLOCKER_CAD_TO_LINK_IDENTITY"

        links.append(
            {
                "robot": robot_name,
                "urdf_path": str(resolved_urdf),
                "urdf_link_name": link_name,
                "link_suffix": suffix,
                "usd_prim_path": (
                    f"/World/{robot_name}/vx300s_{side}/{link_name}"
                ),
                "visual_count": len(visual_meshes),
                "collision_count": len(collision_meshes),
                "existing_visual_meshes": visual_meshes,
                "existing_collision_meshes": collision_meshes,
                "cad_object": (
                    {
                        "name": cad_object["name"],
                        "label": cad_object["label"],
                        "shape_type": (cad_object.get("shape") or {}).get(
                            "shape_type"
                        ),
                        "shape_valid": (cad_object.get("shape") or {}).get(
                            "is_valid"
                        ),
                    }
                    if cad_object is not None
                    else None
                ),
                "source_placement_matrix": source_placement_matrix,
                "source_placement_determinant": (
                    _determinant3(source_placement_matrix)
                    if source_placement_matrix is not None
                    else None
                ),
                "cad_to_link_matrix": cad_to_link_matrix,
                "transform_determinant": (
                    _determinant3(cad_to_link_matrix)
                    if cad_to_link_matrix is not None
                    else None
                ),
                "unit_conversion_mm_to_m": 0.001,
                "mirror_used": False,
                "classification": classification,
                "registration_status": registration_status,
                "invent_collider_allowed": False,
                "helper_semantic_source_class": helper_semantic,
                "evidence_paths": evidence_paths,
                "unresolved": (
                    "CAD subpart identity for this independent URDF link is not "
                    "established by the supplier assembly audit."
                    if classification == "HARD_BLOCKER_CAD_TO_LINK_IDENTITY"
                    else None
                ),
            }
        )
    return links
