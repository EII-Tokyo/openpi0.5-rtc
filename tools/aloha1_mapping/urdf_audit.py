#!/usr/bin/env python3
"""Static, source-order-preserving URDF audit for Stationary ALOHA 1."""

from __future__ import annotations

from collections import Counter
from collections import defaultdict
from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
from typing import Any
from urllib.parse import unquote
from urllib.parse import urlparse
import xml.etree.ElementTree as ET


def _issue(
    issues: list[dict[str, str]],
    code: str,
    message: str,
) -> None:
    issues.append({"severity": "ERROR", "code": code, "message": message})


def _finite_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_mesh(
    uri: str,
    *,
    urdf_path: Path,
    package_map: Mapping[str, Path],
) -> tuple[Path | None, str | None]:
    if uri.startswith("package://"):
        remainder = uri.removeprefix("package://")
        package_name, separator, relative = remainder.partition("/")
        if not separator or package_name not in package_map:
            return None, "UNRESOLVED_PACKAGE_URI"
        return (package_map[package_name] / relative).resolve(), None
    if uri.startswith("file://"):
        parsed = urlparse(uri)
        return Path(unquote(parsed.path)).resolve(), None
    candidate = Path(uri)
    if not candidate.is_absolute():
        candidate = urdf_path.parent / candidate
    return candidate.resolve(), None


def _audit_dynamics(
    links: list[ET.Element],
    *,
    issues: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    missing: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    inertia_fields = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
    for link in links:
        name = link.get("name", "")
        inertial = link.find("inertial")
        if inertial is None:
            missing.append({"link": name, "missing": ["inertial"]})
            _issue(issues, "MISSING_INERTIAL", f"link {name!r} has no inertial")
            continue
        missing_fields: list[str] = []
        origin = inertial.find("origin")
        mass = inertial.find("mass")
        mass_value = _finite_float(mass.get("value") if mass is not None else None)
        if mass_value is None or mass_value <= 0:
            missing_fields.append("mass")
        inertia = inertial.find("inertia")
        if inertia is None:
            missing_fields.append("inertia")
        else:
            missing_fields.extend(
                f"inertia.{field}"
                for field in inertia_fields
                if _finite_float(inertia.get(field)) is None
            )
        origin_xyz = [
            float(value)
            for value in (
                origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            ).split()
        ]
        origin_rpy = [
            float(value)
            for value in (
                origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
            ).split()
        ]
        inventory.append(
            {
                "link": name,
                "origin_explicit": origin is not None,
                "center_of_mass_xyz": origin_xyz,
                "inertial_origin_rpy": origin_rpy,
                "mass": mass_value,
                "inertia": {
                    field: (
                        _finite_float(inertia.get(field))
                        if inertia is not None
                        else None
                    )
                    for field in inertia_fields
                },
            }
        )
        if missing_fields:
            missing.append({"link": name, "missing": missing_fields})
            _issue(
                issues,
                "INVALID_INERTIAL",
                f"link {name!r} has invalid dynamics fields: {missing_fields}",
            )
    return missing, inventory


def _audit_tree(
    link_names: list[str],
    joints: list[ET.Element],
    *,
    issues: list[dict[str, str]],
) -> list[str]:
    link_set = set(link_names)
    parent_joints: dict[str, list[str]] = defaultdict(list)
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    for joint in joints:
        name = joint.get("name", "")
        parent_element = joint.find("parent")
        child_element = joint.find("child")
        parent = parent_element.get("link") if parent_element is not None else None
        child = child_element.get("link") if child_element is not None else None
        if not parent or parent not in link_set:
            _issue(
                issues,
                "INVALID_PARENT_LINK",
                f"joint {name!r} has unknown or missing parent {parent!r}",
            )
        if not child or child not in link_set:
            _issue(
                issues,
                "INVALID_CHILD_LINK",
                f"joint {name!r} has unknown or missing child {child!r}",
            )
        if parent and child:
            parent_joints[child].append(name)
            children_by_parent[parent].append(child)
    for child, names in sorted(parent_joints.items()):
        if len(names) > 1:
            _issue(
                issues,
                "MULTIPLE_PARENT_JOINTS",
                f"link {child!r} has parent joints {names}",
            )
    roots = [name for name in link_names if name not in parent_joints]
    if len(roots) != 1:
        _issue(
            issues,
            "INVALID_ROOT_COUNT",
            f"expected one root link, found {roots}",
        )

    state: dict[str, int] = {}

    def visit(link: str) -> None:
        if state.get(link) == 1:
            _issue(issues, "KINEMATIC_CYCLE", f"cycle reaches link {link!r}")
            return
        if state.get(link) == 2:
            return
        state[link] = 1
        for child in children_by_parent.get(link, []):
            visit(child)
        state[link] = 2

    for root in roots:
        visit(root)
    unreachable = sorted(link_set - set(state))
    if unreachable:
        _issue(
            issues,
            "DISCONNECTED_LINKS",
            f"links are disconnected from roots: {unreachable}",
        )
    return roots


def _audit_joint(
    joint: ET.Element,
    *,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    name = joint.get("name", "")
    joint_type = joint.get("type", "")
    origin = joint.find("origin")
    axis = joint.find("axis")
    limit = joint.find("limit")
    mimic = joint.find("mimic")
    if origin is None:
        _issue(
            issues,
            "MISSING_JOINT_ORIGIN",
            f"joint {name!r} has no explicit origin",
        )
    if joint_type != "fixed":
        if axis is None or not axis.get("xyz"):
            _issue(
                issues,
                "MISSING_JOINT_AXIS",
                f"joint {name!r} has no explicit axis",
            )
        if limit is None:
            _issue(
                issues,
                "MISSING_JOINT_LIMIT",
                f"joint {name!r} has no explicit limit",
            )
        else:
            lower = _finite_float(limit.get("lower"))
            upper = _finite_float(limit.get("upper"))
            effort = _finite_float(limit.get("effort"))
            velocity = _finite_float(limit.get("velocity"))
            if joint_type != "continuous" and (
                lower is None or upper is None or lower > upper
            ):
                _issue(
                    issues,
                    "INVALID_POSITION_LIMIT",
                    f"joint {name!r} has invalid lower/upper limits",
                )
            if effort is None or effort <= 0:
                _issue(
                    issues,
                    "INVALID_EFFORT_LIMIT",
                    f"joint {name!r} effort must be finite and positive",
                )
            if velocity is None or velocity <= 0:
                _issue(
                    issues,
                    "INVALID_VELOCITY_LIMIT",
                    f"joint {name!r} velocity must be finite and positive",
                )
    parent_element = joint.find("parent")
    child_element = joint.find("child")
    return {
        "name": name,
        "type": joint_type,
        "parent": (
            parent_element.get("link") if parent_element is not None else None
        ),
        "child": (
            child_element.get("link") if child_element is not None else None
        ),
        "axis": axis.get("xyz") if axis is not None else None,
        "origin_xyz": origin.get("xyz") if origin is not None else None,
        "origin_rpy": origin.get("rpy") if origin is not None else None,
        "lower": (
            _finite_float(limit.get("lower")) if limit is not None else None
        ),
        "upper": (
            _finite_float(limit.get("upper")) if limit is not None else None
        ),
        "effort": (
            _finite_float(limit.get("effort")) if limit is not None else None
        ),
        "velocity": (
            _finite_float(limit.get("velocity")) if limit is not None else None
        ),
        "mimic_parent": mimic.get("joint") if mimic is not None else None,
        "mimic_multiplier": (
            _finite_float(mimic.get("multiplier", "1"))
            if mimic is not None
            else None
        ),
        "mimic_offset": (
            _finite_float(mimic.get("offset", "0"))
            if mimic is not None
            else None
        ),
    }


def audit_urdf(
    urdf_path: Path,
    *,
    package_map: Mapping[str, Path],
) -> dict[str, Any]:
    resolved_urdf = urdf_path.resolve(strict=True)
    issues: list[dict[str, str]] = []
    try:
        root = ET.parse(resolved_urdf).getroot()
    except ET.ParseError as error:
        return {
            "status": "FAIL",
            "urdf_path": str(resolved_urdf),
            "issues": [
                {
                    "severity": "ERROR",
                    "code": "INVALID_XML",
                    "message": str(error),
                }
            ],
        }
    if root.tag != "robot":
        _issue(issues, "INVALID_ROOT_ELEMENT", "URDF root must be <robot>")

    links = list(root.findall("link"))
    joints = list(root.findall("joint"))
    link_order = [element.get("name", "") for element in links]
    joint_order = [element.get("name", "") for element in joints]
    for label, names in (("LINK", link_order), ("JOINT", joint_order)):
        for name, count in sorted(Counter(names).items()):
            if not name:
                _issue(issues, f"MISSING_{label}_NAME", f"{label} name is empty")
            elif count > 1:
                _issue(
                    issues,
                    f"DUPLICATE_{label}_NAME",
                    f"{label.lower()} {name!r} occurs {count} times",
                )

    roots = _audit_tree(link_order, joints, issues=issues)
    joint_inventory = [
        _audit_joint(joint, issues=issues) for joint in joints
    ]
    missing_dynamics, dynamics = _audit_dynamics(links, issues=issues)

    meshes: list[dict[str, Any]] = []
    for link in links:
        link_name = link.get("name", "")
        for mesh in link.findall("./visual/geometry/mesh") + link.findall(
            "./collision/geometry/mesh"
        ):
            uri = mesh.get("filename", "")
            resolved, resolution_error = _resolve_mesh(
                uri,
                urdf_path=resolved_urdf,
                package_map=package_map,
            )
            exists = bool(resolved and resolved.is_file())
            record = {
                "link": link_name,
                "uri": uri,
                "resolved_path": str(resolved) if resolved else None,
                "exists": exists,
                "sha256": _sha256(resolved) if exists and resolved else None,
            }
            meshes.append(record)
            if resolution_error:
                _issue(
                    issues,
                    resolution_error,
                    f"mesh URI {uri!r} cannot be resolved",
                )
            elif not exists:
                _issue(
                    issues,
                    "MISSING_MESH",
                    f"mesh URI {uri!r} resolves to missing path {resolved}",
                )

    mimic = [
        {
            "joint": item["name"],
            "parent": item["mimic_parent"],
            "multiplier": item["mimic_multiplier"],
            "offset": item["mimic_offset"],
        }
        for item in joint_inventory
        if item["mimic_parent"] is not None
    ]
    return {
        "status": "FAIL" if issues else "PASS",
        "urdf_path": str(resolved_urdf),
        "robot_name": root.get("name"),
        "link_order": link_order,
        "joint_order": joint_order,
        "root_links": roots,
        "joints": joint_inventory,
        "mimic": mimic,
        "meshes": meshes,
        "dynamics": dynamics,
        "missing_dynamics": missing_dynamics,
        "issues": issues,
    }
