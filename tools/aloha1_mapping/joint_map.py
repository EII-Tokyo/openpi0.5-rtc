"""Build an explicit Stationary ALOHA 1 joint mapping from audited evidence."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

ARM_JOINTS = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]
ROS_EXTRA_JOINTS = ["gripper", "left_finger", "right_finger"]


def _control_source_path(project_root: Path) -> Path:
    specifications = json.loads(
        (project_root / "configs/aloha1_source_audit_paths.json").read_text(
            encoding="utf-8"
        )
    )
    scripts_root = next(
        Path(item["root"])
        for item in specifications
        if item.get("role_prefix")
        == "physical_intelligence_aloha_control_or_data_code"
    )
    return scripts_root / "constants.py"


def _literal_assignments(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments: dict[str, Any] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                assignments[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    return assignments


def _urdf_nonfixed_joints(path: Path) -> list[dict[str, Any]]:
    root = ET.parse(path).getroot()
    result: list[dict[str, Any]] = []
    for joint in root.findall("joint"):
        joint_type = joint.attrib["type"]
        if joint_type == "fixed":
            continue
        limit = joint.find("limit")
        mimic = joint.find("mimic")
        axis = joint.find("axis")
        result.append(
            {
                "name": joint.attrib["name"],
                "type": joint_type,
                "axis": (
                    [float(item) for item in axis.attrib["xyz"].split()]
                    if axis is not None
                    else None
                ),
                "position_limit": {
                    "lower": (
                        float(limit.attrib["lower"])
                        if limit is not None and "lower" in limit.attrib
                        else None
                    ),
                    "upper": (
                        float(limit.attrib["upper"])
                        if limit is not None and "upper" in limit.attrib
                        else None
                    ),
                },
                "velocity_limit": float(limit.attrib["velocity"]),
                "effort_limit": float(limit.attrib["effort"]),
                "mimic": (
                    {
                        "parent": mimic.attrib["joint"],
                        "multiplier": float(mimic.attrib.get("multiplier", "1")),
                        "offset": float(mimic.attrib.get("offset", "0")),
                    }
                    if mimic is not None
                    else None
                ),
            }
        )
    return result


def _source_record(manifest: dict[str, Any], path: Path) -> dict[str, Any]:
    resolved = str(path.resolve())
    source = next(
        item for item in manifest["sources"] if item["local_path"] == resolved
    )
    return {
        "local_path": resolved,
        "sha256": source["sha256"],
        "repository": source["repository"],
        "license": source["license"],
    }


def _dataset_indices(side: str, name: str) -> tuple[int | None, int | None]:
    base = 0 if side == "left" else 7
    if name in ARM_JOINTS:
        index = base + ARM_JOINTS.index(name)
        return index, index
    if name == "gripper":
        return None, base + 6
    if name in {"left_finger", "right_finger"}:
        return base + 6, base + 6
    return None, None


def _dataset_transform(name: str) -> dict[str, Any] | None:
    if name in ARM_JOINTS:
        return {"sign": 1.0, "offset": 0.0, "scale": 1.0, "unit": "rad"}
    if name == "gripper":
        return {
            "sign": 1.0,
            "offset": -0.6213,
            "scale": 2.1123,
            "unit": "rad",
            "input": "normalized action, 0=closed, 1=open",
        }
    if name == "left_finger":
        return {
            "sign": 1.0,
            "offset": 0.021,
            "scale": 0.036,
            "unit": "m",
            "input": "normalized state/action, 0=closed, 1=open",
            "status": "engineering_mapping_to_urdf_limits",
        }
    if name == "right_finger":
        return {
            "sign": -1.0,
            "offset": -0.021,
            "scale": -0.036,
            "unit": "m",
            "input": "normalized state/action, 0=closed, 1=open",
            "status": "derived_from_mimic",
        }
    return None


def build_joint_map(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    runtime_report_path = (
        root / "reports/aloha1_mapping/usd_dof_inventory.json"
    )
    runtime_report = json.loads(runtime_report_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(
        (root / "reports/aloha1_mapping/source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    constants_path = _control_source_path(root)
    constants = _literal_assignments(constants_path)
    if constants["JOINT_NAMES"] != ARM_JOINTS:
        raise ValueError(
            f"unexpected ALOHA control joint order: {constants['JOINT_NAMES']}"
        )
    ros_order = list(constants["JOINT_NAMES"]) + ROS_EXTRA_JOINTS
    robots: dict[str, Any] = {}
    for robot_runtime in runtime_report["robots"]:
        name = robot_runtime["name"]
        side = "left" if name.endswith("_left") else "right"
        urdf_path = root / "generated/urdf" / f"{name}.urdf"
        urdf_joints = _urdf_nonfixed_joints(urdf_path)
        urdf_order = [item["name"] for item in urdf_joints]
        isaac_order = robot_runtime["dof_order"]
        if urdf_order != isaac_order:
            raise ValueError(
                f"{name} URDF/Isaac order mismatch: "
                f"{urdf_order} != {isaac_order}"
            )
        if ros_order != isaac_order:
            raise ValueError(
                f"{name} ROS/Isaac order mismatch: {ros_order} != {isaac_order}"
            )
        urdf_by_name = {item["name"]: item for item in urdf_joints}
        runtime_by_name = {
            item["name"]: item for item in robot_runtime["dofs"]
        }
        dofs = []
        for index, dof_name in enumerate(isaac_order):
            urdf = urdf_by_name[dof_name]
            runtime = runtime_by_name[dof_name]
            state_index, action_index = _dataset_indices(side, dof_name)
            dofs.append(
                {
                    "name": dof_name,
                    "isaac_index": index,
                    "ros_index": ros_order.index(dof_name),
                    "dataset_state_index": state_index,
                    "dataset_action_index": action_index,
                    "dataset_mapping": _dataset_transform(dof_name),
                    "joint_type": urdf["type"],
                    "axis": urdf["axis"],
                    "position_limit": urdf["position_limit"],
                    "velocity_limit": urdf["velocity_limit"],
                    "effort_max_force": urdf["effort_limit"],
                    "mimic": urdf["mimic"],
                    "isaac_runtime": {
                        key: runtime[key]
                        for key in (
                            "lower",
                            "upper",
                            "max_velocity",
                            "max_effort",
                            "drive_mode",
                            "stiffness",
                            "damping",
                        )
                    },
                }
            )
        robots[name] = {
            "side": side,
            "urdf_nonfixed_joint_order": urdf_order,
            "isaac_dof_order": isaac_order,
            "ros_joint_state_order": list(ros_order),
            "dataset_order": (
                [f"{side}_{joint}" for joint in ARM_JOINTS]
                + [f"{side}_gripper_normalized"]
            ),
            "dofs": dofs,
        }
    return {
        "schema_version": 1,
        "status": "PARTIAL",
        "scope": "Stationary ALOHA 1 followers",
        "order_policy": "explicit_source_order_never_alphabetical",
        "sources": {
            "urdf_generation_report": str(
                (root / "reports/aloha1_mapping/urdf_generation_manifest.json")
                .resolve()
            ),
            "isaac_runtime_report": str(runtime_report_path.resolve()),
            "control_constants": _source_record(
                source_manifest, constants_path
            ),
        },
        "dataset_14d_order": (
            [f"left_{joint}" for joint in ARM_JOINTS]
            + ["left_gripper_normalized"]
            + [f"right_{joint}" for joint in ARM_JOINTS]
            + ["right_gripper_normalized"]
        ),
        "openpi_internal_sign_flip_mask": [
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
        ],
        "gripper": {
            "calibration_status": "HARD_BLOCKER",
            "reason": (
                "real observed finger endpoints differ from URDF finger limits; "
                "the motor-angle to aperture relation has not been measured on "
                "this Stationary ALOHA 1"
            ),
            "real_observed_finger_position_m": {
                "closed": constants["PUPPET_GRIPPER_POSITION_CLOSE"],
                "open": constants["PUPPET_GRIPPER_POSITION_OPEN"],
                "source_semantics": "real_env.py reads joint_states position[7]",
            },
            "real_command_motor_angle_rad": {
                "closed": constants["PUPPET_GRIPPER_JOINT_CLOSE"],
                "open": constants["PUPPET_GRIPPER_JOINT_OPEN"],
            },
            "urdf_left_finger_limit_m": {
                "closed": 0.021,
                "open": 0.057,
            },
            "temporary_simulation_policy": (
                "map normalized aperture linearly to URDF finger limits and "
                "hold the gripper motor DOF at the corresponding source "
                "endpoint; do not claim sim-to-real calibration"
            ),
        },
        "robots": robots,
    }
