"""Evidence-backed physics configuration plan for Stationary ALOHA 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.joint_map import _control_source_path
from tools.aloha1_mapping.joint_map import _literal_assignments
from tools.aloha1_mapping.joint_map import build_joint_map


def build_physics_plan(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    joint_map = build_joint_map(root)
    constants = _literal_assignments(_control_source_path(root))
    start = constants["START_ARM_POSE"]
    robots = []
    for robot_index, name in enumerate(("follower_left", "follower_right")):
        source_home = start[robot_index * 8 : robot_index * 8 + 8]
        home = source_home[:6] + [0.0] + source_home[6:8]
        robot_map = joint_map["robots"][name]
        base_dir = (
            root
            / "assets/Trossen/ALOHA1/1.0/follower_vx300s"
            / name
        )
        robots.append(
            {
                "name": name,
                "base_usd": str((base_dir / f"{name}.usd").resolve()),
                "profile_dir": str((base_dir / "configuration").resolve()),
                "home_si": home,
                "home_source": (
                    "Physical-Intelligence/aloha constants.py START_ARM_POSE; "
                    "gripper motor DOF has no entry and remains at 0 rad"
                ),
                "dofs": [
                    {
                        "name": dof["name"],
                        "joint_type": dof["joint_type"],
                        "home_si": home[dof["isaac_index"]],
                        "velocity_limit_si": dof["velocity_limit"],
                        "max_force": dof["effort_max_force"],
                        "mimic": dof["mimic"] is not None,
                        "author_drive": dof["mimic"] is None,
                    }
                    for dof in robot_map["dofs"]
                ],
            }
        )
    return {
        "schema_version": 1,
        "status": "PARTIAL",
        "default_profile": "debug_acceleration_drive",
        "profiles": {
            "debug_acceleration_drive": {
                "drive_type": "acceleration",
                "target_type": "position",
                "gain_policy": "preserve_isaac_5_1_importer_authored_values",
                "status": "INTERFACE_DEBUG_ONLY",
                "dynamics_fidelity_claim": False,
            },
            "sim2real_force_drive": {
                "drive_type": "force",
                "target_type": "position",
                "gain_policy": "temporary_copy_of_importer_authored_values",
                "status": "CALIBRATION_PENDING",
                "dynamics_fidelity_claim": False,
                "hard_blocker": (
                    "measured mass/inertia, motor response, friction, and Gain "
                    "Tuner evidence are unavailable"
                ),
            },
        },
        "fingertip_material": {
            "status": "TEMPORARY_PLACEHOLDER",
            "static_friction": 0.5,
            "dynamic_friction": 0.5,
            "restitution": 0.0,
            "hard_blocker": "measured Stationary ALOHA 1 fingertip friction",
        },
        "robots": robots,
    }


def build_missing_dynamics_report(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    urdf_audit = json.loads(
        (root / "reports/aloha1_mapping/urdf_audit.json").read_text(
            encoding="utf-8"
        )
    )
    links = [
        {
            "robot": robot["robot_name"],
            "link": dynamics["link"],
            "urdf_values_present": True,
            "mass": dynamics["mass"],
            "center_of_mass_xyz": dynamics["center_of_mass_xyz"],
            "inertia": dynamics["inertia"],
            "measurement_status": "HARD_BLOCKER",
            "missing_evidence": [
                "physical mass measurement",
                "physical center-of-mass measurement",
                "physical inertia measurement or identified CAD",
            ],
        }
        for robot in urdf_audit["robots"]
        for dynamics in robot["dynamics"]
    ]
    return {
        "schema_version": 1,
        "status": "PARTIAL",
        "default_density_used": False,
        "statement": (
            "URDF dynamics are syntactically complete and imported, but no "
            "Stationary ALOHA 1 measurement evidence was found; values are "
            "not declared calibrated."
        ),
        "links": links,
    }
