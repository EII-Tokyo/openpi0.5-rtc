"""Pure validation planning and status classification for ALOHA 1."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

import yaml


def build_validation_plan(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    joint_map = yaml.safe_load((root / "configs/aloha1_joint_map.yaml").read_text(encoding="utf-8"))
    workcell = root / "assets/Trossen/ALOHA1/1.0/workcell/aloha1_workcell.usd"
    if not workcell.is_file():
        raise FileNotFoundError(f"workcell is unavailable: {workcell}")
    robots = [
        {
            "name": name,
            "articulation_prim": (f"/aloha1_workcell/Robots/{name}/root_joint"),
            "robot_prim": f"/aloha1_workcell/Robots/{name}",
            "source_robot_asset": str(
                (root / "assets/Trossen/ALOHA1/1.0/follower_vx300s" / name / f"{name}.usd").resolve()
            ),
            "dof_order": joint_map["robots"][name]["isaac_dof_order"],
            "home_si": [
                0.0,
                -0.96,
                1.16,
                0.0,
                -0.3,
                0.0,
                0.0,
                0.02239,
                -0.02239,
            ],
        }
        for name in ("follower_left", "follower_right")
    ]
    return {
        "schema_version": 1,
        "workcell": str(workcell.resolve()),
        "expected_articulation_count": 2,
        "robots": robots,
        "official_rule_categories": [
            "IsaacSim.PhysicsRules",
            "IsaacSim.RobotRules",
            "IsaacSim.SimReadyAssetRules",
        ],
        "gripper_validation_report": str((root / "reports/aloha1_mapping/gripper_validation.json").resolve()),
        "required_task5_gripper_statuses": ["PASS", "PARTIAL"],
        "runtime": {
            "physics_dt_s": 1.0 / 60.0,
            "first_frame_jump_tolerance": 0.02,
            "static_steps": 120,
            "static_position_tolerance": 0.03,
            "one_joint_steps": 60,
            "revolute_delta_rad": 0.05,
            "prismatic_delta_m": 0.002,
            "readback_minimum": 1.0e-4,
        },
    }


def classify_validation(
    checks: Sequence[dict[str, Any]],
    hard_blockers: Sequence[str],
) -> str:
    if any(check.get("status") == "FAIL" for check in checks):
        return "FAIL"
    if hard_blockers or any(check.get("status") == "PARTIAL" for check in checks):
        return "PARTIAL"
    return "PASS"


def load_required_machine_report(
    path: Path,
    *,
    name: str,
    accepted_statuses: Sequence[str],
) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        return {
            "name": name,
            "status": "FAIL",
            "evidence": {
                "path": str(resolved),
                "reported_status": "MISSING",
                "accepted_statuses": list(accepted_statuses),
                "report": None,
            },
        }
    report = json.loads(resolved.read_text(encoding="utf-8"))
    reported_status = str(report.get("status", "MISSING"))
    return {
        "name": name,
        "status": ("PASS" if reported_status in accepted_statuses else "FAIL"),
        "evidence": {
            "path": str(resolved),
            "reported_status": reported_status,
            "accepted_statuses": list(accepted_statuses),
            "report": report,
        },
    }
