"""Pure per-issue scope classification for the ALOHA1 Task 7 validators."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

RULE_SOURCE = {
    "JointHasCorrectTransformAndState": {
        "file": "isaacsim/asset/validation/joint_rules.py",
        "line": 36,
        "callable": "JointHasCorrectTransformAndState.CheckPrim",
        "condition": (
            "For revolute/prismatic joints with valid body0/body1, compare the "
            "joint-state-adjusted body0 joint frame with the body1 joint frame."
        ),
    },
    "MimicAPICheck": {
        "file": "isaacsim/asset/validation/joint_rules.py",
        "line": 253,
        "callable": "MimicAPICheck.CheckPrim",
        "condition": (
            "For each PhysxMimicJointAPI instance, require one reference, finite "
            "mimic parameters, and limits accepted by the signed-gearing branch."
        ),
    },
    "RigidBodyHasCollider": {
        "file": "isaacsim/asset/validation/physics_rules.py",
        "line": 104,
        "callable": "RigidBodyHasCollider.CheckPrim",
        "condition": (
            "Every enabled UsdPhysics.RigidBodyAPI prim must contain at least one "
            "UsdPhysics.CollisionAPI prim in its traversed subtree."
        ),
    },
    "RigidBodyHasMassAPI": {
        "file": "isaacsim/asset/validation/physics_rules.py",
        "line": 29,
        "callable": "RigidBodyHasMassAPI.CheckStage/check_rigid_body_prim",
        "condition": (
            "Traverse every RigidBodyAPI prim and require MassAPI plus authored "
            "mass, diagonal inertia, and principal axes."
        ),
    },
    "NoOverrides": {
        "file": "isaacsim/asset/validation/robot_rules.py",
        "line": 108,
        "callable": "NoOverrides.CheckPrim",
        "condition": "Report authored attributes found through the robot asset layer stack.",
    },
    "RobotNaming": {
        "file": "isaacsim/asset/validation/robot_rules.py",
        "line": 25,
        "callable": "RobotNaming.CheckStage",
        "condition": (
            "Require <Manufacturer>/<robot>/<robot.usd> or "
            "<Manufacturer>/<robot>/<version>/<robot.usd> naming."
        ),
    },
    "RobotSchema": {
        "file": "isaacsim/asset/validation/robot_rules.py",
        "line": 129,
        "callable": "RobotSchema.CheckStage",
        "condition": (
            "Require the robot asset default prim to have RobotAPI and non-missing "
            "robotLinks and robotJoints relationships."
        ),
    },
}


def _asset_owner(at: str | None) -> str:
    value = at or ""
    if "/follower_left" in value or "/follower_right" in value:
        return "robot"
    if "/environment/worldBody/floor" in value:
        return "table"
    if "/environment/worldBody/" in value:
        return "diagnostic layer"
    return "workcell wrapper"


def _classification(family: str, issue: Mapping[str, Any]) -> tuple[str, bool, str]:
    rule = str(issue.get("rule"))
    message = str(issue.get("message") or "")
    if family == "IsaacSim.RobotRules":
        return (
            "WRONG_SCOPE",
            False,
            "The installed rule source repeatedly calls the input a robot asset; "
            "the audited input is a two-robot composed workcell. Standalone robot "
            "candidates are the applicable targets.",
        )
    if family != "IsaacSim.PhysicsRules":
        raise ValueError(f"unsupported rule family: {family}")
    if rule == "MimicAPICheck":
        return (
            "INCONCLUSIVE",
            True,
            "The literal signed-limit error remains on both standalone robots, "
            "while the validated runtime mimic motion is symmetric. No source-backed "
            "mimic mutation is authorized.",
        )
    if rule == "RigidBodyHasMassAPI" and (issue.get("at") is None or "Uncaught error" in message):
        return (
            "INCONCLUSIVE",
            True,
            "Asset Validation 1.1.0 raised an internal exception without a target prim; "
            "the affected prim cannot be attributed from this issue record.",
        )
    if rule in {
        "JointHasCorrectTransformAndState",
        "RigidBodyHasCollider",
        "RigidBodyHasMassAPI",
    }:
        return (
            "TRUE_ASSET_DEFECT",
            True,
            "The installed rule is applicable to this enabled body/joint and the "
            "literal finding remains unsuppressed on the correct physical target.",
        )
    raise ValueError(f"unclassified PhysicsRules issue: {rule}")


def audit_blocking_issues(
    *,
    family: str,
    target: str,
    issues: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return one auditable row for every blocking validator issue."""

    rows: list[dict[str, Any]] = []
    for source_index, issue in enumerate(issues):
        if issue.get("severity") not in {"ERROR", "FAILURE"}:
            continue
        rule = str(issue.get("rule"))
        if rule not in RULE_SOURCE:
            raise ValueError(f"missing installed-source mapping for rule: {rule}")
        classification, blocking, evidence = _classification(family, issue)
        rows.append(
            {
                "source_index": source_index,
                "rule_family": family,
                "rule_name": rule,
                "official_source": dict(RULE_SOURCE[rule]),
                "checked_usd": target,
                "target_prim_path": issue.get("at"),
                "asset_owner": _asset_owner(issue.get("at")),
                "applicable_to_current_target": family == "IsaacSim.PhysicsRules",
                "classification": classification,
                "task7_blocking": blocking,
                "literal_severity": issue.get("severity"),
                "literal_message": issue.get("message"),
                "evidence": evidence,
            }
        )
    return rows


def summarize_audit(
    *,
    robot_rows: Sequence[Mapping[str, Any]],
    physics_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the complete historic issue inventory and summarize it."""

    if len(robot_rows) != 63:
        raise ValueError(f"expected 63 RobotRules rows, got {len(robot_rows)}")
    if len(physics_rows) != 26:
        raise ValueError(f"expected 26 PhysicsRules rows, got {len(physics_rows)}")
    rows = [*robot_rows, *physics_rows]
    return {
        "original_robot_blocking_count": len(robot_rows),
        "original_physics_blocking_count": len(physics_rows),
        "total_classified": len(rows),
        "classification_counts": dict(sorted(Counter(row["classification"] for row in rows).items())),
        "task7_blocking_count": sum(bool(row["task7_blocking"]) for row in rows),
    }
