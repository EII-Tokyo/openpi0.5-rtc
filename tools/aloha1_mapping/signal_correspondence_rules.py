"""Classify Isaac Sim 5.1 official rules for the Task 7A boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_PACKAGING_RULES = {
    "NoOverrides",
    "RobotNaming",
    "RobotSchema",
    "ThumbnailExists",
    "VerifyRobotPhysicsAttributesSourceLayer",
    "VerifyRobotPhysicsSchemaSourceLayer",
}


def classify_official_issue_for_task7a(
    issue: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve the official result while defining its Task 7A relevance."""
    rule = str(issue.get("rule"))
    location = str(issue.get("at"))
    result = {
        **dict(issue),
        "official_severity": issue.get("severity"),
        "official_result_suppressed": False,
    }
    if rule == "MimicAPICheck" and location.endswith("/joints/right_finger>"):
        result.update(
            {
                "classification": ("ISAAC_SIM_5_1_MIMIC_VALIDATOR_SCHEMA_CONFLICT"),
                "scope": "TASK7A_CONTROL_SIGNAL",
                "task7a_status": "PARTIAL",
                "runtime_evidence_required": ("right_finger ~= -left_finger with zero active drive"),
            }
        )
    elif rule == "JointHasJointStateAPI" and location.endswith("/joints/gripper>"):
        result.update(
            {
                "classification": ("GRIPPER_JOINT_STATE_SCHEMA_OMISSION"),
                "scope": "TASK7A_CONTROL_SIGNAL",
                "task7a_status": "PARTIAL",
                "runtime_evidence_required": ("gripper target/readback one-joint validation"),
            }
        )
    elif rule == "RigidBodyHasCollider" and any(
        location.endswith(suffix)
        for suffix in (
            "_ee_arm_link>",
            "_ee_gripper_link>",
            "_fingers_link>",
        )
    ):
        result.update(
            {
                "classification": ("SOURCE_MASS_ONLY_LINK_WITHOUT_GEOMETRY"),
                "scope": "TASK7B_GEOMETRY_NOT_RUNTIME_SIGNAL",
                "task7a_status": "PARTIAL",
                "runtime_evidence_required": None,
            }
        )
    elif rule in _PACKAGING_RULES:
        result.update(
            {
                "classification": "ROBOT_SCHEMA_OR_LAYER_PACKAGING",
                "scope": "TASK7B_PACKAGING_NOT_RUNTIME_SIGNAL",
                "task7a_status": "PARTIAL",
                "runtime_evidence_required": None,
            }
        )
    elif issue.get("severity") in {"INFO", "WARNING"}:
        result.update(
            {
                "classification": "OFFICIAL_INFORMATION_OR_WARNING",
                "scope": "RECORDED",
                "task7a_status": "PARTIAL",
                "runtime_evidence_required": None,
            }
        )
    else:
        result.update(
            {
                "classification": "UNCLASSIFIED_TASK7A_RELEVANCE",
                "scope": "TASK7A_REVIEW_REQUIRED",
                "task7a_status": "FAIL",
                "runtime_evidence_required": None,
            }
        )
    return result


def combine_official_rule_fragments(
    fragments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine fresh single-target runs without hiding official failures."""
    targets = []
    classified_issues = []
    for fragment in fragments:
        issues = [classify_official_issue_for_task7a(issue) for issue in fragment.get("issues", [])]
        classified_issues.extend(issues)
        targets.append({**dict(fragment), "classified_issues": issues})
    official_status = (
        "FAIL"
        if any(item.get("official_status") == "FAIL" for item in fragments)
        else ("PARTIAL" if any(item.get("official_status") == "PARTIAL" for item in fragments) else "PASS")
    )
    task7a_status = (
        "FAIL"
        if any(item["task7a_status"] == "FAIL" for item in classified_issues)
        else "PARTIAL"
        if classified_issues
        else "PASS"
    )
    return {
        "schema_version": 1,
        "official_status": official_status,
        "official_status_suppressed": False,
        "task7a_applicable_status": task7a_status,
        "targets": targets,
        "classified_issue_count": len(classified_issues),
        "unclassified_task7a_issue_count": sum(
            item["classification"] == "UNCLASSIFIED_TASK7A_RELEVANCE" for item in classified_issues
        ),
    }
