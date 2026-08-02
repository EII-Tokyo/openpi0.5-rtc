from __future__ import annotations

import pytest

from tools.aloha1_mapping.task7_rule_candidates import classify_candidate_finding
from tools.aloha1_mapping.task7_rule_candidates import classify_rule_targets
from tools.aloha1_mapping.task7_rule_candidates import validate_candidate_contract
from tools.aloha1_mapping.task7_rule_scope_audit import audit_blocking_issues
from tools.aloha1_mapping.task7_rule_scope_audit import summarize_audit
from tools.aloha1_mapping.task7_validator_controls import fresh_runs_match
from tools.aloha1_mapping.task7_validator_controls import issue_signature
from tools.aloha1_mapping.task7_validator_controls import validate_negative_delta
from tools.validate_aloha1_signal_correspondence_official_rules import filter_rule_classes


def test_rule_targets_separate_robot_packages_from_workcell() -> None:
    scopes = classify_rule_targets(
        local_asset_validation_version="1.1.0",
        physics_rules_traverse_entire_stage=True,
        robot_rules_require_robot_default_prim=True,
    )

    assert scopes["IsaacSim.PhysicsRules"] == "STANDALONE_PHYSICAL_ROBOT_ASSET"
    assert scopes["IsaacSim.RobotRules"] == "STANDALONE_ROBOT_PACKAGE"
    assert scopes["IsaacSim.SimReadyAssetRules"] == "COMPOSED_REVIEW_STAGE"


def test_candidate_contract_rejects_hidden_geometry_or_physics_mutation() -> None:
    base = {
        "protected_source_unchanged": True,
        "references_exact_frozen_robot_subtree": True,
        "visual_mesh_count_matches_source": True,
        "collision_prim_count_matches_source": True,
        "joint_body_targets_remapped": True,
        "dof_order_matches_source": True,
        "joint_or_drive_modified": False,
        "mimic_modified": False,
        "mass_or_inertia_modified": False,
        "collider_modified": False,
        "geometry_removed": False,
        "task8": "NOT_RUN",
    }
    assert validate_candidate_contract(base) == "PASS"

    for key in (
        "collider_modified",
        "joint_or_drive_modified",
        "mimic_modified",
        "mass_or_inertia_modified",
        "geometry_removed",
    ):
        invalid = dict(base)
        invalid[key] = True
        with pytest.raises(ValueError, match=key):
            validate_candidate_contract(invalid)


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("JointHasCorrectTransformAndState", "UNRESOLVED"),
        ("RigidBodyHasCollider", "SOURCE_EVIDENCE_BLOCKER"),
        ("RigidBodyHasMassAPI", "SOURCE_EVIDENCE_BLOCKER"),
        ("MimicAPICheck", "LOCAL_VALIDATOR_SCHEMA_CONFLICT"),
        ("NoOverrides", "LOCAL_VALIDATOR_SCHEMA_CONFLICT"),
        ("RobotSchema", "FIXED_WITH_OFFICIAL_5_1_PACKAGE_CHANGE"),
        ("RobotNaming", "FIXED_WITH_OFFICIAL_5_1_PACKAGE_CHANGE"),
    ],
)
def test_current_rule_findings_are_not_suppressed(
    rule: str,
    expected: str,
) -> None:
    assert classify_candidate_finding(rule) == expected


def test_unknown_rule_cannot_be_silently_classified() -> None:
    with pytest.raises(ValueError, match="unclassified rule"):
        classify_candidate_finding("ConvenientGreenReportRule")


def test_workcell_robot_rules_are_classified_per_issue_as_wrong_scope() -> None:
    issues = [
        {
            "severity": "ERROR",
            "rule": "RobotSchema",
            "at": "Prim </World>",
            "message": "default prim lacks RobotAPI",
        },
        {
            "severity": "ERROR",
            "rule": "NoOverrides",
            "at": "Prim </World/follower_left/link>",
            "message": "prim is overridden",
        },
    ]

    rows = audit_blocking_issues(
        family="IsaacSim.RobotRules",
        target="/tmp/workcell.usda",
        issues=issues,
    )

    assert [row["classification"] for row in rows] == ["WRONG_SCOPE", "WRONG_SCOPE"]
    assert all(row["task7_blocking"] is False for row in rows)
    assert rows[0]["asset_owner"] == "workcell wrapper"
    assert rows[1]["asset_owner"] == "robot"


def test_physics_rule_rows_keep_literal_defects_and_internal_error() -> None:
    issues = [
        {
            "severity": "ERROR",
            "rule": "RigidBodyHasCollider",
            "at": "Prim </World/follower_left/helper>",
            "message": "enabled body has no collider",
        },
        {
            "severity": "ERROR",
            "rule": "MimicAPICheck",
            "at": "Prim </World/follower_left/right_finger>",
            "message": "signed limit check",
        },
        {
            "severity": "ERROR",
            "rule": "RigidBodyHasMassAPI",
            "at": None,
            "message": "Uncaught error: NoneType",
        },
    ]

    rows = audit_blocking_issues(
        family="IsaacSim.PhysicsRules",
        target="/tmp/workcell.usda",
        issues=issues,
    )

    assert [row["classification"] for row in rows] == [
        "TRUE_ASSET_DEFECT",
        "INCONCLUSIVE",
        "INCONCLUSIVE",
    ]
    assert all(row["task7_blocking"] is True for row in rows)


def test_scope_audit_summary_requires_exact_original_counts() -> None:
    robot_rows = [
        {"classification": "WRONG_SCOPE", "task7_blocking": False}
        for _ in range(63)
    ]
    physics_rows = [
        {"classification": "TRUE_ASSET_DEFECT", "task7_blocking": True}
        for _ in range(26)
    ]

    summary = summarize_audit(robot_rows=robot_rows, physics_rows=physics_rows)

    assert summary["original_robot_blocking_count"] == 63
    assert summary["original_physics_blocking_count"] == 26
    assert summary["total_classified"] == 89
    assert summary["classification_counts"] == {
        "TRUE_ASSET_DEFECT": 26,
        "WRONG_SCOPE": 63,
    }


def test_scope_audit_summary_rejects_missing_original_issue() -> None:
    with pytest.raises(ValueError, match="expected 63 RobotRules"):
        summarize_audit(
            robot_rows=[{"classification": "WRONG_SCOPE", "task7_blocking": False}] * 62,
            physics_rows=[{"classification": "TRUE_ASSET_DEFECT", "task7_blocking": True}] * 26,
        )


def test_validator_signature_ignores_output_metadata_but_not_issues() -> None:
    base = {
        "category": "IsaacSim.RobotRules",
        "official_status": "PASS",
        "rules": ["RobotSchema"],
        "issues": [],
        "output": "/run/one.json",
    }
    repeat = dict(base, output="/run/two.json")
    assert issue_signature(base) == issue_signature(repeat)
    assert fresh_runs_match(base, repeat)
    changed = dict(repeat, issues=[{"severity": "ERROR", "rule": "RobotSchema"}])
    assert not fresh_runs_match(base, changed)


def test_negative_control_requires_expected_new_rule_and_target() -> None:
    baseline = {"issues": []}
    negative = {
        "issues": [
            {
                "severity": "ERROR",
                "rule": "RobotSchema",
                "at": "Prim </negative_robot_api>",
                "message": "does not have RobotAPI",
            }
        ]
    }
    result = validate_negative_delta(
        baseline=baseline,
        negative=negative,
        expected_rule="RobotSchema",
        expected_target_fragment="negative_robot_api",
    )
    assert result["status"] == "PASS"
    with pytest.raises(ValueError, match="RigidBodyHasMassAPI"):
        validate_negative_delta(
            baseline=baseline,
            negative=negative,
            expected_rule="RigidBodyHasMassAPI",
            expected_target_fragment="base_link",
        )


def test_validator_rule_filter_is_exact_and_rejects_unknown_names() -> None:
    class RigidBodyHasCollider:
        pass

    class RigidBodyHasMassAPI:
        pass

    rules = [RigidBodyHasCollider, RigidBodyHasMassAPI]
    selected = filter_rule_classes(rules, "RigidBodyHasMassAPI")
    assert selected == [RigidBodyHasMassAPI]
    with pytest.raises(ValueError, match="NotARealRule"):
        filter_rule_classes(rules, "NotARealRule")
