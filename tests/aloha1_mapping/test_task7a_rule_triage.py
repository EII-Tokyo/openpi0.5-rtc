import json
from pathlib import Path

from tools.aloha1_mapping.task7a_rule_triage import ALLOWED_CLASSIFICATIONS
from tools.aloha1_mapping.task7a_rule_triage import build_rule_triage
from tools.aloha1_mapping.task7a_rule_triage import classify_rule_issue

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _issue(rule: str, at: str, severity: str = "ERROR") -> dict[str, str]:
    return {
        "rule": rule,
        "at": at,
        "severity": severity,
        "message": f"{rule} diagnostic",
    }


def test_gripper_joint_state_is_child_asset_layer_packaging_defect() -> None:
    result = classify_rule_issue(
        _issue(
            "JointHasJointStateAPI",
            "Prim </follower_left/vx300s_left/joints/gripper>",
        ),
        category="IsaacSim.PhysicsRules",
        target_name="follower_left",
    )

    assert result["classification"] == "LAYER_PACKAGING_DEFECT"
    assert result["workcell_home_layer_has_required_api"] is True
    assert result["official_result_suppressed"] is False
    assert result["closure"] == "FORMALLY_RECORDED_CHILD_TARGET_BOUNDARY"


def test_known_mimic_limit_finding_is_isaac_5_1_schema_conflict() -> None:
    result = classify_rule_issue(
        _issue(
            "MimicAPICheck",
            "Prim </follower_right/vx300s_right/joints/right_finger>",
        ),
        category="IsaacSim.PhysicsRules",
        target_name="follower_right",
    )

    assert result["classification"] == (
        "ISAAC_5_1_VALIDATOR_SCHEMA_CONFLICT"
    )
    assert result["runtime_evidence_required"] == (
        "opposed local-axis limits, mimic target/readback, and zero active drive"
    )


def test_mass_only_helper_link_requires_source_geometry_evidence() -> None:
    result = classify_rule_issue(
        _issue(
            "RigidBodyHasCollider",
            "Prim </follower_left/vx300s_left/follower_left_ee_arm_link>",
        ),
        category="IsaacSim.PhysicsRules",
        target_name="follower_left",
    )

    assert result["classification"] == "MISSING_SOURCE_EVIDENCE"
    assert result["invent_collider_allowed"] is False
    assert result["task7a_applicability"] == "OUT_OF_SCOPE_GEOMETRY_BOUNDARY"


def test_packaging_and_unknown_rules_cover_remaining_classifications() -> None:
    no_override = classify_rule_issue(
        _issue(
            "NoOverrides",
            "Prim </follower_left/vx300s_left/link/collisions/mesh>",
        ),
        category="IsaacSim.RobotRules",
        target_name="follower_left",
    )
    direct_asset = classify_rule_issue(
        _issue("DirectAssetDefect", "Prim </robot>"),
        category="IsaacSim.PhysicsRules",
        target_name="robot",
        proven_asset_authoring_defect=True,
    )
    informational = classify_rule_issue(
        _issue("MaterialsOnTopLevelOnly", "", severity="INFO"),
        category="IsaacSim.SimReadyAssetRules",
        target_name="dual_follower_workcell",
    )
    unknown = classify_rule_issue(
        _issue("UnknownBlockingRule", "Prim </robot>"),
        category="IsaacSim.PhysicsRules",
        target_name="robot",
    )

    assert no_override["classification"] == "LAYER_PACKAGING_DEFECT"
    assert direct_asset["classification"] == "ASSET_AUTHORING_DEFECT"
    assert informational["classification"] == (
        "NON_APPLICABLE_FALSE_POSITIVE"
    )
    assert unknown["classification"] == "INCONCLUSIVE"
    assert {
        no_override["classification"],
        direct_asset["classification"],
        informational["classification"],
        unknown["classification"],
    } <= ALLOWED_CLASSIFICATIONS


def test_current_report_is_covered_exactly_once_without_suppression() -> None:
    official = (
        PROJECT_ROOT
        / "reports/aloha1_mapping/"
        "aloha1_signal_correspondence_official_rules.json"
    )
    report = build_rule_triage(PROJECT_ROOT, official)

    assert report["official_status"] == "FAIL"
    assert report["official_status_suppressed"] is False
    assert report["source_issue_count"] == 37
    assert report["triaged_issue_count"] == 37
    assert report["duplicate_issue_count"] == 0
    assert report["unclassified_issue_count"] == 0
    assert report["classification_counts"] == {
        "ISAAC_5_1_VALIDATOR_SCHEMA_CONFLICT": 2,
        "LAYER_PACKAGING_DEFECT": 28,
        "MISSING_SOURCE_EVIDENCE": 6,
        "NON_APPLICABLE_FALSE_POSITIVE": 1,
    }
    assert all(
        item["classification"] in ALLOWED_CLASSIFICATIONS
        and item["official_result_suppressed"] is False
        for item in report["issues"]
    )


def test_runtime_mimic_probe_is_attached_without_hiding_official_fail(
    tmp_path: Path,
) -> None:
    official = (
        PROJECT_ROOT
        / "reports/aloha1_mapping/"
        "aloha1_signal_correspondence_official_rules.json"
    )
    probe = tmp_path / "mimic_probe.json"
    probe.write_text(
        json.dumps(
            {
                "status": "PASS",
                "stage": {
                    "sha256": (
                        "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533"
                        "c312d7543c788cf"
                    )
                },
                "loaded_rule_module": {
                    "absolute_path": "/local/joint_rules.py",
                    "file_sha256": "rule-file-hash",
                    "MimicAPICheck_source_sha256": "class-source-hash",
                    "source_first_line": 252,
                },
                "robots": {
                    robot: {
                        "left_finger": {
                            "runtime_lower": 0.021,
                            "runtime_upper": 0.057,
                            "usd_lower_limit": 0.021,
                            "usd_upper_limit": 0.057,
                        },
                        "right_finger": {
                            "runtime_lower": -0.0642,
                            "runtime_upper": -0.0138,
                            "usd_lower_limit": -0.0642,
                            "usd_upper_limit": -0.0138,
                            "mimic_attributes": {
                                "physxMimicJoint:rotY:gearing": 1.0
                            },
                        },
                    }
                    for robot in ("follower_left", "follower_right")
                },
                "stage_modified": False,
            }
        ),
        encoding="utf-8",
    )

    report = build_rule_triage(
        PROJECT_ROOT,
        official,
        mimic_probe_path=probe,
    )

    assert report["official_status"] == "FAIL"
    assert report["official_status_suppressed"] is False
    evidence = report["mimic_runtime_probe"]
    assert evidence["status"] == "PASS"
    assert evidence["stage_modified"] is False
    assert evidence["interpretation"] == (
        "OPPOSED_LOCAL_JOINT_AXES_NOT_MODELED_BY_NUMERIC_LIMIT_RULE"
    )
    mimic_issues = [
        issue for issue in report["issues"]
        if issue["rule"] == "MimicAPICheck"
    ]
    assert len(mimic_issues) == 2
    assert all(
        issue["runtime_probe_status"] == "PASS"
        for issue in mimic_issues
    )
