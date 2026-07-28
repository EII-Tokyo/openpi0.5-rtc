from pathlib import Path

from tools.aloha1_mapping.physicsrules import build_physicsrules_report
from tools.aloha1_mapping.physicsrules import classify_physics_rule_category
from tools.aloha1_mapping.validation import build_validation_plan

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _category(*issues: dict[str, str]) -> dict:
    return {
        "category": "IsaacSim.PhysicsRules",
        "status": "FAIL",
        "targets": [
            {
                "target": "follower_left",
                "status": "FAIL",
                "issues": list(issues),
            }
        ],
    }


def test_physics_rules_classifies_only_evidence_backed_known_issues() -> None:
    report = classify_physics_rule_category(
        _category(
            {
                "rule": "JointHasJointStateAPI",
                "severity": "ERROR",
                "at": "Prim </follower_left/joints/gripper>",
                "message": "Has no Joint State API",
            },
            {
                "rule": "MimicAPICheck",
                "severity": "ERROR",
                "at": "Prim </follower_left/joints/right_finger>",
                "message": "limits disagree",
            },
            {
                "rule": "RigidBodyHasCollider",
                "severity": "ERROR",
                "at": "Prim </follower_left/follower_left_fingers_link>",
                "message": "has rigid body api but no collision api",
            },
        )
    )

    assert report["status"] == "PARTIAL"
    assert [item["classification"] for item in report["issues"]] == [
        "CONFIGURATION_LAYER_JOINT_STATE_OMISSION",
        "ISAAC_SIM_5_1_MIMIC_VALIDATOR_SCHEMA_CONFLICT",
        "SOURCE_MASS_ONLY_LINK_WITHOUT_GEOMETRY",
    ]
    assert report["issues"][0]["resolution"] == "FIXED_IN_CONFIGURATION_LAYER"
    assert report["issues"][1]["resolution"] == "FORMALLY_RECORDED"
    assert report["issues"][2]["resolution"] == "HARD_BLOCKER_NO_GEOMETRY_EVIDENCE"
    assert report["unclassified_error_count"] == 0


def test_unknown_physics_rule_error_cannot_be_waived() -> None:
    report = classify_physics_rule_category(
        _category(
            {
                "rule": "UnexpectedRule",
                "severity": "ERROR",
                "at": "Prim </follower_left>",
                "message": "unexpected",
            }
        )
    )

    assert report["status"] == "FAIL"
    assert report["unclassified_error_count"] == 1
    assert report["issues"][0]["classification"] == "UNCLASSIFIED"
    assert report["issues"][0]["resolution"] == "FIX_REQUIRED"


def test_physics_rules_keep_the_importer_source_target_explicit() -> None:
    plan = build_validation_plan(PROJECT_ROOT)

    for robot in plan["robots"]:
        assert Path(robot["source_robot_asset"]).is_file()
        assert "configuration" not in robot["source_robot_asset"]


def test_build_report_selects_exact_official_category() -> None:
    physics = _category()
    report = build_physicsrules_report(
        {
            "categories": [
                {"category": "IsaacSim.RobotRules", "status": "PASS"},
                physics,
            ]
        }
    )

    assert report["status"] == "PASS"
    assert report["official_status"] == "FAIL"
