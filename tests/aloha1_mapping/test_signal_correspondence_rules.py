from tools.aloha1_mapping.signal_correspondence_rules import classify_official_issue_for_task7a
from tools.aloha1_mapping.signal_correspondence_rules import combine_official_rule_fragments


def test_mimic_validator_conflict_stays_official_partial() -> None:
    issue = {
        "severity": "ERROR",
        "rule": "MimicAPICheck",
        "at": "Prim </follower_left/vx300s_left/joints/right_finger>",
        "message": "lower limit should be greater than reference limit",
    }

    classified = classify_official_issue_for_task7a(issue)

    assert classified["official_severity"] == "ERROR"
    assert classified["official_result_suppressed"] is False
    assert classified["task7a_status"] == "PARTIAL"
    assert classified["classification"] == ("ISAAC_SIM_5_1_MIMIC_VALIDATOR_SCHEMA_CONFLICT")


def test_robot_packaging_error_does_not_become_signal_pass() -> None:
    issue = {
        "severity": "ERROR",
        "rule": "RobotSchema",
        "at": "Prim </follower_left>",
        "message": "DefaultPrim does not have a RobotAPI",
    }

    classified = classify_official_issue_for_task7a(issue)

    assert classified["task7a_status"] == "PARTIAL"
    assert classified["scope"] == "TASK7B_PACKAGING_NOT_RUNTIME_SIGNAL"
    assert classified["official_result_suppressed"] is False


def test_combined_rules_keep_official_fail_and_task7a_partial_separate() -> None:
    fragment = {
        "category": "IsaacSim.PhysicsRules",
        "target_name": "follower_left",
        "official_status": "FAIL",
        "issues": [
            {
                "severity": "ERROR",
                "rule": "JointHasJointStateAPI",
                "at": "Prim </follower_left/vx300s_left/joints/gripper>",
                "message": "Has no Joint State API",
            }
        ],
    }

    report = combine_official_rule_fragments([fragment])

    assert report["official_status"] == "FAIL"
    assert report["task7a_applicable_status"] == "PARTIAL"
    assert report["official_status_suppressed"] is False
