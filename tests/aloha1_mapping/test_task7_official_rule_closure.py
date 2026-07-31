import json
from pathlib import Path

import pytest

from tools.aloha1_mapping.task7_official_rule_closure import classify_official_rule_closure

ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "reports/aloha1_mapping"
OFFICIAL_RUNNER = (
    ROOT / "tools/validate_aloha1_signal_correspondence_official_rules.py"
)


def _issues() -> list[dict[str, object]]:
    report = json.loads(
        (REPORT_ROOT / "aloha1_task7a_rule_triage.json").read_text(
            encoding="utf-8"
        )
    )
    return report["issues"]


def test_exact_37_findings_are_partitioned_without_suppression() -> None:
    result = classify_official_rule_closure(_issues())

    assert result["issue_count"] == 37
    assert result["unclassified_issue_count"] == 0
    assert result["official_status"] == "FAIL"
    assert result["official_status_suppressed"] is False
    assert result["classification_counts"] == {
        "ISAAC_5_1_VALIDATOR_SCHEMA_CONFLICT": 2,
        "LAYER_PACKAGING_DEFECT": 28,
        "MISSING_SOURCE_EVIDENCE": 6,
        "NON_APPLICABLE_FALSE_POSITIVE": 1,
    }


def test_packaging_findings_are_the_only_candidate_mutations() -> None:
    result = classify_official_rule_closure(_issues())

    assert result["action_counts"] == {
        "CREATE_ISOLATED_PACKAGING_CANDIDATE": 28,
        "HARD_BLOCKER_NO_SOURCE_GEOMETRY": 6,
        "KEEP_UNSUPPRESSED_VERSION_CONFLICT": 2,
        "RECORD_NON_BLOCKING_INFORMATION": 1,
    }
    assert result["candidate_mutation_issue_count"] == 28
    assert result["source_or_runtime_mutation_issue_count"] == 0
    assert result["task8"] == "NOT_RUN"


def test_packaging_rule_breakdown_is_exact() -> None:
    result = classify_official_rule_closure(_issues())

    assert result["packaging_rule_counts"] == {
        "JointHasJointStateAPI": 2,
        "NoOverrides": 8,
        "RobotNaming": 1,
        "RobotSchema": 3,
        "ThumbnailExists": 2,
        "VerifyRobotPhysicsAttributesSourceLayer": 4,
        "VerifyRobotPhysicsSchemaSourceLayer": 8,
    }


def test_unknown_classification_fails_closed() -> None:
    issues = _issues()
    issues[0] = {**issues[0], "classification": "UNKNOWN"}

    with pytest.raises(ValueError, match="unsupported classifications"):
        classify_official_rule_closure(issues)


def test_generated_report_records_local_version_authority() -> None:
    report = json.loads(
        (
            REPORT_ROOT / "aloha1_task7_official_rule_closure.json"
        ).read_text(encoding="utf-8")
    )

    assert report["status"] == "PARTIAL"
    assert report["local_runtime"]["isaac_sim"] == "5.1.0.0"
    assert report["local_runtime"]["asset_validation"] == "1.1.0"
    assert report["direct_nvidia_mcp_probe"]["asset_validation"] == "1.2.1"
    assert report["version_authority"] == "LOCAL_ISAAC_SIM_5_1_SOURCE"
    assert report["stage_mutated"] is False
    assert report["final_or_default_asset_modified"] is False
    assert report["task8"] == "NOT_RUN"
    right = report["isolated_candidate_results"][
        "follower_right_robot_schema"
    ]
    assert right["official_status"] == "PASS"
    assert right["blocking_issue_count"] == 0
    assert right["warning_count"] == 0
    assert right["deterministic_repeat"] is True
    assert right["physical_stage_modified"] is False
    joint_state = report["isolated_candidate_results"][
        "gripper_joint_state_physics"
    ]
    assert joint_state["status"] == "PASS"
    assert joint_state["validated_packaging_finding_count"] == 2
    assert joint_state["source_stage_modified"] is False
    assert joint_state["final_or_default_asset_modified"] is False
    assert joint_state["task7"] == "PARTIAL"
    assert joint_state["task8"] == "NOT_RUN"


def test_official_runner_records_direct_nvidia_mcp_not_gateway() -> None:
    source = OFFICIAL_RUNNER.read_text(encoding="utf-8")

    assert '"direct_nvidia_mcp_verified": True' in source
    assert "mcpjungle_nvidia_official_api_verified" not in source
