from pathlib import Path

import pytest

from tools.aloha1_mapping.task7_acceptance_matrix import classify_asset_promotion_readiness
from tools.aloha1_mapping.task7_acceptance_matrix import classify_runtime_control
from tools.aloha1_mapping.task7_acceptance_matrix import classify_workcell_physics
from tools.aloha1_mapping.task7_acceptance_matrix import combine_task7a_layers
from tools.aloha1_mapping.task7_acceptance_matrix import verify_file_sha256

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/build_aloha1_task7_acceptance_matrix.py"
REPORT_ROOT = ROOT / "reports/aloha1_mapping"


def _task7a() -> dict[str, str]:
    return {
        "structure_and_runtime_order": "PASS",
        "joint_mapping": "PASS",
        "follower_left_one_joint": "PASS",
        "follower_right_one_joint": "PASS",
        "small_up_down": "PASS",
        "drive_mimic_structure": "PASS",
        "initial_target_readback_first_frame": "PASS",
    }


def _swept() -> dict:
    return {
        "status": "PASS",
        "summary": {
            "case_count": 48,
            "expected_case_count": 48,
            "failed_case_count": 0,
            "partial_case_count": 0,
            "contact_limited_case_count": 4,
            "coverage_status": "PASS",
            "determinism": {
                "status": "PASS",
                "repeat_count": 2,
                "signatures": ["same", "same"],
            },
        },
        "contact_policy": {
            "revision": 2,
            "allowed_pair": ("supplier-CAD finger link <-> user_confirmed_table"),
            "generic_robot_environment_contact": "FAIL",
            "non_adjacent_self_contact": "FAIL",
            "cross_follower_contact": "FAIL",
        },
    }


def _triage() -> dict:
    return {
        "official_status": "FAIL",
        "official_status_suppressed": False,
        "unclassified_issue_count": 0,
        "classification_counts": {
            "ISAAC_5_1_VALIDATOR_SCHEMA_CONFLICT": 2,
            "LAYER_PACKAGING_DEFECT": 28,
            "MISSING_SOURCE_EVIDENCE": 6,
            "NON_APPLICABLE_FALSE_POSITIVE": 1,
        },
        "issues": [],
    }


def _helper_audit() -> dict:
    return {
        "status": "PASS",
        "coverage": {
            "official_findings": 6,
            "helper_records": 6,
            "expected_each": 6,
        },
        "decision": {"asset_promotion_effect": ("PARTIAL_LITERAL_RULE_FAILURE_REMAINS")},
    }


def test_runtime_control_pass_is_independent_of_packaging_findings() -> None:
    result = classify_runtime_control(_task7a())

    assert result["status"] == "PASS"
    assert result["failed_gates"] == []
    assert result["official_asset_rules_included"] is False


def test_allowed_finger_table_contact_preserves_workcell_physics_pass() -> None:
    result = classify_workcell_physics(_swept())

    assert result["status"] == "PASS"
    assert result["contact_limited_case_count"] == 4
    assert result["allowed_pair"] == ("supplier-CAD finger link <-> user_confirmed_table")


def test_any_forbidden_contact_fails_workcell_physics() -> None:
    swept = _swept()
    swept["summary"]["failed_case_count"] = 1

    result = classify_workcell_physics(swept)

    assert result["status"] == "FAIL"


def test_official_fail_remains_visible_while_promotion_is_partial() -> None:
    result = classify_asset_promotion_readiness(
        official={"official_status": "FAIL"},
        triage=_triage(),
        helper_audit=_helper_audit(),
    )

    assert result["status"] == "PARTIAL"
    assert result["ready_for_promotion"] is False
    assert result["official_status"] == "FAIL"
    assert result["official_status_suppressed"] is False


def test_unclassified_promotion_finding_is_fail() -> None:
    triage = _triage()
    triage["unclassified_issue_count"] = 1

    result = classify_asset_promotion_readiness(
        official={"official_status": "FAIL"},
        triage=triage,
        helper_audit=_helper_audit(),
    )

    assert result["status"] == "FAIL"


def test_aggregate_is_partial_when_only_promotion_is_not_ready() -> None:
    assert (
        combine_task7a_layers(
            runtime_status="PASS",
            workcell_status="PASS",
            promotion_status="PARTIAL",
        )
        == "PARTIAL"
    )
    assert (
        combine_task7a_layers(
            runtime_status="FAIL",
            workcell_status="PASS",
            promotion_status="PARTIAL",
        )
        == "FAIL"
    )


def test_hash_mismatch_refuses_report_generation(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_file_sha256(source, "0" * 64)


def test_builder_keeps_task7b_and_task8_not_run() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"task_7b": "NOT_RUN"' in source
    assert '"task_8": "NOT_RUN"' in source
    assert "official_status_suppressed" in source


def test_current_reports_separate_runtime_from_promotion() -> None:
    runtime = REPORT_ROOT / "aloha1_task7_runtime_acceptance.json"
    promotion = REPORT_ROOT / "aloha1_task7_asset_promotion_readiness.json"
    applicability = REPORT_ROOT / "aloha1_task7_official_rule_applicability.json"

    runtime_report = __import__("json").loads(runtime.read_text(encoding="utf-8"))
    promotion_report = __import__("json").loads(promotion.read_text(encoding="utf-8"))
    applicability_report = __import__("json").loads(applicability.read_text(encoding="utf-8"))

    assert runtime_report["runtime_control"]["status"] == "PASS"
    assert runtime_report["workcell_physics"]["status"] == "PASS"
    assert runtime_report["task_7a_aggregate"] == "PARTIAL"
    assert runtime_report["task_7b"] == "NOT_RUN"
    assert runtime_report["task_8"] == "NOT_RUN"
    assert promotion_report["status"] == "PARTIAL"
    assert promotion_report["ready_for_promotion"] is False
    assert promotion_report["official_status"] == "FAIL"
    assert promotion_report["official_status_suppressed"] is False
    assert applicability_report["issue_count"] == 37
