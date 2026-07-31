import json
from pathlib import Path

from tools.aloha1_mapping.task7_post_grasp_acceptance import classify_post_grasp_task7

ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "reports/aloha1_mapping"
SCRIPT = ROOT / "tools/build_aloha1_task7_post_grasp_acceptance.py"


def _passing_inputs() -> dict[str, str]:
    return {
        "runtime_control": "PASS",
        "workcell_physics": "PASS",
        "aloha_6dof_ik_correspondence": "PASS",
        "table_support_alignment": "PASS",
        "static_bottle_hold": "PASS",
        "dynamic_five_pose_grasp": "PASS",
        "visual_model_review": "PASS",
        "user_confirmation": "PASS",
        "asset_promotion_readiness": "PARTIAL",
        "official_rules_literal_status": "FAIL",
        "task8": "NOT_RUN",
    }


def test_runtime_grasp_pass_is_separate_from_asset_promotion() -> None:
    result = classify_post_grasp_task7(_passing_inputs())

    assert result["runtime_grasp_acceptance"] == "PASS"
    assert result["runtime_grasp_gates"][
        "aloha_6dof_ik_correspondence"
    ] == "PASS"
    assert result["asset_promotion_readiness"] == "PARTIAL"
    assert result["official_rules_literal_status"] == "FAIL"
    assert result["task7_aggregate"] == "PARTIAL"
    assert result["task8"] == "NOT_RUN"


def test_dynamic_grasp_failure_fails_task7() -> None:
    inputs = _passing_inputs()
    inputs["dynamic_five_pose_grasp"] = "FAIL"

    result = classify_post_grasp_task7(inputs)

    assert result["runtime_grasp_acceptance"] == "FAIL"
    assert result["task7_aggregate"] == "FAIL"


def test_missing_runtime_gate_preserves_partial() -> None:
    inputs = _passing_inputs()
    inputs["user_confirmation"] = "NOT_RUN"

    result = classify_post_grasp_task7(inputs)

    assert result["runtime_grasp_acceptance"] == "PARTIAL"
    assert result["task7_aggregate"] == "PARTIAL"


def test_task8_cannot_be_promoted_by_post_grasp_classifier() -> None:
    inputs = _passing_inputs()
    inputs["task8"] = "PASS"

    result = classify_post_grasp_task7(inputs)

    assert result["task8"] == "NOT_RUN"
    assert "task8_input_was_ignored" in result["boundaries"]


def test_builder_and_generated_report_preserve_literal_boundaries() -> None:
    assert SCRIPT.is_file()
    report = json.loads(
        (
            REPORT_ROOT
            / "aloha1_task7_post_grasp_acceptance.json"
        ).read_text(encoding="utf-8")
    )

    assert report["runtime_grasp_acceptance"] == "PASS"
    assert report["asset_promotion_readiness"] == "PARTIAL"
    assert report["official_rules_literal_status"] == "FAIL"
    assert report["official_rules_suppressed"] is False
    assert report["task7_aggregate"] == "PARTIAL"
    assert report["task8"] == "NOT_RUN"
    assert report["stage"]["aligned"]["sha256"] == (
        "2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c"
    )
    assert report["stage"]["source"]["sha256"] == (
        "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
    )
    assert report["stage"]["composition_verified"] is True
    assert report["input_immutability"]["all_hashes_unchanged"] is True
