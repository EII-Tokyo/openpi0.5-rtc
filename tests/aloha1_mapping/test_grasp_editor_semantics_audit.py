from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOL = Path("tools/build_aloha1_grasp_editor_semantics_audit.py")


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "grasp_editor_semantics_audit",
        TOOL,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_semantics_audit_separates_fully_closed_from_contact_position() -> None:
    report = _load_module().build_report()

    joint_settings = report["joint_settings"]
    assert joint_settings["fully_closed_position_m"] == pytest.approx(0.021)
    assert joint_settings["cad_contact_candidate_position_m"] == pytest.approx(
        0.048316874538855845
    )
    assert joint_settings["previous_conflation_status"] == "CORRECTED"


def test_semantics_audit_blocks_ik_on_verified_mimic_failure() -> None:
    report = _load_module().build_report()

    assert report["status"] == "PARTIAL"
    assert report["coordinate_transform"]["status"] == "PASS"
    assert report["native_simulate_suitability"]["status"] == "FAIL"
    assert report["mimic_load_comparison"]["classification"] == (
        "OBJECT_CONTACT_AMPLIFIES_PERSISTENT_MIMIC_ERROR"
    )
    assert report["mimic_load_comparison"]["contact_residual_m"] == pytest.approx(
        0.020771507173776627
    )
    assert report["mimic_load_comparison"]["no_contact_residual_m"] == pytest.approx(
        0.001420333981513977
    )
    assert report["next_gates"]["ik"] == "NOT_RUN"
    assert report["next_gates"]["five_random_bottle_videos"] == "NOT_RUN"
    assert report["task8"] == "NOT_RUN"


def test_local_schema_does_not_mislabel_custom_mimic_properties() -> None:
    report = _load_module().build_report()

    schema = report["local_physx_schema"]
    assert schema["version"] == "107.3.26"
    assert "gearing" in schema["declared_mimic_properties"]
    assert "offset" in schema["declared_mimic_properties"]
    assert "naturalFrequency" not in schema["declared_mimic_properties"]
    assert "dampingRatio" not in schema["declared_mimic_properties"]
    assert schema["custom_property_effect_status"] == "INCONCLUSIVE"


def test_external_skip_sim_is_exportable_but_remains_blocked_by_mimic() -> None:
    report = _load_module().build_report()

    external = report["external_programmatic_close_skip_sim"]
    assert external["status"] == "FAIL"
    assert external["bilateral_contact"] == "PASS"
    assert external["native_raw_export"]["status"] == "PASS"
    assert external["derived_export"]["status"] == "PASS"
    assert external["derived_export"]["classification"] == (
        "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
    )
    assert external["mimic_accuracy"] == "FAIL"
    assert external["ik_promotion_allowed"] is False
    assert external["screenshot_review"][
        "all_raw_and_annotated_visual_reviews"
    ] == "PASS"
    assert external["screenshot_review"]["numeric_failure_preserved"] is True
    assert report["next_gates"]["external_programmatic_grasp_then_skip_sim"] == (
        "FAIL_MIMIC_ACCURACY"
    )
