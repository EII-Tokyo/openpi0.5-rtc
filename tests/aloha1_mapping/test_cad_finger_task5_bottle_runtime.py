from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tools.aloha1_mapping.cad_finger_task5_bottle import classify_hold_failure_mode
from tools.aloha1_mapping.cad_finger_task5_bottle import compute_hold_kinematics
from tools.aloha1_mapping.cad_finger_task5_bottle import evaluate_bottle_trial
from tools.aloha1_mapping.cad_finger_task5_bottle import summarize_bottle_trials

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/aloha1_cad_finger_task5_bottle.yaml"
DIAGNOSTIC = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_finger_task5_bottle/"
    "aloha_viperx_supplier_cad_bottle_task5.usda"
)
RUNTIME = ROOT / "tools/validate_aloha_viper_cad_finger_task5_bottle.py"


def _passing_metrics() -> dict[str, object]:
    return {
        "solve_articulation_contact_last_ok": True,
        "left_finger_contact": True,
        "right_finger_contact": True,
        "bilateral_contact_before_release": True,
        "impulses_finite": True,
        "persistent_penetration": False,
        "unexpected_gripper_collision": False,
        "released_without_constraint": True,
        "gravity_enabled_after_release": True,
        "held_for_required_time": True,
        "drop_within_gate": True,
        "finite_state": True,
    }


def test_frozen_supplier_cad_bottle_profile() -> None:
    profile = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert profile["runtime"] == {
        "isaac_sim": "5.1.0.0",
        "kit": "107.3.3",
        "physx": "107.3.26",
    }
    assert profile["approved_source_stage"]["sha256"] == (
        "b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e"
    )
    frozen = profile["frozen"]
    assert frozen["robot"] == "follower_left"
    assert frozen["collider"] == "SUPPLIER_CAD_V2_CONVEX_HULL_DIAGNOSTIC"
    assert frozen["friction"] == 0.7
    assert frozen["restitution"] == 0.0
    assert frozen["bottle_mass_kg"] == 0.020
    assert frozen["bottle_diameter_m"] == 0.065
    assert frozen["physics_frequency_hz"] == 60
    assert frozen["hold_interval_s"] == 2.0
    assert frozen["drop_gate_m"] == 0.010
    assert frozen["solve_articulation_contact_last"] is True
    assert frozen["self_collision"] is False
    assert frozen["surface_gripper"] is False
    assert frozen["fixed_joint"] is False
    assert frozen["parent_attachment"] is False


def test_bottle_diagnostic_is_an_isolated_reference_layer() -> None:
    text = DIAGNOSTIC.read_text(encoding="utf-8")
    assert 'defaultPrim = "workcell"' in text
    assert (
        "@../cad_finger_task5_arm_max_force_over_combined/"
        "aloha_viperx_supplier_cad_arm_max_force_over_combined.usda@"
        "</workcell>"
    ) in text
    assert "BottleProxy" not in text
    assert "SurfaceGripper" not in text
    assert "fixedJoint" not in text


def test_runtime_uses_exact_supplier_paths_and_required_physx_controls() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    assert 'ARTICULATION_PATH = "/workcell/vx300s_left/vx300s_left"' in source
    assert '"vx300s_left_left_finger"' in source
    assert '"vx300s_left_right_finger"' in source
    assert "PhysxContactReportAPI.Apply" in source
    assert "set_solve_articulation_contact_last(True)" in source
    assert "get_solve_articulation_contact_last()" in source
    assert "CreateKinematicEnabledAttr(True)" in source
    assert "GetKinematicEnabledAttr().Set(False)" in source
    assert "subscribe_contact_report_events" in source
    assert "capture_viewport_to_file" in source
    assert "get_image_coords_from_world_points" in source
    assert '"physical_surface_contact"' in source
    assert 'float(contact["separation"]) <= 0.0' in source
    assert "SurfaceGripper" not in source
    assert "FixedJoint.Define" not in source


def test_bottle_gate_requires_real_bilateral_release_and_hold() -> None:
    result = evaluate_bottle_trial(_passing_metrics())
    assert result["status"] == "PASS"
    assert result["failed_checks"] == []

    metrics = _passing_metrics()
    metrics["bilateral_contact_before_release"] = False
    result = evaluate_bottle_trial(metrics)
    assert result["status"] == "FAIL"
    assert result["failed_checks"] == ["bilateral_contact_before_release"]


def test_failure_mode_classification_is_explicit() -> None:
    assert (
        classify_hold_failure_mode(
            {
                "bilateral_contact_before_release": False,
                "contact_lost_after_release": False,
                "continuous_slip_with_bilateral_contact": False,
                "rotation_induced_escape": False,
                "normal_force_decay": False,
                "numerical_penetration_or_ejection": False,
                "drop_within_gate": False,
            }
        )
        == "contact_not_established"
    )
    assert (
        classify_hold_failure_mode(
            {
                "bilateral_contact_before_release": True,
                "contact_lost_after_release": True,
                "continuous_slip_with_bilateral_contact": False,
                "rotation_induced_escape": False,
                "normal_force_decay": False,
                "numerical_penetration_or_ejection": False,
                "drop_within_gate": False,
            }
        )
        == "contact_lost_then_free_fall"
    )
    assert (
        classify_hold_failure_mode(
            {
                "bilateral_contact_before_release": True,
                "contact_lost_after_release": False,
                "continuous_slip_with_bilateral_contact": False,
                "rotation_induced_escape": False,
                "normal_force_decay": False,
                "numerical_penetration_or_ejection": False,
                "drop_within_gate": True,
            }
        )
        == "stable_hold"
    )


def test_repeat_summary_requires_exactly_reproducible_gate_result() -> None:
    metrics = _passing_metrics()
    trials = [
        {
            "status": "PASS",
            "metrics": metrics,
            "deterministic_signature": "same",
            "released_hold": {"drop_m": 0.004},
        },
        {
            "status": "PASS",
            "metrics": metrics,
            "deterministic_signature": "same",
            "released_hold": {"drop_m": 0.004},
        },
    ]
    summary = summarize_bottle_trials(trials, required_repeats=2)
    assert summary["status"] == "PASS"
    assert summary["pass_count"] == 2
    assert summary["deterministic"] is True

    parsed = json.loads(json.dumps(summary, allow_nan=False))
    assert parsed["maximum_drop_m"] == 0.004


def test_hold_drop_uses_worst_frame_and_keeps_final_drop_separate() -> None:
    result = compute_hold_kinematics(
        release_z_m=1.0,
        z_samples_m=[0.999, 0.980, 0.995],
        dt_s=0.5,
    )
    assert result["maximum_drop_m"] == pytest.approx(0.020)
    assert result["final_drop_m"] == pytest.approx(0.005)
    assert result["pose_derived_vertical_velocity_m_s"] == pytest.approx([
        -0.002,
        -0.038,
        0.030,
    ])
