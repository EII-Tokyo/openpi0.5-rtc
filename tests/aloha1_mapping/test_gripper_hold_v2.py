from __future__ import annotations

from tools.aloha1_mapping.gripper_force_diagnosis import classify_hold_failure_mode
from tools.aloha1_mapping.gripper_force_diagnosis import classify_root_cause_v2
from tools.aloha1_mapping.gripper_force_diagnosis import classify_solver_sensitivity
from tools.aloha1_mapping.gripper_force_diagnosis import has_consecutive_true
from tools.aloha1_mapping.gripper_force_diagnosis import select_solver_iteration_frequency
from tools.aloha1_mapping.gripper_force_diagnosis import verify_solver_trial_invariants
from tools.aloha1_mapping.gripper_force_runtime import _summarize_solver_trials


def test_hold_failure_contact_loss_precedes_drop() -> None:
    result = classify_hold_failure_mode(
        {
            "drop_m": 0.03,
            "drop_gate_m": 0.01,
            "contact_loss_frame": 12,
            "drop_gate_crossing_frame": 20,
            "contacts_persist_to_end": False,
            "normal_force_decay_ratio": 0.4,
            "maximum_angular_speed_rad_s": 0.2,
            "persistent_penetration": False,
            "release_linear_speed_m_s": 0.0,
        }
    )

    assert result["mode"] == "CONTACT_LOSS_THEN_FALL"
    assert result["pass"] is False


def test_hold_failure_detects_release_ejection_before_friction_claim() -> None:
    result = classify_hold_failure_mode(
        {
            "drop_m": 0.05,
            "drop_gate_m": 0.01,
            "contact_loss_frame": None,
            "drop_gate_crossing_frame": 8,
            "contacts_persist_to_end": True,
            "normal_force_decay_ratio": 0.9,
            "maximum_angular_speed_rad_s": 0.5,
            "persistent_penetration": False,
            "release_linear_speed_m_s": 0.55,
            "release_ejection_threshold_m_s": 0.1,
        }
    )

    assert result["mode"] == "NUMERICAL_EJECTION_OR_RELEASE_TRANSIENT"


def test_upward_release_ejection_cannot_pass_drop_gate() -> None:
    result = classify_hold_failure_mode(
        {
            "drop_m": -0.05,
            "drop_gate_m": 0.01,
            "persistent_penetration": False,
            "release_linear_speed_m_s": 0.55,
            "release_ejection_threshold_m_s": 0.1,
        }
    )

    assert result == {
        "mode": "NUMERICAL_EJECTION_OR_RELEASE_TRANSIENT",
        "pass": False,
    }


def test_penetration_requires_configured_consecutive_frames() -> None:
    assert has_consecutive_true(
        [False, True, True, True, True, True, False],
        required=5,
    )
    assert not has_consecutive_true(
        [False, True, True, False, True, True, True],
        required=5,
    )


def test_root_cause_v2_uses_only_allowed_category() -> None:
    result = classify_root_cause_v2(
        {
            "contact_semantics": "CONTACT_ENVELOPE_DOMINATED",
            "normal_force": "INSUFFICIENT",
            "material": "SUFFICIENT",
            "friction": "INCONCLUSIVE",
            "solver": "INCONCLUSIVE",
            "hold_failure_mode": "NORMAL_FORCE_DECAY",
            "max_force_observable": True,
            "max_force_saturated": False,
        }
    )

    assert result["root_cause"] == "multiple_contributing_causes"
    assert set(result["contributing_causes"]) == {
        "contact_envelope_or_offset",
        "insufficient_drive_preload",
    }


def test_unresolved_release_transient_prevents_final_root_pass() -> None:
    result = classify_root_cause_v2(
        {
            "contact_semantics": "VERIFIED_PHYSICAL_CONTACT",
            "normal_force": "INSUFFICIENT",
            "material": "SUFFICIENT",
            "friction": "INCONCLUSIVE",
            "solver": "INCONCLUSIVE",
            "hold_failure_mode": "NUMERICAL_EJECTION_OR_RELEASE_TRANSIENT",
            "max_force_observable": False,
            "max_force_saturated": False,
        }
    )

    assert result["root_cause"] == "inconclusive"
    assert result["contributing_causes"] == []
    assert set(result["unresolved_observations"]) == {
        "drive_vs_max_force_not_observable",
        "kinematic_to_dynamic_release_transient",
    }


def test_solver_classification_requires_single_variable_evidence() -> None:
    result = classify_solver_sensitivity(
        [
            {"frequency_hz": 60, "hold_success_rate": 0.0, "invariant_pass": True},
            {"frequency_hz": 120, "hold_success_rate": 1.0, "invariant_pass": True},
            {"frequency_hz": 240, "hold_success_rate": 1.0, "invariant_pass": True},
        ]
    )

    assert result["SOLVER_STATUS"] == "REQUIRES_HIGHER_RATE"


def test_solver_classification_rejects_failed_iteration_invariant() -> None:
    result = classify_solver_sensitivity(
        [
            {
                "frequency_hz": 60,
                "hold_success_rate": 0.0,
                "invariant_pass": True,
            }
        ],
        [
            {
                "hold_success_rate": 1.0,
                "invariant_pass": False,
            }
        ],
    )

    assert result["SOLVER_STATUS"] == "INCONCLUSIVE"
    assert result["reason"] == "single_variable_invariant_failed"


def test_solver_not_run_is_inconclusive() -> None:
    result = classify_solver_sensitivity([])

    assert result["SOLVER_STATUS"] == "INCONCLUSIVE"
    assert result["run"] is False


def test_solver_iteration_frequency_selects_lowest_improving_rate() -> None:
    result = select_solver_iteration_frequency(
        [
            {
                "frequency_hz": 60,
                "hold_success_rate": 0.0,
                "invariant_pass": True,
                "trial_count": 20,
                "successful_trial_count": 20,
            },
            {
                "frequency_hz": 120,
                "hold_success_rate": 1.0,
                "invariant_pass": True,
                "trial_count": 20,
                "successful_trial_count": 20,
            },
            {
                "frequency_hz": 240,
                "hold_success_rate": 1.0,
                "invariant_pass": True,
                "trial_count": 20,
                "successful_trial_count": 20,
            },
        ],
        baseline_frequency_hz=60,
    )

    assert result["status"] == "PASS"
    assert result["selected_frequency_hz"] == 120


def test_solver_iteration_frequency_rejects_incomplete_groups() -> None:
    result = select_solver_iteration_frequency(
        [
            {
                "frequency_hz": 60,
                "hold_success_rate": 0.0,
                "invariant_pass": True,
                "trial_count": 20,
                "successful_trial_count": 19,
            }
        ],
        baseline_frequency_hz=60,
    )

    assert result["status"] == "INCONCLUSIVE"
    assert result["selected_frequency_hz"] is None


def test_solver_invariant_manifest_is_computed_not_hard_coded() -> None:
    trial = {
        "approximation": "convexHull",
        "friction": 0.7,
        "frequency_hz": 120,
        "delta_m": 0.002,
        "contact_target_m": 0.0431,
        "solve_articulation_contact_last": True,
        "input_asset_sha256": "frozen-hash",
        "solver_readback": {
            "position_iterations": 4,
            "velocity_iterations": 1,
        },
    }
    expected = {
        "approximation": "convexHull",
        "friction": 0.7,
        "frequency_hz": 120,
        "delta_m": 0.002,
        "contact_target_m": 0.0431,
        "solve_articulation_contact_last": True,
        "input_asset_sha256": "frozen-hash",
        "solver_readback.position_iterations": 4,
        "solver_readback.velocity_iterations": 1,
    }

    passing = verify_solver_trial_invariants([trial], expected)
    failing = verify_solver_trial_invariants(
        [{**trial, "friction": 1.0}],
        expected,
    )

    assert passing["pass"] is True
    assert passing["mismatches"] == []
    assert failing["pass"] is False
    assert failing["mismatches"][0]["field"] == "friction"


def test_solver_trial_summary_structures_contact_setup_failure() -> None:
    result = _summarize_solver_trials(
        [
            {
                "status": "FAIL",
                "failure": "bilateral_solver_load_bearing_contact_not_found",
                "solver_readback": {
                    "position_iterations": 4,
                    "velocity_iterations": 1,
                },
            }
        ]
    )

    assert result["successful_trial_count"] == 0
    assert result["failed_trial_count"] == 1
    assert result["maximum_drop_m"] is None
    assert result["failure_modes"] == {
        "TRIAL_SETUP_OR_CONTACT_FAILURE": 1
    }
