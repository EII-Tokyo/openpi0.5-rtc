from __future__ import annotations

import math

import pytest

from tools.aloha1_mapping.force_drive_candidate import derive_gain_tuner_pd
from tools.aloha1_mapping.force_drive_candidate import evaluate_force_drive_candidate_gate


def test_gain_tuner_equations_preserve_si_dimensions() -> None:
    result = derive_gain_tuner_pd(
        effective_inertia_si=0.25,
        natural_frequency_hz=2.0,
        damping_ratio=1.0,
        joint_type="prismatic",
    )

    expected_kp = 0.25 * (2.0 * math.pi * 2.0) ** 2
    expected_kd = 2.0 * math.sqrt(0.25 * expected_kp)
    assert result["stiffness_N_per_m"] == pytest.approx(expected_kp)
    assert result["damping_N_s_per_m"] == pytest.approx(expected_kd)
    assert result["equation_source"] == "ISAAC_SIM_5_1_GAIN_TUNER_3_0_6"


def test_force_drive_gate_does_not_invent_missing_physical_inputs() -> None:
    result = evaluate_force_drive_candidate_gate(
        convergence_status="PARTIAL",
        effective_inertia_si=None,
        natural_frequency_hz=None,
        damping_ratio=None,
        continuous_force_limit=None,
        linkage_efficiency=None,
    )

    assert result["status"] == "HARD_BLOCKER"
    assert result["candidate_authored"] is False
    assert result["runtime_scan_allowed"] is False
    assert result["missing_inputs"] == [
        "CONVERGED_TIMESTEP_AND_SOLVER_COUNTS",
        "GRIPPER_EFFECTIVE_MASS_AT_DECLARED_CONFIGURATION",
        "DECLARED_OR_IDENTIFIED_CLOSED_LOOP_NATURAL_FREQUENCY",
        "DECLARED_OR_IDENTIFIED_DAMPING_RATIO",
        "CONTINUOUS_FORCE_LIMIT_NOT_STALL_OR_MOMENTARY_LIMIT",
        "LOADED_GRIPPER_LINKAGE_EFFICIENCY",
    ]


def test_complete_inputs_create_diagnostic_candidate_only() -> None:
    result = evaluate_force_drive_candidate_gate(
        convergence_status="PASS",
        effective_inertia_si=0.25,
        natural_frequency_hz=2.0,
        damping_ratio=1.0,
        continuous_force_limit=1.5,
        linkage_efficiency=0.8,
    )

    assert result["status"] == "DIAGNOSTIC_CANDIDATE_READY"
    assert result["candidate_authored"] is False
    assert result["runtime_scan_allowed"] is True
    assert result["promotion_allowed"] is False
    assert result["candidate"]["max_force"] == pytest.approx(1.2)
