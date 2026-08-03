"""Dimensionally explicit Isaac 5.1 force-drive candidate gates."""

from __future__ import annotations

import math
from typing import Any


def derive_gain_tuner_pd(
    *,
    effective_inertia_si: float,
    natural_frequency_hz: float,
    damping_ratio: float,
    joint_type: str,
) -> dict[str, Any]:
    """Apply the local Gain Tuner 3.0.6 equations with explicit units."""

    inertia = float(effective_inertia_si)
    frequency = float(natural_frequency_hz)
    ratio = float(damping_ratio)
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (inertia, frequency, ratio)
    ):
        raise ValueError("gain derivation inputs must be finite and positive")
    if joint_type not in {"revolute", "prismatic"}:
        raise ValueError("joint_type must be revolute or prismatic")
    omega_n = 2.0 * math.pi * frequency
    stiffness = inertia * omega_n**2
    damping = 2.0 * ratio * math.sqrt(inertia * stiffness)
    unit_fields = (
        {
            "effective_inertia_kg_m2": inertia,
            "stiffness_Nm_per_rad": stiffness,
            "damping_Nm_s_per_rad": damping,
        }
        if joint_type == "revolute"
        else {
            "effective_mass_kg": inertia,
            "stiffness_N_per_m": stiffness,
            "damping_N_s_per_m": damping,
        }
    )
    return {
        "joint_type": joint_type,
        "natural_frequency_hz": frequency,
        "damping_ratio": ratio,
        "angular_natural_frequency_rad_s": omega_n,
        **unit_fields,
        "equation_source": "ISAAC_SIM_5_1_GAIN_TUNER_3_0_6",
        "hardware_integer_gain_direct_mapping": "PROHIBITED",
    }


def evaluate_force_drive_candidate_gate(
    *,
    convergence_status: str,
    effective_inertia_si: float | None,
    natural_frequency_hz: float | None,
    damping_ratio: float | None,
    continuous_force_limit: float | None,
    linkage_efficiency: float | None,
) -> dict[str, Any]:
    """Fail closed unless every physical and numerical input is sourced."""

    missing: list[str] = []
    if convergence_status != "PASS":
        missing.append("CONVERGED_TIMESTEP_AND_SOLVER_COUNTS")
    if effective_inertia_si is None:
        missing.append("GRIPPER_EFFECTIVE_MASS_AT_DECLARED_CONFIGURATION")
    if natural_frequency_hz is None:
        missing.append(
            "DECLARED_OR_IDENTIFIED_CLOSED_LOOP_NATURAL_FREQUENCY"
        )
    if damping_ratio is None:
        missing.append("DECLARED_OR_IDENTIFIED_DAMPING_RATIO")
    if continuous_force_limit is None:
        missing.append(
            "CONTINUOUS_FORCE_LIMIT_NOT_STALL_OR_MOMENTARY_LIMIT"
        )
    if linkage_efficiency is None:
        missing.append("LOADED_GRIPPER_LINKAGE_EFFICIENCY")
    if missing:
        return {
            "status": "HARD_BLOCKER",
            "missing_inputs": missing,
            "candidate": None,
            "candidate_authored": False,
            "runtime_scan_allowed": False,
            "promotion_allowed": False,
        }
    assert effective_inertia_si is not None
    assert natural_frequency_hz is not None
    assert damping_ratio is not None
    assert continuous_force_limit is not None
    assert linkage_efficiency is not None
    efficiency = float(linkage_efficiency)
    force_limit = float(continuous_force_limit)
    if (
        not math.isfinite(efficiency)
        or not 0.0 < efficiency <= 1.0
        or not math.isfinite(force_limit)
        or force_limit <= 0.0
    ):
        raise ValueError("force limit and linkage efficiency are invalid")
    gains = derive_gain_tuner_pd(
        effective_inertia_si=float(effective_inertia_si),
        natural_frequency_hz=float(natural_frequency_hz),
        damping_ratio=float(damping_ratio),
        joint_type="prismatic",
    )
    return {
        "status": "DIAGNOSTIC_CANDIDATE_READY",
        "missing_inputs": [],
        "candidate": {
            **gains,
            "max_force": force_limit * efficiency,
            "max_force_formula": (
                "continuous_actuator_side_equivalent_force * "
                "loaded_linkage_efficiency"
            ),
            "drive_type": "force",
            "classification": "DIAGNOSTIC_ONLY_NOT_FINAL",
        },
        "candidate_authored": False,
        "runtime_scan_allowed": True,
        "promotion_allowed": False,
    }
