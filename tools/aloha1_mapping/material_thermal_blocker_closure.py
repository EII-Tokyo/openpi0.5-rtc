"""Pure evidence gate for ALOHA1 contact material and continuous-duty data."""

from __future__ import annotations

from typing import Any


def classify_material_thermal_gate(
    *,
    material_binding_status: str,
    temporary_material_status: str,
    finger_material_identity: dict[str, Any] | None,
    bottle_material_identity: dict[str, Any] | None,
    pair_friction_measurement: dict[str, Any] | None,
    measured_continuous_thermal_curve: dict[str, Any] | None,
) -> dict[str, Any]:
    """Separate runtime material binding from physical calibration evidence."""

    missing: list[str] = []
    if finger_material_identity is None:
        missing.append("EXACT_FINGER_PAD_MATERIAL_AND_SURFACE_FINISH")
    if bottle_material_identity is None:
        missing.append("EXACT_BOTTLE_MATERIAL_AND_SURFACE_FINISH")
    if pair_friction_measurement is None:
        missing.append("EXACT_PAIR_STATIC_DYNAMIC_FRICTION_AND_RESTITUTION")
    if measured_continuous_thermal_curve is None:
        missing.append(
            "MEASURED_CONTINUOUS_TORQUE_SPEED_CURRENT_THERMAL_CURVE"
        )
    runtime_binding_verified = material_binding_status == "PASS"
    physical_friction_calibrated = (
        runtime_binding_verified
        and temporary_material_status != "TEMPORARY_UNCALIBRATED"
        and finger_material_identity is not None
        and bottle_material_identity is not None
        and pair_friction_measurement is not None
    )
    continuous_force_envelope_verified = (
        measured_continuous_thermal_curve is not None
    )
    return {
        "status": (
            "PASS"
            if physical_friction_calibrated
            and continuous_force_envelope_verified
            else "HARD_BLOCKER"
        ),
        "runtime_binding_verified": runtime_binding_verified,
        "physical_friction_calibrated": physical_friction_calibrated,
        "continuous_force_envelope_verified": continuous_force_envelope_verified,
        "missing_inputs": missing,
        "parameter_scan_allowed": physical_friction_calibrated,
        "promotion_allowed": physical_friction_calibrated
        and continuous_force_envelope_verified,
    }
