from __future__ import annotations

from tools.aloha1_mapping.material_thermal_blocker_closure import classify_material_thermal_gate


def test_material_binding_pass_does_not_calibrate_physical_friction() -> None:
    result = classify_material_thermal_gate(
        material_binding_status="PASS",
        temporary_material_status="TEMPORARY_UNCALIBRATED",
        finger_material_identity=None,
        bottle_material_identity=None,
        pair_friction_measurement=None,
        measured_continuous_thermal_curve=None,
    )

    assert result["status"] == "HARD_BLOCKER"
    assert result["runtime_binding_verified"] is True
    assert result["physical_friction_calibrated"] is False
    assert result["continuous_force_envelope_verified"] is False
    assert result["missing_inputs"] == [
        "EXACT_FINGER_PAD_MATERIAL_AND_SURFACE_FINISH",
        "EXACT_BOTTLE_MATERIAL_AND_SURFACE_FINISH",
        "EXACT_PAIR_STATIC_DYNAMIC_FRICTION_AND_RESTITUTION",
        "MEASURED_CONTINUOUS_TORQUE_SPEED_CURRENT_THERMAL_CURVE",
    ]


def test_complete_physical_sources_close_material_and_thermal_gate() -> None:
    result = classify_material_thermal_gate(
        material_binding_status="PASS",
        temporary_material_status="CALIBRATED",
        finger_material_identity={"model": "exact pad"},
        bottle_material_identity={"model": "exact bottle"},
        pair_friction_measurement={"static": 0.4, "dynamic": 0.3},
        measured_continuous_thermal_curve={"source": "measured"},
    )

    assert result["status"] == "PASS"
    assert result["physical_friction_calibrated"] is True
    assert result["continuous_force_envelope_verified"] is True
