from __future__ import annotations

import json
from pathlib import Path

from tools.aloha1_mapping.actuator_drive_source_boundary import build_report

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "configs/aloha1_official_parameter_sources.yaml"
REPORT = ROOT / "reports/aloha1_mapping/aloha1_actuator_drive_source_boundary.json"


def _build() -> dict[str, object]:
    return build_report(ROOT, SOURCES)


def test_exact_model_continuous_estimates_are_preserved_with_disclosure() -> None:
    report = _build()
    models = report["actuator_models"]

    assert models["XM540-W270"]["estimated_continuous_torque_Nm"] == 2.12
    assert models["XM430-W350"]["estimated_continuous_torque_Nm"] == 0.82
    for model in models.values():
        assert model["reference_voltage_V"] == 12.0
        assert model["continuous_estimate_fraction_of_stall"] == 0.2
        assert model["continuous_value_is_manufacturer_estimate"] is True
        assert model["continuous_value_is_measured_thermal_curve"] is False
        assert model["stall_torque_used_as_continuous"] is False


def test_exact_aloha_modes_and_gripper_limit_semantics_are_not_conflated() -> None:
    report = _build()

    assert report["control_modes"] == {
        "arm": "position",
        "gripper": "pwm",
        "source": "interbotix_xsarm_default_modes",
    }
    actuators = report["joint_actuator_identity"]
    assert actuators["forearm_roll"]["model"] == "XM540-W270"
    assert actuators["wrist_angle"]["model"] == "XM540-W270"
    assert actuators["wrist_rotate"]["model"] == "XM430-W350"
    assert actuators["gripper"]["model"] == "XM430-W350"
    gripper = report["gripper_control_boundary"]
    assert gripper["current_limit_ticks"] == 200
    assert gripper["current_limit_A"] == 0.538
    assert gripper["current_limit_is_physx_max_force"] is False
    assert gripper["pwm_command_to_output_torque_mapping"] == "NOT_DEFINED_BY_OFFICIAL_SOURCES"


def test_integer_controller_gains_are_not_labeled_physical_drive_gains() -> None:
    report = _build()
    mapping = report["physx_drive_mapping"]

    assert mapping["hardware_integer_gain_direct_mapping"] == "PROHIBITED"
    assert mapping["stiffness_Nm_per_rad"] is None
    assert mapping["damping_Nm_s_per_rad"] is None
    assert mapping["status"] == "HARD_BLOCKER"
    assert report["status"] == "PARTIAL"


def test_report_is_deterministic_and_matches_repository_copy() -> None:
    first = _build()
    second = _build()
    frozen = json.loads(REPORT.read_text(encoding="utf-8"))

    assert first["deterministic_signature"] == second["deterministic_signature"]
    assert frozen["deterministic_signature"] == first["deterministic_signature"]
