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
        "arm_static": "position",
        "gripper_static": "pwm",
        "follower_gripper_runtime": "current_based_position",
        "static_source": "interbotix_xsarm_default_modes",
        "runtime_source": "interbotix_aloha_runtime_robot_utils",
        "status": "VERIFIED_RUNTIME_OVERRIDE",
    }
    actuators = report["joint_actuator_identity"]
    assert actuators["forearm_roll"]["model"] == "XM540-W270"
    assert actuators["wrist_angle"]["model"] == "XM540-W270"
    assert actuators["wrist_rotate"]["model"] == "XM430-W350"
    assert actuators["gripper"]["model"] == "XM430-W350"
    gripper = report["gripper_control_boundary"]
    assert gripper["operating_mode"] == "current_based_position"
    assert gripper["configuration_default_current_limit"] == {
        "ticks": 200,
        "ampere": 0.538,
        "source": "interbotix_aloha_vx300s_motor_config",
    }
    assert gripper["pipeline_current_limit_overrides"] == [
        {
            "pipeline": "official_aloha_dual_side_teleop",
            "ticks": 300,
            "ampere": 0.807,
            "applies_to": ["follower_left", "follower_right"],
            "source": "interbotix_aloha_runtime_dual_side_teleop",
        }
    ]
    assert gripper["current_limit_selection"] == "PIPELINE_SCOPED"
    assert gripper["current_limit_is_physx_max_force"] is False
    assert gripper["hardware_current_to_physx_force_mapping"] == "NOT_DIRECT"


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
