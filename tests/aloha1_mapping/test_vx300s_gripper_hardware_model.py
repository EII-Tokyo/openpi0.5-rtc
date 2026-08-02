from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/aloha1_vx300s_gripper_hardware_model.yaml"
TOOL = ROOT / "tools/build_aloha1_vx300s_gripper_hardware_audit.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "build_aloha1_vx300s_gripper_hardware_audit",
        TOOL,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_stationary_aloha1_follower_and_motor_identity() -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert document["scope"]["generation"] == "STATIONARY_ALOHA_1"
    assert document["scope"]["follower_product"] == "ViperX-300 6DOF"
    assert document["scope"]["project_model"] == "aloha_vx300s"
    assert document["scope"]["left_right_relation"] == ("TWO_INSTANCES_OF_THE_SAME_ROBOT_PRODUCT")
    assert document["scope"]["excluded_related_model"]["model"] == "avx300s"
    assert document["scope"]["excluded_related_model"]["reason"] == (
        "ALOHA_2_VARIANT_NOT_APPLICABLE_TO_STATIONARY_ALOHA_1"
    )

    actuator = document["hardware"]["gripper_actuator"]
    assert actuator["manufacturer"] == "ROBOTIS"
    assert actuator["model"] == "XM430-W350"
    assert actuator["dynamixel_id"] == 9
    assert actuator["physical_actuator_count"] == 1
    assert actuator["operating_mode"] == "pwm"
    assert actuator["right_finger_independently_sensed"] is False


def test_linkage_formula_signs_and_urdf_limits_are_explicit() -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    linkage = document["hardware"]["gripper_linkage"]

    assert linkage["horn_radius_m"] == pytest.approx(0.0275)
    assert linkage["arm_length_m"] == pytest.approx(0.035)
    assert linkage["published_finger_positions"] == {
        "left_finger": "+x",
        "right_finger": "-x",
    }
    assert linkage["right_finger_state_kind"] == ("DRIVER_DERIVED_NOT_INDEPENDENT_SENSOR")
    assert linkage["motor_angle_to_linear_formula"] == ("x=r*sin(theta)+sqrt(L^2-(sqrt(r^2-(r*sin(theta))^2))^2)")

    mimic = document["simulation_description"]["urdf_mimic"]
    assert mimic == {
        "parent_joint": "left_finger",
        "child_joint": "right_finger",
        "multiplier": -1.0,
        "offset_m": 0.0,
    }
    limits = document["simulation_description"]["urdf_finger_limits"]
    assert limits["left_lower_m"] == pytest.approx(0.021)
    assert limits["left_upper_m"] == pytest.approx(0.057)
    assert limits["right_lower_m"] == pytest.approx(-0.057)
    assert limits["right_upper_m"] == pytest.approx(-0.021)
    assert limits["velocity_m_per_s"] == pytest.approx(1.0)
    assert limits["effort_source_value"] == pytest.approx(5.0)
    assert limits["effort_physical_calibration_status"] == "NOT_ESTABLISHED"
    assert document["simulation_description"]["joint_dynamics"] == {
        "damping": 0.1,
        "friction": 0.1,
        "mapping_status": "URDF_SOURCE_VALUE_NOT_HARDWARE_CALIBRATION",
    }


def test_register_units_and_voltage_conditions_are_not_conflated_with_physx() -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    actuator = document["hardware"]["gripper_actuator"]

    assert actuator["current_limit"]["raw_value"] == 200
    assert actuator["current_limit"]["unit_ampere_per_tick"] == pytest.approx(0.00269)
    assert actuator["current_limit"]["derived_limit_ampere"] == pytest.approx(0.538)
    assert actuator["current_limit"]["physx_max_force_mapping"] == ("NOT_DIRECTLY_MAPPABLE")
    assert actuator["pwm"]["unit_percent_per_tick"] == pytest.approx(0.113)
    assert actuator["pwm"]["manual_default_limit_raw"] == 885
    assert actuator["pwm"]["pid_behavior"] == ("PID_AND_FEEDFORWARD_DEACTIVATED_IN_PWM_CONTROL_MODE")

    performance = actuator["performance"]
    assert performance["stall_torque_is_continuous_rating"] is False
    assert performance["conditions"] == [
        {"voltage_v": 11.1, "stall_torque_nm": 3.8, "stall_current_a": 2.1},
        {"voltage_v": 12.0, "stall_torque_nm": 4.1, "stall_current_a": 2.3},
        {"voltage_v": 14.8, "stall_torque_nm": 4.8, "stall_current_a": 2.7},
    ]


def test_official_aperture_conflict_remains_fail_closed() -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    conflict = document["hardware"]["maximum_aperture_conflict"]

    assert {item["value_m"] for item in conflict["claims"]} == {0.114, 0.116}
    assert all(item["measurement"] == "CARRIAGE_CENTER_TO_CENTER" for item in conflict["claims"])
    assert conflict["selection_status"] == (
        "RESOLVED_IMPLEMENTED_URDF_AND_CAD_CARRIAGE_DATUM"
    )
    assert conflict["selected_value_m"] == 0.114
    assert conflict["product_page_conflict_retained"] is True


def test_frozen_sources_have_required_provenance_and_hashes() -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    sources = document["frozen_sources"]
    required = {
        "aloha_vx300s_motor_config",
        "xsarm_modes",
        "aloha_vx300s_xacro",
        "xs_driver",
        "xs_sdk_obj",
        "supplier_cad",
    }
    assert required <= sources.keys()

    for name in required:
        source = sources[name]
        assert source["repository"]
        assert source["branch_or_tag"]
        assert source["commit"]
        assert source["license"]
        assert source["local_path"]
        assert len(source["sha256"]) == 64

    assert sources["xs_driver"]["repository"] == ("https://github.com/Interbotix/interbotix_xs_driver.git")
    assert sources["xs_driver"]["commit"] == ("da27b8b2b6c7677844f74581b82c01829a834e1c")
    assert sources["aloha_vx300s_motor_config"]["upstream_drift"]["only_changed_key"] == "sleep_positions"


def test_builder_verifies_sources_formula_and_provenance(tmp_path: Path) -> None:
    module = _load_tool()
    report = module.build_report(project_root=ROOT)

    assert report["status"] == "PASS"
    assert report["official_hardware_model_status"] == "PASS"
    assert report["source_verification"]["all_local_hashes_match"] is True
    assert report["source_verification"]["missing_sources"] == []
    assert report["source_verification"]["hash_mismatches"] == []
    assert report["formula_validation"]["status"] == "PASS"
    assert report["formula_validation"]["monotonic_over_urdf_range"] is True
    assert report["formula_validation"]["right_is_negative_left"] is True
    assert math.isfinite(report["formula_validation"]["lower_motor_angle_rad"])
    assert math.isfinite(report["formula_validation"]["upper_motor_angle_rad"])
    assert report["provenance_classes"]["runtime_readback"] == []
    assert report["unconfirmed_physical_quantities"]
    assert report["task8"] == "NOT_RUN"

    json_path = tmp_path / "audit.json"
    md_path = tmp_path / "audit.md"
    written = module.write_reports(report, json_path=json_path, md_path=md_path)
    assert written == {"json": str(json_path), "markdown": str(md_path)}
    assert json_path.is_file()
    assert md_path.is_file()
