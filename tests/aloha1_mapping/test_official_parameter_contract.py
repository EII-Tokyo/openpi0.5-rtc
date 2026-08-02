from __future__ import annotations

import copy
import json
from pathlib import Path

from tools.aloha1_mapping.official_parameter_contract import REQUIRED_PARAMETER_GROUPS
from tools.aloha1_mapping.official_parameter_contract import build_parameter_matrix
from tools.aloha1_mapping.official_parameter_contract import candidate_gate
from tools.aloha1_mapping.official_parameter_contract import validate_parameter_records
from tools.aloha1_mapping.official_parameter_sources import load_source_manifest

ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = ROOT / "configs/aloha1_official_parameter_sources.yaml"
REPORT = ROOT / "reports/aloha1_mapping/aloha1_official_parameter_matrix.json"


def _record(*, record_id: str = "joint_order", group: str = "joint_kinematics") -> dict:
    return {
        "id": record_id,
        "group": group,
        "status": "OFFICIAL_PINNED_SOURCE",
        "value": ["waist", "shoulder"],
        "units": "ordered_names",
        "frame": "robot_local",
        "sign_convention": "explicit_order_not_alphabetical",
        "applicability": {
            "product": "aloha_vx300s",
            "instances": ["follower_left", "follower_right"],
        },
        "evidence_class": "OFFICIAL_PINNED_SOURCE",
        "source_ids": ["interbotix_aloha_vx300s_motor_config"],
        "source_locator": "joint_order",
        "derivation": {
            "kind": "DIRECT_TRANSCRIPTION",
            "formula": None,
            "inputs": ["interbotix_aloha_vx300s_motor_config:joint_order"],
        },
        "conflict_state": "NONE",
    }


def test_required_groups_cover_geometry_hardware_physics_and_contact() -> None:
    assert {
        "link_geometry",
        "joint_kinematics",
        "link_dynamics",
        "actuator_identity",
        "actuator_performance",
        "register_conversions",
        "operating_modes",
        "gripper_linkage",
        "drive_mapping",
        "collision_geometry",
        "contact_materials",
        "solver_semantics",
    } == REQUIRED_PARAMETER_GROUPS


def test_schema_rejects_missing_units_provenance_and_derivation_inputs() -> None:
    record = _record()
    record.pop("units")
    record["source_ids"] = []
    record["derivation"]["inputs"] = []

    findings = validate_parameter_records([record], approved_source_ids={"unused"})

    codes = {finding["code"] for finding in findings}
    assert "PARAMETER_FIELD_MISSING" in codes
    assert "PARAMETER_SOURCE_MISSING" in codes
    assert "DERIVATION_INPUTS_MISSING" in codes


def test_candidate_gate_rejects_inference_temporary_and_diagnostic_values() -> None:
    records = []
    for index, evidence_class in enumerate(
        ["ENGINEERING_INFERENCE", "TEMPORARY_UNCALIBRATED", "DIAGNOSTIC_ONLY_NOT_FINAL"]
    ):
        record = _record(record_id=f"forbidden_{index}")
        record["evidence_class"] = evidence_class
        records.append(record)

    gate = candidate_gate(records)

    assert gate["status"] == "BLOCKED"
    assert {item["evidence_class"] for item in gate["blocking_records"]} == {
        "ENGINEERING_INFERENCE",
        "TEMPORARY_UNCALIBRATED",
        "DIAGNOSTIC_ONLY_NOT_FINAL",
    }


def test_hard_blocker_is_narrow_and_contains_no_convenient_numeric_value() -> None:
    record = _record(record_id="finger_friction", group="contact_materials")
    record["status"] = "HARD_BLOCKER"
    record["evidence_class"] = "HARD_BLOCKER"
    record.pop("value")
    record["blocker"] = {
        "id": "HARD_BLOCKER_FINGER_BOTTLE_FRICTION_EXACT_VALUE",
        "missing_definition": "exact material pair static/dynamic friction",
        "blocks": ["formal_contact_material"],
        "does_not_block": ["kinematic_contract"],
    }

    findings = validate_parameter_records([record], approved_source_ids={"interbotix_aloha_vx300s_motor_config"})

    assert findings == []
    assert "value" not in record


def test_repository_matrix_is_complete_and_exact_model_scoped() -> None:
    source_manifest = load_source_manifest(SOURCE_MANIFEST)
    matrix = build_parameter_matrix(source_manifest, repository_root=ROOT)

    assert matrix["status"] == "PASS"
    assert set(matrix["coverage"]) == REQUIRED_PARAMETER_GROUPS
    assert all(item["record_count"] > 0 for item in matrix["coverage"].values())
    assert matrix["product"]["project_model"] == "aloha_vx300s"
    assert matrix["formal_parameter_candidate_gate"]["status"] == "BLOCKED"
    assert matrix["hard_blocker_count"] > 0
    assert not matrix["forbidden_formal_values"]

    aperture = next(
        record
        for record in matrix["records"]
        if record["id"] == "gripper_aperture_definition_conflict"
    )
    assert aperture["status"] == "VERIFIED_DERIVATION"
    assert aperture["value"]["implemented_carriage_center_range_m"] == [
        0.042,
        0.114,
    ]
    assert aperture["conflict_state"] == (
        "VERIFIED_OFFICIAL_SOURCE_CONFLICT_PRODUCT_PAGE_NOT_CAD_SUPPORTED"
    )

    actuator_map = next(record for record in matrix["records"] if record["id"] == "actuator_id_model_map")
    assert actuator_map["value"]["1"] == "XM540-W270"
    assert actuator_map["value"]["7"] == "XM540-W270"
    assert actuator_map["value"]["8"] == "XM430-W350"
    assert actuator_map["value"]["9"] == "XM430-W350"

    continuous = {
        record["id"]: record
        for record in matrix["records"]
        if record["id"].startswith("estimated_continuous_torque.")
    }
    assert continuous["estimated_continuous_torque.XM540-W270"]["value"] == {
        "reference_voltage_V": 12.0,
        "estimated_continuous_torque_Nm": 2.12,
        "fraction_of_stall": 0.2,
        "manufacturer_estimate_not_measured_thermal_curve": True,
    }
    assert continuous["estimated_continuous_torque.XM430-W350"]["value"][
        "estimated_continuous_torque_Nm"
    ] == 0.82
    envelope = next(
        record
        for record in matrix["records"]
        if record["id"] == "continuous_actuator_envelope"
    )
    assert envelope["blocker"]["missing_definition"] == (
        "measured continuous torque-speed-current thermal envelope beyond the "
        "official 12 V 20%-of-stall estimates"
    )
    geometry_sources = next(
        record
        for record in matrix["records"]
        if record["id"] == "cad_to_robot_link_geometry_correspondence"
    )
    assert geometry_sources["status"] == "VERIFIED_DERIVATION"
    assert geometry_sources["value"]["physical_link_source_count"] == 11
    collision = next(
        record
        for record in matrix["records"]
        if record["id"] == "formal_collision_geometry"
    )
    assert collision["blocker"]["id"] == (
        "HARD_BLOCKER_COLLIDER_ACCEPTANCE_ERROR_BUDGET"
    )
    assert matrix["hard_blocker_count"] == 5

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["deterministic_signature"] == matrix["deterministic_signature"]


def test_changed_record_changes_deterministic_signature() -> None:
    source_manifest = load_source_manifest(SOURCE_MANIFEST)
    first = build_parameter_matrix(source_manifest, repository_root=ROOT)
    changed_manifest = copy.deepcopy(source_manifest)
    changed_manifest["product"]["manufacturer"] = "not-the-approved-manufacturer"
    second = build_parameter_matrix(changed_manifest, repository_root=ROOT)

    assert first["deterministic_signature"] != second["deterministic_signature"]
