from __future__ import annotations

import json
from pathlib import Path

from tools.derive_aloha1_dynamics_contract import build_contract

ROOT = Path(__file__).resolve().parents[2]
LEFT_URDF = ROOT / "generated/urdf/follower_left.urdf"
RIGHT_URDF = ROOT / "generated/urdf/follower_right.urdf"
SOURCES = ROOT / "configs/aloha1_official_parameter_sources.yaml"
MATRIX = ROOT / "reports/aloha1_mapping/aloha1_official_parameter_matrix.json"
REPORT = ROOT / "reports/aloha1_mapping/aloha1_dynamics_contract.json"


def test_all_authored_inertials_are_finite_positive_and_physically_feasible() -> None:
    contract = build_contract(
        left_urdf=LEFT_URDF,
        right_urdf=RIGHT_URDF,
        source_manifest_path=SOURCES,
        parameter_matrix_path=MATRIX,
    )

    assert contract["inertial_contract"]["status"] == "PASS"
    assert contract["inertial_contract"]["link_count_per_follower"] == 14
    assert contract["inertial_contract"]["minimum_mass_kg"] > 0.0
    assert contract["inertial_contract"]["minimum_principal_moment_kg_m2"] > 0.0
    assert contract["inertial_contract"]["minimum_triangle_margin_kg_m2"] >= 0.0
    assert contract["inertial_contract"]["max_parallel_axis_roundtrip_error"] <= 1e-15


def test_left_and_right_inertial_records_are_identical_not_mirrored() -> None:
    contract = build_contract(
        left_urdf=LEFT_URDF,
        right_urdf=RIGHT_URDF,
        source_manifest_path=SOURCES,
        parameter_matrix_path=MATRIX,
    )

    assert contract["left_right_inertial_identity"]["status"] == "PASS"
    assert contract["left_right_inertial_identity"]["mirrored"] is False


def test_stall_torque_is_not_promoted_to_continuous_or_physx_max_force() -> None:
    contract = build_contract(
        left_urdf=LEFT_URDF,
        right_urdf=RIGHT_URDF,
        source_manifest_path=SOURCES,
        parameter_matrix_path=MATRIX,
    )

    assert contract["actuator_contract"]["manufacturer_tables_status"] == "PASS"
    assert contract["actuator_contract"]["stall_torque_used_as_continuous"] is False
    assert contract["actuator_contract"]["continuous_joint_envelope_status"] == "HARD_BLOCKER"
    assert contract["actuator_contract"]["physx_drive_mapping_status"] == "HARD_BLOCKER"
    assert contract["status"] == "PARTIAL"


def test_shadow_actuators_and_gripper_current_conversion_are_explicit() -> None:
    contract = build_contract(
        left_urdf=LEFT_URDF,
        right_urdf=RIGHT_URDF,
        source_manifest_path=SOURCES,
        parameter_matrix_path=MATRIX,
    )

    assert contract["actuator_contract"]["shadow_semantics"] == {
        "shoulder": {"primary_id": 2, "shadow_id": 3, "secondary_id": 2},
        "elbow": {"primary_id": 4, "shadow_id": 5, "secondary_id": 4},
    }
    assert contract["actuator_contract"]["gripper_current_limit"]["raw_ticks"] == 200
    assert contract["actuator_contract"]["gripper_current_limit"]["derived_ampere"] == 0.538
    assert contract["actuator_contract"]["gripper_current_limit"]["physx_max_force_mapping"] == "NOT_DIRECTLY_MAPPABLE"


def test_repository_report_matches_deterministic_contract() -> None:
    contract = build_contract(
        left_urdf=LEFT_URDF,
        right_urdf=RIGHT_URDF,
        source_manifest_path=SOURCES,
        parameter_matrix_path=MATRIX,
    )
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["deterministic_signature"] == contract["deterministic_signature"]
