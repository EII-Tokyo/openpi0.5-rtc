from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.aloha1_mapping.supplier_cad_grasp_frame import compare_brep_mesh_pad_evidence
from tools.aloha1_mapping.supplier_cad_grasp_frame import derive_supplier_cad_grasp_frame
from tools.aloha1_mapping.supplier_cad_grasp_frame import load_verified_clearance_grasp_frame

ROOT = Path(__file__).resolve().parents[2]
MESH_ROOT = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "viper_gripper/tessellation_angular_controlled/run_a"
)
BREP_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_supplier_cad_grasp_frame_brep.json"
)
CLEARANCE_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_supplier_cad_grasp_clearance.json"
)
SCREENSHOT_REVIEW = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_supplier_cad_grasp_clearance_screenshot_review.json"
)


def test_supplier_cad_pad_pair_selects_largest_connected_inward_planes() -> None:
    report = derive_supplier_cad_grasp_frame(
        left_obj_path=MESH_ROOT / "left_finger.obj",
        right_obj_path=MESH_ROOT / "right_finger.obj",
    )

    assert report["status"] == "PASS"
    left = report["fingers"]["left"]["selected_inner_pad"]
    right = report["fingers"]["right"]["selected_inner_pad"]
    assert left["triangle_count"] == 20
    assert right["triangle_count"] == 20
    assert left["area_m2"] == pytest.approx(0.002020895990580424)
    assert right["area_m2"] == pytest.approx(0.002020896011081235)
    assert left["area_ratio_to_next_candidate"] > 9.0
    assert right["area_ratio_to_next_candidate"] > 9.0
    assert np.dot(left["normal_finger_link"], [0.0, -1.0, 0.0]) > 0.99
    assert np.dot(right["normal_finger_link"], [0.0, 1.0, 0.0]) > 0.99


def test_supplier_cad_pad_midpoint_is_not_the_ee_helper_origin() -> None:
    report = derive_supplier_cad_grasp_frame(
        left_obj_path=MESH_ROOT / "left_finger.obj",
        right_obj_path=MESH_ROOT / "right_finger.obj",
    )

    pair = report["closed_reference_pair"]
    assert pair["left_center_gripper_reference_m"] == pytest.approx(
        [0.1112718957, 0.0072553929, 0.0000513269],
        abs=2e-9,
    )
    assert pair["right_center_gripper_reference_m"] == pytest.approx(
        [0.1112718954, -0.0072553930, -0.0000513334],
        abs=2e-9,
    )
    assert pair["midpoint_gripper_reference_m"] == pytest.approx(
        [0.11127189555, 0.0, 0.0],
        abs=5e-9,
    )
    assert pair["midpoint_ee_gripper_frame_m"] == pytest.approx(
        [0.00407189555, 0.0, 0.0],
        abs=5e-9,
    )
    assert pair["center_line_length_m"] == pytest.approx(
        0.014511149,
        abs=2e-9,
    )
    assert report["frame_semantics"]["official_ee_helper_is_pad_center"] is False


def test_supplier_cad_pad_pair_preserves_handed_symmetry_without_mirroring() -> None:
    report = derive_supplier_cad_grasp_frame(
        left_obj_path=MESH_ROOT / "left_finger.obj",
        right_obj_path=MESH_ROOT / "right_finger.obj",
    )

    symmetry = report["closed_reference_pair"]["symmetry"]
    assert symmetry["mirror_operation_applied"] is False
    assert symmetry["center_x_residual_m"] < 1e-9
    assert symmetry["center_y_sum_abs_m"] < 1e-9
    assert symmetry["center_z_sum_abs_m"] < 1e-8
    assert symmetry["normal_handed_pair_residual"] < 1e-8


def test_brep_and_controlled_mesh_identify_the_same_inner_pad_faces() -> None:
    mesh_report = derive_supplier_cad_grasp_frame(
        left_obj_path=MESH_ROOT / "left_finger.obj",
        right_obj_path=MESH_ROOT / "right_finger.obj",
    )
    brep_report = json.loads(BREP_REPORT.read_text(encoding="utf-8"))

    comparison = compare_brep_mesh_pad_evidence(
        mesh_report=mesh_report,
        brep_report=brep_report,
    )

    assert comparison["status"] == "PASS"
    assert comparison["maximum_centroid_residual_m"] < 2e-8
    assert comparison["maximum_normal_angle_deg"] < 1e-4
    assert comparison["maximum_relative_area_error"] < 1e-6


def test_runtime_frame_requires_frozen_complete_gripper_clearance_gate() -> None:
    frame = load_verified_clearance_grasp_frame(
        clearance_report_path=CLEARANCE_REPORT,
        screenshot_review_path=SCREENSHOT_REVIEW,
        expected_clearance_sha256=(
            "9f23974af362dc92134a38633180360bfff8b54bc0a5eaefae8032e2240b91bc"
        ),
        expected_screenshot_sha256=(
            "c7097b05654a3966c976690e5f0f79c3b2be69eaa51727f430998efae2bbe0f3"
        ),
    )

    assert frame["status"] == "PASS"
    assert frame["classification"] == (
        "FROZEN_SUPPLIER_CAD_COMPLETE_GRIPPER_CLEARANCE_FRAME"
    )
    assert frame["origin_reference_m"] == pytest.approx(
        [0.13552080444282988, 0.0, 0.0],
        abs=1e-12,
    )
    assert frame["bottle_axis_center_from_grasp_m"] == pytest.approx(
        [-0.003365816517218456, 0.0, 0.0],
        abs=1e-12,
    )
    assert frame["finger_targets_m"] == pytest.approx(
        {
            "left_finger": 0.048316874538855845,
            "right_finger": -0.048316874538855845,
        },
        abs=1e-12,
    )
    assert frame["official_ee_helper_semantics"] == "NOT_GRASP_CENTER"
    assert frame["whole_pad_face_centroid_use"] == "REJECTED"
    assert frame["screenshot_gate"]["status"] == "PASS"
    assert frame["screenshot_gate"]["user_confirmed"] is True
    assert frame["rotation_determinant"] == pytest.approx(1.0)
    assert frame["task8"] == "NOT_RUN"
