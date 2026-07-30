from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.aloha1_mapping.supplier_cad_grasp_clearance import build_right_handed_grasp_frame
from tools.aloha1_mapping.supplier_cad_grasp_clearance import select_chebyshev_grasp_station

ROOT = Path(__file__).resolve().parents[2]
RUN13_REJECTION = (
    ROOT
    / "reports/aloha1_mapping/aloha1_grasp_frame_run13_rejection.json"
)
CLEARANCE_REPORT = (
    ROOT
    / "reports/aloha1_mapping/aloha1_supplier_cad_grasp_clearance.json"
)


def test_run13_rejects_whole_pad_face_centroid_as_grasp_center() -> None:
    payload = json.loads(RUN13_REJECTION.read_text(encoding="utf-8"))

    assert payload["status"] == "FAIL"
    assert (
        payload["classification"]
        == "REJECTED_WHOLE_PAD_FACE_CENTROID_NOT_EFFECTIVE_GRASP_CENTER"
    )
    assert payload["task8"] == "NOT_RUN"

    rejected = payload["rejected_frame"]
    assert rejected["selection_rule"] == (
        "MIDPOINT_OF_WHOLE_LARGEST_INWARD_PLANAR_FACE_CENTROIDS"
    )
    assert rejected["ee_endpoint_is_grasp_center"] is False
    assert rejected["complete_gripper_clearance_was_evaluated"] is False

    runtime = payload["runtime_evidence"]
    assert runtime["native_grasp_editor_success"] is True
    assert runtime["bilateral_finger_contact"] is False
    assert runtime["left_finger_contact"] is False
    assert runtime["right_finger_contact"] is False
    assert runtime["gripper_bar_pair_event_count"] > 0
    assert runtime["maximum_gripper_bar_impulse_ns"] > 0.0

    source_use = payload["supplier_cad_use"]
    assert source_use["handed_finger_brep_used"] is True
    assert source_use["complete_gripper_envelope_used"] is False
    assert source_use["project_bottle_brep_clearance_used"] is False

    for key in ("configured_raw", "simulated_raw"):
        record = payload["screenshots"][key]
        assert Path(record["absolute_path"]).is_file()
        assert len(record["sha256"]) == 64
        assert record["visual_review"] == "FAIL"


def test_chebyshev_station_maximizes_complete_gripper_clearance() -> None:
    result = select_chebyshev_grasp_station(
        pad_interval_m=(0.0774000124731113, 0.15895079372311123),
        forbidden_max_x_m={
            "supplier_gripper_shell": 0.0691000000000021,
            "runtime_urdf_gripper_bar": 0.07652499694824219,
        },
        bottle_radius_m=0.0322,
        pad_inward_normal_x=-0.10452846326765405,
        rejected_station_m=0.11127188479610935,
    )

    assert result["status"] == "PASS"
    assert result["selection_rule"] == (
        "CHEBYSHEV_CENTER_OF_COMPLETE_GRIPPER_FEASIBLE_INTERVAL"
    )
    assert result["controlling_forbidden_envelope"] == (
        "runtime_urdf_gripper_bar"
    )
    assert result["feasible_interval_m"] == pytest.approx(
        [0.10872499694824219, 0.15558497720589276]
    )
    assert result["selected_station_m"] == pytest.approx(
        0.13215498707706747
    )
    assert result["selected_minimum_margin_m"] == pytest.approx(
        0.02342999012882528
    )
    assert result["selected_pad_contact_station_m"] == pytest.approx(
        0.13552080359428592
    )
    assert result["pad_normal_bottle_center_offset_m"] == pytest.approx(
        -0.0033658165172184605
    )
    assert result["rejected_station"]["runtime_rejected"] is True
    assert result["rejected_station"]["hard_clearance_m"] == pytest.approx(
        0.002546887847867163
    )
    assert result["selected_minimum_margin_m"] > (
        9.0 * result["rejected_station"]["hard_clearance_m"]
    )


def test_clearance_station_fails_when_bottle_cannot_fit_pad_interval() -> None:
    result = select_chebyshev_grasp_station(
        pad_interval_m=(0.10, 0.12),
        forbidden_max_x_m={"bar": 0.095},
        bottle_radius_m=0.03,
        pad_inward_normal_x=-0.1,
        rejected_station_m=0.11,
    )

    assert result["status"] == "FAIL"
    assert result["classification"] == "NO_COMPLETE_GRIPPER_FEASIBLE_INTERVAL"
    assert result["selected_station_m"] is None


def test_pad_contact_pair_defines_right_handed_non_ee_frame() -> None:
    station_m = 0.13215498707706747
    result = build_right_handed_grasp_frame(
        left_contact_reference_m=(station_m, 0.0322, 0.0),
        right_contact_reference_m=(station_m, -0.0322, 0.0),
        approach_axis_reference=(1.0, 0.0, 0.0),
        bottle_axis_reference=(0.0, 0.0, 1.0),
    )

    matrix = np.asarray(result["reference_from_grasp"], dtype=float)
    assert result["status"] == "PASS"
    assert result["origin_reference_m"] == pytest.approx(
        [station_m, 0.0, 0.0]
    )
    assert result["finger_line_axis_reference"] == pytest.approx(
        [0.0, 1.0, 0.0]
    )
    assert result["bottle_axis_reference"] == pytest.approx([0.0, 0.0, 1.0])
    assert matrix[:3, :3] == pytest.approx(np.eye(3))
    assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1.0)
    assert result["ee_endpoint_is_grasp_center"] is False


def test_complete_gripper_clearance_report_uses_frozen_cad_sources() -> None:
    payload = json.loads(CLEARANCE_REPORT.read_text(encoding="utf-8"))

    assert payload["status"] == "PASS"
    assert payload["task8"] == "NOT_RUN"
    assert payload["toolchain"]["freecad"] == "1.1.1"
    assert payload["toolchain"]["opencascade"] == "7.8.1"
    assert payload["toolchain"]["linear_deflection_mm"] == pytest.approx(0.2)
    assert payload["toolchain"]["angular_deflection_deg"] == pytest.approx(
        20.0
    )
    assert payload["sources"]["supplier_viper_step"]["sha256"] == (
        "337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571"
    )
    assert payload["sources"]["project_bottle_fcstd"]["sha256"] == (
        "3594f60200e54181bc8480a229484293a0d386c146d3f235b32e31a0c16bbf8a"
    )
    assert payload["sources"]["runtime_gripper_bar_stl"]["sha256"] == (
        "a4de62c9a2ed2c78433010e4c05530a1254b1774a7651967f406120c9bf8973e"
    )

    bottle = payload["bottle_section"]
    assert bottle["axial_station_mm"] == pytest.approx(69.0)
    assert bottle["outer_radius_mm"] == pytest.approx(32.2)
    assert bottle["evidence"] == "PROJECT_BOTTLE_BREP_SECTION_READBACK"

    assert payload["forbidden_envelopes"]["supplier_gripper_shell"][
        "source_type"
    ] == "SUPPLIER_STEP_BREP"
    assert payload["forbidden_envelopes"]["runtime_urdf_gripper_bar"][
        "source_type"
    ] == "URDF_COLLISION_STL_CONSERVATIVE_AABB"

    selection = payload["station_selection"]
    assert selection["status"] == "PASS"
    assert selection["selected_station_m"] > 0.13
    assert selection["selected_minimum_margin_m"] > 0.02
    assert selection["rejected_station"]["runtime_rejected"] is True
    assert payload["grasp_frame"]["rotation_determinant"] == pytest.approx(1.0)
    assert payload["grasp_frame"]["ee_endpoint_is_grasp_center"] is False

    determinism = payload["determinism"]
    assert determinism["status"] == "PASS"
    assert determinism["fresh_run_count"] == 2
    assert determinism["semantic_signature_match"] is True
