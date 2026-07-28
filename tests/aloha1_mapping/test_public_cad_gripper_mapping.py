from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.aloha1_mapping.public_cad_gripper_mapping import build_gripper_mapping_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_stationary_cad_has_two_vx_followers_and_two_fingers_each() -> None:
    report = build_gripper_mapping_report(
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_public_cad_assembly_audit.json",
        PROJECT_ROOT / "generated/urdf/follower_left.urdf",
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_widow_gripper_assembly_audit.json",
        PROJECT_ROOT
        / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
        "widow_gripper/widow_gripper_tessellation.json",
    )

    assert report["orientation_mapping_status"] == "PASS"
    assert len(report["stationary_vx_followers"]) == 2
    for follower in report["stationary_vx_followers"]:
        assert len(follower["finger_instances"]) == 2
        assert {
            item["cad_opening_side"] for item in follower["finger_instances"]
        } == {"positive", "negative"}
        assert follower["opening_axis_alignment_to_cad_x"] > 0.9999
        assert follower["closed_inner_gap_mm"] == pytest.approx(
            4.488278416,
            abs=1.0e-6,
        )


def test_cad_positive_x_is_mapped_to_urdf_left_finger_by_source_axes() -> None:
    report = build_gripper_mapping_report(
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_public_cad_assembly_audit.json",
        PROJECT_ROOT / "generated/urdf/follower_left.urdf",
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_widow_gripper_assembly_audit.json",
        PROJECT_ROOT
        / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
        "widow_gripper/widow_gripper_tessellation.json",
    )

    frame = report["cad_to_urdf_frame_mapping"]
    assert frame["source"] == (
        "generated follower URDF gripper_bar visual origin and finger joints"
    )
    assert frame["cad_positive_x_in_urdf"] == pytest.approx([0.0, 1.0, 0.0])
    assert frame["cad_negative_x_in_urdf"] == pytest.approx([0.0, -1.0, 0.0])
    assert frame["positive_cad_x_link"] == "left_finger"
    assert frame["negative_cad_x_link"] == "right_finger"
    assert frame["open_delta_mm"] == pytest.approx(36.0)


def test_mapping_preserves_source_hashes_and_instance_paths() -> None:
    report = build_gripper_mapping_report(
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_public_cad_assembly_audit.json",
        PROJECT_ROOT / "generated/urdf/follower_left.urdf",
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_widow_gripper_assembly_audit.json",
        PROJECT_ROOT
        / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
        "widow_gripper/widow_gripper_tessellation.json",
    )

    assert report["sources"]["stationary_step"]["sha256"] == (
        "6d8f787e30b0b989fafedc09fc630beecb6dd896b3d47de06a92a9a86e133001"
    )
    assert report["sources"]["exact_vx_finger_step"]["sha256"] == (
        "9df09ff9c224edbc33f0b3bcf4f88c26e77103cf6d07a5be391ca32d53b352cb"
    )
    assert report["sources"]["widow_gripper_crosscheck_step"]["sha256"] == (
        "adc6a2c96912ab7973347b6a4587b6001bdc6316ab294dba5c46273139365500"
    )
    instances = {
        item["object_name"]
        for follower in report["stationary_vx_followers"]
        for item in follower["finger_instances"]
    }
    assert instances == {
        "Part__Feature640",
        "Part__Feature641",
        "Part__Feature650",
        "Part__Feature651",
    }


def test_exact_standalone_finger_is_not_claimed_as_installed_instance() -> None:
    report = build_gripper_mapping_report(
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_public_cad_assembly_audit.json",
        PROJECT_ROOT / "generated/urdf/follower_left.urdf",
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_widow_gripper_assembly_audit.json",
        PROJECT_ROOT
        / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
        "widow_gripper/widow_gripper_tessellation.json",
    )

    comparison = report["standalone_vs_installed_geometry"]
    assert comparison["same_revision"] is False
    assert comparison["standalone_label"] == "3D-A1 - Aloha VX Finger v3"
    assert comparison["installed_label_family"].startswith(
        "Aloha VX Fingers 2024-4-21"
    )
    assert comparison["policy"] == (
        "standalone proves supplier shape provenance; purchase-confirmed "
        "Simple Viper proves follower installation; Widow and Stationary "
        "provide cross-checks"
    )


def test_saved_mapping_report_matches_recomputed_report() -> None:
    expected = build_gripper_mapping_report(
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_public_cad_assembly_audit.json",
        PROJECT_ROOT / "generated/urdf/follower_left.urdf",
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_widow_gripper_assembly_audit.json",
        PROJECT_ROOT
        / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
        "widow_gripper/widow_gripper_tessellation.json",
    )
    saved = json.loads(
        (
            PROJECT_ROOT
            / "reports/aloha1_mapping/aloha_public_cad_gripper_mapping.json"
        ).read_text(encoding="utf-8")
    )
    assert saved == expected


def test_simple_viper_is_primary_and_widow_is_shared_gripper_crosscheck() -> None:
    report = build_gripper_mapping_report(
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_public_cad_assembly_audit.json",
        PROJECT_ROOT / "generated/urdf/follower_left.urdf",
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_widow_gripper_assembly_audit.json",
        PROJECT_ROOT
        / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
        "widow_gripper/widow_gripper_tessellation.json",
    )

    primary = report["primary_follower_installation"]
    crosscheck = report["widow_gripper_crosscheck"]
    assert primary["evidence_class"] == (
        "PURCHASE_DRAWING_CONFIRMED_VX300S_FOLLOWER"
    )
    assert primary["root_object"] == "Dummy_Aloha_VX_v3"
    assert primary["gripper_shell_object"] == "Part__Feature006"
    assert primary["finger_group_object"] == (
        "Aloha_VX_Fingers_2024_4_21_v2"
    )
    assert primary["positive_x_finger"]["object_name"] == "Part__Feature007"
    assert primary["positive_x_finger"]["mapped_urdf_joint"] == "left_finger"
    assert primary["negative_x_finger"]["object_name"] == "Part__Feature008"
    assert primary["negative_x_finger"]["mapped_urdf_joint"] == "right_finger"
    assert primary["identical_finger_placement_matrix"] is True
    assert primary["arbitrary_per_side_roll_required"] is False
    assert crosscheck["evidence_class"] == "WX_SHARED_GRIPPER_CROSSCHECK"
    assert crosscheck["mirror_residual_mm"]["maximum"] < 0.30


def test_source_connection_common_volume_is_reported_not_misclassified() -> None:
    report = build_gripper_mapping_report(
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_public_cad_assembly_audit.json",
        PROJECT_ROOT / "generated/urdf/follower_left.urdf",
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_widow_gripper_assembly_audit.json",
        PROJECT_ROOT
        / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
        "widow_gripper/widow_gripper_tessellation.json",
    )

    connection = report["widow_gripper_crosscheck"]["connection_geometry"]
    assert connection["positive_x"]["minimum_shape_distance_mm"] == 0.0
    assert connection["negative_x"]["minimum_shape_distance_mm"] == 0.0
    assert connection["positive_x"]["common_volume_mm3"] == pytest.approx(0.0)
    assert connection["negative_x"]["common_volume_mm3"] == pytest.approx(
        8.345213533,
    )
    assert report["connection_geometry_status"] == "PASS"
    assert report["connection_geometry_finding"] == (
        "SOURCE_CAD_SLIDING_CARRIAGE_COMMON_VOLUME_RECORDED"
    )


def test_primary_mapping_records_toolchain_transform_and_visual_evidence() -> None:
    report = build_gripper_mapping_report(
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_public_cad_assembly_audit.json",
        PROJECT_ROOT / "generated/urdf/follower_left.urdf",
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_widow_gripper_assembly_audit.json",
        PROJECT_ROOT
        / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
        "widow_gripper/widow_gripper_tessellation.json",
    )

    primary = report["primary_follower_installation"]
    assert primary["positive_x_finger"]["assembly_path"] == [
        "Dummy_Aloha_VX_v3",
        "Aloha_VX_Fingers_2024_4_21_v2",
        "Part__Feature007",
    ]
    assert primary["negative_x_finger"]["assembly_path"][-1] == (
        "Part__Feature008"
    )
    assert primary["shared_source_placement_determinant"] == pytest.approx(1.0)
    assert report["unit_conversion"]["cad_mm_to_isaac_m"] == 0.001
    assert report["toolchain"]["freecad"]["version"][0:3] == ["1", "1", "1"]
    assert report["toolchain"]["opencascade"]["version"] == "7.8.0"
    assert report["toolchain"]["blender"]["version"] == "5.2.0 LTS"
    assert report["toolchain"]["production_tessellation_gate"] == (
        "PASS"
    )
    assert report["mounting_datum_registration"]["status"] == "PASS"
    assert report["isaac_visual_evidence"]["status"] == "PASS"
    assert report["isolated_diagnostic_asset"]["status"] == "PASS"
    assert report["cad_to_finger_link_mapping"]["determinant"] == pytest.approx(
        1.0
    )
    assert report["cad_to_finger_link_mapping"]["mirror_used"] is False
    assert report["cad_to_finger_link_mapping"]["local_axis_mapping"] == {
        "cad_local_x": "finger_link_+Y",
        "cad_local_y": "finger_link_+Z",
        "cad_local_z": "finger_link_+X",
    }
    assert report["visual_evidence"]["status"] == "PASS"
    assert report["visual_evidence"]["capture_count"] == 8
    connection = primary["source_connection_common_volume_mm3"]
    assert connection["closed"]["cad_positive_x_finger"] == pytest.approx(
        8.38110343019704
    )
    assert connection["closed"]["cad_negative_x_finger"] == pytest.approx(
        0.03588988364631973
    )
    assert report["standalone_vs_installed_geometry"][
        "replacement_allowed"
    ] is False
