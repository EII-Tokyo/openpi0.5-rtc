from __future__ import annotations

import json
from pathlib import Path

from tools.aloha1_mapping.cad_derived_collision_runtime import canonical_runtime_signature
from tools.aloha1_mapping.cad_derived_collision_runtime import classify_overlap_pair
from tools.aloha1_mapping.cad_derived_collision_runtime import load_frozen_pose_manifest
from tools.aloha1_mapping.cad_derived_collision_runtime import summarize_static_validation

ROOT = Path(__file__).resolve().parents[2]
FIVE_POSE = ROOT / "reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_results_downward_contact_gate_v5.json"
STAGE_REPORT = ROOT / "reports/aloha1_mapping/aloha1_cad_derived_collider_stage.json"
COVERAGE_REPORT = ROOT / "reports/aloha1_mapping/aloha1_cad_derived_collision_coverage.json"


def test_frozen_pose_manifest_has_home_five_starts_and_four_gripper_states() -> None:
    records = load_frozen_pose_manifest(FIVE_POSE)
    assert [record["pose_id"] for record in records] == [
        "home_reference",
        "sample_01",
        "sample_02",
        "sample_03",
        "sample_04",
        "sample_05",
    ]
    assert all(len(record["arm_q_rad"]) == 6 for record in records)
    assert {record["source"] for record in records[1:]} == {"FROZEN_USER_ACCEPTED_FIVE_POSE_RUNTIME"}
    assert records[0]["source"] == "PROJECT_FROZEN_HOME_REFERENCE"


def test_overlap_policy_separates_interfaces_from_forbidden_pairs() -> None:
    adjacent = classify_overlap_pair(
        actor0="/World/follower_left/vx300s_left/follower_left_base_link",
        actor1="/World/follower_left/vx300s_left/follower_left_shoulder_link",
        collider0="/base/mesh",
        collider1="/shoulder/mesh",
        adjacent_body_pairs={
            (
                "/World/follower_left/vx300s_left/follower_left_base_link",
                "/World/follower_left/vx300s_left/follower_left_shoulder_link",
            )
        },
        relation="OVERLAP",
        overlap_volume_m3=1.0e-7,
    )
    assert adjacent["classification"] == "ADJACENT_JOINT_INTERFACE_EXPECTED"
    assert adjacent["allowed"] is True

    assembly = classify_overlap_pair(
        actor0="/World/follower_left/vx300s_left/follower_left_gripper_link",
        actor1=("/World/follower_left/vx300s_left/follower_left_left_finger_link"),
        collider0="/gripper/mesh",
        collider1="/finger/mesh",
        adjacent_body_pairs=set(),
        cad_assembly_interface_pairs={
            (
                "/World/follower_left/vx300s_left/follower_left_gripper_link",
                "/World/follower_left/vx300s_left/follower_left_left_finger_link",
            )
        },
        relation="OVERLAP",
        overlap_volume_m3=8.0e-9,
    )
    assert assembly["classification"] == "CAD_ASSEMBLY_INTERFACE_EXPECTED"
    assert assembly["allowed"] is True

    forbidden = classify_overlap_pair(
        actor0="/World/follower_left/vx300s_left/follower_left_base_link",
        actor1="/World/follower_left/vx300s_left/follower_left_gripper_link",
        collider0="/base/mesh",
        collider1="/gripper/mesh",
        adjacent_body_pairs=set(),
        relation="OVERLAP",
        overlap_volume_m3=2.0e-6,
    )
    assert forbidden["classification"] == "UNEXPECTED_SELF_COLLISION"
    assert forbidden["allowed"] is False

    table = classify_overlap_pair(
        actor0="/World/follower_left/vx300s_left/follower_left_left_finger_link",
        actor1="/World/environment/worldBody/user_confirmed_table",
        collider0="/finger/mesh",
        collider1="/table/collider",
        adjacent_body_pairs=set(),
        relation="OVERLAP",
        overlap_volume_m3=3.0e-8,
    )
    assert table["classification"] == "ENVIRONMENT_CONTACT_REQUIRES_RUNTIME_EFFECT_REVIEW"
    assert table["allowed"] is None


def test_static_summary_fails_for_forbidden_overlap_or_first_frame_jump() -> None:
    passing = {
        "pose_id": "home_reference",
        "finite": True,
        "within_joint_limits": True,
        "unexpected_overlap_count": 0,
        "unresolved_environment_contact_count": 0,
        "first_frame_jump_max_abs_rad": 0.001,
        "first_frame_jump_gate_rad": 0.02,
        "nonfinite_contact_count": 0,
    }
    summary = summarize_static_validation([passing, {**passing, "pose_id": "sample_01"}])
    assert summary["status"] == "PASS"

    overlap = {**passing, "pose_id": "sample_02", "unexpected_overlap_count": 1}
    summary = summarize_static_validation([passing, overlap])
    assert summary["status"] == "FAIL"

    jump = {**passing, "pose_id": "sample_03", "first_frame_jump_max_abs_rad": 0.021}
    summary = summarize_static_validation([passing, jump])
    assert summary["status"] == "FAIL"


def test_runtime_signature_is_order_stable_and_sensitive() -> None:
    report = {
        "poses": [
            {
                "pose_id": "sample_01",
                "status": "PASS",
                "first_frame_jump_max_abs_rad": 0.001,
                "overlaps": [
                    {
                        "actor_pair": ["/b", "/a"],
                        "collider_pair": ["/d", "/c"],
                        "classification": "ADJACENT_JOINT_INTERFACE_EXPECTED",
                        "overlap_volume_m3": 1.234567891e-7,
                    }
                ],
            }
        ]
    }
    first = canonical_runtime_signature(report)
    report["poses"][0]["overlaps"].reverse()
    second = canonical_runtime_signature(report)
    assert first == second
    report["poses"][0]["first_frame_jump_max_abs_rad"] = 0.003
    assert canonical_runtime_signature(report) != first


def test_phase4_stage_contract_stays_isolated() -> None:
    report = json.loads(STAGE_REPORT.read_text(encoding="utf-8"))
    assert report["source_hashes_unchanged"] is True
    assert report["source_or_imported_asset_modified"] is False
    assert report["final_or_default_asset_modified"] is False
    assert report["task8"] == "NOT_RUN"


def test_native_runtime_coverage_contains_all_compound_and_fallback_shapes() -> None:
    report = json.loads(COVERAGE_REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["collider_count"] == 42
    assert report["source_kind_counts"] == {
        "CAD_DERIVED": 34,
        "IMPORTER_BASELINE_FALLBACK": 4,
        "SUPPLIER_CAD_FINGER": 4,
    }
