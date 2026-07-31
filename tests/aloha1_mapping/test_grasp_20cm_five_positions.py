from __future__ import annotations

import copy

import numpy as np
import pytest

from tools.aloha1_mapping.grasp_20cm_sampling import derive_legal_offset_bounds
from tools.aloha1_mapping.grasp_20cm_sampling import extend_profile_for_clearance_lift
from tools.aloha1_mapping.grasp_20cm_sampling import sample_candidate_offsets
from tools.aloha1_mapping.grasp_20cm_sampling import translate_horizontal_bottle_profile
from tools.run_aloha1_grasp_20cm_five_positions import build_five_position_summary


def test_legal_offsets_keep_full_bottle_inside_obstacle_free_table() -> None:
    bounds = derive_legal_offset_bounds(
        table_xy_bounds={
            "minimum": [-0.55, -0.3],
            "maximum": [0.55, 0.3],
        },
        left_base_aabb={
            "minimum": [-0.667, -0.121],
            "maximum": [-0.3675, 0.083],
        },
        right_base_aabb={
            "minimum": [0.3675, -0.121],
            "maximum": [0.667, 0.083],
        },
        nominal_bottle_xy_bounds={
            "minimum": [-0.0674041129, -0.1985098658],
            "maximum": [0.1427072305, -0.1177366805],
        },
    )

    assert bounds["free_surface_xy"]["minimum"] == [-0.3675, -0.3]
    assert bounds["free_surface_xy"]["maximum"] == [0.3675, 0.3]
    assert bounds["offset_xy_m"]["minimum"] == pytest.approx(
        [-0.3000958871, -0.1014901342]
    )
    assert bounds["offset_xy_m"]["maximum"] == pytest.approx(
        [0.2247927695, 0.4177366805]
    )


def test_fixed_seed_candidate_offsets_are_unique_and_bounded() -> None:
    bounds = {
        "minimum": [-0.3, -0.1],
        "maximum": [0.22, 0.41],
    }

    first = sample_candidate_offsets(
        offset_xy_bounds=bounds,
        seed=20260731,
        count=64,
    )
    second = sample_candidate_offsets(
        offset_xy_bounds=bounds,
        seed=20260731,
        count=64,
    )

    assert first == second
    assert len(first) == 64
    assert len({tuple(item["offset_xy_m"]) for item in first}) == 64
    assert all(
        bounds["minimum"][0] <= item["offset_xy_m"][0] <= bounds["maximum"][0]
        and bounds["minimum"][1]
        <= item["offset_xy_m"][1]
        <= bounds["maximum"][1]
        for item in first
    )


def test_profile_translation_changes_only_world_xy_positions() -> None:
    profile = {
        "kinematics": {
            "placement": {
                "placement_matrix": np.eye(4).tolist(),
                "bottle_axis": {
                    "a_world_m": [0.0, 1.0, 2.0],
                    "b_world_m": [0.0, 1.0, 3.0],
                    "grasp_point_world_m": [0.0, 1.0, 2.5],
                },
                "target_poses": {
                    "pregrasp_ee_position_world_m": [1.0, 2.0, 3.0],
                    "grasp_ee_position_world_m": [1.0, 2.0, 2.0],
                    "lift_ee_position_world_m": [1.0, 2.0, 4.0],
                    "object_from_gripper": np.eye(4).tolist(),
                },
            }
        }
    }
    original = copy.deepcopy(profile)

    translated = translate_horizontal_bottle_profile(
        profile,
        offset_xy_m=[0.1, -0.2],
    )

    placement = translated["kinematics"]["placement"]
    assert placement["placement_matrix"][0][3] == pytest.approx(0.1)
    assert placement["placement_matrix"][1][3] == pytest.approx(-0.2)
    assert placement["bottle_axis"]["a_world_m"] == [0.1, 0.8, 2.0]
    assert placement["target_poses"][
        "grasp_ee_position_world_m"
    ] == [1.1, 1.8, 2.0]
    assert placement["target_poses"]["object_from_gripper"] == (
        original["kinematics"]["placement"]["target_poses"][
            "object_from_gripper"
        ]
    )
    assert profile == original


def test_preflight_profile_uses_the_formal_clearance_lift_distance() -> None:
    profile = {
        "kinematics": {
            "placement": {
                "target_poses": {
                    "grasp_ee_position_world_m": [0.1, -0.2, 0.056],
                    "lift_ee_position_world_m": [0.1, -0.2, 0.060],
                }
            }
        }
    }
    original = copy.deepcopy(profile)

    extended = extend_profile_for_clearance_lift(
        profile,
        target_clearance_m=0.200,
        hold_drop_gate_m=0.010,
    )

    targets = extended["kinematics"]["placement"]["target_poses"]
    assert targets["lift_ee_position_world_m"] == pytest.approx(
        [0.1, -0.2, 0.266]
    )
    assert extended["formal_lift_distance_m"] == pytest.approx(0.210)
    assert profile == original


def test_diagnostic_lift_margin_changes_only_the_lift_endpoint() -> None:
    profile = {
        "kinematics": {
            "placement": {
                "target_poses": {
                    "grasp_ee_position_world_m": [0.1, -0.2, 0.056],
                    "lift_ee_position_world_m": [0.1, -0.2, 0.060],
                }
            }
        }
    }

    extended = extend_profile_for_clearance_lift(
        profile,
        target_clearance_m=0.200,
        hold_drop_gate_m=0.010,
        additional_lift_margin_m=0.002,
    )

    targets = extended["kinematics"]["placement"]["target_poses"]
    assert targets["grasp_ee_position_world_m"] == [0.1, -0.2, 0.056]
    assert targets["lift_ee_position_world_m"] == pytest.approx(
        [0.1, -0.2, 0.268]
    )
    assert extended["formal_lift_distance_m"] == pytest.approx(0.212)
    assert extended["additional_lift_margin_m"] == pytest.approx(0.002)


def test_five_position_summary_requires_paired_machine_passes() -> None:
    records = []
    for index in range(5):
        signature = f"signature-{index}"
        records.append(
            {
                "position_id": f"position_{index + 1:02d}",
                "offset_xy_m": [index * 0.01, -index * 0.01],
                "primary": {
                    "process_id": 100 + index,
                    "exit_code": 0,
                    "machine_status": "PASS",
                    "deterministic_signature": signature,
                    "video_count": 2,
                },
                "collider_repeat": {
                    "process_id": 200 + index,
                    "exit_code": 0,
                    "machine_status": "PASS",
                    "deterministic_signature": signature,
                    "collision_record_count": 24,
                },
            }
        )

    summary = build_five_position_summary(records)

    assert summary["machine_status"] == "PASS"
    assert summary["status"] == "PARTIAL"
    assert summary["machine_pass_count"] == 5
    assert summary["video_count"] == 5
    assert summary["fresh_process_count"] == 10
    assert summary["visual_model_review"] == "NOT_RUN"
    assert summary["user_confirmation"] == "NOT_RUN"
    assert summary["task8"] == "NOT_RUN"
