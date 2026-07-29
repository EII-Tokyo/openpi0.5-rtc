from pathlib import Path

import numpy as np

from tools.aloha1_mapping.signal_correspondence import ACTIVE_ONE_JOINT_TESTS
from tools.aloha1_mapping.signal_correspondence import build_fixed_oblique_camera_spec
from tools.aloha1_mapping.signal_correspondence import build_signal_mapping_plan
from tools.aloha1_mapping.signal_correspondence import build_small_up_down_targets
from tools.aloha1_mapping.signal_correspondence import canonical_dof_name
from tools.aloha1_mapping.signal_correspondence import classify_task7a_status

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_names_are_mapped_explicitly_without_sorting() -> None:
    assert (
        canonical_dof_name(
            "follower_left",
            "vx300s_left_wrist_angle",
        )
        == "wrist_angle"
    )
    assert canonical_dof_name("follower_right", "wrist_angle") == "wrist_angle"

    plan = build_signal_mapping_plan(PROJECT_ROOT)
    assert plan["order_policy"] == "EXPLICIT_SOURCE_ORDER_NEVER_ALPHABETICAL"
    assert plan["dataset_14d_order"][6] == "left_gripper_normalized"
    assert plan["dataset_14d_order"][13] == "right_gripper_normalized"
    assert plan["robots"]["follower_left"]["runtime_expected_order"] == [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
        "left_finger",
        "right_finger",
    ]
    assert plan["robots"]["follower_right"]["runtime_expected_order"][6] == ("gripper")


def test_small_up_down_targets_are_symmetric_safe_and_return_home() -> None:
    targets = build_small_up_down_targets()

    assert targets["joint"] == "shoulder"
    assert targets["unit"] == "rad"
    assert targets["home"][1] == -0.96
    assert targets["small_up"][1] == -1.04
    assert targets["return_home"] == targets["home"]
    assert targets["maximum_absolute_delta_rad"] == 0.08


def test_one_joint_plan_includes_arm_gripper_and_mimic_driver() -> None:
    assert ACTIVE_ONE_JOINT_TESTS == [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
        "left_finger",
    ]


def test_fixed_oblique_camera_is_derived_from_robot_geometry() -> None:
    points = np.asarray(
        [
            [-0.70, -0.12, 0.02],
            [0.10, 0.08, 0.49],
            [-0.20, 0.00, 0.30],
        ],
        dtype=np.float64,
    )

    left = build_fixed_oblique_camera_spec(points, "follower_left")
    right = build_fixed_oblique_camera_spec(points, "follower_right")

    assert left["target_world_m"] == [-0.3, -0.019999999999999997, 0.255]
    assert left["robot_aabb_min_world_m"] == [-0.7, -0.12, 0.02]
    assert left["robot_aabb_max_world_m"] == [0.1, 0.08, 0.49]
    assert left["fixed_for_robot_phase_group"] is True
    assert left["position_world_m"][0] < left["target_world_m"][0]
    assert right["position_world_m"][0] > right["target_world_m"][0]
    assert left["position_world_m"] != right["position_world_m"]


def test_task7a_status_requires_swept_collision_and_preserves_rules() -> None:
    common = {
        "mapping_status": "PASS",
        "structure_status": "PASS",
        "drive_mimic_status": "PASS",
        "small_up_down_status": "PASS",
    }

    assert (
        classify_task7a_status(
            **common,
            swept_collision_status="FAIL",
            official_task7a_status="PARTIAL",
        )
        == "FAIL"
    )
    assert (
        classify_task7a_status(
            **common,
            swept_collision_status="PASS",
            official_task7a_status="PARTIAL",
        )
        == "PARTIAL"
    )
    assert (
        classify_task7a_status(
            **common,
            swept_collision_status="PASS",
            official_task7a_status="PASS",
        )
        == "PASS"
    )
