from pathlib import Path

from tools.aloha1_mapping.joint_map import build_joint_map

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_joint_map_uses_runtime_order_and_explicit_dataset_indices() -> None:
    mapping = build_joint_map(PROJECT_ROOT)

    left = mapping["robots"]["follower_left"]
    assert left["isaac_dof_order"] == [
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
    assert left["urdf_nonfixed_joint_order"] == left["isaac_dof_order"]
    assert left["ros_joint_state_order"] == left["isaac_dof_order"]
    assert [dof["isaac_index"] for dof in left["dofs"]] == list(range(9))
    assert left["dofs"][0]["dataset_state_index"] == 0
    assert left["dofs"][7]["dataset_state_index"] == 6
    assert left["dofs"][8]["mimic"] == {
        "parent": "left_finger",
        "multiplier": -1.0,
        "offset": 0.0,
    }


def test_joint_map_records_gripper_semantic_mismatch_without_hiding_it() -> None:
    mapping = build_joint_map(PROJECT_ROOT)

    gripper = mapping["gripper"]
    assert gripper["real_observed_finger_position_m"]["closed"] == 0.01844
    assert gripper["urdf_left_finger_limit_m"]["closed"] == 0.021
    assert gripper["calibration_status"] == "HARD_BLOCKER"
    assert mapping["status"] == "PARTIAL"


def test_joint_map_names_robot_frames_and_prefixes_explicitly() -> None:
    mapping = build_joint_map(PROJECT_ROOT)

    left = mapping["robots"]["follower_left"]
    right = mapping["robots"]["follower_right"]
    assert left["robot_prefix"] == "follower_left"
    assert right["robot_prefix"] == "follower_right"
    assert left["base_frame"] == "follower_left_base_link"
    assert right["base_frame"] == "follower_right_base_link"
    assert left["end_effector_frame"] == "follower_left_ee_gripper_link"
    assert right["end_effector_frame"] == "follower_right_ee_gripper_link"


def test_joint_map_can_select_the_frozen_signal_runtime_inventory() -> None:
    relative = Path("reports/aloha1_mapping/aloha1_signal_correspondence_runtime_inventory.json")
    mapping = build_joint_map(
        PROJECT_ROOT,
        runtime_report_relative=relative,
    )

    assert mapping["sources"]["isaac_runtime_report"] == str((PROJECT_ROOT / relative).resolve())
    assert mapping["robots"]["follower_left"]["isaac_dof_order"] == [
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
