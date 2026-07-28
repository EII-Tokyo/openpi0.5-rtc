from __future__ import annotations

from tools.aloha1_mapping.model_compare import compare_urdf_audits


def test_compare_urdf_audits_reports_order_limits_gripper_and_mesh_hashes() -> None:
    aloha = {
        "robot_name": "follower_left",
        "link_order": ["base", "finger"],
        "joint_order": ["waist", "left_finger_joint"],
        "joints": [
            {
                "name": "waist",
                "type": "revolute",
                "axis": "0 0 1",
                "origin_xyz": "0 0 0",
                "origin_rpy": "0 0 0",
                "lower": -1.0,
                "upper": 1.0,
                "effort": 10.0,
                "velocity": 2.0,
                "mimic_parent": None,
                "mimic_multiplier": None,
                "mimic_offset": None,
            },
            {
                "name": "left_finger_joint",
                "type": "prismatic",
                "axis": "0 1 0",
                "origin_xyz": "0 0 0",
                "origin_rpy": "0 0 0",
                "lower": 0.02,
                "upper": 0.05,
                "effort": 5.0,
                "velocity": 1.0,
                "mimic_parent": None,
                "mimic_multiplier": None,
                "mimic_offset": None,
            },
        ],
        "meshes": [
            {
                "resolved_path": "/aloha/base.stl",
                "sha256": "same",
            },
            {
                "resolved_path": "/aloha/finger.stl",
                "sha256": "aloha-finger",
            },
        ],
    }
    standard = {
        **aloha,
        "robot_name": "standard_vx300s",
        "joints": [
            aloha["joints"][0],
            {
                **aloha["joints"][1],
                "lower": 0.01,
                "upper": 0.04,
            },
        ],
        "meshes": [
            {
                "resolved_path": "/standard/base.stl",
                "sha256": "same",
            },
            {
                "resolved_path": "/standard/finger.stl",
                "sha256": "standard-finger",
            },
        ],
    }

    comparison = compare_urdf_audits(aloha, standard)

    assert comparison["joint_order_equal"] is True
    assert comparison["link_order_equal"] is True
    assert comparison["joint_differences"] == [
        {
            "joint": "left_finger_joint",
            "fields": {
                "lower": {"aloha": 0.02, "standard": 0.01},
                "upper": {"aloha": 0.05, "standard": 0.04},
            },
        }
    ]
    assert comparison["gripper_joint_differences"] == comparison[
        "joint_differences"
    ]
    assert comparison["mesh_hash_differences"] == [
        {
            "mesh": "finger.stl",
            "aloha_sha256": "aloha-finger",
            "standard_sha256": "standard-finger",
        }
    ]
    assert comparison["all_mesh_hashes_equal"] is False


def test_compare_urdf_audits_ignores_only_instance_prefix_in_link_references() -> None:
    aloha = {
        "robot_name": "follower_left",
        "link_order": ["follower_left_base", "follower_left_arm"],
        "joint_order": ["waist"],
        "joints": [
            {
                "name": "waist",
                "type": "revolute",
                "parent": "follower_left_base",
                "child": "follower_left_arm",
                "axis": "0 0 1",
                "origin_xyz": "0 0 0",
                "origin_rpy": "0 0 0",
                "lower": -1.0,
                "upper": 1.0,
                "effort": 10.0,
                "velocity": 2.0,
                "mimic_parent": None,
                "mimic_multiplier": None,
                "mimic_offset": None,
            }
        ],
        "meshes": [],
    }
    standard = {
        **aloha,
        "robot_name": "standard_vx300s",
        "link_order": ["standard_vx300s_base", "standard_vx300s_arm"],
        "joints": [
            {
                **aloha["joints"][0],
                "parent": "standard_vx300s_base",
                "child": "standard_vx300s_arm",
            }
        ],
    }

    comparison = compare_urdf_audits(aloha, standard)

    assert comparison["link_order_equal"] is True
    assert comparison["joint_differences"] == []
