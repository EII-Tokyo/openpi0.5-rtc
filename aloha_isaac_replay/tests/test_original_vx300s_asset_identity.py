from __future__ import annotations

from aloha_isaac_replay.assets.urdf_audit import audit_urdf


def test_archived_puppet_urdfs_are_original_vx300s_like() -> None:
    for path in (
        "reports/aloha_model_audit/raw/robot_descriptions/puppet_left_robot_description.urdf",
        "reports/aloha_model_audit/raw/robot_descriptions/puppet_right_robot_description.urdf",
    ):
        audit = audit_urdf(path)
        assert audit.is_vx300s_like, f"{path} identity errors: {audit.identity_errors}"
        assert audit.arm_joint_names_present == ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
        assert audit.finger_joint_names_present == ["left_finger", "right_finger"]
        assert any(link.endswith("ee_gripper_link") for link in audit.ee_links)

