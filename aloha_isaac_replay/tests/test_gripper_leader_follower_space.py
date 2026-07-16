from __future__ import annotations

from pathlib import Path


def test_gripper_state_and_command_use_puppet_space_while_leader_path_is_separate() -> None:
    constants = Path("examples/aloha_real/constants.py").read_text()
    real_env = Path("examples/aloha_real/real_env.py").read_text()
    report = Path("reports/aloha_isaac_replay/action_provenance/gripper_semantics.md").read_text()
    assert "MASTER_GRIPPER_JOINT_NORMALIZE_FN" in constants
    assert "PUPPET_GRIPPER_JOINT_NORMALIZE_FN" in constants
    assert "PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN" in constants
    assert "PUPPET_GRIPPER_JOINT_NORMALIZE_FN(left_qpos_raw[6])" in real_env
    assert "PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(left_gripper_desired_pos_normalized)" in real_env
    assert "MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_left.dxl.joint_states.position[6])" in real_env
    assert "do not score gripper action as if it were observed gripper qpos" in report
