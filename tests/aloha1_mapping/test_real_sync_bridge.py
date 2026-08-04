from __future__ import annotations

import pytest

from tools.aloha1_mapping.real_sync_bridge import build_remote_publisher_command
from tools.aloha1_mapping.real_sync_bridge import format_initial_pose_check
from tools.aloha1_mapping.real_sync_bridge import initial_pose_error_rad


def test_initial_pose_error_uses_arm_only() -> None:
    assert initial_pose_error_rad([0, 1, 2, 3, 4, 5], [0, 1, 2.01, 3, 4, 5, 99]) == pytest.approx(0.01)


def test_initial_pose_error_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        initial_pose_error_rad([0, 1], [0, 1])


def test_remote_command_is_follower_left_only() -> None:
    command = build_remote_publisher_command(
        manifest_path="/app/.codex/runtime/sleep_home_sleep_50hz_smooth_manifest.json",
        output_path="/app/.codex/runtime/integrated_result.json",
    )
    text = command[-1]
    assert "puppet_left" not in text or "aloha_ros_nodes" in text
    assert "run_aloha1_home_sleep_real_publisher.py" in text
    assert "--execute-real" in text
    assert "puppet_right" not in text


def test_pose_dialog_reports_blocked_state() -> None:
    text = format_initial_pose_check(
        max_error_rad=0.03, gate_rad=0.02, real_position=[0, 0, 0, 0, 0, 0]
    )
    assert "BLOCKED" in text
