from __future__ import annotations

import json
from pathlib import Path


def test_corrected_action_replay_baseline_report_exists() -> None:
    report_dir = Path("reports/aloha_isaac_replay/controller_system_id")
    summary = json.loads((report_dir / "summary.json").read_text())
    assert summary["additional_openpi_transform_applied"] is False
    assert summary["action_type"] == "absolute_follower_joint_target"
    assert summary["uses_gripper_action"] is False
    assert summary["delay_scan_range"] == [0, 15]
    assert summary["ready_for_contact_reward"] is False
    assert summary["ready_for_rl"] is False

