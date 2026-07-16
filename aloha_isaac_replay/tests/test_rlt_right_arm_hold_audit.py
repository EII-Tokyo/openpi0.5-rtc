from __future__ import annotations

import json
from pathlib import Path


REPORT_DIR = Path("reports/aloha_isaac_replay/controller_system_id")


def test_rlt_right_arm_hold_report_marks_selected_actor_data_not_usable_for_right_arm_id() -> None:
    summary = json.loads((REPORT_DIR / "summary.json").read_text())
    hold = summary["right_arm_hold_summary"]

    assert hold["episode_count"] == 10
    assert hold["right_arm_hold_detected_count"] == 6
    assert hold["right_arm_hold_or_static_detected_count"] == 10
    assert hold["right_arm_controller_id_usable_count"] == 0
    assert summary["right_arm_data_gate"] == "BLOCKED_RLT_RIGHT_ARM_HOLD_OR_STATIC_COMMAND"
    assert summary["ready_for_contact_reward"] is False
    assert summary["ready_for_rl"] is False


def test_rlt_right_arm_hold_audit_keeps_isaac_runtime_blocker_separate() -> None:
    summary = json.loads((REPORT_DIR / "summary.json").read_text())

    assert summary["isaac_runtime_gate"] == "BLOCKED_ISAAC_RIGHT_SHOULDER"
    assert summary["gate"] == "BLOCKED_RLT_RIGHT_ARM_HOLD_OR_STATIC_DATA_AND_ISAAC_RIGHT_SHOULDER_RUNTIME"

    report = (REPORT_DIR / "rlt_right_arm_hold_audit.md").read_text()
    assert "not the same as zeroing the action" in report
    assert "Usable for right-arm controller ID: `0`" in report
