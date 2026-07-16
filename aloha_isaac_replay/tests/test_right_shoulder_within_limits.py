from __future__ import annotations

import json
from pathlib import Path


def test_right_shoulder_within_limits_or_blocked() -> None:
    summary_path = Path("reports/aloha_isaac_replay/controller_system_id/summary.json")
    assert summary_path.exists(), "Run controller system-id report generation first"
    summary = json.loads(summary_path.read_text())
    assert "right_shoulder" in summary["per_joint_delays"]
    assert summary["right_shoulder_max_error"] < 3.5 or summary["gate"] == "BLOCKED_RIGHT_SHOULDER"

