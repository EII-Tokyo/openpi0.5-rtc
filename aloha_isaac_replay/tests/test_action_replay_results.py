from __future__ import annotations

import json
from pathlib import Path


def test_action_replay_entrypoint_exists() -> None:
    wrapper = Path("scripts/replay_aloha_action.py")
    implementation = Path("aloha_isaac_replay/scripts/replay_aloha_action.py")
    assert wrapper.exists()
    assert implementation.exists()
    assert "replay_aloha_action" in wrapper.read_text()
    text = implementation.read_text()
    assert "--episode" in text
    assert "--output-dir" in text


def test_action_replay_metrics_record_controller_failure_baseline() -> None:
    path = Path(
        "reports/aloha_isaac_replay/action_replay/"
        "key_region_00590092c6824332a8770a49ffc6dc31/action_replay_metrics.json"
    )
    assert path.exists(), "Run scripts/replay_aloha_action.py before claiming action replay status"
    metrics = json.loads(path.read_text())
    assert metrics["status"] == "ANALYZED"
    assert metrics["uses_controller"] is True
    assert metrics["uses_action"] is True
    assert metrics["uses_gripper_action"] is False
    assert metrics["mode"] == "arm_action_position_targets_only"
    assert metrics["steps_replayed"] > 0
    assert metrics["arm_rmse"] > 0.5, "Current baseline should not be reported as passing"
    assert "right_shoulder" in metrics["per_joint"]
