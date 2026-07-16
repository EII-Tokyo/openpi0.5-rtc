from __future__ import annotations

import json
from pathlib import Path


REPORT_DIR = Path("reports/aloha_isaac_replay/action_provenance")


def test_isaac_recommendation_uses_raw_hdf5_arm_action_as_absolute_target() -> None:
    text = (REPORT_DIR / "isaac_adapter_recommendation.md").read_text()
    assert "HDF5 action[t, arm]" in text
    assert "standard ALOHA absolute follower joint target" in text
    assert "d = 0..10 frames" in text
    assert "no `adapt_to_pi` sign flip" in text
    assert "no delta integration" in text


def test_summary_gates_allow_controller_identification_but_not_reward_or_rl() -> None:
    summary = json.loads((REPORT_DIR / "summary.json").read_text())
    assert summary["ready_to_resume_controller_identification"] is True
    assert summary["ready_for_reward"] is False
    assert summary["ready_for_rl"] is False
    assert "delay-aware comparison" in summary["correct_isaac_adapter_chain"]

