from __future__ import annotations

import json
from pathlib import Path

from aloha_isaac_replay.data.gripper_semantics import analyze_episode_grippers


def test_first_selected_episode_requires_gripper_qpos_action_separation() -> None:
    selected = json.loads(Path("reports/aloha_isaac_replay/selected_success_hdf5.json").read_text())
    episode = selected["selected"][0]["path"]
    payload = analyze_episode_grippers(episode)
    assert payload["interpretation"]["qpos_action_must_remain_separate"] is True
    assert payload["left"]["command_and_observation_same_space"] is False

