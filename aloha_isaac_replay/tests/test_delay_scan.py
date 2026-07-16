from __future__ import annotations

import numpy as np

from aloha_isaac_replay.controller_system_id.delay_scan import scan_action_qpos_delays


def test_delay_scan_finds_known_lag() -> None:
    action = np.arange(20, dtype=np.float64).reshape(20, 1)
    qpos = np.vstack([np.zeros((3, 1)), action, np.zeros((3, 1))])
    result = scan_action_qpos_delays(action, qpos, max_delay=6, joint_names=("j0",))
    assert result["aggregate"]["best_delay"] == 3
    assert result["per_joint"]["j0"]["best_delay"] == 3

