from __future__ import annotations

import numpy as np

from aloha_isaac_replay.controller_system_id.continuous_joints import nearest_equivalent_targets


def test_continuous_nearest_equivalent_wraps_only_continuous_joints() -> None:
    names = ("left_shoulder", "left_forearm_roll")
    raw = np.asarray([3.0, 3.2], dtype=np.float64)
    reference = np.asarray([3.0, -3.0], dtype=np.float64)
    adjusted, events = nearest_equivalent_targets(raw, reference, names)
    assert adjusted[0] == 3.0
    assert abs(adjusted[1] + 3.083185307179586) < 1e-9
    assert events == ["left_forearm_roll"]

