from __future__ import annotations

import numpy as np

from aloha_isaac_replay.controller_system_id.continuous_joints import nearest_equivalent_targets


def test_no_wrap_for_shoulder_elbow() -> None:
    names = ("right_shoulder", "right_elbow")
    raw = np.asarray([3.2, -3.3], dtype=np.float64)
    reference = np.asarray([-3.0, 3.0], dtype=np.float64)
    adjusted, events = nearest_equivalent_targets(raw, reference, names)
    assert np.array_equal(adjusted, raw)
    assert events == []

