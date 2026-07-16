from __future__ import annotations

import numpy as np

from aloha_isaac_replay.controller_system_id.action_semantics import arm_action_from_raw_hdf5_action


def test_raw_hdf5_action_no_openpi_flip() -> None:
    frame = np.arange(14, dtype=np.float64)
    arm = arm_action_from_raw_hdf5_action(frame)
    assert np.array_equal(arm, np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.float64))

