from __future__ import annotations

import numpy as np

from aloha_isaac_replay.controller_system_id.action_semantics import canonical_absolute_targets


def test_absolute_action_not_integrated() -> None:
    raw_actions = np.asarray(
        [
            [0, 1, 2, 3, 4, 5, 0.1, 7, 8, 9, 10, 11, 12, 0.2],
            [1, 2, 3, 4, 5, 6, 0.1, 8, 9, 10, 11, 12, 13, 0.2],
        ],
        dtype=np.float64,
    )
    targets = canonical_absolute_targets(raw_actions)
    assert np.array_equal(targets[1], np.asarray([1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13], dtype=np.float64))

