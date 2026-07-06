from __future__ import annotations

import numpy as np

from scripts import rebuild_online_rollout_paper_anchor_replay as rebuild


def test_compute_anchor_starts_appends_final_anchor_when_stride_misses_it() -> None:
    starts = rebuild.compute_anchor_starts(num_frames=59, train_horizon=10, chunk_stride=2)

    assert starts.tolist() == list(range(0, 40, 2)) + [39]


def test_compute_anchor_starts_matches_full_stride_grid() -> None:
    starts = rebuild.compute_anchor_starts(num_frames=118, train_horizon=10, chunk_stride=2)

    assert len(starts) == 50
    assert starts[0] == 0
    assert starts[-1] == 98


def test_build_action_windows_uses_anchor_and_horizon() -> None:
    actions = np.arange(8 * 2, dtype=np.float32).reshape(8, 2)
    starts = np.asarray([0, 2, 3], dtype=np.int64)

    windows = rebuild.build_action_windows(actions, starts, train_horizon=3)

    np.testing.assert_array_equal(windows[0], actions[0:3])
    np.testing.assert_array_equal(windows[1], actions[2:5])
    np.testing.assert_array_equal(windows[2], actions[3:6])


def test_adjacent_exact_fraction_flattens_nonbatch_axes() -> None:
    array = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 2.0], [3.0, 5.0]],
        ],
        dtype=np.float32,
    )

    assert rebuild.adjacent_exact_fraction(array) == 0.5
