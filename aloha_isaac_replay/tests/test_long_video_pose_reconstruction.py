from __future__ import annotations

import numpy as np

from aloha_isaac_replay.data.long_video_pose_reconstruction import detect_open_close_lift_candidates


def test_detect_open_close_lift_ignores_initial_closed_segment() -> None:
    qpos = np.zeros((220, 14), dtype=np.float64)
    qpos[:, 6] = 0.03
    qpos[80:121, 6] = 0.9
    qpos[121:126, 6] = np.linspace(0.8, 0.2, 5)
    qpos[126:, 6] = 0.04
    qpos[126:170, 1] = np.linspace(0.0, 1.0, 44)
    qpos[170:, 1] = 1.0

    rows = detect_open_close_lift_candidates(
        qpos,
        open_threshold=0.65,
        close_threshold=0.35,
        lock_threshold=0.10,
        approach_offset_frames=20,
        post_close_window_frames=80,
        lift_motion_threshold=0.01,
        lift_motion_consecutive=2,
    )

    assert len(rows) == 1
    candidate = rows[0]
    assert candidate.open_segment_start == 80
    assert candidate.open_segment_end == 121
    assert candidate.close_frame == 124
    assert candidate.approach_frame == 104
    assert candidate.lift_start_frame is not None
    assert candidate.lift_start_frame >= candidate.close_frame


def test_detect_open_close_lift_reports_missing_lock() -> None:
    qpos = np.zeros((160, 14), dtype=np.float64)
    qpos[:, 6] = 0.5
    qpos[20:80, 6] = 0.9
    qpos[80:, 6] = 0.2
    qpos[85:120, 0] = np.linspace(0.0, 0.8, 35)

    rows = detect_open_close_lift_candidates(
        qpos,
        close_threshold=0.35,
        lock_threshold=0.05,
        lift_motion_threshold=0.01,
    )

    assert len(rows) == 1
    assert rows[0].close_frame == 80
    assert rows[0].grasp_lock_frame is None
    assert "no_grasp_lock_below_lock_threshold" in rows[0].reasons
