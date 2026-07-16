from __future__ import annotations

import numpy as np

from aloha_isaac_replay.validation.time_axis import validate_50hz_timestamps


def test_validate_50hz_timestamps_accepts_nominal_20ms_spacing() -> None:
    result = validate_50hz_timestamps(np.array([0.0, 0.02, 0.04, 0.06]))
    assert result["valid"] is True
    assert result["mean_dt"] == 0.02


def test_validate_50hz_timestamps_rejects_large_timing_gap() -> None:
    result = validate_50hz_timestamps(np.array([0.0, 0.02, 0.12]))
    assert result["valid"] is False
    assert result["max_abs_dt_error"] > 0.006

