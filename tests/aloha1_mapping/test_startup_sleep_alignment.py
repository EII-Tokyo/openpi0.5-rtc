from __future__ import annotations

import pytest

from tools.aloha1_mapping.startup_sleep_alignment import interpolate_targets
from tools.aloha1_mapping.startup_sleep_alignment import max_step_velocity
from tools.aloha1_mapping.startup_sleep_alignment import validate_sleep_manifest


def test_validate_sleep_manifest_pins_arm_order_and_six_values() -> None:
    target, order = validate_sleep_manifest({
        "joint_order": ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"],
        "sleep_rad": [0, -1.8, 1.55, 0, -1.57, 0],
    })
    assert target == [0.0, -1.8, 1.55, 0.0, -1.57, 0.0]
    assert order[-1] == "wrist_rotate"


def test_interpolation_includes_current_and_sleep_endpoints() -> None:
    samples = interpolate_targets([0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6], rate_hz=50, move_seconds=2)
    assert len(samples) == 101
    assert samples[0] == [0.0] * 6
    assert samples[-1] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_max_velocity_uses_move_time_not_sample_count() -> None:
    assert max_step_velocity([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], rate_hz=50, move_seconds=5) == pytest.approx(0.2)


def test_manifest_rejects_wrong_order() -> None:
    with pytest.raises(ValueError, match="unexpected arm joint order"):
        validate_sleep_manifest({"joint_order": ["elbow"] * 6, "sleep_rad": [0] * 6})
