from __future__ import annotations

import numpy as np
import pytest

from aloha_isaac_replay.adapters.gripper_mapping import isaac_gripper_to_standard
from aloha_isaac_replay.adapters.gripper_mapping import standard_gripper_to_isaac
from aloha_isaac_replay.adapters.gripper_mapping import validate_gripper_direction


def test_standard_gripper_zero_is_closed_and_one_is_open_for_vx300s_fingers() -> None:
    closed = standard_gripper_to_isaac(0.0)
    opened = standard_gripper_to_isaac(1.0)
    assert float(closed["left_finger"]) == pytest.approx(0.021)
    assert float(opened["left_finger"]) == pytest.approx(0.057)
    assert float(closed["right_finger"]) == pytest.approx(-0.021)
    assert float(opened["right_finger"]) == pytest.approx(-0.057)


def test_gripper_mapping_round_trip_accepts_opposite_mimic_direction() -> None:
    values = np.array([0.0, 0.25, 0.5, 1.0])
    fingers = standard_gripper_to_isaac(values)
    recovered = isaac_gripper_to_standard(fingers["left_finger"], fingers["right_finger"])
    assert np.allclose(recovered, values)


def test_gripper_mapping_rejects_invalid_normalized_values() -> None:
    with pytest.raises(ValueError, match="normalized gripper command"):
        standard_gripper_to_isaac(1.2)


def test_gripper_direction_summary_is_valid() -> None:
    summary = validate_gripper_direction()
    assert summary["valid_mimic_opposite_direction"] is True

