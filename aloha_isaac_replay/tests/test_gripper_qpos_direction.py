from __future__ import annotations

import numpy as np

from aloha_isaac_replay.adapters.gripper_mapping import (
    standard_gripper_qpos_to_isaac_fingers,
    standard_gripper_to_isaac,
)


def test_normalized_qpos_opening_direction_maps_to_opposite_finger_motion() -> None:
    closed = standard_gripper_to_isaac(0.0)
    opened = standard_gripper_to_isaac(1.0)
    assert opened["left_finger"] > closed["left_finger"]
    assert opened["right_finger"] < closed["right_finger"]


def test_observed_qpos_mapping_is_side_scoped_and_not_action_specific() -> None:
    left = standard_gripper_qpos_to_isaac_fingers(np.array([0.0, 0.5, 1.0]), side="left")
    right = standard_gripper_qpos_to_isaac_fingers(np.array([0.0, 0.5, 1.0]), side="right")

    assert set(left) == {"left/left_finger", "left/right_finger"}
    assert set(right) == {"right/left_finger", "right/right_finger"}
    np.testing.assert_allclose(left["left/left_finger"], [0.021, 0.039, 0.057])
    np.testing.assert_allclose(right["right/right_finger"], [-0.021, -0.039, -0.057])


def test_observed_qpos_mapping_rejects_unknown_side() -> None:
    try:
        standard_gripper_qpos_to_isaac_fingers(0.5, side="leader")
    except ValueError as exc:
        assert "side" in str(exc)
    else:
        raise AssertionError("expected unknown gripper side to be rejected")
