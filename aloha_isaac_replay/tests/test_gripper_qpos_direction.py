from __future__ import annotations

import numpy as np

from aloha_isaac_replay.adapters.gripper_mapping import (
    gripper_qpos_calibration_from_loaded_contact,
    standard_gripper_value_for_symmetric_finger_gap,
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


def test_loaded_soft_bottle_qpos_calibration_maps_plateau_to_effective_gap() -> None:
    calibration = gripper_qpos_calibration_from_loaded_contact(
        raw_open_value=0.9473305344581604,
        raw_contact_value=0.5712134838104248,
        effective_contact_width=0.052,
        source="episode18_loaded_plateau_test",
    )

    contact = standard_gripper_qpos_to_isaac_fingers(0.5712134838104248, side="left", calibration=calibration)
    opened = standard_gripper_qpos_to_isaac_fingers(0.9473305344581604, side="left", calibration=calibration)

    contact_gap = abs(float(contact["left/left_finger"]) - float(contact["left/right_finger"]))
    open_gap = abs(float(opened["left/left_finger"]) - float(opened["left/right_finger"]))

    np.testing.assert_allclose(contact_gap, 0.052)
    np.testing.assert_allclose(open_gap, 0.114)
    assert calibration.standard_closed_value == standard_gripper_value_for_symmetric_finger_gap(0.052)
    assert "episode18" in calibration.source
