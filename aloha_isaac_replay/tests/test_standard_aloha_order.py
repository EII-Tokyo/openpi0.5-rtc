from __future__ import annotations

import numpy as np

from aloha_isaac_replay.adapters.standard_aloha import OPENPI_JOINT_FLIP_MASK
from aloha_isaac_replay.adapters.standard_aloha import STANDARD_ALOHA_14D_NAMES
from aloha_isaac_replay.adapters.standard_aloha import openpi_internal_to_standard
from aloha_isaac_replay.adapters.standard_aloha import split_left_right
from aloha_isaac_replay.adapters.standard_aloha import standard_to_openpi_internal


def test_standard_aloha_14d_order_is_explicit() -> None:
    assert STANDARD_ALOHA_14D_NAMES == (
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    )


def test_openpi_sign_flip_round_trip_preserves_standard_values() -> None:
    standard = np.arange(14, dtype=np.float64)
    internal = standard_to_openpi_internal(standard)
    assert np.array_equal(internal, standard * OPENPI_JOINT_FLIP_MASK)
    assert np.array_equal(openpi_internal_to_standard(internal), standard)


def test_split_left_right_uses_standard_seven_dim_halves() -> None:
    left, right = split_left_right(np.arange(14, dtype=np.float64))
    assert np.array_equal(left, np.arange(7, dtype=np.float64))
    assert np.array_equal(right, np.arange(7, 14, dtype=np.float64))

