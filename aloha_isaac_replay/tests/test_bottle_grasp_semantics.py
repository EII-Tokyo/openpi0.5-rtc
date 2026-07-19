from __future__ import annotations

import numpy as np

from aloha_isaac_replay.validation.bottle_grasp_semantics import evaluate_grasp_semantics
from aloha_isaac_replay.validation.bottle_grasp_semantics import evaluate_axis_aligned_finger_rear_quarter


PERPENDICULAR_GRASP = {
    "name": "grasp_rear_quarter",
    "position": [-0.000006338189, 0.056161419263, 0.037819468343],
    "orientation": {
        "w": 0.707106725188,
        "xyz": [-0.000281413091, 0.000281413091, -0.707106725188],
    },
}


OLD_PARALLEL_MID_GRASP = {
    "name": "grasp_mid",
    "position": [0.0, 0.052, 0.105],
    "orientation": {
        "w": 0.500198949508,
        "xyz": [0.499800971299, -0.499800971299, -0.500198949508],
    },
}


def test_rear_quarter_perpendicular_grasp_passes() -> None:
    row = evaluate_grasp_semantics(PERPENDICULAR_GRASP)

    assert row["pass"] is True
    assert np.isclose(row["rear_fraction_from_bottom"], 0.052 / 0.206)
    assert row["closing_long_axis_dot_abs"] < 1e-3
    assert row["approach_side_dot"] > 0.99
    assert row["finger_midpoint_radial_distance_m"] < 1e-4


def test_old_mid_grasp_fails_rear_quarter_and_closing_direction() -> None:
    row = evaluate_grasp_semantics(OLD_PARALLEL_MID_GRASP)

    assert row["pass"] is False
    assert row["rear_quarter_ok"] is False
    assert row["closing_perpendicular_ok"] is False


def test_axis_aligned_finger_rear_quarter_passes_for_gap_center_offset() -> None:
    finger_center = np.asarray([-0.194288097965299, 0.368509901617476, 0.2551739379285274])
    object_center = finger_center + np.asarray([0.0515, 0.0, 0.0])
    object_box = {
        "bbox_valid": True,
        "min": (object_center - np.asarray([0.103, 0.034, 0.034])).tolist(),
        "max": (object_center + np.asarray([0.103, 0.034, 0.034])).tolist(),
    }

    row = evaluate_axis_aligned_finger_rear_quarter(
        finger_contact_center_world=finger_center,
        object_bbox=object_box,
        object_axis="X",
        finger_gap_axis="Z",
    )

    assert row["pass"] is True
    assert np.isclose(row["fraction_from_axis_min"], 0.25)
    assert row["closing_perpendicular_ok"] is True


def test_finger_rear_quarter_rejects_center_vector_parallel_to_bottle_axis() -> None:
    row = evaluate_axis_aligned_finger_rear_quarter(
        finger_contact_center_world=[0.25, 0.0, 0.0],
        object_bbox={"bbox_valid": True, "min": [0.0, -0.034, -0.034], "max": [1.0, 0.034, 0.034]},
        object_axis="X",
        finger_gap_axis="Y",
        finger_gap_axis_vector_world=[1.0, 0.0, 0.0],
    )

    assert row["rear_quarter_ok"] is True
    assert row["closing_perpendicular_ok"] is False
    assert row["closing_long_axis_dot_abs"] == 1.0
    assert row["pass"] is False


def test_axis_aligned_finger_rear_quarter_rejects_mid_body_contact() -> None:
    object_center = np.asarray([0.0, 0.0, 0.0])
    object_box = {
        "bbox_valid": True,
        "min": [-0.103, -0.034, -0.034],
        "max": [0.103, 0.034, 0.034],
    }

    row = evaluate_axis_aligned_finger_rear_quarter(
        finger_contact_center_world=object_center,
        object_bbox=object_box,
        object_axis="X",
        finger_gap_axis="Z",
    )

    assert row["pass"] is False
    assert row["rear_quarter_ok"] is False
