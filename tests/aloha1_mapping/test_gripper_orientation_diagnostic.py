from __future__ import annotations

import numpy as np

from tools.aloha1_mapping.gripper_orientation import classify_orientation
from tools.aloha1_mapping.gripper_orientation import expected_capture_names
from tools.aloha1_mapping.gripper_orientation import finger_state_targets
from tools.aloha1_mapping.gripper_orientation import inward_surface_normal_y
from tools.aloha1_mapping.gripper_orientation import obj_text
from tools.aloha1_mapping.gripper_orientation import physical_side_order
from tools.aloha1_mapping.gripper_orientation import surface_normal_gate


def test_physical_side_order_requires_left_above_right_y() -> None:
    assert physical_side_order(0.51, 0.49) is True
    assert physical_side_order(0.49, 0.51) is False


def test_surface_normal_gate_requires_opposed_inward_normals() -> None:
    assert surface_normal_gate(-0.999, 0.999) is True
    assert surface_normal_gate(0.999, -0.999) is False


def test_orientation_classification_requires_monotonic_aperture() -> None:
    result = classify_orientation(
        side_order_ok=True,
        inward_normals_ok=True,
        closed_aperture_m=0.001,
        open_aperture_m=0.073,
        crossed_centerline=False,
    )

    assert result["status"] == "ASSEMBLY_ORIENTATION_CONFIRMED"
    assert result["gates"]["aperture_monotonic"] is True


def test_finger_state_targets_use_imported_asymmetric_limits() -> None:
    limits = {
        "left": (0.021, 0.057),
        "right": (-0.057, -0.021),
    }

    assert finger_state_targets(limits, "closed") == {
        "left": 0.021,
        "right": -0.021,
    }
    assert finger_state_targets(limits, "open") == {
        "left": 0.057,
        "right": -0.057,
    }


def test_obj_text_is_one_based_and_deterministic() -> None:
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.asarray([[0, 1, 2]], dtype=int)

    assert obj_text("finger", points, faces) == ("o finger\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")


def test_inward_surface_normal_follows_physical_side() -> None:
    left_points = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 0, 1]],
        dtype=float,
    )
    right_points = left_points[[0, 2, 1]]
    faces = np.asarray([[0, 1, 2]], dtype=int)

    assert inward_surface_normal_y(left_points, faces, "left") == -1.0
    assert inward_surface_normal_y(right_points, faces, "right") == 1.0


def test_expected_capture_names_cover_both_states_and_three_views() -> None:
    assert expected_capture_names() == [
        "closed_closing_axis.png",
        "closed_top.png",
        "closed_isometric.png",
        "open_closing_axis.png",
        "open_top.png",
        "open_isometric.png",
    ]
