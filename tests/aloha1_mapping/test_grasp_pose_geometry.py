from __future__ import annotations

import numpy as np
import pytest

from tools.aloha1_mapping.grasp_pose_geometry import derive_gripper_pose
from tools.aloha1_mapping.grasp_pose_geometry import evaluate_pre_ik_grasp


def _point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return (transform @ np.asarray([*point, 1.0], dtype=np.float64))[:3]


def test_grasp_pose_maps_contact_midpoint_and_radial_line() -> None:
    left_gripper = np.asarray([-0.038, 0.0, 0.0], dtype=np.float64)
    right_gripper = np.asarray([0.038, 0.0, 0.0], dtype=np.float64)
    bottle_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    grasp_point = np.asarray([0.069, 0.0, 0.033], dtype=np.float64)

    world_from_gripper = derive_gripper_pose(
        left_contact_gripper_m=left_gripper,
        right_contact_gripper_m=right_gripper,
        gripper_approach_axis=[0.0, 0.0, -1.0],
        bottle_axis_world=bottle_axis,
        grasp_point_world_m=grasp_point,
        table_up_world=[0.0, 0.0, 1.0],
    )

    left_world = _point(world_from_gripper, left_gripper)
    right_world = _point(world_from_gripper, right_gripper)
    assert (left_world + right_world) / 2.0 == pytest.approx(grasp_point)
    assert abs(float(np.dot(right_world - left_world, bottle_axis))) < 1e-12
    assert float(np.linalg.det(world_from_gripper[:3, :3])) == pytest.approx(1.0)


def test_real_supplier_cad_sample_preserves_handedness() -> None:
    left_gripper = np.asarray(
        [0.005540657957598394, -0.1298559135889289, -0.0012842316550211036],
        dtype=np.float64,
    )
    right_gripper = np.asarray(
        [0.015102521266211703, -0.2018600770598328, 0.022714400640992588],
        dtype=np.float64,
    )
    target = np.asarray([0.010321589611905047, -0.1658579953243808, 0.033], dtype=np.float64)

    world_from_gripper = derive_gripper_pose(
        left_contact_gripper_m=left_gripper,
        right_contact_gripper_m=right_gripper,
        gripper_approach_axis=[0.0, 0.0, -1.0],
        bottle_axis_world=[0.9912975457874654, 0.13164032708766643, 0.0],
        grasp_point_world_m=target,
        table_up_world=[0.0, 0.0, 1.0],
    )

    left_world = _point(world_from_gripper, left_gripper)
    right_world = _point(world_from_gripper, right_gripper)
    assert (left_world + right_world) / 2.0 == pytest.approx(target)
    assert np.linalg.det(world_from_gripper[:3, :3]) == pytest.approx(1.0)
    assert np.linalg.norm(right_world - left_world) == pytest.approx(
        np.linalg.norm(right_gripper - left_gripper)
    )


def test_same_side_fingers_fail_closed() -> None:
    result = evaluate_pre_ik_grasp(
        left_contact_world_m=[0.0, 0.04, 0.03],
        right_contact_world_m=[0.0, 0.06, 0.03],
        bottle_axis_a_world_m=[-0.1, 0.0, 0.03],
        bottle_axis_b_world_m=[0.1, 0.0, 0.03],
        expected_axis_coordinate_m=0.1,
        open_aperture_m=0.08,
        section_diameter_m=0.068,
    )

    assert result.status == "FAIL"
    assert "same_radial_side" in result.failed_gates


def test_opposite_sides_at_body_section_pass() -> None:
    result = evaluate_pre_ik_grasp(
        left_contact_world_m=[0.0, -0.038, 0.03],
        right_contact_world_m=[0.0, 0.038, 0.03],
        bottle_axis_a_world_m=[-0.1, 0.0, 0.03],
        bottle_axis_b_world_m=[0.1, 0.0, 0.03],
        expected_axis_coordinate_m=0.1,
        open_aperture_m=0.076,
        section_diameter_m=0.068,
    )

    assert result.status == "PASS"
    assert result.failed_gates == ()
    assert result.metrics["gripper_line_to_axis_deg"] == pytest.approx(90.0)
    assert result.metrics["left_radial_signed_m"] < 0.0
    assert result.metrics["right_radial_signed_m"] > 0.0


def test_aperture_and_section_mismatch_fail_independently() -> None:
    result = evaluate_pre_ik_grasp(
        left_contact_world_m=[-0.02, -0.038, 0.03],
        right_contact_world_m=[0.02, 0.038, 0.03],
        bottle_axis_a_world_m=[-0.1, 0.0, 0.03],
        bottle_axis_b_world_m=[0.1, 0.0, 0.03],
        expected_axis_coordinate_m=0.1,
        open_aperture_m=0.06,
        section_diameter_m=0.068,
        axial_tolerance_m=0.005,
    )

    assert result.status == "FAIL"
    assert "open_aperture_not_larger_than_section" in result.failed_gates
    assert "finger_axial_mismatch" in result.failed_gates


@pytest.mark.parametrize(
    ("left", "right", "approach"),
    [
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([-0.03, 0.0, 0.0], [0.03, 0.0, 0.0], [1.0, 0.0, 0.0]),
    ],
)
def test_degenerate_gripper_basis_is_rejected(
    left: list[float],
    right: list[float],
    approach: list[float],
) -> None:
    with pytest.raises(ValueError, match="degenerate"):
        derive_gripper_pose(
            left_contact_gripper_m=left,
            right_contact_gripper_m=right,
            gripper_approach_axis=approach,
            bottle_axis_world=[1.0, 0.0, 0.0],
            grasp_point_world_m=[0.0, 0.0, 0.0],
            table_up_world=[0.0, 0.0, 1.0],
        )
