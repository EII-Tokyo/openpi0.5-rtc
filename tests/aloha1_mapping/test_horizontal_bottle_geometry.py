from __future__ import annotations

import math

import numpy as np
import pytest

from tools.aloha1_mapping.horizontal_bottle_geometry import canonical_bottle_axis
from tools.aloha1_mapping.horizontal_bottle_geometry import derive_horizontal_support_placement
from tools.aloha1_mapping.horizontal_bottle_geometry import evaluate_geometry
from tools.aloha1_mapping.horizontal_bottle_geometry import point_on_axis
from tools.aloha1_mapping.horizontal_bottle_geometry import shortest_arc_rotation
from tools.aloha1_mapping.horizontal_bottle_geometry import transform_directed_axis
from tools.aloha1_mapping.horizontal_bottle_geometry import transform_points


def test_transform_axis_preserves_directed_length() -> None:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    axis = transform_directed_axis(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.206],
        transform,
    )

    assert axis.a_world == pytest.approx([1.0, 2.0, 3.0])
    assert axis.b_world == pytest.approx([1.0, 2.0, 3.206])
    assert axis.unit == pytest.approx([0.0, 0.0, 1.0])
    assert axis.length_m == pytest.approx(0.206)


@pytest.mark.parametrize(
    "transform",
    [
        np.full((4, 4), np.nan),
        np.diag([1.0, 1.0, 1.0, 2.0]),
        np.diag([-1.0, 1.0, 1.0, 1.0]),
    ],
)
def test_transform_axis_rejects_invalid_affine_transform(
    transform: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="finite|affine|determinant"):
        transform_directed_axis(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.206],
            transform,
        )


def test_transform_axis_rejects_zero_length_axis() -> None:
    with pytest.raises(ValueError, match="zero-length"):
        transform_directed_axis(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            np.eye(4),
        )


@pytest.mark.parametrize(
    ("gripper_line", "expected"),
    [
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0]),
    ],
)
def test_canonical_horizontal_axis_is_perpendicular_and_signed(
    gripper_line: list[float],
    expected: list[float],
) -> None:
    axis = canonical_bottle_axis(gripper_line)
    assert axis == pytest.approx(expected)
    assert float(np.dot(axis, gripper_line)) == pytest.approx(0.0)
    assert axis[2] == pytest.approx(0.0)


def test_canonical_horizontal_axis_rejects_vertical_line() -> None:
    with pytest.raises(ValueError, match="XY projection"):
        canonical_bottle_axis([0.0, 0.0, 1.0])


def test_shortest_arc_rotation_maps_axis_without_reflection() -> None:
    rotation = shortest_arc_rotation(
        source=[0.0, 0.0, 1.0],
        target=[1.0, 0.0, 0.0],
    )

    assert rotation @ np.array([0.0, 0.0, 1.0]) == pytest.approx(
        [1.0, 0.0, 0.0]
    )
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert rotation.T @ rotation == pytest.approx(np.eye(3))


def test_shortest_arc_rotation_handles_antiparallel_deterministically() -> None:
    first = shortest_arc_rotation(
        source=[0.0, 0.0, 1.0],
        target=[0.0, 0.0, -1.0],
    )
    second = shortest_arc_rotation(
        source=[0.0, 0.0, 1.0],
        target=[0.0, 0.0, -1.0],
    )

    assert first == pytest.approx(second)
    assert first @ np.array([0.0, 0.0, 1.0]) == pytest.approx(
        [0.0, 0.0, -1.0]
    )
    assert np.linalg.det(first) == pytest.approx(1.0)


def test_support_placement_uses_collision_samples_and_grasp_coordinate() -> None:
    local_points = np.asarray(
        [
            [-0.034, 0.0, 0.0],
            [0.034, 0.0, 0.0],
            [0.0, 0.0, 0.206],
        ],
        dtype=np.float64,
    )
    rotation = shortest_arc_rotation(
        source=[0.0, 0.0, 1.0],
        target=[1.0, 0.0, 0.0],
    )
    placement = derive_horizontal_support_placement(
        local_collision_points=local_points,
        rotation=rotation,
        grasp_center_world_xy=[0.40, -0.12],
        grasp_coordinate_m=0.069,
        table_top_z=0.75,
        setup_gap_m=0.002,
        axis_a_local=[0.0, 0.0, 0.0],
        axis_b_local=[0.0, 0.0, 0.206],
    )

    world_points = transform_points(local_points, placement.matrix)
    assert world_points[:, 2].min() == pytest.approx(0.752)
    assert point_on_axis(
        placement.a_world,
        placement.axis_unit,
        0.069,
    )[:2] == pytest.approx([0.40, -0.12])
    assert placement.axis_unit == pytest.approx([1.0, 0.0, 0.0])
    assert placement.lowest_point_world_z == pytest.approx(0.752)


def test_support_placement_rejects_negative_setup_gap() -> None:
    with pytest.raises(ValueError, match="setup gap"):
        derive_horizontal_support_placement(
            local_collision_points=np.asarray(
                [[-0.01, 0.0, 0.0], [0.01, 0.0, 0.1]]
            ),
            rotation=np.eye(3),
            grasp_center_world_xy=[0.0, 0.0],
            grasp_coordinate_m=0.05,
            table_top_z=0.0,
            setup_gap_m=-0.001,
            axis_a_local=[0.0, 0.0, 0.0],
            axis_b_local=[0.0, 0.0, 0.1],
        )


def test_geometry_gate_accepts_horizontal_perpendicular_vertical_approach() -> (
    None
):
    result = evaluate_geometry(
        axis_unit=[1.0, 0.0, 0.0],
        table_normal=[0.0, 0.0, 1.0],
        gripper_line=[0.0, 1.0, 0.0],
        approach_delta=[0.0, 0.0, -0.01],
        axis_vertical_angle_gate_deg=1.0,
        gripper_perpendicular_gate_deg=3.0,
        approach_direction_gate_deg=3.0,
    )

    assert result["status"] == "PASS"
    assert result["axis_to_table_normal_deg"] == pytest.approx(90.0)
    assert result["gripper_line_to_axis_deg"] == pytest.approx(90.0)
    assert result["approach_to_negative_z_deg"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("axis", "gripper_line", "approach", "failed_gate"),
    [
        (
            [math.cos(math.radians(2.0)), 0.0, math.sin(math.radians(2.0))],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -0.01],
            "axis_horizontal",
        ),
        (
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -0.01],
            "gripper_axis_perpendicular",
        ),
        (
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.01, 0.0, -0.01],
            "vertical_approach",
        ),
    ],
)
def test_geometry_gate_reports_specific_failure(
    axis: list[float],
    gripper_line: list[float],
    approach: list[float],
    failed_gate: str,
) -> None:
    result = evaluate_geometry(
        axis_unit=axis,
        table_normal=[0.0, 0.0, 1.0],
        gripper_line=gripper_line,
        approach_delta=approach,
        axis_vertical_angle_gate_deg=1.0,
        gripper_perpendicular_gate_deg=3.0,
        approach_direction_gate_deg=3.0,
    )

    assert result["status"] == "FAIL"
    assert failed_gate in result["failed_gates"]
