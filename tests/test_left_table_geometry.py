import math

import pytest

from tools.isaac_sim.left_table_geometry import (
    maximum_point_error,
    minimum_local_z,
    points_in_table_footprint,
)


IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def test_identity_table_filters_world_points_by_local_footprint():
    result = points_in_table_footprint(
        [(0.0, 0.0, -0.001), (0.56, 0.0, -0.002), (0.0, 0.31, 0.0)],
        IDENTITY,
        (0.55, 0.30),
    )

    assert result == [(0.0, 0.0, -0.001)]
    assert minimum_local_z(result) == pytest.approx(-0.001)


def test_rotated_translated_table_uses_table_local_coordinates():
    table_from_world = [
        [0.0, 1.0, 0.0, -2.0],
        [-1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, -3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    result = points_in_table_footprint(
        [(0.8, 2.1, 2.999), (0.8, 2.7, 2.9)],
        table_from_world,
        (0.55, 0.30),
    )

    assert len(result) == 1
    assert result[0] == pytest.approx((0.1, 0.2, -0.001))


def test_minimum_local_z_rejects_empty_point_set():
    with pytest.raises(ValueError, match="no points inside table footprint"):
        minimum_local_z([])


def test_point_correspondence_is_ordered_and_metric():
    visual = [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)]
    collision = [(0.0, 0.0, 0.0), (1.0002, 2.0, 3.0)]

    assert maximum_point_error(visual, visual) == 0.0
    assert maximum_point_error(visual, collision) == pytest.approx(0.0002)


def test_point_correspondence_rejects_empty_or_different_topology():
    with pytest.raises(ValueError, match="visual/collision topology mismatch"):
        maximum_point_error([], [])
    with pytest.raises(ValueError, match="visual/collision topology mismatch"):
        maximum_point_error([(0.0, 0.0, 0.0)], [])


def test_geometry_rejects_malformed_or_nonfinite_transform():
    with pytest.raises(ValueError, match="table_from_world must be 4x4"):
        points_in_table_footprint([(0.0, 0.0, 0.0)], [[1.0]], (1.0, 1.0))

    nonfinite = [row[:] for row in IDENTITY]
    nonfinite[0][0] = math.nan
    with pytest.raises(ValueError, match="non-finite homogeneous point"):
        points_in_table_footprint([(0.0, 0.0, 0.0)], nonfinite, (1.0, 1.0))
