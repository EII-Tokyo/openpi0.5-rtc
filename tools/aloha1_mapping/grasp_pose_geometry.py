"""CAD-derived pre-IK geometry for an ALOHA parallel-finger grasp."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PreIkGraspEvaluation:
    """Fail-closed result for a candidate grasp before any IK call."""

    status: str
    failed_gates: tuple[str, ...]
    metrics: dict[str, Any]


def _vector(value: Sequence[float], *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return result


def _unit(value: Sequence[float], *, name: str) -> np.ndarray:
    result = _vector(value, name=name)
    norm = float(np.linalg.norm(result))
    if norm <= 1e-12:
        raise ValueError(f"degenerate {name}")
    return result / norm


def _orthogonal_component(
    vector: Sequence[float],
    axis: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    candidate = _vector(vector, name=name)
    candidate = candidate - axis * float(np.dot(candidate, axis))
    norm = float(np.linalg.norm(candidate))
    if norm <= 1e-12:
        raise ValueError(f"degenerate {name}")
    return candidate / norm


def derive_gripper_pose(
    *,
    left_contact_gripper_m: Sequence[float],
    right_contact_gripper_m: Sequence[float],
    gripper_approach_axis: Sequence[float],
    bottle_axis_world: Sequence[float],
    grasp_point_world_m: Sequence[float],
    table_up_world: Sequence[float],
) -> np.ndarray:
    """Map verified gripper contact geometry to a horizontal bottle section.

    The returned homogeneous matrix maps points in the selected Grasp Editor
    gripper frame into the supplied world/task frame.  It preserves the
    left/right handed geometry; it does not mirror or swap fingers.
    """

    left = _vector(left_contact_gripper_m, name="left contact")
    right = _vector(right_contact_gripper_m, name="right contact")
    local_radial = _unit(right - left, name="finger contact line")
    local_approach = _orthogonal_component(
        gripper_approach_axis,
        local_radial,
        name="gripper approach axis",
    )
    local_third = _unit(
        np.cross(local_radial, local_approach),
        name="local third axis",
    )
    local_basis = np.column_stack(
        (local_radial, local_approach, local_third)
    )

    bottle_axis = _unit(bottle_axis_world, name="bottle axis")
    table_up = _unit(table_up_world, name="table up")
    world_radial = _unit(
        np.cross(table_up, bottle_axis),
        name="bottle radial axis",
    )
    world_approach = _orthogonal_component(
        -table_up,
        world_radial,
        name="world approach axis",
    )
    world_third = _unit(
        np.cross(world_radial, world_approach),
        name="world third axis",
    )
    world_basis = np.column_stack(
        (world_radial, world_approach, world_third)
    )

    rotation = world_basis @ local_basis.T
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10):
        raise ValueError("derived gripper rotation is not orthogonal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=1e-10):
        raise ValueError(
            f"derived gripper rotation determinant is {determinant}"
        )

    local_midpoint = (left + right) / 2.0
    target_midpoint = _vector(grasp_point_world_m, name="grasp point")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_midpoint - rotation @ local_midpoint
    return transform


def _perpendicular_angle_degrees(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    cosine = float(
        np.clip(
            abs(float(np.dot(_unit(first, name="first angle vector"), _unit(second, name="second angle vector")))),
            0.0,
            1.0,
        )
    )
    return float(np.degrees(np.arccos(cosine)))


def evaluate_pre_ik_grasp(
    *,
    left_contact_world_m: Sequence[float],
    right_contact_world_m: Sequence[float],
    bottle_axis_a_world_m: Sequence[float],
    bottle_axis_b_world_m: Sequence[float],
    expected_axis_coordinate_m: float,
    open_aperture_m: float,
    section_diameter_m: float,
    table_up_world: Sequence[float] = (0.0, 0.0, 1.0),
    body_interval_m: Sequence[float] = (0.018, 0.120),
    axial_tolerance_m: float = 0.005,
    perpendicular_tolerance_deg: float = 3.0,
    contact_envelope_allowance_m: float = 0.0,
) -> PreIkGraspEvaluation:
    """Evaluate whether a candidate can geometrically pinch the bottle."""

    left = _vector(left_contact_world_m, name="left contact")
    right = _vector(right_contact_world_m, name="right contact")
    axis_a = _vector(bottle_axis_a_world_m, name="bottle axis A")
    axis_b = _vector(bottle_axis_b_world_m, name="bottle axis B")
    axis = _unit(axis_b - axis_a, name="bottle axis")
    table_up = _unit(table_up_world, name="table up")
    radial = _unit(np.cross(table_up, axis), name="bottle radial axis")
    line = right - left

    left_axial = float(np.dot(left - axis_a, axis))
    right_axial = float(np.dot(right - axis_a, axis))
    left_axis_point = axis_a + axis * left_axial
    right_axis_point = axis_a + axis * right_axial
    left_radial = float(np.dot(left - left_axis_point, radial))
    right_radial = float(np.dot(right - right_axis_point, radial))
    line_angle = _perpendicular_angle_degrees(line, axis)
    lower, upper = (float(value) for value in body_interval_m)

    failed: list[str] = []
    if left_radial * right_radial >= 0.0:
        failed.append("same_radial_side")
    if abs(left_axial - right_axial) > float(axial_tolerance_m):
        failed.append("finger_axial_mismatch")
    if (
        abs(left_axial - float(expected_axis_coordinate_m))
        > float(axial_tolerance_m)
        or abs(right_axial - float(expected_axis_coordinate_m))
        > float(axial_tolerance_m)
    ):
        failed.append("grasp_section_mismatch")
    if not (
        lower <= left_axial <= upper and lower <= right_axial <= upper
    ):
        failed.append("outside_bottle_body_interval")
    if abs(line_angle - 90.0) > float(perpendicular_tolerance_deg):
        failed.append("contact_line_not_perpendicular_to_bottle_axis")
    required_aperture = (
        float(section_diameter_m) + float(contact_envelope_allowance_m)
    )
    if float(open_aperture_m) <= required_aperture:
        failed.append("open_aperture_not_larger_than_section")

    metrics: dict[str, Any] = {
        "left_axis_coordinate_m": left_axial,
        "right_axis_coordinate_m": right_axial,
        "expected_axis_coordinate_m": float(expected_axis_coordinate_m),
        "left_radial_signed_m": left_radial,
        "right_radial_signed_m": right_radial,
        "gripper_line_to_axis_deg": line_angle,
        "open_aperture_m": float(open_aperture_m),
        "section_diameter_m": float(section_diameter_m),
        "contact_envelope_allowance_m": float(
            contact_envelope_allowance_m
        ),
        "body_interval_m": [lower, upper],
        "axial_tolerance_m": float(axial_tolerance_m),
        "perpendicular_tolerance_deg": float(
            perpendicular_tolerance_deg
        ),
    }
    return PreIkGraspEvaluation(
        status="PASS" if not failed else "FAIL",
        failed_gates=tuple(failed),
        metrics=metrics,
    )
