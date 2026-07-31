"""Deterministic legal-position sampling for the Bottle500 grasp test."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import math
from typing import Any

import numpy as np


def _xy_bounds(record: Mapping[str, Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.asarray(record["minimum"], dtype=np.float64)
    maximum = np.asarray(record["maximum"], dtype=np.float64)
    if minimum.shape != (2,) or maximum.shape != (2,):
        raise ValueError("XY bounds must contain two-vectors")
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        raise ValueError("XY bounds must be finite")
    if np.any(maximum <= minimum):
        raise ValueError("XY maximum must exceed minimum")
    return minimum, maximum


def derive_legal_offset_bounds(
    *,
    table_xy_bounds: Mapping[str, Sequence[float]],
    left_base_aabb: Mapping[str, Sequence[float]],
    right_base_aabb: Mapping[str, Sequence[float]],
    nominal_bottle_xy_bounds: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Derive the central unobstructed tabletop offset envelope."""

    table_min, table_max = _xy_bounds(table_xy_bounds)
    bottle_min, bottle_max = _xy_bounds(nominal_bottle_xy_bounds)
    left_max_x = float(left_base_aabb["maximum"][0])
    right_min_x = float(right_base_aabb["minimum"][0])
    free_min = np.asarray(
        [max(float(table_min[0]), left_max_x), float(table_min[1])],
        dtype=np.float64,
    )
    free_max = np.asarray(
        [min(float(table_max[0]), right_min_x), float(table_max[1])],
        dtype=np.float64,
    )
    if np.any(free_max <= free_min):
        raise ValueError("robot bases leave no central tabletop region")
    offset_min = free_min - bottle_min
    offset_max = free_max - bottle_max
    if np.any(offset_max <= offset_min):
        raise ValueError("Bottle500 does not fit in central tabletop region")
    if np.any(offset_min > 0.0) or np.any(offset_max < 0.0):
        raise ValueError("accepted nominal Bottle500 pose is outside region")
    return {
        "free_surface_xy": {
            "minimum": free_min.tolist(),
            "maximum": free_max.tolist(),
            "derivation": (
                "TABLE_Y_BOUNDS_AND_INNER_X_FACES_OF_COMPOSED_"
                "LEFT_RIGHT_ROBOT_BASE_AABBS"
            ),
        },
        "nominal_bottle_xy_bounds": {
            "minimum": bottle_min.tolist(),
            "maximum": bottle_max.tolist(),
        },
        "offset_xy_m": {
            "minimum": offset_min.tolist(),
            "maximum": offset_max.tolist(),
        },
        "clearance_margin_m": 0.0,
        "clearance_margin_status": (
            "NO_GUESSED_MARGIN;_CANDIDATES_REQUIRE_RUNTIME_IK_AND_"
            "PHYSICS_VALIDATION"
        ),
    }


def sample_candidate_offsets(
    *,
    offset_xy_bounds: Mapping[str, Sequence[float]],
    seed: int,
    count: int,
) -> list[dict[str, Any]]:
    """Draw a reproducible candidate sequence without changing physics."""

    minimum, maximum = _xy_bounds(offset_xy_bounds)
    if count < 1:
        raise ValueError("candidate count must be positive")
    generator = np.random.default_rng(int(seed))
    samples = generator.uniform(minimum, maximum, size=(count, 2))
    return [
        {
            "candidate_index": index,
            "offset_xy_m": [float(value) for value in sample],
            "seed": int(seed),
        }
        for index, sample in enumerate(samples)
    ]


def _translate_xyz(value: Sequence[float], offset: np.ndarray) -> list[float]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("translated world position must be a finite 3-vector")
    vector[:2] += offset
    return vector.tolist()


def translate_horizontal_bottle_profile(
    profile: Mapping[str, Any],
    *,
    offset_xy_m: Sequence[float],
) -> dict[str, Any]:
    """Translate the accepted object pose while preserving T_O_G."""

    offset = np.asarray(offset_xy_m, dtype=np.float64)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError("Bottle500 XY offset must be a finite two-vector")
    translated = copy.deepcopy(dict(profile))
    placement = translated["kinematics"]["placement"]
    matrix = np.asarray(placement["placement_matrix"], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("placement matrix must be finite 4x4")
    matrix[:2, 3] += offset
    placement["placement_matrix"] = matrix.tolist()
    axis = placement["bottle_axis"]
    for key in ("a_world_m", "b_world_m", "grasp_point_world_m"):
        axis[key] = _translate_xyz(axis[key], offset)
    targets = placement["target_poses"]
    for key in (
        "pregrasp_ee_position_world_m",
        "grasp_ee_position_world_m",
        "lift_ee_position_world_m",
    ):
        targets[key] = _translate_xyz(targets[key], offset)
    placement["random_position_offset_xy_m"] = offset.tolist()
    placement["randomization_semantics"] = (
        "WORLD_XY_TRANSLATION_ONLY;_BOTTLE_ROTATION_AND_T_O_G_UNCHANGED"
    )
    if not math.isclose(
        float(np.linalg.det(matrix[:3, :3])),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise ValueError("translated bottle rotation determinant changed")
    return translated


def extend_profile_for_clearance_lift(
    profile: Mapping[str, Any],
    *,
    target_clearance_m: float,
    hold_drop_gate_m: float,
    additional_lift_margin_m: float = 0.0,
) -> dict[str, Any]:
    """Bind IK preflight and runtime to the same formal lift distance."""

    values = (
        float(target_clearance_m),
        float(hold_drop_gate_m),
        float(additional_lift_margin_m),
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("formal lift inputs must be finite and non-negative")
    extended = copy.deepcopy(dict(profile))
    targets = extended["kinematics"]["placement"]["target_poses"]
    grasp = np.asarray(
        targets["grasp_ee_position_world_m"],
        dtype=np.float64,
    )
    lift = np.asarray(
        targets["lift_ee_position_world_m"],
        dtype=np.float64,
    )
    if (
        grasp.shape != (3,)
        or lift.shape != (3,)
        or not np.isfinite(grasp).all()
        or not np.isfinite(lift).all()
    ):
        raise ValueError("grasp/lift targets must be finite 3-vectors")
    formal_lift = sum(values)
    lift[2] = grasp[2] + formal_lift
    targets["lift_ee_position_world_m"] = lift.tolist()
    extended["formal_lift_distance_m"] = formal_lift
    extended["formal_lift_derivation"] = (
        "TARGET_CLEARANCE_PLUS_HOLD_DROP_GATE_PLUS_DIAGNOSTIC_MARGIN"
    )
    extended["additional_lift_margin_m"] = values[2]
    return extended
