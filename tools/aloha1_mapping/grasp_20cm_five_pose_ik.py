"""Pure geometry and sampling for the frozen five-pose ALOHA grasp gate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def _finite_vector(
    value: Sequence[float] | np.ndarray,
    *,
    size: int,
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {size}-vector")
    return result


def require_rigid_transform(
    value: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str = "transform",
) -> np.ndarray:
    """Return a finite right-handed rigid 4x4 transform."""

    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(
        result[3],
        [0.0, 0.0, 0.0, 1.0],
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = result[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise ValueError(f"{name} rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError(f"{name} rotation determinant is {determinant}")
    return result.copy()


def _transform_point(transform: np.ndarray, point: Sequence[float]) -> np.ndarray:
    local = _finite_vector(point, size=3, name="point")
    return transform[:3, :3] @ local + transform[:3, 3]


def _unit(value: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    vector = _finite_vector(value, size=3, name=name)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} is degenerate")
    return vector / norm


def _angle_degrees(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
) -> float:
    cosine = float(
        np.clip(
            np.dot(_unit(first, name="first vector"), _unit(second, name="second vector")),
            -1.0,
            1.0,
        )
    )
    return float(math.degrees(math.acos(cosine)))


def place_bottle_center_and_yaw(
    *,
    nominal_world_from_object: Sequence[Sequence[float]] | np.ndarray,
    geometric_center_local_m: Sequence[float],
    desired_center_xy_m: Sequence[float],
    yaw_delta_rad: float,
) -> np.ndarray:
    """Place the CAD center at world XY and rotate the horizontal pose in Z."""

    source = require_rigid_transform(
        nominal_world_from_object,
        name="nominal_world_from_object",
    )
    center_local = _finite_vector(
        geometric_center_local_m,
        size=3,
        name="geometric_center_local_m",
    )
    desired_xy = _finite_vector(
        desired_center_xy_m,
        size=2,
        name="desired_center_xy_m",
    )
    yaw = float(yaw_delta_rad)
    if not math.isfinite(yaw):
        raise ValueError("yaw_delta_rad must be finite")
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotate_z = np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    result = source.copy()
    result[:3, :3] = rotate_z @ source[:3, :3]
    source_center_world = _transform_point(source, center_local)
    desired_center_world = source_center_world.copy()
    desired_center_world[:2] = desired_xy
    result[:3, 3] = desired_center_world - result[:3, :3] @ center_local
    return require_rigid_transform(result, name="world_from_object")


def line_yaw_deg(axis_world: Sequence[float] | np.ndarray) -> float:
    """Return an unoriented XY line angle in the half-open range [0, 180)."""

    axis = _finite_vector(axis_world, size=3, name="axis_world")
    horizontal = axis[:2]
    if float(np.linalg.norm(horizontal)) <= 1.0e-12:
        raise ValueError("axis_world has no XY projection")
    return float(math.degrees(math.atan2(horizontal[1], horizontal[0])) % 180.0)


def line_yaw_distance_deg(first: float, second: float) -> float:
    """Return the shortest angular distance between two unoriented XY lines."""

    first_value = float(first)
    second_value = float(second)
    if not math.isfinite(first_value) or not math.isfinite(second_value):
        raise ValueError("line yaws must be finite")
    difference = abs((first_value - second_value) % 180.0)
    return float(min(difference, 180.0 - difference))


def derive_sample_geometry(
    *,
    world_from_object: Sequence[Sequence[float]] | np.ndarray,
    a_local_m: Sequence[float],
    b_local_m: Sequence[float],
    object_from_gripper: Sequence[Sequence[float]] | np.ndarray,
) -> dict[str, Any]:
    """Compose CAD bottle-axis and Grasp Editor transforms in world space."""

    world_object = require_rigid_transform(
        world_from_object,
        name="world_from_object",
    )
    object_gripper = require_rigid_transform(
        object_from_gripper,
        name="object_from_gripper",
    )
    a_world = _transform_point(world_object, a_local_m)
    b_world = _transform_point(world_object, b_local_m)
    axis = _unit(b_world - a_world, name="bottle axis")
    world_gripper = require_rigid_transform(
        world_object @ object_gripper,
        name="world_from_gripper",
    )
    return {
        "a_world_m": a_world.tolist(),
        "b_world_m": b_world.tolist(),
        "axis_unit_world": axis.tolist(),
        "axis_length_m": float(np.linalg.norm(b_world - a_world)),
        "line_yaw_deg": line_yaw_deg(axis),
        "axis_to_world_z_deg": _angle_degrees(axis, [0.0, 0.0, 1.0]),
        "world_from_gripper": world_gripper.tolist(),
    }


def _xy_bounds(
    bounds: Mapping[str, Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    minimum = _finite_vector(bounds["minimum"], size=2, name="minimum XY")
    maximum = _finite_vector(bounds["maximum"], size=2, name="maximum XY")
    if np.any(maximum <= minimum):
        raise ValueError("maximum XY must exceed minimum XY")
    return minimum, maximum


def sample_initial_arm_joint_candidates(
    *,
    lower_limits: Sequence[float] | np.ndarray,
    upper_limits: Sequence[float] | np.ndarray,
    seed: int,
    count: int,
) -> np.ndarray:
    """Draw fixed-seed six-DOF candidates within explicit joint limits."""

    lower = _finite_vector(lower_limits, size=6, name="lower_limits")
    upper = _finite_vector(upper_limits, size=6, name="upper_limits")
    if np.any(upper <= lower):
        raise ValueError("upper joint limits must exceed lower limits")
    if int(count) < 1:
        raise ValueError("count must be positive")
    generator = np.random.default_rng(int(seed))
    return generator.uniform(lower, upper, size=(int(count), 6))


def compose_initial_command(
    baseline_command: Sequence[float] | np.ndarray,
    sampled_arm_q_rad: Sequence[float] | np.ndarray,
    *,
    arm_dof_indices: Sequence[int],
) -> np.ndarray:
    """Replace only explicitly mapped arm DOFs in an existing command."""

    baseline = np.asarray(baseline_command, dtype=np.float64)
    if baseline.ndim != 1 or not np.isfinite(baseline).all():
        raise ValueError("baseline_command must be a finite vector")
    arm = _finite_vector(
        sampled_arm_q_rad,
        size=6,
        name="sampled_arm_q_rad",
    )
    indices = [int(value) for value in arm_dof_indices]
    if len(indices) != 6:
        raise ValueError("arm_dof_indices must contain six entries")
    if len(set(indices)) != len(indices):
        raise ValueError("arm_dof_indices must be unique")
    if any(index < 0 or index >= baseline.size for index in indices):
        raise ValueError("arm_dof_indices contains an out of range index")
    result = baseline.copy()
    result[np.asarray(indices, dtype=np.int64)] = arm
    return result


def sample_bottle_center_yaw_candidates(
    *,
    center_xy_bounds: Mapping[str, Sequence[float]],
    yaw_domain_deg: Sequence[float],
    seed: int,
    count: int,
    formal_sample_index: int,
) -> list[dict[str, Any]]:
    """Draw candidates that obey the accepted sample 1/4 centerline layout."""

    minimum, maximum = _xy_bounds(center_xy_bounds)
    yaw_domain = _finite_vector(
        yaw_domain_deg,
        size=2,
        name="yaw_domain_deg",
    )
    if yaw_domain[0] < 0.0 or yaw_domain[1] > 180.0:
        raise ValueError("line-yaw domain must be within [0, 180]")
    if yaw_domain[1] <= yaw_domain[0]:
        raise ValueError("line-yaw maximum must exceed minimum")
    sample_index = int(formal_sample_index)
    if sample_index not in range(5):
        raise ValueError("formal_sample_index must be in range(5)")
    candidate_count = int(count)
    if candidate_count < 1:
        raise ValueError("count must be positive")
    if not minimum[0] <= 0.0 <= maximum[0]:
        raise ValueError("center bounds do not include world x=0")

    sample_minimum = minimum.copy()
    sample_maximum = maximum.copy()
    fixed_x: float | None = None
    if sample_index == 0:
        fixed_x = 0.0
        sample_minimum[1] = max(sample_minimum[1], np.nextafter(0.0, 1.0))
    elif sample_index == 3:
        fixed_x = 0.0
        sample_maximum[1] = min(sample_maximum[1], np.nextafter(0.0, -1.0))
    else:
        sample_maximum[0] = min(sample_maximum[0], np.nextafter(0.0, -1.0))
    if np.any(sample_maximum <= sample_minimum):
        raise ValueError("bounds do not support the formal sample structure")

    generator = np.random.default_rng(
        np.random.SeedSequence([int(seed), sample_index])
    )
    centers = generator.uniform(
        sample_minimum,
        sample_maximum,
        size=(candidate_count, 2),
    )
    if fixed_x is not None:
        centers[:, 0] = fixed_x
    yaws = generator.uniform(
        float(yaw_domain[0]),
        float(yaw_domain[1]),
        size=candidate_count,
    )
    return [
        {
            "candidate_index": index,
            "formal_sample_index": sample_index,
            "seed": int(seed),
            "bottle_center_xy_m": [float(value) for value in centers[index]],
            "bottle_line_yaw_deg": float(yaws[index]),
        }
        for index in range(candidate_count)
    ]


def _record_ee(record: Mapping[str, Any]) -> np.ndarray:
    return _finite_vector(
        record["initial_ee_position_world_m"],
        size=3,
        name="initial_ee_position_world_m",
    )


def _record_yaw(record: Mapping[str, Any]) -> float:
    value = float(record["bottle_line_yaw_deg"])
    if not math.isfinite(value):
        raise ValueError("bottle_line_yaw_deg must be finite")
    return value


def pairwise_ee_distances_m(records: Sequence[Mapping[str, Any]]) -> list[float]:
    """Return all pairwise initial-EE Euclidean distances."""

    return [
        float(np.linalg.norm(_record_ee(first) - _record_ee(second)))
        for index, first in enumerate(records)
        for second in records[index + 1 :]
    ]


def minimum_pairwise_ee_distance_m(
    records: Sequence[Mapping[str, Any]],
) -> float:
    distances = pairwise_ee_distances_m(records)
    return min(distances, default=math.inf)


def minimum_pairwise_line_yaw_separation_deg(
    records: Sequence[Mapping[str, Any]],
) -> float:
    distances = [
        line_yaw_distance_deg(_record_yaw(first), _record_yaw(second))
        for index, first in enumerate(records)
        for second in records[index + 1 :]
    ]
    return min(distances, default=math.inf)


def select_diverse_records(
    *,
    records: Sequence[Mapping[str, Any]],
    count: int,
    minimum_line_yaw_separation_deg: float,
    minimum_ee_separation_m: float,
) -> list[dict[str, Any]]:
    """Select preflight passes in order without consulting runtime outcomes."""

    required = int(count)
    yaw_gate = float(minimum_line_yaw_separation_deg)
    ee_gate = float(minimum_ee_separation_m)
    if required < 1:
        raise ValueError("count must be positive")
    if not math.isfinite(yaw_gate) or yaw_gate < 0.0:
        raise ValueError("minimum line-yaw separation must be non-negative")
    if not math.isfinite(ee_gate) or ee_gate < 0.0:
        raise ValueError("minimum EE separation must be non-negative")

    selected: list[dict[str, Any]] = []
    for source in records:
        if source.get("preflight_status") != "PASS":
            continue
        yaw = _record_yaw(source)
        ee = _record_ee(source)
        yaw_margin = min(
            (
                line_yaw_distance_deg(yaw, _record_yaw(previous))
                for previous in selected
            ),
            default=math.inf,
        )
        ee_margin = min(
            (
                float(np.linalg.norm(ee - _record_ee(previous)))
                for previous in selected
            ),
            default=math.inf,
        )
        if yaw_margin + 1.0e-12 < yaw_gate:
            continue
        if ee_margin + 1.0e-12 < ee_gate:
            continue
        accepted = copy.deepcopy(dict(source))
        accepted["minimum_prior_yaw_separation_deg"] = yaw_margin
        accepted["minimum_prior_ee_distance_m"] = ee_margin
        selected.append(accepted)
        if len(selected) == required:
            break
    return selected


def canonical_five_pose_signature(
    records: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the frozen numerical sample identity, independent of JSON spacing."""

    canonical = [
        {
            "sample_id": record["sample_id"],
            "candidate_index": int(record["candidate_index"]),
            "bottle_geometric_center_world_m": record[
                "bottle_geometric_center_world_m"
            ],
            "bottle_line_yaw_deg": float(record["bottle_line_yaw_deg"]),
            "world_from_object": record["world_from_object"],
            "initial_arm_q_rad": record["initial_arm_q_rad"],
            "initial_ee_position_world_m": record[
                "initial_ee_position_world_m"
            ],
        }
        for record in records
    ]
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def apply_frozen_bottle_transform(
    profile: Mapping[str, Any],
    *,
    world_from_object: Sequence[Sequence[float]] | np.ndarray,
) -> dict[str, Any]:
    """Apply a full frozen object transform while preserving validated T_O_G."""

    transformed = copy.deepcopy(dict(profile))
    placement = transformed["kinematics"]["placement"]
    source_world_object = require_rigid_transform(
        placement["placement_matrix"],
        name="source placement matrix",
    )
    frozen_world_object = require_rigid_transform(
        world_from_object,
        name="world_from_object",
    )
    object_world_source = np.linalg.inv(source_world_object)
    axis = placement["bottle_axis"]

    def local_from_source_world(key: str) -> np.ndarray:
        source_world = _finite_vector(axis[key], size=3, name=key)
        return _transform_point(object_world_source, source_world)

    a_local = local_from_source_world("a_world_m")
    b_local = local_from_source_world("b_world_m")
    grasp_local = local_from_source_world("grasp_point_world_m")
    a_world = _transform_point(frozen_world_object, a_local)
    b_world = _transform_point(frozen_world_object, b_local)
    grasp_world = _transform_point(frozen_world_object, grasp_local)
    axis_vector = b_world - a_world
    axis_unit = _unit(axis_vector, name="bottle axis")

    targets = placement["target_poses"]
    object_gripper = require_rigid_transform(
        targets["object_from_gripper"],
        name="object_from_gripper",
    )
    source_grasp = _finite_vector(
        targets["grasp_ee_position_world_m"],
        size=3,
        name="source grasp target",
    )
    pregrasp_offset = (
        _finite_vector(
            targets["pregrasp_ee_position_world_m"],
            size=3,
            name="source pregrasp target",
        )
        - source_grasp
    )
    lift_offset = (
        _finite_vector(
            targets["lift_ee_position_world_m"],
            size=3,
            name="source lift target",
        )
        - source_grasp
    )
    world_gripper = require_rigid_transform(
        frozen_world_object @ object_gripper,
        name="world_from_gripper",
    )
    new_grasp = world_gripper[:3, 3]
    quaternion_wxyz = Rotation.from_matrix(
        world_gripper[:3, :3]
    ).as_quat(canonical=True, scalar_first=True)

    placement["placement_matrix"] = frozen_world_object.tolist()
    axis["a_world_m"] = a_world.tolist()
    axis["b_world_m"] = b_world.tolist()
    axis["grasp_point_world_m"] = grasp_world.tolist()
    if "unit_world" in axis:
        axis["unit_world"] = axis_unit.tolist()
    if "length_m" in axis:
        axis["length_m"] = float(np.linalg.norm(axis_vector))
    targets["grasp_ee_position_world_m"] = new_grasp.tolist()
    targets["pregrasp_ee_position_world_m"] = (
        new_grasp + pregrasp_offset
    ).tolist()
    targets["lift_ee_position_world_m"] = (
        new_grasp + lift_offset
    ).tolist()
    targets["orientation_world_wxyz"] = quaternion_wxyz.tolist()
    placement["sampled_bottle_pose_semantics"] = (
        "FROZEN_CAD_GEOMETRIC_CENTER_AND_WORLD_Z_YAW;_"
        "VALIDATED_T_O_G_PRESERVED"
    )
    placement["sampled_bottle_line_yaw_deg"] = line_yaw_deg(axis_unit)
    return transformed
