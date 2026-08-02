"""Isaac Sim 5.1 bindings for the ALOHA Bottle500 20 cm diagnostic."""

# Isaac Sim 5.1 native APIs use positional boolean arguments.
# ruff: noqa: FBT003

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image

from tools.aloha1_mapping.convex_geometry_audit import convex_pair_relation
from tools.aloha1_mapping.grasp_20cm_controller import Phase
from tools.aloha1_mapping.grasp_20cm_controller import RunObservation
from tools.aloha1_mapping.grasp_20cm_controller import canonical_run_signature
from tools.aloha1_mapping.grasp_20cm_runtime import EXPECTED_DOF_ORDER
from tools.aloha1_mapping.grasp_20cm_runtime import sha256_file
from tools.aloha1_mapping.grasp_initialization_contract import canonical_initialization_signature
from tools.aloha1_mapping.grasp_initialization_contract import evaluate_finger_initialization
from tools.aloha1_mapping.grasp_initialization_contract import evaluate_finger_runtime_frame

PHASE_TIMEOUT_FRAMES = {
    Phase.VALIDATE: 60,
    Phase.SETUP_KINEMATIC: 60,
    Phase.RELEASE_DYNAMIC: 60,
    Phase.SETTLE: 600,
    Phase.OPEN_PREGRASP: 900,
    Phase.VERTICAL_DESCENT: 900,
    Phase.BILATERAL_CONTACT: 300,
    Phase.CLOSE_PRELOAD: 300,
    Phase.VERTICAL_LIFT: 1800,
    Phase.HEIGHT_REACHED: 60,
    Phase.HOLD: 240,
}
COLLIDER_OVERLAY_RENDER_FLUSH_UPDATES = 20
BOTTLE_TENSOR_LIFECYCLE_MODES = {
    "BASELINE",
    "INITIALIZE_KINEMATIC_BODIES",
    "RECREATE_AFTER_DYNAMIC",
    "RECREATE_AFTER_DYNAMIC_STEP",
}


def single_body_tensor_indices(*, count: int) -> np.ndarray:
    """Return the explicit index required by local PhysX tensor setters."""

    if int(count) != 1:
        raise ValueError("diagnostic rigid-body view must contain exactly one")
    return np.asarray([0], dtype=np.int32)


def reset_body_transition_plan(
    *,
    initially_kinematic: bool,
) -> tuple[str, ...]:
    """Describe the local PhysX-safe Reset ordering."""

    prefix = ("set_dynamic",) if initially_kinematic else ()
    return (
        *prefix,
        "set_transform",
        "set_velocity",
        "set_kinematic",
    )


def bottle_tensor_lifecycle_plan(mode: str) -> tuple[str, ...]:
    """Return the one-variable local PhysX tensor lifecycle experiment."""

    normalized = str(mode)
    if normalized not in BOTTLE_TENSOR_LIFECYCLE_MODES:
        raise ValueError(
            f"unsupported bottle tensor lifecycle: {normalized}"
        )
    prefix = (
        ("initialize_kinematic_bodies",)
        if normalized == "INITIALIZE_KINEMATIC_BODIES"
        else ()
    )
    if normalized == "RECREATE_AFTER_DYNAMIC":
        suffix = ("recreate_after_dynamic",)
    elif normalized == "RECREATE_AFTER_DYNAMIC_STEP":
        suffix = (
            "wait_one_dynamic_physics_step",
            "recreate_after_dynamic_step",
        )
    else:
        suffix = ()
    return (*prefix, "create_initial_view", *suffix)


def delayed_tensor_recreation_due(
    *,
    mode: str,
    pending: bool,
    current_frame: int,
    transition_frame: int | None,
) -> bool:
    """Return whether the post-dynamic-step tensor view must be rebuilt."""

    return bool(
        mode == "RECREATE_AFTER_DYNAMIC_STEP"
        and pending
        and transition_frame is not None
        and int(current_frame) > int(transition_frame)
    )


def tensor_view_identity_record(
    view: Any,
    *,
    expected_prim_path: str,
) -> dict[str, Any]:
    """Prove that a single-body tensor view binds the expected rigid prim."""

    count = int(view.count)
    prim_paths = [str(path) for path in view.prim_paths]
    exact = count == 1 and prim_paths == [str(expected_prim_path)]
    if not exact:
        raise ValueError(
            "PhysX tensor view does not bind exact bottle path: "
            f"count={count}, prim_paths={prim_paths}"
        )
    return {
        "count": count,
        "prim_paths": prim_paths,
        "expected_prim_path": str(expected_prim_path),
        "exact_path_match": True,
    }


def normalize_direct_physx_transform(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize the local PhysX direct-transform diagnostic readback."""

    if not bool(value.get("ret_val")):
        raise ValueError("direct PhysX rigid-body transform is unavailable")
    position = np.asarray(list(value["position"]), dtype=np.float64)
    rotation = np.asarray(list(value["rotation"]), dtype=np.float64)
    if (
        position.shape != (3,)
        or rotation.shape != (4,)
        or not np.isfinite(position).all()
        or not np.isfinite(rotation).all()
    ):
        raise ValueError("direct PhysX rigid-body transform is invalid")
    return {
        "available": True,
        "position_world_m": position.tolist(),
        "orientation_xyzw": rotation.tolist(),
    }


def normalize_usd_velocity_readback(
    *,
    linear_velocity: Sequence[float],
    angular_velocity_deg_s: Sequence[float],
) -> dict[str, Any]:
    """Normalize USD Physics velocity attrs to the tensor report units."""

    linear = np.asarray(linear_velocity, dtype=np.float64)
    angular_deg = np.asarray(angular_velocity_deg_s, dtype=np.float64)
    if (
        linear.shape != (3,)
        or angular_deg.shape != (3,)
        or not np.isfinite(linear).all()
        or not np.isfinite(angular_deg).all()
    ):
        raise ValueError("USD velocity readback must contain finite 3-vectors")
    return {
        "linear_velocity_world_m_s": linear.tolist(),
        "angular_velocity_world_deg_s": angular_deg.tolist(),
        "angular_velocity_world_rad_s": np.deg2rad(angular_deg).tolist(),
        "angular_source_units": "degrees_per_second",
        "update_velocities_to_usd": True,
        "output_velocities_local_space": False,
    }


def derive_gripper_closeup_camera_geometry(
    *,
    grasp_point_world_m: Sequence[float],
    bottle_axis_world: Sequence[float],
    nominal_lift_m: float,
    axial_distance_m: float = 1.25,
    elevation_m: float = 0.75,
    axial_side: int = 1,
) -> dict[str, list[float] | float | int | str]:
    """Derive an evidence view along AB, centered on the lift interval."""

    grasp = np.asarray(grasp_point_world_m, dtype=np.float64)
    axis = np.asarray(bottle_axis_world, dtype=np.float64)
    if grasp.shape != (3,) or axis.shape != (3,):
        raise ValueError("grasp point and bottle axis must be 3-vectors")
    if not np.isfinite(grasp).all() or not np.isfinite(axis).all():
        raise ValueError("camera geometry inputs must be finite")
    if abs(float(axis[2])) > 1e-6:
        raise ValueError("Bottle500 AB must be horizontal")
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise ValueError("bottle axis must be nonzero")
    axis /= norm
    if (
        nominal_lift_m <= 0.0
        or axial_distance_m <= 0.0
        or elevation_m <= 0.0
    ):
        raise ValueError("camera distances must be positive")
    if axial_side not in {-1, 1}:
        raise ValueError("axial_side must be either -1 or 1")
    target = grasp + np.asarray(
        [0.0, 0.0, nominal_lift_m / 2.0],
        dtype=np.float64,
    )
    position = (
        target
        + axis * float(axial_distance_m) * int(axial_side)
        + np.asarray([0.0, 0.0, elevation_m])
    )
    forward = target - position
    forward /= np.linalg.norm(forward)
    return {
        "position_world_m": position.tolist(),
        "target_world_m": target.tolist(),
        "camera_forward_world": forward.tolist(),
        "bottle_axis_world": axis.tolist(),
        "nominal_lift_m": float(nominal_lift_m),
        "axial_distance_m": float(axial_distance_m),
        "axial_side": int(axial_side),
        "elevation_m": float(elevation_m),
        "derivation": (
            "LOOK_ALONG_BOTTLE_AB_AND_CENTER_NOMINAL_VERTICAL_LIFT"
        ),
    }


def derive_subject_bounding_closeup_camera_geometry(
    *,
    subject_points_world_m: Sequence[Sequence[float]],
    bottle_axis_world: Sequence[float],
    horizontal_fov_rad: float,
    vertical_fov_rad: float,
    near_clipping_m: float,
    frame_margin_fraction: float = 0.15,
    axial_side: int = 1,
) -> dict[str, Any]:
    """Frame current Bottle500 A/B and both finger origins without guessing.

    A sphere around the supplied world points is fitted inside the smaller
    usable camera half-FOV.  The near-plane constraint is evaluated against
    the same sphere.  This is an evidence-camera operation only; it does not
    modify physics, controls, or the simulated state.
    """

    points = np.asarray(subject_points_world_m, dtype=np.float64)
    axis = np.asarray(bottle_axis_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("subject points must contain at least two 3-vectors")
    if axis.shape != (3,):
        raise ValueError("bottle axis must be a 3-vector")
    if not np.isfinite(points).all() or not np.isfinite(axis).all():
        raise ValueError("closeup framing inputs must be finite")
    if abs(float(axis[2])) > 1e-6:
        raise ValueError("Bottle500 axis must remain horizontal")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0:
        raise ValueError("bottle axis must be nonzero")
    axis /= axis_norm
    if axial_side not in {-1, 1}:
        raise ValueError("axial_side must be either -1 or 1")
    if (
        not math.isfinite(horizontal_fov_rad)
        or not math.isfinite(vertical_fov_rad)
        or min(horizontal_fov_rad, vertical_fov_rad) <= 0.0
        or max(horizontal_fov_rad, vertical_fov_rad) >= math.pi
    ):
        raise ValueError("camera FOV values must be finite angles in (0, pi)")
    if not math.isfinite(near_clipping_m) or near_clipping_m < 0.0:
        raise ValueError("near clipping distance must be finite and non-negative")
    if not 0.0 <= frame_margin_fraction < 1.0:
        raise ValueError("frame margin fraction must be in [0, 1)")

    target = (points.min(axis=0) + points.max(axis=0)) / 2.0
    radius = max(float(np.linalg.norm(point - target)) for point in points)
    if radius <= 0.0:
        raise ValueError("subject bounding sphere must have positive radius")
    usable_half_fov = (
        0.5
        * min(float(horizontal_fov_rad), float(vertical_fov_rad))
        * (1.0 - float(frame_margin_fraction))
    )
    fov_distance = radius / math.tan(usable_half_fov)
    near_plane_distance = float(near_clipping_m) + radius
    camera_distance = max(fov_distance, near_plane_distance)

    baseline_offset = (
        axis * float(axial_side) * 1.25
        + np.asarray([0.0, 0.0, 0.75], dtype=np.float64)
    )
    offset_direction = baseline_offset / np.linalg.norm(baseline_offset)
    position = target + offset_direction * camera_distance
    forward = target - position
    forward /= np.linalg.norm(forward)
    return {
        "position_world_m": position.tolist(),
        "target_world_m": target.tolist(),
        "camera_forward_world": forward.tolist(),
        "bottle_axis_world": axis.tolist(),
        "subject_point_count": len(points),
        "subject_bounds_world_m": {
            "minimum": points.min(axis=0).tolist(),
            "maximum": points.max(axis=0).tolist(),
        },
        "bounding_sphere_radius_m": radius,
        "camera_distance_m": camera_distance,
        "fov_distance_m": fov_distance,
        "near_plane_distance_m": near_plane_distance,
        "horizontal_fov_rad": float(horizontal_fov_rad),
        "vertical_fov_rad": float(vertical_fov_rad),
        "usable_half_fov_rad": usable_half_fov,
        "near_clipping_m": float(near_clipping_m),
        "frame_margin_fraction": float(frame_margin_fraction),
        "axial_side": int(axial_side),
        "derivation": (
            "CURRENT_FRAME_BOTTLE_AB_AND_BILATERAL_FINGER_BOUNDING_SPHERE"
        ),
        "scope": "DIAGNOSTIC_EVIDENCE_CAMERA_ONLY_NO_PHYSICS_CHANGE",
    }


def derive_overview_camera_geometry(
    *,
    base_position_world_m: Sequence[float],
    initial_ee_position_world_m: Sequence[float],
    grasp_position_world_m: Sequence[float],
    lift_position_world_m: Sequence[float],
    minimum_distance_m: float = 3.6,
) -> dict[str, Any]:
    """Frame the base, random initial EE, grasp, and lift anchors."""

    anchors = np.asarray(
        [
            base_position_world_m,
            initial_ee_position_world_m,
            grasp_position_world_m,
            lift_position_world_m,
        ],
        dtype=np.float64,
    )
    if anchors.shape != (4, 3) or not np.isfinite(anchors).all():
        raise ValueError("overview anchors must be four finite 3-vectors")
    distance_floor = float(minimum_distance_m)
    if not math.isfinite(distance_floor) or distance_floor <= 0.0:
        raise ValueError("overview minimum distance must be positive")
    anchor_min = anchors.min(axis=0)
    anchor_max = anchors.max(axis=0)
    target = (anchor_min + anchor_max) / 2.0
    target[2] = max(
        float(target[2]),
        float(anchors[0, 2]) + 0.30,
    )
    radius = float(
        np.max(np.linalg.norm(anchors - target, axis=1))
    )
    distance = max(distance_floor, 4.0 * radius)
    viewing_ray = np.asarray([1.85, -1.75, 1.45], dtype=np.float64)
    viewing_ray /= np.linalg.norm(viewing_ray)
    position = target + viewing_ray * distance
    return {
        "position_world_m": position.tolist(),
        "target_world_m": target.tolist(),
        "distance_m": distance,
        "anchor_min_world_m": anchor_min.tolist(),
        "anchor_max_world_m": anchor_max.tolist(),
        "anchors_world_m": anchors.tolist(),
        "derivation": (
            "RUNTIME_BASE_INITIAL_EE_GRASP_LIFT_ANCHOR_BOUNDS"
        ),
        "framing_status": (
            "NUMERIC_FRUSTUM_GATE_STILL_REQUIRED_PER_CAPTURED_FRAME"
        ),
    }


def solver_active_contacts(
    contacts: Sequence[Mapping[str, Any]],
    *,
    tokens: Sequence[str],
) -> list[Mapping[str, Any]]:
    """Return reported pairs carrying a finite nonzero solver impulse.

    Positive separation is retained here because PhysX can generate and
    solve contacts inside the contact-offset envelope.  Callers must keep
    this distinct from geometric contact (`separation <= 0`).
    """

    records: list[Mapping[str, Any]] = []
    for contact in contacts:
        pair_text = "\n".join(
            str(contact.get(key, ""))
            for key in (
                "actor0_path",
                "actor1_path",
                "collider0_path",
                "collider1_path",
            )
        )
        if not all(token in pair_text for token in tokens):
            continue
        try:
            separation_m = float(contact["separation_m"])
            impulse_ns = float(contact["impulse_ns"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            math.isfinite(separation_m)
            and math.isfinite(impulse_ns)
            and impulse_ns > 0.0
        ):
            records.append(contact)
    return records


def bilateral_observation_contact(
    *,
    bilateral_geometric: bool,
    bilateral_solver_active: bool,
) -> bool:
    """Gate motion on bilateral force-carrying contact-report pairs.

    ``bilateral_geometric`` remains an independent diagnostic.  PhysX may
    solve a reported contact while its separation is slightly positive
    inside the contact-offset envelope, so its sign must not suppress a
    finite positive solver impulse.
    """

    del bilateral_geometric
    return bool(bilateral_solver_active)


def open_pregrasp_evidence_ready(
    *,
    open_target_reached: bool,
    already_captured: bool,
) -> bool:
    """Capture the open evidence after motion reaches its commanded target."""

    return bool(open_target_reached and not already_captured)


def required_collider_phase_labels(
    *,
    phase: str,
    terminal: bool,
    observation: Mapping[str, Any],
    contact: Mapping[str, Any],
    captured: set[str],
) -> list[str]:
    """Return the exact sparse collision-evidence milestones still needed."""

    labels: list[str] = []
    if phase == Phase.RELEASE_DYNAMIC.value and "RELEASE_DYNAMIC" not in captured:
        labels.append("RELEASE_DYNAMIC")
    if (
        phase == Phase.OPEN_PREGRASP.value
        and open_pregrasp_evidence_ready(
            open_target_reached=bool(observation.get("open_target_reached")),
            already_captured="OPEN_PREGRASP" in captured,
        )
    ):
        labels.append("OPEN_PREGRASP")
    if (
        phase == Phase.BILATERAL_CONTACT.value
        and bool(contact.get("bilateral_solver_active_contact"))
        and "BILATERAL_CONTACT" not in captured
    ):
        labels.append("BILATERAL_CONTACT")
    if (
        phase == Phase.VERTICAL_LIFT.value
        and float(observation.get("clearance_m", 0.0)) > 0.001
        and "FIRST_SUPPORT_CLEARANCE" not in captured
    ):
        labels.append("FIRST_SUPPORT_CLEARANCE")
    if phase == Phase.HEIGHT_REACHED.value and "HEIGHT_REACHED" not in captured:
        labels.append("HEIGHT_REACHED")
    if terminal and phase == Phase.HOLD.value and "HOLD_END" not in captured:
        labels.append("HOLD_END")
    return labels


def physics_sample_duration_s(
    *,
    sample_count: int,
    physics_dt_s: float,
) -> float:
    """Measure evidence duration represented by fixed-rate physics samples."""

    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if not math.isfinite(physics_dt_s) or physics_dt_s <= 0.0:
        raise ValueError("physics_dt_s must be finite and positive")
    return float(sample_count) * float(physics_dt_s)


def initial_pose_hold_complete(
    *,
    observed_frame_count: int,
    required_frame_count: int,
) -> bool:
    """Return whether setup has observed the exact minimum hold duration."""

    observed = int(observed_frame_count)
    required = int(required_frame_count)
    if observed < 0:
        raise ValueError("observed_frame_count must be non-negative")
    if required < 1:
        raise ValueError("required_frame_count must be positive")
    return observed >= required


def formal_phase_bottle_dynamic(
    observations: Sequence[RunObservation],
    telemetry: Sequence[Mapping[str, Any]],
) -> bool:
    """Exclude setup-only kinematics from the formal dynamic-bottle gate."""

    if len(observations) != len(telemetry):
        raise ValueError("observation and telemetry counts must match")
    formal_phases = {
        Phase.RELEASE_DYNAMIC.value,
        Phase.SETTLE.value,
        Phase.OPEN_PREGRASP.value,
        Phase.VERTICAL_DESCENT.value,
        Phase.BILATERAL_CONTACT.value,
        Phase.CLOSE_PRELOAD.value,
        Phase.VERTICAL_LIFT.value,
        Phase.HEIGHT_REACHED.value,
        Phase.HOLD.value,
    }
    formal_observations = [
        observation
        for observation, record in zip(
            observations,
            telemetry,
            strict=True,
        )
        if record.get("phase") in formal_phases
    ]
    return bool(formal_observations) and all(
        observation.bottle_dynamic
        for observation in formal_observations
    )


def build_lula_cspace_phase_targets(
    *,
    generator_factory: Any,
    robot_description_path: str,
    urdf_path: str,
    waypoint_positions: Sequence[Sequence[float]],
    physics_dt_s: float,
    velocity_limits_rad_s: Sequence[float],
    acceleration_limits_rad_s2: Sequence[float],
) -> dict[str, Any]:
    """Time-parameterize one six-DOF ALOHA arm phase with local Lula."""

    waypoints = np.asarray(waypoint_positions, dtype=np.float64)
    velocity_limits = np.asarray(
        velocity_limits_rad_s,
        dtype=np.float64,
    )
    acceleration_limits = np.asarray(
        acceleration_limits_rad_s2,
        dtype=np.float64,
    )
    dt = float(physics_dt_s)
    if (
        waypoints.ndim != 2
        or waypoints.shape[0] < 2
        or waypoints.shape[1] != 6
        or velocity_limits.shape != (6,)
        or acceleration_limits.shape != (6,)
        or not np.isfinite(waypoints).all()
        or not np.isfinite(velocity_limits).all()
        or not np.isfinite(acceleration_limits).all()
        or np.any(velocity_limits <= 0.0)
        or np.any(acceleration_limits <= 0.0)
        or not math.isfinite(dt)
        or dt <= 0.0
    ):
        raise ValueError("invalid six-DOF Lula trajectory inputs")
    generator = generator_factory(
        robot_description_path=robot_description_path,
        urdf_path=urdf_path,
    )
    active_joints = list(generator.get_active_joints())
    expected_joints = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
    ]
    if active_joints != expected_joints:
        raise RuntimeError(
            f"unexpected Lula active-joint order: {active_joints}"
        )
    generator.set_c_space_velocity_limits(velocity_limits)
    generator.set_c_space_acceleration_limits(acceleration_limits)
    trajectory = generator.compute_c_space_trajectory(waypoints)
    if trajectory is None:
        raise RuntimeError("Lula could not time-parameterize arm phase")
    start_time = float(trajectory.start_time)
    end_time = float(trajectory.end_time)
    if (
        not math.isfinite(start_time)
        or not math.isfinite(end_time)
        or end_time <= start_time
    ):
        raise RuntimeError("Lula returned an invalid trajectory domain")
    sample_times = list(np.arange(start_time, end_time, dt))
    if not sample_times or not math.isclose(
        sample_times[-1],
        end_time,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        sample_times.append(end_time)
    targets = []
    positions = []
    velocities = []
    for time_s in sample_times:
        position, velocity = trajectory.get_joint_targets(float(time_s))
        position_array = np.asarray(position, dtype=np.float64)
        velocity_array = np.asarray(velocity, dtype=np.float64)
        if (
            position_array.shape != (6,)
            or velocity_array.shape != (6,)
            or not np.isfinite(position_array).all()
            or not np.isfinite(velocity_array).all()
        ):
            raise RuntimeError("Lula returned non-finite phase target")
        positions.append(position_array)
        velocities.append(velocity_array)
        targets.append(
            {
                "time_s": float(time_s),
                "position_rad": position_array.tolist(),
                "velocity_rad_s": velocity_array.tolist(),
            }
        )
    position_array = np.asarray(positions, dtype=np.float64)
    velocity_array = np.asarray(velocities, dtype=np.float64)
    time_array = np.asarray(sample_times, dtype=np.float64)
    sample_intervals = np.diff(time_array)
    sampled_acceleration = np.diff(velocity_array, axis=0) / (
        sample_intervals[:, None]
    )
    maximum_velocity = np.max(np.abs(velocity_array), axis=0)
    maximum_acceleration = np.max(
        np.abs(sampled_acceleration),
        axis=0,
    )
    endpoint_velocity_zero = bool(
        np.max(np.abs(velocity_array[[0, -1]])) <= 1.0e-8
    )
    return {
        "targets": targets,
        "audit": {
            "active_joints": active_joints,
            "waypoint_count": int(waypoints.shape[0]),
            "sample_count": len(targets),
            "physics_dt_s": dt,
            "start_time_s": start_time,
            "end_time_s": end_time,
            "duration_s": end_time - start_time,
            "finite": bool(
                np.isfinite(position_array).all()
                and np.isfinite(velocity_array).all()
                and np.isfinite(sampled_acceleration).all()
            ),
            "endpoint_velocity_zero": endpoint_velocity_zero,
            "maximum_abs_velocity_rad_s": maximum_velocity.tolist(),
            "maximum_abs_sampled_acceleration_rad_s2": (
                maximum_acceleration.tolist()
            ),
            "velocity_within_limits": bool(
                np.all(maximum_velocity <= velocity_limits + 1.0e-8)
            ),
            "sampled_acceleration_within_limits": bool(
                np.all(
                    maximum_acceleration
                    <= acceleration_limits + 1.0e-6
                )
            ),
            "velocity_limits_rad_s": velocity_limits.tolist(),
            "acceleration_limits_rad_s2": (
                acceleration_limits.tolist()
            ),
            "jerk_limit_status": (
                "NOT_SET_NO_EXACT_MODEL_OFFICIAL_VALUE"
            ),
        },
    }


def arm_phase_target_reached(
    *,
    trajectory_exhausted: bool,
    joint_readback: Sequence[float],
    joint_target: Sequence[float],
    arm_dof_indices: Sequence[int],
    tolerance_rad: float,
) -> bool:
    """Require physical arm readback, not only command-sequence exhaustion."""

    readback = np.asarray(joint_readback, dtype=np.float64)
    target = np.asarray(joint_target, dtype=np.float64)
    indices = np.asarray(arm_dof_indices, dtype=np.int64)
    tolerance = float(tolerance_rad)
    if (
        readback.ndim != 1
        or target.shape != readback.shape
        or indices.shape != (6,)
        or len(set(indices.tolist())) != 6
        or np.any(indices < 0)
        or np.any(indices >= readback.size)
        or not np.isfinite(readback).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("arm readback gate inputs are invalid")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("arm readback tolerance must be finite/non-negative")
    return bool(
        trajectory_exhausted
        and np.max(np.abs(readback[indices] - target[indices]))
        <= tolerance
    )


def arm_phase_timeout_reached(
    *,
    phase_frame_count: int,
    trajectory_sample_count: int,
    readback_settle_timeout_frames: int,
    trajectory_exhausted: bool,
) -> bool:
    """Start the existing readback timeout only after trajectory playback."""

    phase_frames = int(phase_frame_count)
    trajectory_samples = int(trajectory_sample_count)
    settle_frames = int(readback_settle_timeout_frames)
    if (
        phase_frames < 0
        or trajectory_samples <= 0
        or settle_frames <= 0
    ):
        raise ValueError("arm phase timeout inputs must be positive")
    return bool(
        trajectory_exhausted
        and phase_frames > trajectory_samples + settle_frames
    )


def preload_solver_contact_ready(
    *,
    close_exhausted: bool,
    left_solver_active: bool,
    right_solver_active: bool,
    coupling_residual_m: float,
    coupling_gate_m: float,
) -> bool:
    """Require bilateral force-carrying PhysX contacts for preload."""

    residual = float(coupling_residual_m)
    gate = float(coupling_gate_m)
    return bool(
        close_exhausted
        and left_solver_active
        and right_solver_active
        and math.isfinite(residual)
        and math.isfinite(gate)
        and gate >= 0.0
        and residual <= gate
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class IsaacGrasp20cmBindings:
    """Own session-only USD, PhysX, trajectory, and machine evidence state."""

    def __init__(
        self,
        *,
        app: Any,
        profile: Mapping[str, Any],
        artifact_root: Path,
        delegate_readback: Mapping[str, Any],
        bottle_xy_offset_m: Sequence[float] = (0.0, 0.0),
        bottle_world_from_object: Sequence[Sequence[float]] | None = None,
        initial_arm_q_rad: Sequence[float] | None = None,
        initial_pose_hold_frames: int = 60,
        arm_phase_readback_tolerance_rad: float | None = None,
        arm_trajectory_mode: str = "LEGACY_VELOCITY_STEP",
        arm_acceleration_limits_rad_s2: Sequence[float] | None = None,
        additional_lift_margin_m: float = 0.0,
        capture_collider_evidence: bool = True,
        closeup_axial_side: int = 1,
        bottle_tensor_lifecycle: str = "BASELINE",
        bottle_usd_velocity_readback: bool = False,
    ) -> None:
        import carb.settings
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.xforms import get_world_pose
        from omni.physx import get_physx_interface
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysxSchema
        from pxr import Usd
        from pxr import UsdGeom
        from pxr import UsdPhysics

        from tools.aloha1_mapping.bottle_com_velocity import build_sample
        from tools.aloha1_mapping.grasp_20cm_five_pose_ik import apply_frozen_bottle_transform
        from tools.aloha1_mapping.grasp_20cm_five_pose_ik import compose_initial_command
        from tools.aloha1_mapping.grasp_20cm_sampling import translate_horizontal_bottle_profile
        from tools.audit_aloha1_bottle_collision_runtime import _create_bottle_render_evidence
        from tools.audit_aloha1_bottle_collision_runtime import _quaternion_matrix_wxyz
        from tools.audit_aloha1_bottle_collision_runtime import _update_bottle_render_evidence
        from tools.run_aloha1_grasp_editor_variant_b_gui import build_external_close_targets
        from tools.validate_aloha1_follower_finger_collision_runtime import _create_finger_render_evidence
        from tools.validate_aloha1_follower_finger_collision_runtime import _update_finger_render_evidence
        from tools.validate_aloha1_gripper_coupling_ab import author_coupling_variant
        from tools.validate_aloha1_task7b2_horizontal_grasp import DIAGNOSTIC_COUPLING_CLASSIFICATION
        from tools.validate_aloha1_task7b2_horizontal_grasp import _author_session_finger_drive_type
        from tools.validate_aloha1_task7b2_horizontal_grasp import _command_positions
        from tools.validate_aloha1_task7b2_horizontal_grasp import _create_session_bottle
        from tools.validate_aloha1_task7b2_horizontal_grasp import _load_profile
        from tools.validate_aloha1_task7b2_horizontal_grasp import _look_at_quaternion
        from tools.validate_aloha1_task7b2_horizontal_grasp import _physical_contacts
        from tools.validate_aloha1_task7b2_horizontal_grasp import _serialize_contacts
        from tools.validate_aloha1_task7b2_horizontal_grasp import _solve_settled_bottle_runtime_ik
        from tools.validate_aloha1_task7b2_horizontal_grasp import _world_bounds
        from tools.validate_aloha1_task7b2_horizontal_grasp import derive_pose_finite_difference_velocity
        from tools.validate_aloha1_task7b2_horizontal_grasp import read_physx_bottle_state
        from tools.validate_aloha1_task7b2_horizontal_grasp import transform_local_points_to_world_bounds

        self.app = app
        self.profile = dict(profile)
        self.config = self.profile["config"]
        self.artifact_root = artifact_root.resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.report_path = (
            self.artifact_root / "aloha1_grasp_20cm_runtime.json"
        )
        self.telemetry_path = (
            self.artifact_root / "aloha1_grasp_20cm_telemetry.jsonl"
        )
        self.delegate_readback = dict(delegate_readback)
        self.stage_path = Path(
            self.profile["frozen_inputs"]["stage"]["absolute_path"]
        )
        self.stage_hash_before = sha256_file(self.stage_path)
        self.dt = 1.0 / float(self.config["physics"]["frequency_hz"])
        self._get_world_pose = get_world_pose
        self._command_positions = _command_positions
        self._physical_contacts = _physical_contacts
        self._serialize_contacts = _serialize_contacts
        self._solve_settled_ik = _solve_settled_bottle_runtime_ik
        self._read_bottle_state = read_physx_bottle_state
        self._derive_pose_velocity = derive_pose_finite_difference_velocity
        self._build_com_velocity_sample = build_sample
        self._transform_collision_bounds = (
            transform_local_points_to_world_bounds
        )
        self._quaternion_matrix_wxyz = _quaternion_matrix_wxyz
        self._physx = get_physx_interface()
        self._physx_sim = get_physx_simulation_interface()
        self._settings = carb.settings.get_settings()
        self._collider_display_setting = str(
            self.config["evidence"]["collider_overlay"][
                "display_setting"
            ]
        )
        self._collider_display_value = int(
            self.config["evidence"]["collider_overlay"][
                "display_value"
            ]
        )
        self._collider_display_before = int(
            self._settings.get(self._collider_display_setting) or 0
        )
        self.finger_safety_config = dict(self.config["finger_safety"])
        self.finger_dof_names = [
            str(name) for name in self.finger_safety_config["dof_names"]
        ]
        self.finger_dof_indices = np.asarray(
            self.finger_safety_config["dof_indices"],
            dtype=np.int32,
        )
        self.source_finger_limits = {
            str(name): {
                "lower": float(record["lower"]),
                "upper": float(record["upper"]),
            }
            for name, record in self.finger_safety_config[
                "source_limits_m"
            ].items()
        }
        self.abort_on_first_runtime_violation = bool(
            self.finger_safety_config[
                "abort_on_first_runtime_violation"
            ]
        )

        task_profile_path = Path(
            self.profile["frozen_inputs"][
                "task7b2_runtime_profile"
            ]["absolute_path"]
        )
        self.task_profile = _load_profile(task_profile_path)
        self.bottle_xy_offset_m = [
            float(value) for value in bottle_xy_offset_m
        ]
        if (
            len(self.bottle_xy_offset_m) != 2
            or not np.isfinite(self.bottle_xy_offset_m).all()
        ):
            raise ValueError("Bottle500 XY offset must be a finite two-vector")
        if bottle_world_from_object is None:
            self.task_profile = translate_horizontal_bottle_profile(
                self.task_profile,
                offset_xy_m=self.bottle_xy_offset_m,
            )
            self.bottle_pose_mode = "LEGACY_WORLD_XY_TRANSLATION_ONLY"
        else:
            if not np.allclose(
                self.bottle_xy_offset_m,
                [0.0, 0.0],
                rtol=0.0,
                atol=0.0,
            ):
                raise ValueError(
                    "frozen bottle transform cannot be combined with "
                    "legacy XY offsets"
                )
            self.task_profile = apply_frozen_bottle_transform(
                self.task_profile,
                world_from_object=bottle_world_from_object,
            )
            self.bottle_pose_mode = "FROZEN_CENTER_AND_YAW_TRANSFORM"
        self.bottle_world_from_object = np.asarray(
            self.task_profile["kinematics"]["placement"][
                "placement_matrix"
            ],
            dtype=np.float64,
        )
        self.initial_pose_hold_frames = int(initial_pose_hold_frames)
        if self.initial_pose_hold_frames < 1:
            raise ValueError("initial_pose_hold_frames must be positive")
        self.arm_phase_readback_tolerance_rad = (
            None
            if arm_phase_readback_tolerance_rad is None
            else float(arm_phase_readback_tolerance_rad)
        )
        if (
            self.arm_phase_readback_tolerance_rad is not None
            and (
                not math.isfinite(
                    self.arm_phase_readback_tolerance_rad
                )
                or self.arm_phase_readback_tolerance_rad < 0.0
            )
        ):
            raise ValueError(
                "arm phase readback tolerance must be finite/non-negative"
            )
        self.arm_trajectory_mode = str(arm_trajectory_mode)
        allowed_trajectory_modes = {
            "LEGACY_VELOCITY_STEP",
            "LULA_CSPACE_ACCELERATION_LIMITED",
        }
        if self.arm_trajectory_mode not in allowed_trajectory_modes:
            raise ValueError(
                f"unsupported arm trajectory mode: "
                f"{self.arm_trajectory_mode}"
            )
        self.arm_acceleration_limits_rad_s2 = (
            None
            if arm_acceleration_limits_rad_s2 is None
            else np.asarray(
                arm_acceleration_limits_rad_s2,
                dtype=np.float64,
            )
        )
        if self.arm_trajectory_mode == "LULA_CSPACE_ACCELERATION_LIMITED":
            limits = self.arm_acceleration_limits_rad_s2
            if (
                limits is None
                or limits.shape != (6,)
                or not np.isfinite(limits).all()
                or np.any(limits <= 0.0)
            ):
                raise ValueError(
                    "acceleration-limited Lula mode requires six finite "
                    "positive arm acceleration limits"
                )
        self.additional_lift_margin_m = float(
            additional_lift_margin_m
        )
        if (
            not math.isfinite(self.additional_lift_margin_m)
            or self.additional_lift_margin_m < 0.0
        ):
            raise ValueError(
                "additional lift margin must be finite and non-negative"
            )
        self.capture_collider_evidence_enabled = bool(
            capture_collider_evidence
        )
        self.closeup_axial_side = int(closeup_axial_side)
        if self.closeup_axial_side not in {-1, 1}:
            raise ValueError("closeup_axial_side must be either -1 or 1")
        self.bottle_tensor_lifecycle_mode = str(
            bottle_tensor_lifecycle
        )
        self.bottle_usd_velocity_readback = bool(
            bottle_usd_velocity_readback
        )
        self.bottle_tensor_lifecycle_plan = (
            bottle_tensor_lifecycle_plan(
                self.bottle_tensor_lifecycle_mode
            )
        )
        self.bottle_tensor_lifecycle: dict[str, Any] = {
            "mode": self.bottle_tensor_lifecycle_mode,
            "operation_plan": list(self.bottle_tensor_lifecycle_plan),
            "initialize_kinematic_bodies_called": False,
            "initialize_kinematic_bodies_return": None,
            "rigid_body_view_creation_count": 0,
            "rigid_body_view_identities": [],
            "kinematic_to_dynamic_transition": None,
            "delayed_recreation_pending": False,
            "delayed_recreation_frame": None,
            "delayed_recreation_time_s": None,
            "usd_velocity_writeback_enabled": (
                self.bottle_usd_velocity_readback
            ),
            "classification": "DIAGNOSTIC_ONE_VARIABLE_LIFECYCLE_ONLY",
        }
        self.task_profile["config"]["bottle"]["session_path"] = str(
            self.config["bottle"]["session_prim"]
        )
        self.task_profile["diagnostic_preload_delta_m"] = float(
            self.config["physics"]["preload_delta_m"]
        )
        self.task_profile["diagnostic_finger_drive_type"] = str(
            self.config["physics"]["finger_drive_type"]
        )
        if (
            self.task_profile["hashes"]["task7a_stage"]
            != self.stage_hash_before
        ):
            raise RuntimeError("Task7B.2 profile does not bind approved Stage")
        verified_lula = Path(
            self.profile["frozen_inputs"]["lula_descriptor"][
                "absolute_path"
            ]
        )
        if (
            self.task_profile["inputs"]["lula_descriptor"]
            != verified_lula
            or self.task_profile["hashes"]["lula_descriptor"]
            != sha256_file(verified_lula)
        ):
            raise RuntimeError("runtime IK does not bind frozen Lula descriptor")

        self.stage = get_current_stage()
        self.stage.SetEditTarget(self.stage.GetSessionLayer())
        session_root = str(
            Path(self.config["bottle"]["session_prim"]).parent
        ).replace("\\", "/")
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            if self.stage.GetPrimAtPath(session_root).IsValid():
                self.stage.RemovePrim(session_root)
            coupling = author_coupling_variant(
                stage=self.stage,
                variant="official_symmetric_adapter",
                physx_schema=PhysxSchema,
                usd_physics=UsdPhysics,
            )
            if (
                coupling["classification"]
                != DIAGNOSTIC_COUPLING_CLASSIFICATION
            ):
                raise RuntimeError("unexpected diagnostic coupling")
            self.coupling_readback = coupling
            self.drive_readback = _author_session_finger_drive_type(
                stage=self.stage,
                usd_physics=UsdPhysics,
                requested_type=str(
                    self.config["physics"]["finger_drive_type"]
                ),
            )
            (
                self.bottle_prim,
                self.bottle_session,
                self.bottle_collision_points_local,
            ) = _create_session_bottle(self.stage, self.task_profile)
            (
                self._bottle_render_evidence,
                self._bottle_render_handles,
            ) = _create_bottle_render_evidence(
                self.stage,
                bottle_path=str(self.bottle_prim.GetPath()),
            )
            (
                self._finger_render_evidence,
                self._finger_render_handles,
            ) = _create_finger_render_evidence(
                self.stage,
                finger_paths=self.config["evidence"][
                    "collider_overlay"
                ]["finger_colliders"],
            )
            UsdGeom.Imageable(
                self.stage.GetPrimAtPath(
                    self._bottle_render_evidence["visual_root"]
                )
            ).MakeInvisible()
            UsdGeom.Imageable(
                self.stage.GetPrimAtPath(
                    self._bottle_render_evidence["collider_root"]
                )
            ).MakeInvisible()
            pusher_visual = self.stage.GetPrimAtPath(
                self._bottle_render_evidence["pusher_visual_prim"]
            )
            if pusher_visual.IsValid():
                UsdGeom.Imageable(pusher_visual).MakeInvisible()
            for side, paths in self.config["evidence"][
                "collider_overlay"
            ]["finger_colliders"].items():
                clone_base = (
                    f"{self._finger_render_evidence['root']}/{side}"
                )
                UsdGeom.Imageable(
                    self.stage.GetPrimAtPath(
                        f"{clone_base}/ExactVisualAtPhysxPose"
                    )
                ).MakeInvisible()
                UsdGeom.Imageable(
                    self.stage.GetPrimAtPath(
                        f"{clone_base}/AuthoredColliderAtPhysxPose"
                    )
                ).MakeInvisible()
                source_visual_parent = self.stage.GetPrimAtPath(
                    str(paths["visual"])
                ).GetParent()
                if source_visual_parent.IsValid():
                    UsdGeom.Imageable(
                        source_visual_parent
                    ).MakeVisible()
            self._render_evidence = {
                "authored_geometry_clone": bool(
                    self.config["evidence"]["collider_overlay"][
                        "authored_geometry_clone"
                    ]
                ),
                "semantics": str(
                    self.config["evidence"]["collider_overlay"][
                        "semantics"
                    ]
                ),
                "bottle": self._bottle_render_evidence,
                "fingers": self._finger_render_evidence,
                "physics_schemas_copied": False,
                "collision_schemas_copied": False,
            }
            self._update_bottle_render_evidence = (
                _update_bottle_render_evidence
            )
            self._update_finger_render_evidence = (
                _update_finger_render_evidence
            )

        World.clear_instance()
        self.world = World(
            stage_units_in_meters=1.0,
            backend="numpy",
            device="cpu",
            physics_dt=self.dt,
            rendering_dt=self.dt,
        )
        self.physics_context = self.world.get_physics_context()
        self.physics_context.set_solve_articulation_contact_last(True)
        self.articulation = SingleArticulation(
            prim_path=str(self.config["robot"]["articulation_prim"]),
            name="aloha1_grasp_20cm_follower_left",
            reset_xform_properties=False,
        )
        self.world.scene.add(self.articulation)
        self.world.reset()
        if list(self.articulation.dof_names) != EXPECTED_DOF_ORDER:
            raise RuntimeError(
                f"unexpected DOF order: {self.articulation.dof_names}"
            )
        if [
            list(self.articulation.dof_names)[int(index)]
            for index in self.finger_dof_indices
        ] != self.finger_dof_names:
            raise RuntimeError("finger DOF names/indices do not match runtime order")
        composed_limits = np.asarray(
            self.articulation._articulation_view.get_dof_limits(),  # noqa: SLF001
            dtype=np.float64,
        )
        if composed_limits.ndim == 3 and composed_limits.shape[0] == 1:
            composed_limits = composed_limits[0]
        if composed_limits.shape != (len(EXPECTED_DOF_ORDER), 2):
            raise RuntimeError(
                f"unexpected composed DOF limit shape: {composed_limits.shape}"
            )
        self.composed_finger_limits = {
            name: {
                "lower": float(composed_limits[int(index), 0]),
                "upper": float(composed_limits[int(index), 1]),
            }
            for name, index in zip(
                self.finger_dof_names,
                self.finger_dof_indices,
                strict=True,
            )
        }
        frozen_waypoints = self.task_profile["kinematics"]["ik"][
            "waypoints"
        ]
        pregrasp = [
            waypoint
            for waypoint in frozen_waypoints
            if waypoint["phase"] == "move_to_pregrasp"
        ]
        if not pregrasp:
            raise RuntimeError("frozen pregrasp waypoint missing")
        initial_arm = np.asarray(
            pregrasp[-1]["joint_positions_rad"],
            dtype=np.float64,
        )
        frozen_pregrasp_command = np.asarray(
            [
                *initial_arm,
                0.0,
                *self.config["robot"].get(
                    "open_targets_m",
                    self.task_profile["config"]["robot"][
                        "open_targets_m"
                    ],
                ),
            ],
            dtype=np.float64,
        )
        arm_dof_names = list(
            self.task_profile["config"]["robot"][
                "cspace_joint_order"
            ]
        )
        if len(arm_dof_names) != 6:
            raise RuntimeError("explicit arm DOF order must contain six names")
        verified_arm_dof_indices = [
            list(self.articulation.dof_names).index(name)
            for name in arm_dof_names
        ]
        sampled_arm = (
            initial_arm
            if initial_arm_q_rad is None
            else np.asarray(initial_arm_q_rad, dtype=np.float64)
        )
        self.initial_command = compose_initial_command(
            frozen_pregrasp_command,
            sampled_arm,
            arm_dof_indices=verified_arm_dof_indices,
        )
        zero_command = np.zeros_like(self.initial_command)
        self.articulation.set_joints_default_state(
            positions=self.initial_command,
            velocities=zero_command,
            efforts=zero_command,
        )
        self.articulation.post_reset()
        self.articulation.set_joint_positions(self.initial_command)
        self.articulation.set_joint_velocities(zero_command)
        immediate_readback = np.asarray(
            self.articulation.get_joint_positions(),
            dtype=np.float64,
        )
        if immediate_readback.shape != self.initial_command.shape:
            raise RuntimeError("initial joint readback shape mismatch")
        immediate_error = np.abs(
            immediate_readback - self.initial_command
        )
        if not np.isfinite(immediate_error).all():
            raise RuntimeError("initial joint readback is non-finite")
        self.arm_dof_indices = verified_arm_dof_indices
        self.initial_pose_evidence: dict[str, Any] = {
            "arm_dof_names": arm_dof_names,
            "arm_dof_indices": verified_arm_dof_indices,
            "initial_arm_q_target_rad": self.initial_command[
                verified_arm_dof_indices
            ].tolist(),
            "initial_joint_readback_immediate": (
                immediate_readback.tolist()
            ),
            "initial_arm_max_readback_error_rad": float(
                immediate_error[verified_arm_dof_indices].max()
            ),
            "initial_pose_hold_frames_required": (
                self.initial_pose_hold_frames
            ),
            "initial_pose_hold_frames_observed": 0,
            "first_frame_jump_rad": None,
            "initial_ee_position_world_m": None,
            "initial_ee_orientation_world_wxyz": None,
        }
        self._initial_pose_reference_q = immediate_readback.copy()

        controller = self.articulation.get_articulation_controller()
        finger_indices = np.asarray([7, 8], dtype=np.int32)
        copied_max_force = float(
            coupling["copied_left_drive_parameters"]["max_force"]
        )
        controller.set_max_efforts(
            np.asarray(
                [copied_max_force, copied_max_force],
                dtype=np.float32,
            ),
            finger_indices,
        )
        max_efforts = np.asarray(
            controller.get_max_efforts(),
            dtype=np.float64,
        )
        if not np.allclose(
            max_efforts[finger_indices],
            copied_max_force,
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError("finger max-force readback mismatch")

        self.command = self.initial_command.copy()
        self.command_velocity = np.zeros_like(self.command)
        self._target_write_count = 0
        self._write_joint_command()

        simulation_view = SimulationManager.get_physics_sim_view()
        if simulation_view is None or not simulation_view.is_valid:
            raise RuntimeError("PhysX tensor SimulationView unavailable")
        bottle_path = str(self.config["bottle"]["session_prim"])
        self._simulation_view = simulation_view
        self._bottle_tensor_path = bottle_path
        if (
            self.bottle_tensor_lifecycle_mode
            == "INITIALIZE_KINEMATIC_BODIES"
        ):
            initialization_result = (
                simulation_view.initialize_kinematic_bodies()
            )
            self.bottle_tensor_lifecycle[
                "initialize_kinematic_bodies_called"
            ] = True
            self.bottle_tensor_lifecycle[
                "initialize_kinematic_bodies_return"
            ] = self._jsonable_render_value(initialization_result)
        self.bottle = simulation_view.create_rigid_body_view(bottle_path)
        self.bottle_tensor_lifecycle[
            "rigid_body_view_creation_count"
        ] = 1
        if self.bottle is None or int(self.bottle.count) != 1:
            raise RuntimeError("Bottle500 PhysX rigid-body view unavailable")
        initial_identity = tensor_view_identity_record(
            self.bottle,
            expected_prim_path=bottle_path,
        )
        self.bottle_tensor_lifecycle[
            "rigid_body_view_identities"
        ].append(
            {
                "creation_index": 1,
                "physics_frame": 0,
                **initial_identity,
            }
        )
        self._finger_link_views = {
            side: simulation_view.create_rigid_body_view(
                str(paths["link"])
            )
            for side, paths in self.config["evidence"][
                "collider_overlay"
            ]["finger_colliders"].items()
        }
        if any(
            view is None or int(view.count) != 1
            for view in self._finger_link_views.values()
        ):
            raise RuntimeError(
                "finger PhysX rigid-body view unavailable for "
                "collider evidence"
            )
        table_bounds = _world_bounds(
            self.stage,
            str(self.config["stage"]["table_prim"]),
        )
        self.table_top_z_m = float(table_bounds["maximum"][2])
        self.base_position, self.base_orientation = self._get_world_pose(
            "/World/follower_left/vx300s_left/follower_left_base_link"
        )
        self._look_at_quaternion = _look_at_quaternion
        self._capture_attempt_index = 0
        self._initialize_video_capture()

        clearance_report = json.loads(
            self.task_profile["inputs"][
                "supplier_cad_clearance_report"
            ].read_text(encoding="utf-8")
        )
        contact_target = float(
            clearance_report["contact_solution"]["left_finger_q_m"]
        )
        close_target = contact_target - float(
            self.config["physics"]["preload_delta_m"]
        )
        self.close_targets = build_external_close_targets(
            open_position_m=float(
                self.task_profile["config"]["robot"][
                    "open_targets_m"
                ][0]
            ),
            contact_target_m=close_target,
            speed_m_s=0.02,
            physics_dt_s=self.dt,
        )

        self._contact_buffer: list[dict[str, Any]] = []
        self.all_contacts: list[dict[str, Any]] = []
        self._contact_frame = 0
        self._contact_phase = Phase.IDLE

        def on_contact(
            headers: Sequence[Any],
            data: Sequence[Any],
        ) -> None:
            records = self._serialize_contacts(
                headers,
                data,
                frame=self._contact_frame,
                time_s=self._contact_frame * self.dt,
                phase=self._contact_phase.value,
                dt=self.dt,
            )
            self._contact_buffer.extend(records)
            self.all_contacts.extend(records)

        self.contact_subscription = (
            self._physx_sim.subscribe_contact_report_events(on_contact)
        )
        self._initial_bottle_transform = np.asarray(
            self.bottle.get_transforms()[0],
            dtype=np.float64,
        )
        self._reset_runtime_records()

    def _finger_collider_world_points(self) -> dict[str, np.ndarray]:
        """Return authored finger-collider points at live PhysX link poses."""

        points_by_side: dict[str, np.ndarray] = {}
        for handle in self._finger_render_handles:
            if str(handle["category"]) != "collider":
                continue
            side = str(handle["side"])
            transform = np.asarray(
                self._finger_link_views[side].get_transforms()[0],
                dtype=np.float64,
            )
            quaternion_wxyz = [
                float(transform[6]),
                float(transform[3]),
                float(transform[4]),
                float(transform[5]),
            ]
            rotation = self._quaternion_matrix_wxyz(quaternion_wxyz)
            points_by_side[side] = (
                np.asarray(handle["local_points"], dtype=np.float64)
                @ rotation.T
                + transform[:3]
            )
        if set(points_by_side) != {"left", "right"}:
            raise RuntimeError("exactly two live finger colliders are required")
        return points_by_side

    def _finger_pair_geometry(self) -> dict[str, Any]:
        """Measure the live authored convex-hull pair without changing physics."""

        points = self._finger_collider_world_points()
        left = points["left"]
        right = points["right"]
        intersection_extent = np.minimum(
            np.max(left, axis=0),
            np.max(right, axis=0),
        ) - np.maximum(
            np.min(left, axis=0),
            np.min(right, axis=0),
        )
        if np.any(intersection_extent <= 0.0):
            return {
                "relation": "AABB_SEPARATED",
                "overlap_volume_m3": 0.0,
                "method": (
                    "world AABB broad phase; convex halfspace test is only "
                    "needed when all AABB extents overlap"
                ),
                "aabb_intersection_extent_m": intersection_extent.tolist(),
            }
        relation = convex_pair_relation(left, right)
        return {
            **relation,
            "aabb_intersection_extent_m": intersection_extent.tolist(),
        }

    def _evaluate_current_finger_initialization(self) -> dict[str, Any]:
        self._physx.update_transformations(True, True, False, False)
        readback = np.asarray(
            self.articulation.get_joint_positions(),
            dtype=np.float64,
        )
        geometry = self._finger_pair_geometry()
        contract = evaluate_finger_initialization(
            reset_complete=True,
            dof_order=self.finger_dof_names,
            targets=self.command[self.finger_dof_indices].tolist(),
            readback=readback[self.finger_dof_indices].tolist(),
            source_limits=self.source_finger_limits,
            overlap_volume_m3=float(geometry["overlap_volume_m3"]),
        )
        contract.update(
            {
                "source_limits_m": self.source_finger_limits,
                "composed_limits_m": self.composed_finger_limits,
                "pair_geometry": geometry,
                "world_reset_completed": True,
                "immediate_readback_required": bool(
                    self.finger_safety_config["require_immediate_readback"]
                ),
            }
        )
        contract["signature"] = canonical_initialization_signature(contract)
        return contract

    def _initialize_video_capture(self) -> None:
        """Create independent local-5.1 camera/render-product pairs."""

        from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices
        from isaacsim.sensors.camera import Camera

        targets = self.task_profile["kinematics"]["placement"][
            "target_poses"
        ]
        grasp = np.asarray(
            targets["grasp_ee_position_world_m"],
            dtype=np.float64,
        )
        base = np.asarray(self.base_position, dtype=np.float64)
        initial_ee, _ = self._get_world_pose(
            str(self.config["robot"]["end_effector_prim"])
        )
        overview = derive_overview_camera_geometry(
            base_position_world_m=base,
            initial_ee_position_world_m=initial_ee,
            grasp_position_world_m=grasp,
            lift_position_world_m=targets[
                "lift_ee_position_world_m"
            ],
        )
        specs = {
            "overview": {
                "position": np.asarray(
                    overview["position_world_m"],
                    dtype=np.float64,
                ),
                "target": np.asarray(
                    overview["target_world_m"],
                    dtype=np.float64,
                ),
                "derivation": overview,
            },
        }
        closeup = derive_gripper_closeup_camera_geometry(
            grasp_point_world_m=grasp,
            bottle_axis_world=self.task_profile["kinematics"][
                "placement"
            ]["bottle_axis"]["unit_world"],
            nominal_lift_m=(
                float(self.config["target"]["clearance_m"])
                + float(self.config["target"]["hold_drop_gate_m"])
            ),
            axial_side=self.closeup_axial_side,
        )
        specs["gripper_closeup"] = {
            "position": np.asarray(
                closeup["position_world_m"],
                dtype=np.float64,
            ),
            "target": np.asarray(
                closeup["target_world_m"],
                dtype=np.float64,
            ),
            "derivation": closeup,
        }
        self.video_cameras: dict[str, dict[str, Any]] = {}
        for view, spec in specs.items():
            orientation = self._look_at_quaternion(
                spec["position"],
                spec["target"],
            )
            camera = Camera(
                prim_path=(
                    "/World/ALOHA1Grasp20cmSession/Cameras/"
                    f"{view}"
                ),
                name=f"aloha1_grasp_20cm_{view}",
                position=spec["position"],
                orientation=orientation,
                resolution=(960, 540),
                annotator_device="cpu",
            )
            camera.initialize(attach_rgb_annotator=True)
            camera.set_world_pose(
                position=spec["position"],
                orientation=orientation,
                camera_axes="usd",
            )
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = quats_to_rot_matrices(orientation)
            matrix[:3, 3] = spec["position"]
            self.video_cameras[view] = {
                "camera": camera,
                "position_world_m": spec["position"].tolist(),
                "target_world_m": spec["target"].tolist(),
                "orientation_wxyz": orientation.tolist(),
                "camera_world_matrix": matrix.tolist(),
                "resolution": [960, 540],
                "render_product_path": str(
                    camera.get_render_product_path()
                ),
                "view_status": (
                    "ENGINEERING_EVIDENCE_VIEW_NOT_CALIBRATED"
                ),
                "derivation": spec.get(
                    "derivation",
                    {
                        "derivation": (
                            "FULL_ARM_OVERVIEW_FROM_FROZEN_BASE_AND_GRASP"
                        )
                    },
                ),
            }
        render_products = {
            record["render_product_path"]
            for record in self.video_cameras.values()
        }
        if len(render_products) != 2:
            raise RuntimeError(
                "video views do not have independent render products"
            )

    def _reset_runtime_records(self) -> None:
        self.started_at = time.perf_counter()
        self.observations: list[RunObservation] = []
        self.telemetry: list[dict[str, Any]] = []
        self._phase = Phase.IDLE
        self._phase_frames = 0
        self._trajectory: dict[Phase, list[np.ndarray]] = {}
        self._trajectory_velocity: dict[Phase, list[np.ndarray]] = {}
        self._trajectory_cursor: dict[Phase, int] = {}
        self._trajectory_audit: dict[str, Any] = {}
        self._ik_report: dict[str, Any] = {"status": "NOT_RUN"}
        self._setup_complete = False
        self._initial_pose_hold_observed_frames = 0
        if hasattr(self, "initial_pose_evidence"):
            self.initial_pose_evidence[
                "initial_pose_hold_frames_observed"
            ] = 0
            self.initial_pose_evidence["first_frame_jump_rad"] = None
            self.initial_pose_evidence[
                "initial_ee_position_world_m"
            ] = None
            self.initial_pose_evidence[
                "initial_ee_orientation_world_wxyz"
            ] = None
        self._preload_stable_frames = 0
        self._bilateral_before_lift = False
        self._bilateral_through_hold = True
        self._support_contact_ever = False
        self._height_reached = False
        self._hold_reference_clearance_m: float | None = None
        self._maximum_clearance_m = -math.inf
        self._initial_ee_z_m: float | None = None
        self._deep_penetration_frames: list[int] = []
        self._last_snapshot = {
            "clearance_m": 0.0,
            "maximum_clearance_m": 0.0,
            "left_contact": False,
            "right_contact": False,
            "ee_position_world_m": None,
            "ik": "NOT_RUN",
            "fingers": None,
            "bottle_velocity": None,
            "hold_drop_m": 0.0,
        }
        self._pending_capture_frames: list[dict[str, Any]] = []
        self._captured_frame_records: list[dict[str, Any]] = []
        self._video_capture_finalized = False
        self._video_capture_error: str | None = None
        self._video_attempt_root: Path | None = None
        self._collider_overlay_records: list[dict[str, Any]] = []
        self._captured_overlay_phases: set[str] = set()
        self._previous_bottle_pose_state: dict[str, Any] | None = None
        self._previous_bottle_com_state: dict[str, Any] | None = None
        self.initialization_contract = (
            self._evaluate_current_finger_initialization()
        )
        self._finger_safety_records: list[dict[str, Any]] = []
        self._finger_safety_first_violation: dict[str, Any] | None = None
        self._finger_environment_contact_count = 0
        if hasattr(self, "bottle_tensor_lifecycle"):
            self.bottle_tensor_lifecycle[
                "kinematic_to_dynamic_transition"
            ] = None
            self.bottle_tensor_lifecycle[
                "delayed_recreation_pending"
            ] = False
            self.bottle_tensor_lifecycle[
                "delayed_recreation_frame"
            ] = None
            self.bottle_tensor_lifecycle[
                "delayed_recreation_time_s"
            ] = None

    def prepare_run(self) -> None:
        if sha256_file(self.stage_path) != self.stage_hash_before:
            raise RuntimeError("approved Stage hash changed before Run")
        if not bool(
            self.physics_context.get_solve_articulation_contact_last()
        ):
            raise RuntimeError(
                "solve_articulation_contact_last readback is false"
            )
        self._reset_runtime_records()
        if self.initialization_contract["status"] != "PASS":
            raise RuntimeError(
                "FAIL_INITIALIZATION_CONTRACT: "
                + ",".join(
                    self.initialization_contract["failure_codes"]
                )
            )
        while True:
            self._capture_attempt_index += 1
            candidate = (
                self.artifact_root
                / f"video_attempt_{self._capture_attempt_index:03d}"
            )
            if not candidate.exists():
                self._video_attempt_root = candidate
                break
        assert self._video_attempt_root is not None
        self._video_attempt_root.mkdir(parents=True, exist_ok=False)
        self._phase = Phase.VALIDATE

    def _set_phase(self, phase: Phase) -> None:
        if phase is not self._phase:
            self._phase = phase
            self._phase_frames = 0
            self._contact_phase = phase
        self._phase_frames += 1

    def _build_runtime_trajectories(self) -> None:
        if self._ik_report.get("status") == "PASS":
            return
        bottle_state = self._read_bottle_state(self.bottle)
        ee_position, ee_orientation = self._get_world_pose(
            str(self.config["robot"]["end_effector_prim"])
        )
        current_q = np.asarray(
            self.articulation.get_joint_positions(),
            dtype=np.float64,
        )
        nominal_lift_m = float(
            self.config["target"]["clearance_m"]
            + self.config["target"]["hold_drop_gate_m"]
            + self.additional_lift_margin_m
        )
        from tools.aloha1_mapping.grasp_20cm_sampling import extend_profile_for_clearance_lift

        extended_profile = extend_profile_for_clearance_lift(
            self.task_profile,
            target_clearance_m=float(
                self.config["target"]["clearance_m"]
            ),
            hold_drop_gate_m=float(
                self.config["target"]["hold_drop_gate_m"]
            ),
            additional_lift_margin_m=self.additional_lift_margin_m,
        )
        result = self._solve_settled_ik(
            extended_profile,
            base_position=np.asarray(
                self.base_position,
                dtype=np.float64,
            ),
            base_orientation=np.asarray(
                self.base_orientation,
                dtype=np.float64,
            ),
            bottle_state=bottle_state,
            current_ee_position=np.asarray(
                ee_position,
                dtype=np.float64,
            ),
            current_ee_orientation=np.asarray(
                ee_orientation,
                dtype=np.float64,
            ),
            current_arm_q=current_q[:6],
        )
        self._ik_report = {
            **result,
            "nominal_vertical_lift_m": nominal_lift_m,
            "actual_gate": (
                "BOTTLE_COLLISION_MIN_WORLD_Z_MINUS_TABLE_TOP_WORLD_Z"
            ),
        }
        if result["status"] != "PASS":
            raise RuntimeError(
                "settled Bottle500 runtime IK failed: "
                f"{result.get('failure_phase')}"
            )
        phase_map = {
            "move_to_pregrasp": Phase.OPEN_PREGRASP,
            "vertical_descent": Phase.VERTICAL_DESCENT,
            "vertical_lift": Phase.VERTICAL_LIFT,
        }
        previous_q = current_q[:6].copy()
        velocity_limits = np.asarray(
            extended_profile["kinematics"]["ik"][
                "joint_velocity_limits_rad_s"
            ],
            dtype=np.float64,
        )
        for phase_name, phase in phase_map.items():
            raw_targets = [
                np.asarray(
                    waypoint["joint_positions_rad"],
                    dtype=np.float64,
                )
                for waypoint in result["waypoints"]
                if waypoint["phase"] == phase_name
            ]
            if not raw_targets:
                raise RuntimeError(
                    f"runtime IK has no {phase_name} waypoints"
                )
            if (
                self.arm_trajectory_mode
                == "LULA_CSPACE_ACCELERATION_LIMITED"
            ):
                from isaacsim.robot_motion.motion_generation import LulaCSpaceTrajectoryGenerator

                assert self.arm_acceleration_limits_rad_s2 is not None
                timed = build_lula_cspace_phase_targets(
                    generator_factory=LulaCSpaceTrajectoryGenerator,
                    robot_description_path=str(
                        extended_profile["inputs"]["lula_descriptor"]
                    ),
                    urdf_path=str(
                        extended_profile["inputs"]["follower_left_urdf"]
                    ),
                    waypoint_positions=[
                        previous_q,
                        *raw_targets,
                    ],
                    physics_dt_s=self.dt,
                    velocity_limits_rad_s=velocity_limits,
                    acceleration_limits_rad_s2=(
                        self.arm_acceleration_limits_rad_s2
                    ),
                )
                audit = timed["audit"]
                if not all(
                    bool(audit[key])
                    for key in (
                        "finite",
                        "endpoint_velocity_zero",
                        "velocity_within_limits",
                        "sampled_acceleration_within_limits",
                    )
                ):
                    raise RuntimeError(
                        f"Lula trajectory audit failed for {phase_name}: "
                        f"{audit}"
                    )
                self._trajectory[phase] = [
                    np.asarray(target["position_rad"], dtype=np.float64)
                    for target in timed["targets"]
                ]
                self._trajectory_velocity[phase] = [
                    np.asarray(
                        target["velocity_rad_s"],
                        dtype=np.float64,
                    )
                    for target in timed["targets"]
                ]
                self._trajectory_audit[phase_name] = audit
            else:
                self._trajectory[phase] = raw_targets
                self._trajectory_velocity[phase] = [
                    np.zeros(6, dtype=np.float64)
                    for _ in raw_targets
                ]
                self._trajectory_audit[phase_name] = {
                    "mode": "LEGACY_VELOCITY_STEP",
                    "sample_count": len(raw_targets),
                    "acceleration_limit_status": "NOT_APPLIED",
                }
            self._trajectory_cursor[phase] = 0
            previous_q = self._trajectory[phase][-1].copy()
        self._trajectory[Phase.BILATERAL_CONTACT] = []
        self._trajectory_velocity[Phase.BILATERAL_CONTACT] = []
        self._trajectory_cursor[Phase.BILATERAL_CONTACT] = 0
        self._trajectory_cursor[Phase.CLOSE_PRELOAD] = 0
        self._ik_report["trajectory_time_parameterization"] = {
            "mode": self.arm_trajectory_mode,
            "phase_audits": self._trajectory_audit,
            "jerk_limit_status": (
                "NOT_SET_NO_EXACT_MODEL_OFFICIAL_VALUE"
                if self.arm_trajectory_mode
                == "LULA_CSPACE_ACCELERATION_LIMITED"
                else "NOT_APPLICABLE_LEGACY_BASELINE"
            ),
        }

    def _advance_arm(self, phase: Phase) -> None:
        targets = self._trajectory.get(phase, [])
        cursor = self._trajectory_cursor.get(phase, 0)
        if cursor >= len(targets):
            return
        self.command[:6] = targets[cursor]
        self.command_velocity[:6] = self._trajectory_velocity[phase][cursor]
        self._trajectory_cursor[phase] = cursor + 1
        self._write_joint_command()

    def _advance_close(self, phase: Phase) -> None:
        cursor = self._trajectory_cursor.get(
            Phase.BILATERAL_CONTACT,
            0,
        )
        if cursor < len(self.close_targets):
            left = float(self.close_targets[cursor])
            self.command[7] = left
            self.command[8] = -left
            self._trajectory_cursor[Phase.BILATERAL_CONTACT] = cursor + 1
        self._trajectory_cursor[phase] = self._trajectory_cursor.get(
            phase,
            0,
        ) + 1
        self._write_joint_command()

    def _write_joint_command(self) -> None:
        if (
            self.arm_trajectory_mode
            == "LULA_CSPACE_ACCELERATION_LIMITED"
        ):
            from isaacsim.core.utils.types import ArticulationAction

            arm_indices = np.asarray(
                self.arm_dof_indices,
                dtype=np.int32,
            )
            self.articulation.apply_action(
                ArticulationAction(
                    joint_positions=np.asarray(
                        self.command[self.arm_dof_indices],
                        dtype=np.float32,
                    ),
                    joint_velocities=np.asarray(
                        self.command_velocity[self.arm_dof_indices],
                        dtype=np.float32,
                    ),
                    joint_indices=arm_indices,
                )
            )
            self.articulation.apply_action(
                ArticulationAction(
                    joint_positions=np.asarray(
                        self.command[[7, 8]],
                        dtype=np.float32,
                    ),
                    joint_indices=np.asarray([7, 8], dtype=np.int32),
                )
            )
        else:
            self._command_positions(self.articulation, self.command)
        self._target_write_count += 1

    def apply_phase_target(self, phase: Phase) -> None:
        self._set_phase(phase)
        if phase in {
            Phase.VALIDATE,
            Phase.SETUP_KINEMATIC,
            Phase.RELEASE_DYNAMIC,
            Phase.SETTLE,
            Phase.HEIGHT_REACHED,
            Phase.HOLD,
        }:
            self._write_joint_command()
        elif phase is Phase.OPEN_PREGRASP:
            self._build_runtime_trajectories()
            self._advance_arm(phase)
        elif phase is Phase.VERTICAL_DESCENT:
            self._advance_arm(phase)
        elif phase in {
            Phase.BILATERAL_CONTACT,
            Phase.CLOSE_PRELOAD,
        }:
            self._advance_close(phase)
        elif phase is Phase.VERTICAL_LIFT:
            self._advance_arm(phase)
        if phase is Phase.HEIGHT_REACHED:
            self._height_reached = True
            self._hold_reference_clearance_m = float(
                self._last_snapshot["clearance_m"]
            )

    def _recreate_bottle_tensor_view(self) -> None:
        refreshed = self._simulation_view.create_rigid_body_view(
            self._bottle_tensor_path
        )
        if refreshed is None or int(refreshed.count) != 1:
            raise RuntimeError(
                "Bottle500 PhysX rigid-body view recreation failed"
            )
        self.bottle = refreshed
        self.bottle_tensor_lifecycle[
            "rigid_body_view_creation_count"
        ] = int(
            self.bottle_tensor_lifecycle[
                "rigid_body_view_creation_count"
            ]
        ) + 1
        identity = tensor_view_identity_record(
            self.bottle,
            expected_prim_path=self._bottle_tensor_path,
        )
        self.bottle_tensor_lifecycle[
            "rigid_body_view_identities"
        ].append(
            {
                "creation_index": int(
                    self.bottle_tensor_lifecycle[
                        "rigid_body_view_creation_count"
                    ]
                ),
                "physics_frame": int(self._contact_frame),
                **identity,
            }
        )

    def set_bottle_kinematic(self, *, enabled: bool) -> None:
        from pxr import UsdPhysics

        rigid = UsdPhysics.RigidBodyAPI(self.bottle_prim)
        before = bool(rigid.GetKinematicEnabledAttr().Get())
        rigid.GetKinematicEnabledAttr().Set(bool(enabled))
        self._physx_sim.flush_changes()
        recreated = False
        if (
            not enabled
            and self.bottle_tensor_lifecycle_mode
            == "RECREATE_AFTER_DYNAMIC"
        ):
            self._recreate_bottle_tensor_view()
            recreated = True
        elif (
            not enabled
            and self.bottle_tensor_lifecycle_mode
            == "RECREATE_AFTER_DYNAMIC_STEP"
        ):
            self.bottle_tensor_lifecycle[
                "delayed_recreation_pending"
            ] = True
        self.bottle_tensor_lifecycle[
            "kinematic_to_dynamic_transition"
        ] = {
            "physics_frame": int(self._contact_frame),
            "time_s": float(self._contact_frame * self.dt),
            "phase": self._phase.value,
            "before_kinematic_enabled": before,
            "after_kinematic_enabled": bool(
                rigid.GetKinematicEnabledAttr().Get()
            ),
            "rigid_body_view_recreated": recreated,
        }

    def _phase_done(self, phase: Phase) -> bool:
        targets = self._trajectory.get(phase, [])
        return bool(targets) and self._trajectory_cursor.get(
            phase,
            0,
        ) >= len(targets)

    def read_observation(
        self,
        *,
        frame: int,
        time_s: float,
    ) -> RunObservation:
        from pxr import UsdPhysics

        self._contact_frame = frame
        transition = self.bottle_tensor_lifecycle.get(
            "kinematic_to_dynamic_transition"
        )
        transition_frame = (
            int(transition["physics_frame"])
            if isinstance(transition, Mapping)
            else None
        )
        if delayed_tensor_recreation_due(
            mode=self.bottle_tensor_lifecycle_mode,
            pending=bool(
                self.bottle_tensor_lifecycle[
                    "delayed_recreation_pending"
                ]
            ),
            current_frame=frame,
            transition_frame=transition_frame,
        ):
            self._recreate_bottle_tensor_view()
            self.bottle_tensor_lifecycle[
                "delayed_recreation_pending"
            ] = False
            self.bottle_tensor_lifecycle[
                "delayed_recreation_frame"
            ] = int(frame)
            self.bottle_tensor_lifecycle[
                "delayed_recreation_time_s"
            ] = float(time_s)
        self._physx.update_transformations(
            True,
            True,
            self.bottle_usd_velocity_readback,
            False,
        )
        bottle_state = self._read_bottle_state(self.bottle)
        physics_dt_readback = float(
            self.physics_context.get_physics_dt()
        )
        bottle_state["synchronized_com_velocity_sample"] = (
            self._build_com_velocity_sample(
                step_index=frame,
                state_boundary_index=frame,
                dt_s=physics_dt_readback,
                sampling_phase="POST_PHYSICS_STEP",
                actor_prim_path=self._bottle_tensor_path,
                tensor_index=0,
                actor_position_world_m=(
                    bottle_state["position_world_m"]
                ),
                actor_orientation_world_wxyz=(
                    bottle_state["orientation_wxyz"]
                ),
                center_of_mass_local_m=(
                    bottle_state["center_of_mass_local_m"]
                ),
                linear_velocity_com_world_m_s=(
                    bottle_state["linear_velocity_world_m_s"]
                ),
                angular_velocity_world_rad_s=(
                    bottle_state["angular_velocity_world_rad_s"]
                ),
            )
        )
        bottle_state["sampling_contract"] = {
            "callback_phase": "POST_PHYSICS_STEP",
            "subscription_api": (
                "omni.physx.IPhysx.subscribe_physics_on_step_events"
            ),
            "subscription_pre_step_argument": False,
            "physics_dt_s": physics_dt_readback,
            "source_evidence": (
                "omni.physx.tests/PhysxInterfaceSimulationEvents.py"
            ),
        }
        direct_physx_transform = normalize_direct_physx_transform(
            self._physx.get_rigidbody_transformation(
                self._bottle_tensor_path
            )
        )
        direct_position = np.asarray(
            direct_physx_transform["position_world_m"],
            dtype=np.float64,
        )
        tensor_position = np.asarray(
            bottle_state["position_world_m"],
            dtype=np.float64,
        )
        direct_physx_transform[
            "tensor_position_delta_norm_m"
        ] = float(np.linalg.norm(direct_position - tensor_position))
        bottle_state["direct_physx_transform"] = direct_physx_transform
        bottle_state["tensor_view_identity"] = (
            self.bottle_tensor_lifecycle[
                "rigid_body_view_identities"
            ][-1]
        )
        if self.bottle_usd_velocity_readback:
            rigid_body_api = UsdPhysics.RigidBodyAPI(self.bottle_prim)
            usd_velocity = normalize_usd_velocity_readback(
                linear_velocity=rigid_body_api.GetVelocityAttr().Get(),
                angular_velocity_deg_s=(
                    rigid_body_api.GetAngularVelocityAttr().Get()
                ),
            )
            tensor_linear = np.asarray(
                bottle_state["linear_velocity_world_m_s"],
                dtype=np.float64,
            )
            tensor_angular = np.asarray(
                bottle_state["angular_velocity_world_rad_s"],
                dtype=np.float64,
            )
            usd_velocity["tensor_linear_delta_norm_m_s"] = float(
                np.linalg.norm(
                    np.asarray(
                        usd_velocity["linear_velocity_world_m_s"],
                        dtype=np.float64,
                    )
                    - tensor_linear
                )
            )
            usd_velocity["tensor_angular_delta_norm_rad_s"] = float(
                np.linalg.norm(
                    np.asarray(
                        usd_velocity["angular_velocity_world_rad_s"],
                        dtype=np.float64,
                    )
                    - tensor_angular
                )
            )
            bottle_state["usd_velocity_readback"] = usd_velocity
        pose_finite_difference_velocity = (
            self._derive_pose_velocity(
                previous=self._previous_bottle_pose_state,
                current=bottle_state,
                dt_s=self.dt,
            )
            if self._previous_bottle_pose_state is not None
            else None
        )
        current_com_pose_state = {
            "position_world_m": list(
                bottle_state["center_of_mass_world_m"]
            ),
            "orientation_wxyz": list(bottle_state["orientation_wxyz"]),
        }
        center_of_mass_pose_finite_difference_velocity = (
            self._derive_pose_velocity(
                previous=self._previous_bottle_com_state,
                current=current_com_pose_state,
                dt_s=self.dt,
            )
            if self._previous_bottle_com_state is not None
            else None
        )
        self._previous_bottle_pose_state = {
            "position_world_m": list(bottle_state["position_world_m"]),
            "orientation_wxyz": list(bottle_state["orientation_wxyz"]),
        }
        self._previous_bottle_com_state = current_com_pose_state
        position = np.asarray(
            bottle_state["position_world_m"],
            dtype=np.float64,
        )
        orientation = np.asarray(
            bottle_state["orientation_wxyz"],
            dtype=np.float64,
        )
        collision_bounds = self._transform_collision_bounds(
            local_points=self.bottle_collision_points_local,
            position_world=position,
            orientation_world_wxyz=orientation,
        )
        clearance_m = (
            float(collision_bounds["minimum"][2])
            - self.table_top_z_m
        )
        self._maximum_clearance_m = max(
            self._maximum_clearance_m,
            clearance_m,
        )
        current_contacts = self._contact_buffer
        self._contact_buffer = []
        bottle_token = str(self.config["bottle"]["session_prim"])
        left_contacts = self._physical_contacts(
            current_contacts,
            tokens=(
                bottle_token,
                "diagnostic_supplier_cad_left_finger",
            ),
        )
        right_contacts = self._physical_contacts(
            current_contacts,
            tokens=(
                bottle_token,
                "diagnostic_supplier_cad_right_finger",
            ),
        )
        left_solver_contacts = solver_active_contacts(
            current_contacts,
            tokens=(
                bottle_token,
                "diagnostic_supplier_cad_left_finger",
            ),
        )
        right_solver_contacts = solver_active_contacts(
            current_contacts,
            tokens=(
                bottle_token,
                "diagnostic_supplier_cad_right_finger",
            ),
        )
        support_contacts = self._physical_contacts(
            current_contacts,
            tokens=(
                bottle_token,
                str(self.config["stage"]["table_prim"]).rsplit(
                    "/",
                    maxsplit=1,
                )[-1],
            ),
        )
        if support_contacts:
            self._support_contact_ever = True
        support_contact = bool(support_contacts) or bool(
            self._support_contact_ever
            and clearance_m
            <= float(
                self.config["target"][
                    "support_contact_latch_clearance_m"
                ]
            )
        )
        bilateral_geometric = bool(left_contacts and right_contacts)
        bilateral_solver_active = bool(
            left_solver_contacts and right_solver_contacts
        )
        bilateral = bilateral_observation_contact(
            bilateral_geometric=bilateral_geometric,
            bilateral_solver_active=bilateral_solver_active,
        )
        if bilateral and self._phase in {
            Phase.BILATERAL_CONTACT,
            Phase.CLOSE_PRELOAD,
        }:
            self._bilateral_before_lift = True
        if self._phase is Phase.HOLD:
            self._bilateral_through_hold &= bilateral

        qpos = np.asarray(
            self.articulation.get_joint_positions(),
            dtype=np.float64,
        )
        qvel = np.asarray(
            self.articulation.get_joint_velocities(),
            dtype=np.float64,
        )
        finger_pair_geometry = self._finger_pair_geometry()
        finger_safety = evaluate_finger_runtime_frame(
            frame=frame,
            phase=self._phase.value,
            targets=self.command[self.finger_dof_indices].tolist(),
            readback=qpos[self.finger_dof_indices].tolist(),
            source_limits=self.source_finger_limits,
            pair_overlap_volume_m3=float(
                finger_pair_geometry["overlap_volume_m3"]
            ),
            contacts=current_contacts,
            finger_paths={
                "left_finger": str(
                    self.config["robot"]["left_finger_prim"]
                ),
                "right_finger": str(
                    self.config["robot"]["right_finger_prim"]
                ),
            },
        )
        finger_safety["pair_geometry"] = finger_pair_geometry
        self._finger_safety_records.append(finger_safety)
        self._finger_environment_contact_count += len(
            finger_safety["finger_environment_contacts"]
        )
        if (
            finger_safety["status"] == "FAIL"
            and self._finger_safety_first_violation is None
        ):
            self._finger_safety_first_violation = dict(finger_safety)
        arm_target_error_rad = float(
            np.max(
                np.abs(
                    qpos[self.arm_dof_indices]
                    - self.command[self.arm_dof_indices]
                )
            )
        )
        def phase_target_reached(phase: Phase) -> bool:
            exhausted = self._phase_done(phase)
            if self.arm_phase_readback_tolerance_rad is None:
                return exhausted
            return arm_phase_target_reached(
                trajectory_exhausted=exhausted,
                joint_readback=qpos,
                joint_target=self.command,
                arm_dof_indices=self.arm_dof_indices,
                tolerance_rad=self.arm_phase_readback_tolerance_rad,
            )

        ee_position, ee_orientation = self._get_world_pose(
            str(self.config["robot"]["end_effector_prim"])
        )
        ee_position = np.asarray(ee_position, dtype=np.float64)
        ee_orientation = np.asarray(
            ee_orientation,
            dtype=np.float64,
        )
        if self._phase is Phase.SETUP_KINEMATIC:
            self._initial_pose_hold_observed_frames += 1
            self._setup_complete = initial_pose_hold_complete(
                observed_frame_count=(
                    self._initial_pose_hold_observed_frames
                ),
                required_frame_count=self.initial_pose_hold_frames,
            )
            self.initial_pose_evidence[
                "initial_pose_hold_frames_observed"
            ] = self._initial_pose_hold_observed_frames
            if self._initial_pose_hold_observed_frames == 1:
                self.initial_pose_evidence["first_frame_jump_rad"] = float(
                    np.max(
                        np.abs(
                            qpos[self.arm_dof_indices]
                            - self._initial_pose_reference_q[
                                self.arm_dof_indices
                            ]
                        )
                    )
                )
                self.initial_pose_evidence[
                    "initial_ee_position_world_m"
                ] = ee_position.tolist()
                self.initial_pose_evidence[
                    "initial_ee_orientation_world_wxyz"
                ] = ee_orientation.tolist()
        if self._initial_ee_z_m is None:
            self._initial_ee_z_m = float(ee_position[2])
        ee_displacement = (
            float(ee_position[2]) - self._initial_ee_z_m
        )
        coupling_residual = abs(float(qpos[7] + qpos[8]))
        close_exhausted = (
            self._trajectory_cursor.get(
                Phase.BILATERAL_CONTACT,
                0,
            )
            >= len(self.close_targets)
        )
        if self._phase is Phase.CLOSE_PRELOAD and preload_solver_contact_ready(
            close_exhausted=close_exhausted,
            left_solver_active=bool(left_solver_contacts),
            right_solver_active=bool(right_solver_contacts),
            coupling_residual_m=coupling_residual,
            coupling_gate_m=0.001,
        ):
            self._preload_stable_frames += 1
        elif self._phase is Phase.CLOSE_PRELOAD:
            self._preload_stable_frames = 0
        preload_complete = self._preload_stable_frames >= 5
        hold_drop_m = (
            max(
                0.0,
                self._hold_reference_clearance_m - clearance_m,
            )
            if self._hold_reference_clearance_m is not None
            else 0.0
        )
        deep = [
            contact
            for contact in current_contacts
            if bottle_token in (
                f"{contact.get('collider0_path', '')} "
                f"{contact.get('collider1_path', '')}"
            )
            and float(contact["separation_m"]) < -0.005
        ]
        if deep:
            self._deep_penetration_frames.append(frame)
        persistent_penetration = all(
            frame - offset in self._deep_penetration_frames
            for offset in (0, 1, 2)
        )
        finite_values = np.concatenate(
            (
                qpos,
                qvel,
                position,
                np.asarray(
                    bottle_state["linear_velocity_world_m_s"],
                    dtype=np.float64,
                ),
                np.asarray(
                    bottle_state["angular_velocity_world_rad_s"],
                    dtype=np.float64,
                ),
            )
        )
        maximum_speed = float(
            np.linalg.norm(
                bottle_state["linear_velocity_world_m_s"]
            )
        )
        maximum_angular = float(
            bottle_state["angular_speed_rad_s"]
        )
        phase_timeout = PHASE_TIMEOUT_FRAMES.get(self._phase, 600)
        if self._phase is Phase.SETUP_KINEMATIC:
            phase_timeout = max(
                phase_timeout,
                self.initial_pose_hold_frames + 10,
            )
        if self._phase in {
            Phase.OPEN_PREGRASP,
            Phase.VERTICAL_DESCENT,
            Phase.VERTICAL_LIFT,
        }:
            phase_timed_out = arm_phase_timeout_reached(
                phase_frame_count=self._phase_frames,
                trajectory_sample_count=len(
                    self._trajectory.get(self._phase, [])
                ),
                readback_settle_timeout_frames=phase_timeout,
                trajectory_exhausted=self._phase_done(self._phase),
            )
        else:
            phase_timed_out = self._phase_frames > phase_timeout
        dynamic = not bool(
            UsdPhysics.RigidBodyAPI(self.bottle_prim)
            .GetKinematicEnabledAttr()
            .Get()
        )
        observation = RunObservation(
            frame=frame,
            time_s=time_s,
            clearance_m=clearance_m,
            bottle_dynamic=dynamic,
            support_contact=support_contact,
            bottle_linear_speed_m_s=maximum_speed,
            bottle_angular_speed_rad_s=maximum_angular,
            stage_contract_valid=(
                sha256_file(self.stage_path) == self.stage_hash_before
            ),
            setup_complete=self._setup_complete,
            open_target_reached=phase_target_reached(
                Phase.OPEN_PREGRASP
            ),
            descent_complete=phase_target_reached(
                Phase.VERTICAL_DESCENT
            ),
            bilateral_contact=bilateral,
            preload_complete=preload_complete,
            lift_waypoint_exhausted=phase_target_reached(
                Phase.VERTICAL_LIFT
            ),
            hold_drop_m=hold_drop_m,
            finite_state=bool(np.isfinite(finite_values).all()),
            persistent_penetration=persistent_penetration,
            numerical_ejection=bool(
                maximum_speed > 5.0 or maximum_angular > 50.0
            ),
            forbidden_constraint=bool(
                self.abort_on_first_runtime_violation
                and finger_safety["status"] == "FAIL"
            ),
            phase_timed_out=phase_timed_out,
            ee_vertical_displacement_m=ee_displacement,
        )
        self.observations.append(observation)
        record = {
            "frame": frame,
            "time_s": time_s,
            "phase": self._phase.value,
            "observation": asdict(observation),
            "joint_target": self.command.tolist(),
            "joint_velocity_target": self.command_velocity.tolist(),
            "joint_readback": qpos.tolist(),
            "joint_velocity": qvel.tolist(),
            "finger_safety": finger_safety,
            "arm_target_max_readback_error_rad": arm_target_error_rad,
            "bottle": {
                **bottle_state,
                "collision_bounds": collision_bounds,
                "pose_finite_difference_velocity": (
                    pose_finite_difference_velocity
                ),
                "center_of_mass_pose_finite_difference_velocity": (
                    center_of_mass_pose_finite_difference_velocity
                ),
            },
            "contact_semantics": {
                "geometric_contact_definition": "separation_m <= 0",
                "solver_active_definition": (
                    "finite impulse_ns > 0 inside reported contact pair"
                ),
                "left_geometric_contact": bool(left_contacts),
                "right_geometric_contact": bool(right_contacts),
                "bilateral_geometric_contact": bilateral_geometric,
                "left_solver_active_contact": bool(
                    left_solver_contacts
                ),
                "right_solver_active_contact": bool(
                    right_solver_contacts
                ),
                "bilateral_solver_active_contact": (
                    bilateral_solver_active
                ),
                "observation_contact_gate": (
                    "BILATERAL_REPORTED_PAIR_WITH_FINITE_POSITIVE_"
                    "SOLVER_IMPULSE"
                ),
            },
            "contacts": current_contacts,
        }
        self.telemetry.append(record)
        self._pending_capture_frames.append(
            {
                "physics_frame": frame,
                "time_s": time_s,
                "phase": self._phase.value,
                "telemetry_index": len(self.telemetry) - 1,
            }
        )
        self._last_snapshot = {
            "clearance_m": clearance_m,
            "maximum_clearance_m": self._maximum_clearance_m,
            "left_contact": (
                bool(left_solver_contacts)
                if self._phase in {Phase.HEIGHT_REACHED, Phase.HOLD}
                else bool(left_contacts)
            ),
            "right_contact": (
                bool(right_solver_contacts)
                if self._phase in {Phase.HEIGHT_REACHED, Phase.HOLD}
                else bool(right_contacts)
            ),
            "left_geometric_contact": bool(left_contacts),
            "right_geometric_contact": bool(right_contacts),
            "left_solver_active_contact": bool(left_solver_contacts),
            "right_solver_active_contact": bool(right_solver_contacts),
            "ee_position_world_m": ee_position.tolist(),
            "ik": self._ik_report.get("status", "NOT_RUN"),
            "fingers": {
                "target_m": self.command[[7, 8]].tolist(),
                "readback_m": qpos[[7, 8]].tolist(),
                "coupling_residual_m": coupling_residual,
                "safety_status": finger_safety["status"],
            },
            "bottle_velocity": {
                "linear_world_m_s": bottle_state[
                    "linear_velocity_world_m_s"
                ],
                "angular_world_rad_s": bottle_state[
                    "angular_velocity_world_rad_s"
                ],
            },
            "hold_drop_m": hold_drop_m,
        }
        return observation

    @staticmethod
    def _jsonable_render_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): IsaacGrasp20cmBindings._jsonable_render_value(
                    item
                )
                for key, item in value.items()
            }
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, str | int | float | bool) or value is None:
            return value
        return str(value)

    def prepare_pending_evidence_cameras(
        self,
        *,
        render_settle_updates: int,
    ) -> dict[str, Any]:
        """Refit only the sparse closeup camera around current evidence.

        The collision-only pipeline calls this with the timeline paused and
        performs one application update before reading the render product.
        The overview camera and all simulation state remain unchanged.
        """

        if len(self._pending_capture_frames) != 1:
            raise RuntimeError(
                "evidence camera refit requires exactly one pending frame"
            )
        if render_settle_updates < 1:
            raise ValueError("render_settle_updates must be positive")
        pending = self._pending_capture_frames[0]
        telemetry = self.telemetry[int(pending["telemetry_index"])]
        bottle = telemetry["bottle"]
        from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices

        bottle_position = np.asarray(
            bottle["position_world_m"],
            dtype=np.float64,
        )
        bottle_rotation = quats_to_rot_matrices(
            np.asarray(bottle["orientation_wxyz"], dtype=np.float64)
        )
        bottle_axis_config = self.task_profile["config"]["bottle"]["axis"]
        a_world = (
            bottle_rotation
            @ np.asarray(bottle_axis_config["a_local_m"], dtype=np.float64)
            + bottle_position
        )
        b_world = (
            bottle_rotation
            @ np.asarray(bottle_axis_config["b_local_m"], dtype=np.float64)
            + bottle_position
        )
        left_origin, _ = self._get_world_pose(
            str(
                self.task_profile["config"]["robot"]
                ["left_finger_collider"]
            )
        )
        right_origin, _ = self._get_world_pose(
            str(
                self.task_profile["config"]["robot"]
                ["right_finger_collider"]
            )
        )
        finger_points = self._finger_collider_world_points()
        subject_points = np.concatenate(
            (
                np.asarray([a_world, b_world], dtype=np.float64),
                finger_points["left"],
                finger_points["right"],
            ),
            axis=0,
        )
        camera_record = self.video_cameras["gripper_closeup"]
        camera = camera_record["camera"]
        clipping = camera.get_clipping_range()
        geometry = derive_subject_bounding_closeup_camera_geometry(
            subject_points_world_m=subject_points,
            bottle_axis_world=(
                self.task_profile["kinematics"]["placement"]["bottle_axis"]
                ["unit_world"]
            ),
            horizontal_fov_rad=float(camera.get_horizontal_fov()),
            vertical_fov_rad=float(camera.get_vertical_fov()),
            near_clipping_m=float(clipping[0]),
            axial_side=self.closeup_axial_side,
        )
        position = np.asarray(geometry["position_world_m"], dtype=np.float64)
        target = np.asarray(geometry["target_world_m"], dtype=np.float64)
        orientation = self._look_at_quaternion(position, target)
        camera.set_world_pose(
            position=position,
            orientation=orientation,
            camera_axes="usd",
        )
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = quats_to_rot_matrices(orientation)
        matrix[:3, 3] = position
        camera_record.update(
            {
                "position_world_m": position.tolist(),
                "target_world_m": target.tolist(),
                "orientation_wxyz": orientation.tolist(),
                "camera_world_matrix": matrix.tolist(),
                "derivation": {
                    **geometry,
                    "physics_frame": int(pending["physics_frame"]),
                    "time_s": float(pending["time_s"]),
                    "runtime_phase": str(pending["phase"]),
                    "clipping_range_m": [
                        float(clipping[0]),
                        float(clipping[1]),
                    ],
                    "render_settle_updates": int(render_settle_updates),
                },
            }
        )
        return {
            "status": "PASS",
            "view": "gripper_closeup",
            "physics_frame": int(pending["physics_frame"]),
            "runtime_phase": str(pending["phase"]),
            "render_settle_updates": int(render_settle_updates),
            "derivation": camera_record["derivation"],
            "scope": "DIAGNOSTIC_EVIDENCE_CAMERA_ONLY_NO_PHYSICS_CHANGE",
        }

    def capture_pending_render_frame(self) -> bool:
        """Save both camera products after the enclosing app update."""

        if not self._pending_capture_frames:
            return False
        if len(self._pending_capture_frames) != 1:
            raise RuntimeError(
                "render capture fell behind physics: "
                f"{len(self._pending_capture_frames)} pending frames"
            )
        if self._video_attempt_root is None:
            raise RuntimeError("video attempt root is not initialized")
        from tools.validate_aloha1_task7b2_horizontal_grasp import FULL_ARM_LINK_PRIMS
        from tools.validate_aloha1_task7b2_horizontal_grasp import _full_arm_framing_evidence

        pending = self._pending_capture_frames[0]
        frame = int(pending["physics_frame"])
        telemetry = self.telemetry[int(pending["telemetry_index"])]
        from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices

        bottle = telemetry["bottle"]
        bottle_position = np.asarray(
            bottle["position_world_m"],
            dtype=np.float64,
        )
        bottle_rotation = quats_to_rot_matrices(
            np.asarray(
                bottle["orientation_wxyz"],
                dtype=np.float64,
            )
        )
        bottle_axis_config = self.task_profile["config"]["bottle"][
            "axis"
        ]
        a_world = (
            bottle_rotation
            @ np.asarray(
                bottle_axis_config["a_local_m"],
                dtype=np.float64,
            )
            + bottle_position
        )
        b_world = (
            bottle_rotation
            @ np.asarray(
                bottle_axis_config["b_local_m"],
                dtype=np.float64,
            )
            + bottle_position
        )
        left_origin, _ = self._get_world_pose(
            str(
                self.task_profile["config"]["robot"][
                    "left_finger_collider"
                ]
            )
        )
        right_origin, _ = self._get_world_pose(
            str(
                self.task_profile["config"]["robot"][
                    "right_finger_collider"
                ]
            )
        )
        projection_points = {
            "bottle_a": a_world.tolist(),
            "bottle_b": b_world.tolist(),
            "left_finger_collider_origin": np.asarray(
                left_origin,
                dtype=np.float64,
            ).tolist(),
            "right_finger_collider_origin": np.asarray(
                right_origin,
                dtype=np.float64,
            ).tolist(),
        }
        selected_contacts: dict[str, dict[str, Any] | None] = {}
        for side, token in (
            ("left", "diagnostic_supplier_cad_left_finger"),
            ("right", "diagnostic_supplier_cad_right_finger"),
        ):
            candidates = [
                item
                for item in telemetry["contacts"]
                if token
                in (
                    f"{item.get('collider0_path', '')} "
                    f"{item.get('collider1_path', '')}"
                )
                and math.isfinite(float(item["impulse_ns"]))
            ]
            selected = (
                max(candidates, key=lambda item: float(item["impulse_ns"]))
                if candidates
                else None
            )
            selected_contacts[side] = selected
            if selected is not None:
                point = np.asarray(
                    selected["position_world_m"],
                    dtype=np.float64,
                )
                normal = np.asarray(
                    selected["normal_world"],
                    dtype=np.float64,
                )
                projection_points[f"{side}_contact"] = point.tolist()
                projection_points[f"{side}_normal_endpoint"] = (
                    point + 0.030 * normal
                ).tolist()
        camera_samples: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
        for view, camera_record in self.video_cameras.items():
            camera = camera_record["camera"]
            rgba = camera.get_rgba(device="cpu")
            if rgba is None:
                return False
            pixels = np.asarray(rgba)
            if pixels.size == 0:
                return False
            current = camera.get_current_frame(clone=True)
            rendering_time = float(current.get("rendering_time", -1.0))
            if rendering_time < 0.0:
                return False
            camera_samples[view] = (pixels, current)
        view_records: dict[str, dict[str, Any]] = {}
        rendering_times: list[float] = []
        for view, camera_record in self.video_cameras.items():
            camera = camera_record["camera"]
            pixels, current = camera_samples[view]
            if pixels.shape != (540, 960, 4):
                raise RuntimeError(
                    f"{view} unexpected RGBA shape {pixels.shape}"
                )
            if pixels.dtype != np.uint8:
                if not np.isfinite(pixels).all():
                    raise RuntimeError(f"{view} contains non-finite pixels")
                pixels = np.clip(pixels, 0, 255).astype(np.uint8)
            destination = (
                self._video_attempt_root
                / "frames"
                / view
                / f"{frame:06d}.png"
            ).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels, mode="RGBA").convert("RGB").save(
                destination
            )
            rendering_time = float(current.get("rendering_time", -1.0))
            if not math.isfinite(rendering_time):
                raise RuntimeError(
                    f"{view} has invalid rendering_time"
                )
            rendering_times.append(rendering_time)
            record = {
                "absolute_path": str(destination),
                "sha256": sha256_file(destination),
                "resolution": [960, 540],
                "physics_frame": frame,
                "time_s": float(pending["time_s"]),
                "runtime_signature": None,
                "rendering_time_s": rendering_time,
                "rendering_frame": self._jsonable_render_value(
                    current.get("rendering_frame")
                ),
                "camera_prim_path": str(camera.prim_path),
                "render_product_path": str(
                    camera.get_render_product_path()
                ),
                "camera_world_matrix": camera_record[
                    "camera_world_matrix"
                ],
                "camera_pose": {
                    "position_world_m": camera_record[
                        "position_world_m"
                    ],
                    "orientation_wxyz": camera_record[
                        "orientation_wxyz"
                    ],
                    "target_world_m": camera_record["target_world_m"],
                },
                "view_status": camera_record["view_status"],
                "camera_derivation": camera_record["derivation"],
                "projection_world_points": projection_points,
                "projection_pixels_xy": {
                    label: pixel.tolist()
                    for label, pixel in zip(
                        projection_points,
                        np.asarray(
                            camera.get_image_coords_from_world_points(
                                np.asarray(
                                    list(projection_points.values()),
                                    dtype=np.float64,
                                )
                            ),
                            dtype=np.float64,
                        ),
                        strict=True,
                    )
                    if np.isfinite(pixel).all()
                },
                "selected_contact_records": selected_contacts,
            }
            if view == "overview":
                framing = _full_arm_framing_evidence(
                    stage=self.stage,
                    camera=camera,
                    camera_world_matrix=camera_record[
                        "camera_world_matrix"
                    ],
                    resolution=(960, 540),
                    required_link_prims=FULL_ARM_LINK_PRIMS,
                    required_scene_prims=(
                        str(self.config["bottle"]["session_prim"]),
                        str(self.config["stage"]["table_prim"]),
                    ),
                )
                record["framing_evidence"] = {
                    **framing,
                    "required_full_arm_links_in_frame": list(
                        framing["projected_in_frame_links"]
                    ),
                    "occlusion_status": (
                        "PENDING_VISUAL_MODEL_REVIEW"
                    ),
                }
            view_records[view] = record
        if max(rendering_times) - min(rendering_times) > 1e-9:
            raise RuntimeError(
                "two camera products are not from the same render time: "
                f"{rendering_times}"
            )
        self._captured_frame_records.append(
            {
                "physics_frame": frame,
                "time_s": float(pending["time_s"]),
                "phase": str(pending["phase"]),
                "telemetry_index": int(pending["telemetry_index"]),
                "views": view_records,
            }
        )
        self._pending_capture_frames.pop(0)
        return True

    @property
    def video_capture_finalized(self) -> bool:
        return self._video_capture_finalized

    @property
    def has_pending_video_frame(self) -> bool:
        return bool(self._pending_capture_frames)

    def pending_requires_collider_evidence(self, *, terminal: bool) -> bool:
        """Read whether the newest physics sample is a required milestone."""

        if not self.capture_collider_evidence_enabled:
            return False
        if len(self._pending_capture_frames) != 1:
            return False
        pending = self._pending_capture_frames[0]
        telemetry = self.telemetry[int(pending["telemetry_index"])]
        return bool(
            required_collider_phase_labels(
                phase=str(pending["phase"]),
                terminal=terminal,
                observation=telemetry["observation"],
                contact=telemetry["contact_semantics"],
                captured=self._captured_overlay_phases,
            )
        )

    def discard_pending_video_frame(self) -> bool:
        """Drop one uncaptured frame in sparse collision-evidence mode."""

        if not self._pending_capture_frames:
            return False
        if len(self._pending_capture_frames) != 1:
            raise RuntimeError(
                "sparse collision capture fell behind physics: "
                f"{len(self._pending_capture_frames)} pending frames"
            )
        self._pending_capture_frames.pop(0)
        return True

    def capture_required_collider_evidence(
        self,
        *,
        terminal: bool,
    ) -> list[str]:
        """Capture exact-pose viewport overlays for required milestones."""

        if not self.capture_collider_evidence_enabled:
            return []
        if not self._captured_frame_records or self._video_attempt_root is None:
            return []
        frame_record = self._captured_frame_records[-1]
        telemetry = self.telemetry[int(frame_record["telemetry_index"])]
        phase = str(frame_record["phase"])
        observation = telemetry["observation"]
        contact = telemetry["contact_semantics"]
        labels = required_collider_phase_labels(
            phase=phase,
            terminal=terminal,
            observation=observation,
            contact=contact,
            captured=self._captured_overlay_phases,
        )
        if not labels:
            return []

        from omni.kit.viewport.utility import get_active_viewport

        from tools.validate_aloha1_task7b2_horizontal_grasp import _capture_viewport_png

        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("active viewport unavailable for collider capture")
        frame = int(frame_record["physics_frame"])
        captured: list[str] = []
        try:
            from pxr import UsdGeom

            bottle_state = self._read_bottle_state(self.bottle)
            self._update_bottle_render_evidence(
                self.stage,
                handles=self._bottle_render_handles,
                position_world=bottle_state["position_world_m"],
                orientation_world_wxyz=bottle_state[
                    "orientation_wxyz"
                ],
            )
            self._update_finger_render_evidence(
                self.stage,
                handles=self._finger_render_handles,
                link_transforms={
                    side: np.asarray(
                        view.get_transforms()[0],
                        dtype=np.float64,
                    ).tolist()
                    for side, view in self._finger_link_views.items()
                },
            )
            UsdGeom.Imageable(
                self.stage.GetPrimAtPath(
                    self._bottle_render_evidence["collider_root"]
                )
            ).MakeVisible()
            for side in self._finger_link_views:
                UsdGeom.Imageable(
                    self.stage.GetPrimAtPath(
                        f"{self._finger_render_evidence['root']}/"
                        f"{side}/AuthoredColliderAtPhysxPose"
                    )
                ).MakeVisible()
            self._settings.set_int(
                self._collider_display_setting,
                self._collider_display_value,
            )
            readback = int(
                self._settings.get(self._collider_display_setting) or 0
            )
            if readback != self._collider_display_value:
                raise RuntimeError(
                    "collider visualization setting readback mismatch"
                )
            for label in labels:
                for view, camera_record in self.video_cameras.items():
                    destination = (
                        self._video_attempt_root
                        / "collider_overlay"
                        / label
                        / f"{view}_raw.png"
                    ).resolve()
                    width, height = _capture_viewport_png(
                        self.app,
                        viewport,
                        camera_path=str(camera_record["camera"].prim_path),
                        destination=destination,
                    )
                    normal = Path(
                        frame_record["views"][view]["absolute_path"]
                    )
                    self._collider_overlay_records.append(
                        {
                            "phase_label": label,
                            "runtime_phase": phase,
                            "physics_frame": frame,
                            "time_s": float(frame_record["time_s"]),
                            "view": view,
                            "normal_absolute_path": str(normal),
                            "normal_sha256": sha256_file(normal),
                            "collider_overlay_absolute_path": str(
                                destination
                            ),
                            "collider_overlay_sha256": sha256_file(
                                destination
                            ),
                            "resolution": [width, height],
                            "camera_prim_path": str(
                                camera_record["camera"].prim_path
                            ),
                            "camera_world_matrix": camera_record[
                                "camera_world_matrix"
                            ],
                            "setting": {
                                "path": self._collider_display_setting,
                                "requested": (
                                    self._collider_display_value
                                ),
                                "readback": readback,
                            },
                            "render_evidence": {
                                "semantics": self._render_evidence[
                                    "semantics"
                                ],
                                "authored_geometry_clone": True,
                                "physics_schemas_copied": False,
                                "collision_schemas_copied": False,
                                "bottle_collider_mesh_count": (
                                    self._bottle_render_evidence[
                                        "collider_mesh_count"
                                    ]
                                ),
                                "finger_collider_mesh_count": sum(
                                    1
                                    for record in (
                                        self._finger_render_evidence[
                                            "records"
                                        ]
                                    )
                                    if record["category"]
                                    == "collider"
                                ),
                            },
                        }
                    )
                self._captured_overlay_phases.add(label)
                captured.append(label)
        finally:
            from pxr import UsdGeom

            UsdGeom.Imageable(
                self.stage.GetPrimAtPath(
                    self._bottle_render_evidence["collider_root"]
                )
            ).MakeInvisible()
            for side in self._finger_link_views:
                UsdGeom.Imageable(
                    self.stage.GetPrimAtPath(
                        f"{self._finger_render_evidence['root']}/"
                        f"{side}/AuthoredColliderAtPhysxPose"
                    )
                ).MakeInvisible()
            self._settings.set_int(
                self._collider_display_setting,
                self._collider_display_before,
            )
            for _ in range(COLLIDER_OVERLAY_RENDER_FLUSH_UPDATES):
                self.app.update()
            restored = int(
                self._settings.get(self._collider_display_setting) or 0
            )
            if restored != self._collider_display_before:
                raise RuntimeError(
                    "collider visualization setting was not restored"
                )
        return captured

    def finalize_video_capture(self) -> dict[str, Any]:
        """Bind frames to the terminal signature and build candidates."""

        if self._video_capture_finalized:
            candidate_path = (
                self._video_attempt_root / "video" / "candidate_manifest.json"
            )
            return json.loads(candidate_path.read_text(encoding="utf-8"))
        if self._pending_capture_frames:
            raise RuntimeError("cannot finalize with pending video frames")
        if self._video_attempt_root is None:
            raise RuntimeError("video attempt root is not initialized")
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        signature = str(report["deterministic_signature"])
        for record in self._captured_frame_records:
            for view_record in record["views"].values():
                view_record["runtime_signature"] = signature
        manifest = {
            "schema_version": 1,
            "runtime_signature": signature,
            "required_full_arm_links": [
                "base",
                "shoulder",
                "elbow",
                "forearm",
                "wrist",
                "gripper",
            ],
            "capture_api": {
                "class": "isaacsim.sensors.camera.Camera",
                "extension": "isaacsim.sensors.camera",
                "version_scope": "ISAAC_SIM_5_1_0_0_KIT_107_3_3",
                "independent_render_products": True,
                "capture_location": (
                    "OUTER_APP_LOOP_AFTER_PHYSICS_AND_RENDER_UPDATE"
                ),
            },
            "collision_evidence": {
                "enabled": self.capture_collider_evidence_enabled,
                "purpose": (
                    "PAIRED_COLLIDER_SCREENSHOT_REPEAT"
                    if self.capture_collider_evidence_enabled
                    else "PRIMARY_CLEAN_VIDEO"
                ),
                "setting_path": self._collider_display_setting,
                "setting_before": self._collider_display_before,
                "setting_after": int(
                    self._settings.get(
                        self._collider_display_setting
                    )
                    or 0
                ),
                "required_phase_labels": (
                    [
                        "RELEASE_DYNAMIC",
                        "OPEN_PREGRASP",
                        "BILATERAL_CONTACT",
                        "FIRST_SUPPORT_CLEARANCE",
                        "HEIGHT_REACHED",
                        "HOLD_END",
                    ]
                    if self.capture_collider_evidence_enabled
                    else []
                ),
                "captured_phase_labels": sorted(
                    self._captured_overlay_phases
                ),
                "render_evidence": self._render_evidence,
                "records": self._collider_overlay_records,
            },
            "frames": self._captured_frame_records,
        }
        manifest_path = (
            self._video_attempt_root / "frame_manifest.json"
        )
        _atomic_json(manifest_path, manifest)
        from tools.build_aloha1_grasp_20cm_video import build_video_evidence

        candidate = build_video_evidence(
            report_path=self.report_path,
            telemetry_path=self.telemetry_path,
            frame_manifest_path=manifest_path,
            output_root=self._video_attempt_root / "video",
        )
        self._video_capture_finalized = True
        return candidate

    def finalize_collision_evidence_capture(self) -> dict[str, Any]:
        """Finalize a sparse deterministic repeat without encoding video."""

        if self._video_capture_finalized:
            candidate_path = (
                self._video_attempt_root / "video" / "candidate_manifest.json"
            )
            return json.loads(candidate_path.read_text(encoding="utf-8"))
        if self._pending_capture_frames:
            raise RuntimeError("cannot finalize with pending sparse frame")
        if self._video_attempt_root is None:
            raise RuntimeError("video attempt root is not initialized")
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        signature = str(report["deterministic_signature"])
        for record in self._captured_frame_records:
            for view_record in record["views"].values():
                view_record["runtime_signature"] = signature
        required_labels = [
            "RELEASE_DYNAMIC",
            "OPEN_PREGRASP",
            "BILATERAL_CONTACT",
            "FIRST_SUPPORT_CLEARANCE",
            "HEIGHT_REACHED",
            "HOLD_END",
        ]
        manifest = {
            "schema_version": 1,
            "runtime_signature": signature,
            "required_full_arm_links": [
                "base",
                "shoulder",
                "elbow",
                "forearm",
                "wrist",
                "gripper",
            ],
            "capture_api": {
                "class": "isaacsim.sensors.camera.Camera",
                "extension": "isaacsim.sensors.camera",
                "version_scope": "ISAAC_SIM_5_1_0_0_KIT_107_3_3",
                "independent_render_products": True,
                "capture_location": "SPARSE_REQUIRED_PHYSICS_MILESTONES_ONLY",
            },
            "collision_evidence": {
                "enabled": True,
                "purpose": "PAIRED_COLLIDER_SCREENSHOT_REPEAT",
                "setting_path": self._collider_display_setting,
                "setting_before": self._collider_display_before,
                "setting_after": int(
                    self._settings.get(self._collider_display_setting) or 0
                ),
                "required_phase_labels": required_labels,
                "captured_phase_labels": sorted(
                    self._captured_overlay_phases
                ),
                "render_evidence": self._render_evidence,
                "records": self._collider_overlay_records,
            },
            "frames": self._captured_frame_records,
        }
        manifest_path = self._video_attempt_root / "frame_manifest.json"
        _atomic_json(manifest_path, manifest)
        from tools.build_aloha1_grasp_20cm_video import annotate_collision_evidence

        output_root = self._video_attempt_root / "video"
        collision = annotate_collision_evidence(
            manifest=manifest,
            report=report,
            output_dir=output_root / "collision_annotated",
        )
        candidate = {
            "schema_version": 1,
            "status": collision["status"],
            "promotion_status": collision["status"],
            "runtime_signature": signature,
            "videos": [],
            "collision_evidence": collision,
            "capture_mode": "SPARSE_COLLISION_EVIDENCE_ONLY",
            "source_report": {
                "absolute_path": str(self.report_path.resolve()),
                "sha256": sha256_file(self.report_path),
            },
            "source_telemetry": {
                "absolute_path": str(self.telemetry_path.resolve()),
                "sha256": sha256_file(self.telemetry_path),
            },
            "source_frame_manifest": {
                "absolute_path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
            },
            "task8": "NOT_RUN",
        }
        candidate_path = output_root / "candidate_manifest.json"
        _atomic_json(candidate_path, candidate)
        self._video_capture_finalized = True
        return candidate

    def save_video_capture_exception(self, exception_text: str) -> None:
        self._video_capture_error = exception_text[-12000:]
        root = self._video_attempt_root or self.artifact_root
        _atomic_json(
            root / "video_capture_error.json",
            {
                "schema_version": 1,
                "status": "FAIL",
                "reason": "video_capture_exception",
                "exception": self._video_capture_error,
                "captured_frame_count": len(
                    self._captured_frame_records
                ),
                "pending_frame_count": len(
                    self._pending_capture_frames
                ),
                "machine_report_preserved": str(self.report_path),
                "task8": "NOT_RUN",
            },
        )

    def ui_snapshot(self) -> dict[str, Any]:
        return dict(self._last_snapshot)

    def abort_reset_snapshot(self, *, phase: str) -> dict[str, Any]:
        """Read the minimal mutation audit state for Abort/Reset."""

        from pxr import UsdPhysics

        kinematic = bool(
            UsdPhysics.RigidBodyAPI(self.bottle_prim)
            .GetKinematicEnabledAttr()
            .Get()
        )
        return {
            "phase": str(phase),
            "target_write_count": int(self._target_write_count),
            "telemetry_count": len(self.telemetry),
            "joint_target": self.command.tolist(),
            "bottle_kinematic_enabled": kinematic,
            "stage_sha256": sha256_file(self.stage_path),
        }

    def _terminal_metrics(self, phase: Phase) -> dict[str, Any]:
        dynamic_formal = formal_phase_bottle_dynamic(
            self.observations,
            self.telemetry,
        )
        hold_samples = [
            item
            for item in self.observations
            if self.telemetry[item.frame - 1]["phase"] == Phase.HOLD.value
        ]
        hold_records = [
            self.telemetry[item.frame - 1]
            for item in hold_samples
        ]
        hold_duration = physics_sample_duration_s(
            sample_count=len(hold_samples),
            physics_dt_s=self.dt,
        )
        return {
            "status": phase.value,
            "aborted": phase is Phase.ABORTED,
            "forbidden_constraint": False,
            "finite_state": all(
                item.finite_state for item in self.observations
            ),
            "persistent_penetration": any(
                item.persistent_penetration
                for item in self.observations
            ),
            "numerical_ejection": any(
                item.numerical_ejection for item in self.observations
            ),
            "dynamic_during_formal_phases": dynamic_formal,
            "bilateral_contact_before_lift": (
                self._bilateral_before_lift
            ),
            "bilateral_contact_through_hold": (
                self._bilateral_through_hold
            ),
            "height_reached": self._height_reached,
            "maximum_clearance_m": self._maximum_clearance_m,
            "hold_duration_s": hold_duration,
            "hold_physics_frame_count": len(hold_samples),
            "hold_bilateral_geometric_frame_count": sum(
                bool(
                    record["contact_semantics"][
                        "bilateral_geometric_contact"
                    ]
                )
                for record in hold_records
            ),
            "hold_bilateral_solver_active_frame_count": sum(
                bool(
                    record["contact_semantics"][
                        "bilateral_solver_active_contact"
                    ]
                )
                for record in hold_records
            ),
            "hold_drop_m": float(
                self._last_snapshot["hold_drop_m"]
            ),
            "ee_vertical_displacement_m": (
                self.observations[-1].ee_vertical_displacement_m
                if self.observations
                else 0.0
            ),
        }

    def finalize_run(self, phase: Phase, reason: str) -> None:
        metrics = self._terminal_metrics(phase)
        signature = canonical_run_signature(
            self.observations,
            metrics,
        )
        report = {
            "schema_version": 1,
            "status": phase.value,
            "reason": reason,
            "classification": (
                "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
            ),
            "runtime": {
                "isaac_sim": "5.1.0.0",
                "kit": "107.3.3",
                "physx": "107.3.26",
                "delegate": self.delegate_readback,
                "session_sublayer_application": self.profile.get(
                    "session_sublayer_application",
                    {
                        "status": "NOT_APPLIED",
                        "inserted_paths": [],
                        "root_layer_saved": False,
                    },
                ),
                "solve_articulation_contact_last": bool(
                    self.physics_context
                    .get_solve_articulation_contact_last()
                ),
                "dof_order": list(self.articulation.dof_names),
                "ik": self._ik_report,
                "coupling": self.coupling_readback,
                "finger_drive": self.drive_readback,
                "initialization_contract": self.initialization_contract,
                "finger_safety": {
                    "status": (
                        "PASS"
                        if self._finger_safety_first_violation is None
                        else "FAIL"
                    ),
                    "violation_count": sum(
                        record["status"] == "FAIL"
                        for record in self._finger_safety_records
                    ),
                    "first_violation": self._finger_safety_first_violation,
                    "classified_environment_contact_count": (
                        self._finger_environment_contact_count
                    ),
                    "abort_on_first_runtime_violation": (
                        self.abort_on_first_runtime_violation
                    ),
                    "frame_count": len(self._finger_safety_records),
                },
                "initial_pose": self.initial_pose_evidence,
                "arm_phase_readback_tolerance_rad": (
                    self.arm_phase_readback_tolerance_rad
                ),
                "trajectory": {
                    "arm_trajectory_mode": self.arm_trajectory_mode,
                    "phase_audits": self._trajectory_audit,
                    "arm_acceleration_limits_rad_s2": (
                        None
                        if self.arm_acceleration_limits_rad_s2 is None
                        else self.arm_acceleration_limits_rad_s2.tolist()
                    ),
                    "jerk_limit_status": (
                        "NOT_SET_NO_EXACT_MODEL_OFFICIAL_VALUE"
                        if self.arm_trajectory_mode
                        == "LULA_CSPACE_ACCELERATION_LIMITED"
                        else "NOT_APPLICABLE_LEGACY_BASELINE"
                    ),
                    "additional_lift_margin_m": (
                        self.additional_lift_margin_m
                    ),
                    "classification": (
                        "DIAGNOSTIC_CALCULATED_RUNTIME_CLEARANCE_"
                        "COMPENSATION"
                        if self.additional_lift_margin_m > 0.0
                        else "BASELINE_ZERO_ADDITIONAL_MARGIN"
                    ),
                },
                "bottle_tensor_lifecycle": (
                    self.bottle_tensor_lifecycle
                ),
                "bottle_velocity_sampling": {
                    "callback_phase": "POST_PHYSICS_STEP",
                    "sampling_phase": "POST_PHYSICS_STEP",
                    "physics_dt_s": float(
                        self.physics_context.get_physics_dt()
                    ),
                    "subscription_pre_step_argument": False,
                },
            },
            "stage": {
                "absolute_path": str(self.stage_path),
                "sha256_before": self.stage_hash_before,
                "sha256_after": sha256_file(self.stage_path),
                "root_prim": str(
                    self.stage.GetDefaultPrim().GetPath()
                ),
                "sublayers": list(
                    self.stage.GetRootLayer().subLayerPaths
                ),
                "session_only": True,
            },
            "bottle": self.bottle_session,
            "bottle_random_position": {
                "offset_xy_m": self.bottle_xy_offset_m,
                "pose_mode": self.bottle_pose_mode,
                "world_from_object": (
                    self.bottle_world_from_object.tolist()
                ),
                "changed_variable": (
                    "BOTTLE_INITIAL_WORLD_XY_TRANSLATION_ONLY"
                    if self.bottle_pose_mode
                    == "LEGACY_WORLD_XY_TRANSLATION_ONLY"
                    else "BOTTLE_GEOMETRIC_CENTER_XY_AND_WORLD_Z_YAW"
                ),
                "rotation_unchanged": (
                    self.bottle_pose_mode
                    == "LEGACY_WORLD_XY_TRANSLATION_ONLY"
                ),
                "object_from_gripper_unchanged": True,
            },
            "table_top_z_m": self.table_top_z_m,
            "target_clearance_m": float(
                self.config["target"]["clearance_m"]
            ),
            "metrics": metrics,
            "deterministic_signature": signature,
            "telemetry_absolute_path": str(self.telemetry_path),
            "runtime_seconds": time.perf_counter() - self.started_at,
            "boundaries": {
                "real_robot": False,
                "remote_103": False,
                "surface_gripper": False,
                "fixed_joint": False,
                "parent_attachment": False,
                "source_stage_modified": False,
                "final_collider_modified": False,
                "task8": "NOT_RUN",
            },
        }
        if report["stage"]["sha256_after"] != self.stage_hash_before:
            report["status"] = "FAIL"
            report["reason"] = "approved_stage_hash_changed"
        _atomic_json(self.report_path, report)
        self.telemetry_path.write_text(
            "".join(
                json.dumps(record, sort_keys=True) + "\n"
                for record in self.telemetry
            ),
            encoding="utf-8",
        )

    def save_exception(self, exception_text: str) -> None:
        _atomic_json(
            self.report_path,
            {
                "schema_version": 1,
                "status": "FAIL",
                "reason": "exception",
                "exception": exception_text[-12000:],
                "runtime": {
                    "initialization_contract": getattr(
                        self,
                        "initialization_contract",
                        {"status": "NOT_RUN"},
                    ),
                    "finger_safety": {
                        "status": (
                            "FAIL"
                            if getattr(
                                self,
                                "_finger_safety_first_violation",
                                None,
                            )
                            is not None
                            else "NOT_RUN"
                        ),
                        "violation_count": sum(
                            record["status"] == "FAIL"
                            for record in getattr(
                                self,
                                "_finger_safety_records",
                                [],
                            )
                        ),
                        "first_violation": getattr(
                            self,
                            "_finger_safety_first_violation",
                            None,
                        ),
                    },
                },
                "stage": {
                    "absolute_path": str(self.stage_path),
                    "sha256_before": self.stage_hash_before,
                    "sha256_after": sha256_file(self.stage_path),
                },
                "boundaries": {
                    "real_robot": False,
                    "remote_103": False,
                    "surface_gripper": False,
                    "fixed_joint": False,
                    "parent_attachment": False,
                    "source_stage_modified": False,
                    "final_collider_modified": False,
                    "task8": "NOT_RUN",
                },
            },
        )

    def reset_session(self) -> None:
        from pxr import UsdPhysics

        self.world.pause()
        rigid = UsdPhysics.RigidBodyAPI(self.bottle_prim)
        kinematic_attr = rigid.GetKinematicEnabledAttr()
        initially_kinematic = bool(kinematic_attr.Get())
        operation_plan = reset_body_transition_plan(
            initially_kinematic=initially_kinematic
        )
        if operation_plan[0] == "set_dynamic":
            kinematic_attr.Set(False)
            self._physx_sim.flush_changes()
        bottle_indices = single_body_tensor_indices(
            count=int(self.bottle.count)
        )
        self.bottle.set_transforms(
            self._initial_bottle_transform.reshape(1, 7),
            bottle_indices,
        )
        self.bottle.set_velocities(
            np.zeros((1, 6), dtype=np.float32),
            bottle_indices,
        )
        kinematic_attr.Set(True)
        self._physx_sim.flush_changes()
        self.command = self.initial_command.copy()
        self.command_velocity = np.zeros_like(self.command)
        self.articulation.set_joint_positions(self.command)
        self.articulation.set_joint_velocities(
            np.zeros_like(self.command)
        )
        self._write_joint_command()
        self._contact_buffer = []
        self.all_contacts = []
        self._reset_runtime_records()
