"""Pure COM-frame and sampling-time analysis for Bottle500 telemetry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

ALLOWED_RESULTS = {
    "COM_FRAME_SEMANTICS_EXPLAINS_DISAGREEMENT",
    "VERIFIED_LOCAL_PHYSX_VELOCITY_TRANSFORM_DISAGREEMENT",
    "SAMPLING_TIME_MISMATCH",
    "INCONCLUSIVE",
}


def isolated_control_profile(variant: str) -> dict[str, Any]:
    """Return the frozen one-variable V1/V2 no-contact command."""

    commands = {
        "V1": {
            "description": "NO_CONTACT_PURE_TRANSLATION",
            "linear_velocity_com_world_m_s": [0.12, -0.08, 0.05],
            "angular_velocity_world_rad_s": [0.0, 0.0, 0.0],
        },
        "V2": {
            "description": "NO_CONTACT_PURE_ROTATION",
            "linear_velocity_com_world_m_s": [0.0, 0.0, 0.0],
            "angular_velocity_world_rad_s": [0.0, 2.0, 0.0],
        },
    }
    try:
        command = commands[str(variant)]
    except KeyError as exc:
        raise ValueError(f"unsupported isolated variant: {variant}") from exc
    return {
        "variant": str(variant),
        **command,
        "gravity_enabled": False,
        "collisions_enabled": False,
        "linear_damping": 0.0,
        "angular_damping": 0.0,
        "preserve_authored_center_of_mass": True,
        "source_asset_modified": False,
        "task8": "NOT_RUN",
    }


def _vector(value: Sequence[float], *, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {length}-vector")
    return array


def rotate_local_vector_wxyz(
    vector: Sequence[float],
    orientation_wxyz: Sequence[float],
) -> np.ndarray:
    """Rotate a local vector into world coordinates."""

    value = _vector(vector, length=3, name="vector")
    quaternion = _vector(
        orientation_wxyz,
        length=4,
        name="orientation_wxyz",
    )
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise ValueError("orientation_wxyz must be nonzero")
    quaternion /= norm
    scalar = float(quaternion[0])
    imaginary = quaternion[1:]
    twice_cross = 2.0 * np.cross(imaginary, value)
    return value + scalar * twice_cross + np.cross(
        imaginary,
        twice_cross,
    )


def build_sample(
    *,
    step_index: int,
    dt_s: float,
    actor_prim_path: str,
    tensor_index: int,
    actor_position_world_m: Sequence[float],
    actor_orientation_world_wxyz: Sequence[float],
    center_of_mass_local_m: Sequence[float],
    linear_velocity_com_world_m_s: Sequence[float],
    angular_velocity_world_rad_s: Sequence[float],
    state_boundary_index: int | None = None,
    sampling_phase: str = "PRE_PHYSICS_STEP",
) -> dict[str, Any]:
    """Normalize one physics-boundary rigid-body sample."""

    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    boundary_index = (
        int(step_index)
        if state_boundary_index is None
        else int(state_boundary_index)
    )
    if (
        int(step_index) < 0
        or int(tensor_index) < 0
        or boundary_index < 0
    ):
        raise ValueError("step and tensor indices must be non-negative")
    if not str(actor_prim_path).startswith("/"):
        raise ValueError("actor_prim_path must be absolute")
    phase = str(sampling_phase)
    if phase not in {"PRE_PHYSICS_STEP", "POST_PHYSICS_STEP"}:
        raise ValueError("unsupported sampling_phase")
    origin = _vector(
        actor_position_world_m,
        length=3,
        name="actor_position_world_m",
    )
    orientation = _vector(
        actor_orientation_world_wxyz,
        length=4,
        name="actor_orientation_world_wxyz",
    )
    com_local = _vector(
        center_of_mass_local_m,
        length=3,
        name="center_of_mass_local_m",
    )
    velocity_com = _vector(
        linear_velocity_com_world_m_s,
        length=3,
        name="linear_velocity_com_world_m_s",
    )
    angular = _vector(
        angular_velocity_world_rad_s,
        length=3,
        name="angular_velocity_world_rad_s",
    )
    com_offset_world = rotate_local_vector_wxyz(com_local, orientation)
    com_world = origin + com_offset_world
    velocity_origin_prediction = velocity_com - np.cross(
        angular,
        com_offset_world,
    )
    return {
        "physics_step_index": int(step_index),
        "physics_dt_s": dt,
        "sampling_phase": phase,
        "state_boundary_index": boundary_index,
        "actor_prim_path": str(actor_prim_path),
        "tensor_index": int(tensor_index),
        "p_O_world_m": origin.tolist(),
        "R_O_world_wxyz": orientation.tolist(),
        "r_OC_local_m": com_local.tolist(),
        "r_OC_world_m": com_offset_world.tolist(),
        "p_C_world_m": com_world.tolist(),
        "v_C_world_m_s": velocity_com.tolist(),
        "omega_world_rad_s": angular.tolist(),
        "v_O_pred_world_m_s": velocity_origin_prediction.tolist(),
    }


def _error_metrics(error: np.ndarray) -> dict[str, Any]:
    norms = np.linalg.norm(error, axis=1)
    return {
        "rmse_m_s": float(np.sqrt(np.mean(np.square(error)))),
        "max_error_m_s": float(np.max(norms)),
        "signed_mean_error_m_s": np.mean(error, axis=0).tolist(),
        "component_rmse_m_s": np.sqrt(
            np.mean(np.square(error), axis=0)
        ).tolist(),
        "component_max_abs_error_m_s": np.max(
            np.abs(error), axis=0
        ).tolist(),
    }


def analyze_samples(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare pre-step velocity with forward, backward and midpoint FD."""

    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    paths = {str(item["actor_prim_path"]) for item in samples}
    indices = {int(item["tensor_index"]) for item in samples}
    phases = {str(item["sampling_phase"]) for item in samples}
    if len(paths) != 1 or len(indices) != 1:
        raise ValueError("actor path/index changed within run")
    if len(phases) != 1 or not phases.issubset(
        {"PRE_PHYSICS_STEP", "POST_PHYSICS_STEP"}
    ):
        raise ValueError("samples must use one supported sampling phase")
    dt = np.asarray(
        [float(item["physics_dt_s"]) for item in samples[:-1]],
        dtype=np.float64,
    )
    if not np.isfinite(dt).all() or np.any(dt <= 0.0):
        raise ValueError("invalid per-step dt")
    p_origin = np.asarray(
        [item["p_O_world_m"] for item in samples],
        dtype=np.float64,
    )
    p_com = np.asarray(
        [item["p_C_world_m"] for item in samples],
        dtype=np.float64,
    )
    v_com = np.asarray(
        [item["v_C_world_m_s"] for item in samples],
        dtype=np.float64,
    )
    v_origin = np.asarray(
        [item["v_O_pred_world_m_s"] for item in samples],
        dtype=np.float64,
    )
    omega = np.asarray(
        [item["omega_world_rad_s"] for item in samples],
        dtype=np.float64,
    )
    arrays = (p_origin, p_com, v_com, v_origin, omega)
    if any(array.shape != (len(samples), 3) for array in arrays):
        raise ValueError("sample vectors have inconsistent shapes")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("sample vectors must be finite")

    fd_origin = np.diff(p_origin, axis=0) / dt[:, None]
    fd_com = np.diff(p_com, axis=0) / dt[:, None]
    com_forward_error = fd_com - v_com[:-1]
    com_backward_error = fd_com - v_com[1:]
    com_midpoint_error = fd_com - 0.5 * (v_com[:-1] + v_com[1:])
    origin_forward_error = fd_origin - v_origin[:-1]
    origin_backward_error = fd_origin - v_origin[1:]
    origin_midpoint_error = fd_origin - 0.5 * (
        v_origin[:-1] + v_origin[1:]
    )
    origin_forward_vs_com_error = fd_origin - v_com[:-1]
    origin_backward_vs_com_error = fd_origin - v_com[1:]
    origin_midpoint_vs_com_error = fd_origin - 0.5 * (
        v_com[:-1] + v_com[1:]
    )
    signed_integral = np.sum(v_com[:-1] * dt[:, None], axis=0)
    speed = np.linalg.norm(v_com, axis=1)
    position_norm = np.maximum(
        np.linalg.norm(p_origin, axis=1),
        np.linalg.norm(p_com, axis=1),
    )
    return {
        "actor_prim_path": next(iter(paths)),
        "tensor_index": next(iter(indices)),
        "sampling_phase": next(iter(phases)),
        "sample_count": len(samples),
        "transition_count": len(samples) - 1,
        "dt_s": {
            "minimum": float(np.min(dt)),
            "maximum": float(np.max(dt)),
            "mean": float(np.mean(dt)),
        },
        "vx_min_m_s": float(np.min(v_com[:-1, 0])),
        "vx_max_m_s": float(np.max(v_com[:-1, 0])),
        "vy_min_m_s": float(np.min(v_com[:-1, 1])),
        "vy_max_m_s": float(np.max(v_com[:-1, 1])),
        "vz_min_m_s": float(np.min(v_com[:-1, 2])),
        "vz_max_m_s": float(np.max(v_com[:-1, 2])),
        "signed_velocity_mean_m_s": np.mean(v_com[:-1], axis=0).tolist(),
        "signed_vz_mean_m_s": float(np.mean(v_com[:-1, 2])),
        "mean_abs_velocity_m_s": np.mean(
            np.abs(v_com[:-1]), axis=0
        ).tolist(),
        "mean_abs_vz_m_s": float(np.mean(np.abs(v_com[:-1, 2]))),
        "signed_velocity_integral_vector_m": signed_integral.tolist(),
        "signed_velocity_integral_m": float(signed_integral[2]),
        "actor_origin_delta_m": (p_origin[-1] - p_origin[0]).tolist(),
        "com_delta_m": (p_com[-1] - p_com[0]).tolist(),
        "com_forward_fd_vs_velocity": _error_metrics(com_forward_error),
        "com_backward_fd_vs_velocity": _error_metrics(com_backward_error),
        "com_midpoint_fd_vs_velocity": _error_metrics(com_midpoint_error),
        "actor_origin_forward_fd_vs_prediction": _error_metrics(
            origin_forward_error
        ),
        "actor_origin_backward_fd_vs_prediction": _error_metrics(
            origin_backward_error
        ),
        "actor_origin_midpoint_fd_vs_prediction": _error_metrics(
            origin_midpoint_error
        ),
        "actor_origin_forward_fd_vs_com_velocity_uncorrected": (
            _error_metrics(origin_forward_vs_com_error)
        ),
        "actor_origin_backward_fd_vs_com_velocity_uncorrected": (
            _error_metrics(origin_backward_vs_com_error)
        ),
        "actor_origin_midpoint_fd_vs_com_velocity_uncorrected": (
            _error_metrics(origin_midpoint_vs_com_error)
        ),
        "maximum_com_speed_m_s": float(np.max(speed)),
        "maximum_angular_speed_rad_s": float(
            np.max(np.linalg.norm(omega, axis=1))
        ),
        "maximum_position_norm_m": float(np.max(position_norm)),
    }


def evaluate_post_step_alignment(
    metrics: Mapping[str, Any],
    *,
    velocity_tolerance_m_s: float,
) -> dict[str, Any]:
    """Evaluate actual PhysX post-step sampling without hiding shifts."""

    tolerance = float(velocity_tolerance_m_s)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("velocity_tolerance_m_s must be finite and positive")

    def passes(key: str) -> bool:
        return float(metrics[key]["max_error_m_s"]) <= tolerance

    com_backward = passes("com_backward_fd_vs_velocity")
    com_forward = passes("com_forward_fd_vs_velocity")
    com_midpoint = passes("com_midpoint_fd_vs_velocity")
    origin_backward = passes(
        "actor_origin_backward_fd_vs_prediction"
    )
    uncorrected_origin_backward = passes(
        "actor_origin_backward_fd_vs_com_velocity_uncorrected"
    )
    return {
        "sampling_phase": str(metrics["sampling_phase"]),
        "velocity_tolerance_m_s": tolerance,
        "declared_post_step_backward_alignment": com_backward,
        "shifted_forward_alignment": com_forward,
        "midpoint_alignment": com_midpoint,
        "actor_origin_prediction_backward_alignment": origin_backward,
        "uncorrected_actor_origin_backward_alignment": (
            uncorrected_origin_backward
        ),
        "com_frame_explains_origin_disagreement": bool(
            com_backward
            and origin_backward
            and not uncorrected_origin_backward
        ),
        "definition": {
            "declared": (
                "POST_STEP v[k] compared with (p[k]-p[k-1])/dt"
            ),
            "shifted": (
                "POST_STEP v[k] compared with (p[k+1]-p[k])/dt"
            ),
            "midpoint": (
                "finite difference compared with 0.5*(v[k]+v[k+1])"
            ),
        },
    }


def build_velocity_diagnosis(
    *,
    v1_metrics: Mapping[str, Any],
    v2_metrics: Mapping[str, Any],
    v3_metrics: Mapping[str, Any],
    v1_runtime_valid: bool,
    v2_runtime_valid: bool,
    v3_signature_unchanged: bool,
    dt_s: float,
) -> dict[str, Any]:
    """Combine V1/V2 numerical baselines with the contact-rich V3 run."""

    tolerance = derive_baseline_tolerance(
        v1=v1_metrics,
        v2=v2_metrics,
        dt_s=dt_s,
    )
    threshold = float(tolerance["velocity_tolerance_m_s"])
    v1_alignment = evaluate_post_step_alignment(
        v1_metrics,
        velocity_tolerance_m_s=threshold,
    )
    v2_alignment = evaluate_post_step_alignment(
        v2_metrics,
        velocity_tolerance_m_s=threshold,
    )
    v3_alignment = evaluate_post_step_alignment(
        v3_metrics,
        velocity_tolerance_m_s=threshold,
    )
    v1_pass = bool(
        v1_runtime_valid
        and v1_alignment["declared_post_step_backward_alignment"]
        and v1_alignment["actor_origin_prediction_backward_alignment"]
    )
    v2_pass = bool(
        v2_runtime_valid
        and v2_alignment["declared_post_step_backward_alignment"]
        and v2_alignment["actor_origin_prediction_backward_alignment"]
    )
    conclusion = classify_velocity_result(
        v1_pass=v1_pass,
        v2_pass=v2_pass,
        v3_current_alignment=bool(
            v3_alignment["declared_post_step_backward_alignment"]
        ),
        v3_shifted_alignment=bool(
            v3_alignment["shifted_forward_alignment"]
        ),
        v3_com_frame_explains=bool(
            v3_alignment["com_frame_explains_origin_disagreement"]
        ),
    )
    gates = {
        "v1_control_pass": v1_pass,
        "v2_control_pass": v2_pass,
        "v3_physical_signature_unchanged": bool(
            v3_signature_unchanged
        ),
        "four_choice_conclusion_reached": conclusion != "INCONCLUSIVE",
    }
    return {
        "status": "PASS" if all(gates.values()) else "PARTIAL",
        "conclusion": conclusion,
        "tolerance": tolerance,
        "alignments": {
            "V1": v1_alignment,
            "V2": v2_alignment,
            "V3": v3_alignment,
        },
        "gates": gates,
    }


def derive_baseline_tolerance(
    *,
    v1: Mapping[str, Any],
    v2: Mapping[str, Any],
    dt_s: float,
) -> dict[str, Any]:
    """Derive a velocity tolerance from measured controls and float32 ULP."""

    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    metric_keys = (
        "com_forward_fd_vs_velocity",
        "com_backward_fd_vs_velocity",
        "com_midpoint_fd_vs_velocity",
        "actor_origin_forward_fd_vs_prediction",
        "actor_origin_backward_fd_vs_prediction",
        "actor_origin_midpoint_fd_vs_prediction",
    )
    measured = max(
        float(run[key]["max_error_m_s"])
        for run in (v1, v2)
        for key in metric_keys
    )
    characteristic_position = max(
        float(v1["maximum_position_norm_m"]),
        float(v2["maximum_position_norm_m"]),
    )
    float32_position_ulp_per_dt = (
        float(np.finfo(np.float32).eps)
        * characteristic_position
        / dt
    )
    tolerance = measured + float32_position_ulp_per_dt
    return {
        "velocity_tolerance_m_s": tolerance,
        "measured_baseline_max_error_m_s": measured,
        "float32_position_ulp_per_dt_m_s": (
            float32_position_ulp_per_dt
        ),
        "characteristic_position_m": characteristic_position,
        "physics_dt_s": dt,
        "source": "V1_V2_BASELINE_PLUS_FLOAT32_POSITION_ULP_PER_DT",
        "classification": "NUMERICAL_DIAGNOSTIC_NOT_HARDWARE_CALIBRATION",
    }


def classify_velocity_result(
    *,
    v1_pass: bool,
    v2_pass: bool,
    v3_current_alignment: bool,
    v3_shifted_alignment: bool,
    v3_com_frame_explains: bool,
) -> str:
    """Return only one of the four user-approved velocity conclusions."""

    if not (v1_pass and v2_pass):
        return "INCONCLUSIVE"
    if v3_com_frame_explains:
        return "COM_FRAME_SEMANTICS_EXPLAINS_DISAGREEMENT"
    if not v3_current_alignment and v3_shifted_alignment:
        return "SAMPLING_TIME_MISMATCH"
    if not v3_current_alignment and not v3_shifted_alignment:
        return "VERIFIED_LOCAL_PHYSX_VELOCITY_TRANSFORM_DISAGREEMENT"
    return "INCONCLUSIVE"
