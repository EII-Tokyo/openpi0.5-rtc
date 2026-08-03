"""Pure helpers for the predeclared Isaac 5.1 numerical-convergence study."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def build_predeclared_convergence_plan() -> dict[str, Any]:
    return {
        "frequency_sweep": [
            {
                "frequency_hz": frequency_hz,
                "position_iterations": 64,
                "velocity_iterations": 8,
            }
            for frequency_hz in (60, 120, 240, 480)
        ],
        "position_iteration_sweep": [4, 8, 16, 32, 64],
        "velocity_iteration_sweep": [1, 2, 4, 8],
        "optional_frequency_hz": 960,
        "selection_order": [
            "frequency_sweep",
            "position_iteration_sweep",
            "velocity_iteration_sweep",
        ],
    }


def validate_numerical_override(
    *,
    frequency_hz: float,
    position_iterations: int,
    velocity_iterations: int,
) -> dict[str, float | int]:
    frequency = float(frequency_hz)
    position = int(position_iterations)
    velocity = int(velocity_iterations)
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency_hz must be finite and positive")
    if position < 1:
        raise ValueError("position_iterations must be positive")
    if velocity < 1:
        raise ValueError("velocity_iterations must be positive")
    return {
        "frequency_hz": frequency,
        "physics_dt_s": 1.0 / frequency,
        "position_iterations": position,
        "velocity_iterations": velocity,
    }


def scaled_frame_count(
    *,
    base_frames: int,
    frequency_hz: float,
    baseline_frequency_hz: float = 60.0,
) -> int:
    """Scale a frame-count gate while preserving its physical duration."""

    frames = int(base_frames)
    frequency = float(frequency_hz)
    baseline = float(baseline_frequency_hz)
    if frames < 1:
        raise ValueError("base_frames must be positive")
    if (
        not math.isfinite(frequency)
        or frequency <= 0.0
        or not math.isfinite(baseline)
        or baseline <= 0.0
    ):
        raise ValueError("frequencies must be finite and positive")
    return max(1, round(frames * frequency / baseline))


def physical_model_signature(config: dict[str, Any]) -> str:
    physical = copy.deepcopy(config)
    physics = physical.get("physics")
    if isinstance(physics, dict):
        physics.pop("frequency_hz", None)
    encoded = json.dumps(
        physical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def apply_articulation_solver_override(
    articulation: Any,
    *,
    position_iterations: int,
    velocity_iterations: int,
) -> dict[str, int | str]:
    """Apply Isaac 5.1 SingleArticulation solver counts and verify readback."""

    position = int(position_iterations)
    velocity = int(velocity_iterations)
    if position < 1 or velocity < 1:
        raise ValueError("solver iteration counts must be positive")
    articulation.set_solver_position_iteration_count(position)
    articulation.set_solver_velocity_iteration_count(velocity)
    effective_position = int(
        articulation.get_solver_position_iteration_count()
    )
    effective_velocity = int(
        articulation.get_solver_velocity_iteration_count()
    )
    if (effective_position, effective_velocity) != (position, velocity):
        raise RuntimeError(
            "solver iteration readback mismatch: "
            f"requested=({position}, {velocity}) "
            f"effective=({effective_position}, {effective_velocity})"
        )
    return {
        "requested_position_iterations": position,
        "requested_velocity_iterations": velocity,
        "effective_position_iterations": effective_position,
        "effective_velocity_iterations": effective_velocity,
        "readback_status": "PASS",
    }


def _finite_max(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return max(finite) if finite else None


def extract_runtime_cell_metrics(
    *,
    report: dict[str, Any],
    telemetry: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reduce one frozen-model runtime trace without discarding signs."""

    if not telemetry:
        raise ValueError("telemetry must not be empty")
    numerical = report.get("runtime", {}).get("numerical_convergence", {})
    readback = numerical.get("readback", {})
    contact_totals = {
        "left_signed_normal_impulse_ns": 0.0,
        "right_signed_normal_impulse_ns": 0.0,
        "left_scalar_impulse_ns": 0.0,
        "right_scalar_impulse_ns": 0.0,
    }
    separations: list[float] = []
    bilateral_times: list[float] = []
    target_errors: list[float] = []
    work: list[float] = []
    absolute_work: list[float] = []
    residuals: dict[str, list[float]] = {
        "position_max": [],
        "position_rms": [],
        "velocity_max": [],
        "velocity_rms": [],
    }
    phase_first_time: dict[str, float] = {}
    for row in telemetry:
        time_s = float(row["time_s"])
        phase = str(row["phase"])
        phase_first_time.setdefault(phase, time_s)
        semantics = row.get("contact_semantics", {})
        if bool(semantics.get("bilateral_solver_active_contact")):
            bilateral_times.append(time_s)
        target = np.asarray(row.get("joint_target", []), dtype=np.float64)
        actual = np.asarray(row.get("joint_readback", []), dtype=np.float64)
        if target.shape == actual.shape and target.size:
            target_errors.append(float(np.max(np.abs(target - actual))))
        increment = row.get("drive_work_increment_j")
        if increment is not None and math.isfinite(float(increment)):
            work.append(float(increment))
            power = np.asarray(row.get("joint_power_w", []), dtype=np.float64)
            dt = float(readback.get("effective_physics_dt_s", 0.0))
            if power.size and dt > 0.0:
                absolute_work.append(float(np.sum(np.abs(power)) * dt))
        row_residuals = row.get("solver_residuals", {})
        for name, values in residuals.items():
            value = row_residuals.get(name)
            if value is not None and math.isfinite(float(value)):
                values.append(float(value))
        for contact in row.get("contacts", []):
            separation = float(contact.get("separation_m", math.nan))
            if math.isfinite(separation):
                separations.append(separation)
            paths = " ".join(
                str(contact.get(name, "")).lower()
                for name in (
                    "actor0_path",
                    "actor1_path",
                    "collider0_path",
                    "collider1_path",
                )
            )
            if "bottle500" not in paths:
                continue
            side = "left" if "left_finger" in paths else (
                "right" if "right_finger" in paths else None
            )
            if side is None:
                continue
            vector = np.asarray(
                contact.get("impulse_vector_ns", [math.nan] * 3),
                dtype=np.float64,
            )
            normal = np.asarray(
                contact.get("normal_world", [math.nan] * 3),
                dtype=np.float64,
            )
            if vector.shape == (3,) and normal.shape == (3,) and (
                np.isfinite(vector).all() and np.isfinite(normal).all()
            ):
                contact_totals[f"{side}_signed_normal_impulse_ns"] += float(
                    np.dot(vector, normal)
                )
            scalar = float(contact.get("impulse_ns", math.nan))
            if math.isfinite(scalar):
                contact_totals[f"{side}_scalar_impulse_ns"] += scalar
    minimum_separation = min(separations) if separations else None
    final_bottle = telemetry[-1].get("bottle", {})
    return {
        "status": str(report.get("status", "UNKNOWN")),
        "reason": str(report.get("reason", "")),
        "physical_model_signature": numerical.get(
            "physical_model_signature"
        ),
        "numerical_readback": readback,
        "sample_count": len(telemetry),
        "phase_first_time_s": phase_first_time,
        "first_bilateral_solver_contact_time_s": (
            min(bilateral_times) if bilateral_times else None
        ),
        "contact": {
            **contact_totals,
            "minimum_separation_m": minimum_separation,
            "maximum_penetration_m": (
                None
                if minimum_separation is None
                else max(0.0, -minimum_separation)
            ),
        },
        "drive": {
            "signed_work_j": float(sum(work)),
            "absolute_work_j": float(sum(absolute_work)),
            "maximum_joint_target_error": _finite_max(target_errors),
        },
        "solver_residuals": {
            name: _finite_max(values) for name, values in residuals.items()
        },
        "bottle_final": {
            "position_world_m": final_bottle.get("position_world_m"),
            "orientation_wxyz": final_bottle.get("orientation_wxyz"),
            "pose_finite_difference_velocity": final_bottle.get(
                "pose_finite_difference_velocity"
            ),
        },
        "hold_drop_m": report.get("metrics", {}).get("hold_drop_m"),
    }


def _phase_signal(
    rows: list[dict[str, Any]],
    *,
    phase: str,
    path: tuple[str, ...],
) -> np.ndarray:
    values: list[np.ndarray] = []
    for row in rows:
        if str(row.get("phase")) != phase:
            continue
        value: Any = row
        for name in path:
            value = value[name]
        values.append(np.asarray(value, dtype=np.float64))
    if not values:
        return np.empty((0, 0), dtype=np.float64)
    return np.stack(values)


def _normalized_interpolation(values: np.ndarray, points: int = 101) -> np.ndarray:
    if values.shape[0] == 1:
        return np.repeat(values, points, axis=0)
    source = np.linspace(0.0, 1.0, values.shape[0])
    target = np.linspace(0.0, 1.0, points)
    return np.stack(
        [np.interp(target, source, values[:, index]) for index in range(values.shape[1])],
        axis=1,
    )


def compare_runtime_cells(
    *,
    coarse: list[dict[str, Any]],
    fine: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare phase-normalized traces from two numerical discretizations."""

    coarse_phases = list(dict.fromkeys(str(row["phase"]) for row in coarse))
    fine_phases = {str(row["phase"]) for row in fine}
    common = [phase for phase in coarse_phases if phase in fine_phases]
    signal_specs = {
        "joint_position": ("joint_readback",),
        "joint_velocity": ("joint_velocity",),
        "bottle_position": ("bottle", "position_world_m"),
    }
    maxima = dict.fromkeys(signal_specs, 0.0)
    compared = 0
    for phase in common:
        for name, path in signal_specs.items():
            coarse_values = _phase_signal(coarse, phase=phase, path=path)
            fine_values = _phase_signal(fine, phase=phase, path=path)
            if not coarse_values.size or coarse_values.shape[1:] != fine_values.shape[1:]:
                continue
            difference = _normalized_interpolation(coarse_values) - (
                _normalized_interpolation(fine_values)
            )
            if name == "bottle_position":
                metric = float(np.max(np.linalg.norm(difference, axis=1)))
            else:
                metric = float(np.max(np.abs(difference)))
            maxima[name] = max(maxima[name], metric)
            compared += 1
    return {
        "common_phases": common,
        "compared_signal_phase_count": compared,
        "joint_position_max_abs_difference": maxima["joint_position"],
        "joint_velocity_max_abs_difference": maxima["joint_velocity"],
        "bottle_position_max_norm_difference_m": maxima["bottle_position"],
    }


def select_coarsest_converged_value(
    *,
    ordered_values: list[int],
    comparisons: list[dict[str, Any]],
) -> int | None:
    """Select only when every successive pair from a value to finest passes."""

    if len(ordered_values) < 2:
        raise ValueError("at least two ordered values are required")
    pair_by_values = {
        (int(record["coarse"]), int(record["fine"])): record
        for record in comparisons
    }
    for start in range(len(ordered_values) - 1):
        required = []
        complete = True
        for index in range(start, len(ordered_values) - 1):
            pair = (int(ordered_values[index]), int(ordered_values[index + 1]))
            record = pair_by_values.get(pair)
            if record is None:
                complete = False
                break
            gates = record.get("gates", {})
            required.append(bool(gates) and all(bool(value) for value in gates.values()))
        if complete and required and all(required):
            return int(ordered_values[start])
    return None


def should_continue_solver_sweeps(
    *,
    selected_frequency_hz: int | None,
) -> bool:
    """Solver-iteration sweeps require a previously frozen timestep."""

    return selected_frequency_hz is not None


def build_runtime_cell_command(
    *,
    isaac_python: Path,
    launcher: Path,
    runtime_config: Path,
    artifact_root: Path,
    bottle_transform_path: Path,
    initial_arm_q_rad: list[float],
    frequency_hz: int,
    position_iterations: int,
    velocity_iterations: int,
    capture_failure_evidence: bool,
) -> list[str]:
    """Build one fresh-process command with physical-time gates invariant."""

    validated = validate_numerical_override(
        frequency_hz=frequency_hz,
        position_iterations=position_iterations,
        velocity_iterations=velocity_iterations,
    )
    if len(initial_arm_q_rad) != 6 or not all(
        math.isfinite(float(value)) for value in initial_arm_q_rad
    ):
        raise ValueError("initial_arm_q_rad must contain six finite values")
    hold_frames = scaled_frame_count(
        base_frames=60,
        frequency_hz=float(frequency_hz),
    )
    command = [
        str(isaac_python),
        str(launcher),
        "--config",
        str(runtime_config),
        "--artifact-root",
        str(artifact_root),
        "--autorun",
        "--close-after-terminal",
        "--bottle-world-from-object-json",
        str(bottle_transform_path),
        "--initial-arm-q-rad",
        *(repr(float(value)) for value in initial_arm_q_rad),
        "--initial-pose-hold-frames",
        str(hold_frames),
        "--arm-phase-readback-tolerance-rad",
        "0.02",
        "--arm-trajectory-mode",
        "LULA_CSPACE_ACCELERATION_LIMITED",
        "--arm-acceleration-limits-rad-s2",
        "5.0",
        "5.0",
        "5.0",
        "5.0",
        "5.0",
        "5.0",
        "--physics-frequency-hz",
        str(int(validated["frequency_hz"])),
        "--solver-position-iterations",
        str(validated["position_iterations"]),
        "--solver-velocity-iterations",
        str(validated["velocity_iterations"]),
        "--enable-solver-residual-reporting",
    ]
    if not capture_failure_evidence:
        command.extend(("--skip-collider-evidence", "--skip-video-capture"))
    return command
