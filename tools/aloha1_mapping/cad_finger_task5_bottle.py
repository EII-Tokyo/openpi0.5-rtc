"""Pure gates for supplier-CAD follower gripper bottle Task 5."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

REQUIRED_HOLD_CHECKS = (
    "solve_articulation_contact_last_ok",
    "left_finger_contact",
    "right_finger_contact",
    "bilateral_contact_before_release",
    "impulses_finite",
    "persistent_penetration",
    "unexpected_gripper_collision",
    "released_without_constraint",
    "gravity_enabled_after_release",
    "held_for_required_time",
    "drop_within_gate",
    "finite_state",
)

_INVERTED_CHECKS = {
    "persistent_penetration",
    "unexpected_gripper_collision",
}


def compute_hold_kinematics(
    *,
    release_z_m: float,
    z_samples_m: Sequence[float],
    dt_s: float,
) -> dict[str, Any]:
    """Compute the drop gate and pose-derived velocity over the full hold."""

    if not z_samples_m:
        raise ValueError("hold trajectory requires at least one z sample")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    previous = float(release_z_m)
    velocities = []
    samples = [float(value) for value in z_samples_m]
    for value in samples:
        velocities.append((value - previous) / dt_s)
        previous = value
    return {
        "release_z_m": float(release_z_m),
        "minimum_z_m": min(samples),
        "maximum_z_m": max(samples),
        "maximum_drop_m": float(release_z_m) - min(samples),
        "maximum_rise_m": max(samples) - float(release_z_m),
        "final_drop_m": float(release_z_m) - samples[-1],
        "pose_derived_vertical_velocity_m_s": velocities,
        "maximum_abs_pose_derived_vertical_velocity_m_s": max(
            abs(value) for value in velocities
        ),
        "final_pose_derived_vertical_velocity_m_s": velocities[-1],
    }


def _passed(name: str, metrics: Mapping[str, Any]) -> bool:
    value = bool(metrics.get(name, False))
    return not value if name in _INVERTED_CHECKS else value


def evaluate_bottle_trial(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the unchanged 20 g bottle hold gate."""

    failed = [
        name for name in REQUIRED_HOLD_CHECKS if not _passed(name, metrics)
    ]
    return {
        "status": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "metrics": dict(metrics),
    }


def classify_hold_failure_mode(metrics: Mapping[str, Any]) -> str:
    """Classify the dominant observed release/hold outcome."""

    if not bool(metrics.get("bilateral_contact_before_release")):
        return "contact_not_established"
    if bool(metrics.get("numerical_penetration_or_ejection")):
        return "numerical_penetration_or_ejection"
    if bool(metrics.get("contact_lost_after_release")):
        return "contact_lost_then_free_fall"
    if bool(metrics.get("rotation_induced_escape")):
        return "rotation_induced_escape"
    if bool(metrics.get("normal_force_decay")):
        return "normal_force_decay"
    if bool(metrics.get("continuous_slip_with_bilateral_contact")):
        return "bilateral_contact_but_continuous_slip"
    if bool(metrics.get("drop_within_gate")):
        return "stable_hold"
    return "inconclusive"


def summarize_bottle_trials(
    trials: Sequence[Mapping[str, Any]],
    *,
    required_repeats: int,
) -> dict[str, Any]:
    """Summarize fresh-reset trials without weakening the physical gate."""

    if required_repeats <= 0:
        raise ValueError("required_repeats must be positive")
    pass_count = sum(trial.get("status") == "PASS" for trial in trials)
    signatures = {
        str(trial.get("deterministic_signature")) for trial in trials
    }
    drops = [
        float(trial["released_hold"]["drop_m"])
        for trial in trials
        if math.isfinite(float(trial["released_hold"]["drop_m"]))
    ]
    deterministic = len(trials) >= 2 and len(signatures) == 1
    complete = len(trials) == required_repeats
    all_pass = complete and pass_count == required_repeats
    return {
        "status": "PASS" if all_pass and deterministic else "FAIL",
        "trial_count": len(trials),
        "required_repeats": required_repeats,
        "complete": complete,
        "pass_count": pass_count,
        "fail_count": len(trials) - pass_count,
        "all_trials_pass": all_pass,
        "deterministic": deterministic,
        "unique_signature_count": len(signatures),
        "minimum_drop_m": min(drops) if drops else None,
        "maximum_drop_m": max(drops) if drops else None,
        "mean_drop_m": sum(drops) / len(drops) if drops else None,
        "failure_modes": sorted(
            {
                str(trial.get("failure_mode", "inconclusive"))
                for trial in trials
                if trial.get("status") != "PASS"
            }
        ),
    }
