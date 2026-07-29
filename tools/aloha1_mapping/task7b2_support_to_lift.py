"""Pure gates for ALOHA1 Task 7B.2 support-to-lift pickup."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any

import numpy as np


def _finite_bounds(
    bounds: Mapping[str, Sequence[float]],
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.asarray(bounds["minimum"], dtype=np.float64)
    maximum = np.asarray(bounds["maximum"], dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError(f"{name} bounds must contain two 3-vectors")
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        raise ValueError(f"{name} bounds must be finite")
    if np.any(maximum <= minimum):
        raise ValueError(f"{name} bounds must have positive extent")
    return minimum, maximum


def derive_supported_bottle_translation(
    *,
    table_bounds: Mapping[str, Sequence[float]],
    bottle_bounds: Mapping[str, Sequence[float]],
    aperture_midpoint: Sequence[float],
) -> list[float]:
    """Place the bottle bottom on the table top at the aperture midpoint X/Y."""

    _, table_maximum = _finite_bounds(table_bounds, name="table")
    bottle_minimum, _ = _finite_bounds(bottle_bounds, name="bottle")
    midpoint = np.asarray(aperture_midpoint, dtype=np.float64)
    if midpoint.shape != (3,) or not np.isfinite(midpoint).all():
        raise ValueError("aperture midpoint must be one finite 3-vector")
    return [
        float(midpoint[0]),
        float(midpoint[1]),
        float(table_maximum[2] - bottle_minimum[2]),
    ]


_CHECKS = (
    "support_settle_pass",
    "support_contact_before_lift",
    "bilateral_contact_before_lift",
    "non_target_arm_drift_within_gate",
    "bottle_left_support",
    "bilateral_contact_through_hold",
    "finite_state",
)


def _failed_checks(metrics: Mapping[str, Any]) -> list[str]:
    failed = [name for name in _CHECKS if not bool(metrics.get(name))]
    if not math.isclose(
        float(metrics["shoulder_delta_rad"]),
        float(metrics["expected_shoulder_delta_rad"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        failed.append("shoulder_delta_matches_validated_signal")
    if float(metrics["minimum_support_clearance_m"]) < float(
        metrics["required_clearance_m"]
    ):
        failed.append("support_clearance")
    if bool(metrics["support_recontact_after_clear"]):
        failed.append("support_recontact_after_clear")
    if float(metrics["hold_drop_m"]) > float(metrics["drop_gate_m"]):
        failed.append("hold_drop_within_gate")
    if bool(metrics["persistent_penetration"]):
        failed.append("no_persistent_penetration")
    if bool(metrics["forbidden_contact"]):
        failed.append("no_forbidden_contact")
    if bool(metrics["constraint_found"]):
        failed.append("no_fixed_constraint")
    if bool(metrics["surface_gripper_used"]):
        failed.append("no_surface_gripper")
    if bool(metrics["parent_attachment_used"]):
        failed.append("no_parent_attachment")
    return failed


def _failure_mode(metrics: Mapping[str, Any]) -> str:
    if not bool(metrics["support_settle_pass"]) or not bool(
        metrics["support_contact_before_lift"]
    ):
        return "support_settle_failed"
    if not bool(metrics["bilateral_contact_before_lift"]):
        return "bilateral_contact_not_established"
    if bool(metrics["forbidden_contact"]):
        return "forbidden_contact"
    if bool(metrics["persistent_penetration"]):
        return "numerical_penetration_or_ejection"
    if not bool(metrics["bottle_left_support"]) or float(
        metrics["minimum_support_clearance_m"]
    ) < float(metrics["required_clearance_m"]):
        return "bottle_never_left_support"
    if bool(metrics["support_recontact_after_clear"]):
        return "support_recontact_after_lift"
    if not bool(metrics["bilateral_contact_through_hold"]):
        return "contact_lost_during_lift"
    if float(metrics["hold_drop_m"]) > float(metrics["drop_gate_m"]):
        return "continuous_slip_during_hold"
    return "inconclusive"


def evaluate_pickup_trial(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one fresh support-to-lift trial."""

    failed = _failed_checks(metrics)
    return {
        "status": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "failure_mode": (
            "stable_support_to_lift_pickup"
            if not failed
            else _failure_mode(metrics)
        ),
    }


def summarize_pickup_trials(
    trials: Sequence[Mapping[str, Any]],
    *,
    required_repeats: int,
) -> dict[str, Any]:
    """Require a complete, all-pass, fresh and deterministic acceptance group."""

    signatures = {
        str(trial.get("deterministic_signature"))
        for trial in trials
    }
    pass_count = sum(trial.get("status") == "PASS" for trial in trials)
    complete = len(trials) == required_repeats
    all_fresh = all(
        bool(trial.get("fresh_world_reset")) for trial in trials
    )
    deterministic = complete and len(signatures) == 1
    status = (
        "PASS"
        if complete
        and pass_count == required_repeats
        and all_fresh
        and deterministic
        else "FAIL"
    )
    return {
        "status": status,
        "trial_count": len(trials),
        "required_trial_count": required_repeats,
        "pass_count": pass_count,
        "fresh_world_reset_count": sum(
            bool(trial.get("fresh_world_reset")) for trial in trials
        ),
        "deterministic": deterministic,
        "unique_signature_count": len(signatures),
        "failure_modes": sorted(
            {
                str(trial.get("failure_mode"))
                for trial in trials
                if trial.get("status") != "PASS"
            }
        ),
    }


def canonical_pickup_signature(trial: Mapping[str, Any]) -> str:
    payload = json.dumps(
        trial,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def render_pickup_markdown(report: Mapping[str, Any]) -> str:
    """Render a bounded report while preserving acceptance boundaries."""

    boundaries = report["boundaries"]
    summary = report["summary"]
    lines = [
        "# ALOHA1 Task 7B.2 Support-to-Lift Pickup",
        "",
        f"- Status: `{report['status']}`",
        f"- Conclusion: `{report['conclusion']}`",
        f"- Trials: `{summary['pass_count']}/{summary['trial_count']}`",
        f"- Deterministic: `{summary['deterministic']}`",
        f"- Task 7B static hold: `{boundaries['task7b_static_hold']}`",
        f"- Asset promotion: `{boundaries['asset_promotion']}`",
        f"- Task 8: `{boundaries['task8']}`",
        "",
        "PASS means the bottle started on the user-confirmed support, left "
        "that support under the validated shoulder signal, and completed "
        "the two-second hold gate.",
    ]
    return "\n".join(lines) + "\n"
