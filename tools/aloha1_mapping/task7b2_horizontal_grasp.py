"""Pure acceptance gates for the horizontal Bottle500 pickup diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

REQUIRED_TRIAL_FIELDS = {
    "fresh_world_reset",
    "bottle_dynamic_during_settle",
    "support_contact_before_grasp",
    "axis_horizontal_pass",
    "gripper_axis_perpendicular_pass",
    "coupling_accuracy_pass",
    "vertical_descent_pass",
    "ik_reachable",
    "left_physical_contact_before_lift",
    "right_physical_contact_before_lift",
    "contact_points_in_body_interval",
    "bottle_left_support",
    "bilateral_contact_through_hold",
    "hold_drop_m",
    "drop_gate_m",
    "finite_state",
    "persistent_penetration",
    "numerical_ejection",
    "forbidden_contact",
    "forbidden_constraint",
    "surface_gripper_used",
    "parent_attachment_used",
    "contact_lost_before_hold",
    "free_fall_after_contact_loss",
    "rotation_induced_escape",
    "normal_force_decay",
    "continuous_slip",
}


def classify_horizontal_failure(trial: dict[str, Any]) -> str:
    """Return one exact failure mode using the approved fail-closed precedence."""
    if REQUIRED_TRIAL_FIELDS - trial.keys():
        return "inconclusive"
    if not trial["axis_horizontal_pass"]:
        return "horizontal_geometry_failed"
    if not trial["gripper_axis_perpendicular_pass"]:
        return "gripper_axis_correspondence_failed"
    if not trial["coupling_accuracy_pass"]:
        return "gripper_coupling_accuracy_failed"
    if not trial["vertical_descent_pass"] or not trial["ik_reachable"]:
        return "vertical_ik_unreachable"
    if (
        not trial["bottle_dynamic_during_settle"]
        or not trial["support_contact_before_grasp"]
    ):
        return "support_settle_failed"
    if (
        not trial["left_physical_contact_before_lift"]
        or not trial["right_physical_contact_before_lift"]
    ):
        return "contact_not_established"
    if (
        not trial["finite_state"]
        or trial["persistent_penetration"]
        or trial["numerical_ejection"]
    ):
        return "numerical_penetration_or_ejection"
    if (
        trial["forbidden_contact"]
        or trial["forbidden_constraint"]
        or trial["surface_gripper_used"]
        or trial["parent_attachment_used"]
        or not trial["contact_points_in_body_interval"]
    ):
        return "forbidden_contact"
    if not trial["bottle_left_support"]:
        return "support_clearance_failed"
    if (
        trial["contact_lost_before_hold"]
        and trial["free_fall_after_contact_loss"]
    ):
        return "contact_lost_then_free_fall"
    if trial["rotation_induced_escape"]:
        return "rotation_induced_escape"
    if trial["normal_force_decay"]:
        return "normal_force_decay"
    hold_drop = trial["hold_drop_m"]
    drop_gate = trial["drop_gate_m"]
    if (
        not isinstance(hold_drop, int | float)
        or not isinstance(drop_gate, int | float)
        or not math.isfinite(float(hold_drop))
        or not math.isfinite(float(drop_gate))
        or float(drop_gate) <= 0.0
    ):
        return "inconclusive"
    if trial["continuous_slip"] or (
        trial["bilateral_contact_through_hold"]
        and float(hold_drop) > float(drop_gate)
    ):
        return "bilateral_contact_continuous_slip"
    if not trial["bilateral_contact_through_hold"]:
        return "inconclusive"
    return "stable_hold"


def evaluate_horizontal_trial(trial: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one runtime record without mutating it."""
    failure_mode = classify_horizontal_failure(trial)
    status = "PASS" if failure_mode == "stable_hold" else "FAIL"
    missing = sorted(REQUIRED_TRIAL_FIELDS - trial.keys())
    return {
        "status": status,
        "failure_mode": failure_mode,
        "trial_index": trial.get("trial_index"),
        "fresh_world_reset": bool(trial.get("fresh_world_reset", False)),
        "missing_required_fields": missing,
        "hold_drop_m": trial.get("hold_drop_m"),
        "drop_gate_m": trial.get("drop_gate_m"),
        "task8": "NOT_RUN",
    }


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonicalize(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, list | tuple):
        return [_canonicalize(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NON_FINITE"
        return round(value, 9)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "NON_FINITE"
    return round(numeric, 9)


def canonical_horizontal_signature(trial: dict[str, Any]) -> str:
    """Hash deterministic physical signals, excluding runtime and file paths."""
    evaluation = evaluate_horizontal_trial(trial)
    payload = {
        "failure_mode": evaluation["failure_mode"],
        "status": evaluation["status"],
        "fresh_world_reset": trial.get("fresh_world_reset"),
        "phase_frame_counts": trial.get("phase_frame_counts", {}),
        "joint_trajectories": trial.get("joint_trajectories", []),
        "contacts": trial.get("contacts", []),
        "bottle_poses": trial.get("bottle_poses", []),
        "hold_drop_m": trial.get("hold_drop_m"),
        "drop_gate_m": trial.get("drop_gate_m"),
        "bottle_left_support": trial.get("bottle_left_support"),
        "bilateral_contact_through_hold": trial.get(
            "bilateral_contact_through_hold"
        ),
        "persistent_penetration": trial.get("persistent_penetration"),
        "numerical_ejection": trial.get("numerical_ejection"),
    }
    encoded = json.dumps(
        _canonicalize(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def summarize_horizontal_trials(
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate smoke or 20-trial acceptance without relaxing any gate."""
    if not trials:
        return {
            "status": "NOT_RUN",
            "trial_count": 0,
            "pass_count": 0,
            "fresh_world_reset_count": 0,
            "unique_deterministic_signature_count": 0,
            "determinism_status": "NOT_RUN",
            "acceptance_eligible": False,
            "task8": "NOT_RUN",
        }
    evaluations = [evaluate_horizontal_trial(trial) for trial in trials]
    signatures = [
        canonical_horizontal_signature(trial) for trial in trials
    ]
    trial_count = len(trials)
    pass_count = sum(item["status"] == "PASS" for item in evaluations)
    fresh_count = sum(
        bool(trial.get("fresh_world_reset", False)) for trial in trials
    )
    unique_count = len(set(signatures))
    complete_acceptance = (
        trial_count == 20
        and pass_count == 20
        and fresh_count == 20
        and unique_count == 1
    )
    if complete_acceptance:
        status = "PASS"
    elif trial_count == 1 or trial_count < 20:
        status = "PARTIAL"
    else:
        status = "FAIL"
    failure_counts: dict[str, int] = {}
    for evaluation in evaluations:
        mode = str(evaluation["failure_mode"])
        failure_counts[mode] = failure_counts.get(mode, 0) + 1
    return {
        "status": status,
        "trial_count": trial_count,
        "pass_count": pass_count,
        "fresh_world_reset_count": fresh_count,
        "unique_deterministic_signature_count": unique_count,
        "determinism_status": "PASS" if unique_count == 1 else "FAIL",
        "acceptance_eligible": complete_acceptance,
        "failure_mode_counts": failure_counts,
        "signatures": signatures,
        "evaluations": evaluations,
        "task8": "NOT_RUN",
    }


def render_horizontal_markdown(summary: dict[str, Any]) -> str:
    """Render a bounded human-readable view of the machine aggregate."""
    lines = [
        "# ALOHA1 Horizontal Bottle500 Pickup",
        "",
        f"- Status: `{summary.get('status', 'NOT_RUN')}`",
        f"- Trials: `{summary.get('trial_count', 0)}`",
        f"- PASS: `{summary.get('pass_count', 0)}`",
        (
            "- Fresh resets: "
            f"`{summary.get('fresh_world_reset_count', 0)}`"
        ),
        (
            "- Deterministic signatures: "
            f"`{summary.get('unique_deterministic_signature_count', 0)}`"
        ),
        "- Task 8: `NOT_RUN`",
        "",
        "## Failure classifications",
        "",
    ]
    counts = summary.get("failure_mode_counts", {})
    if counts:
        lines.extend(
            f"- `{mode}`: {count}" for mode, count in sorted(counts.items())
        )
    else:
        lines.append("- No trials were run.")
    lines.extend(
        [
            "",
            "Screenshots and video are supporting evidence only; runtime "
            "contact, pose, velocity, drop, and deterministic data are "
            "authoritative.",
            "",
        ]
    )
    return "\n".join(lines)
