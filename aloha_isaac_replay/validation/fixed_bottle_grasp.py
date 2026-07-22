from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from typing import Any

PASS_FIXED_BOTTLE_GRASP = "PASS_FIXED_BOTTLE_GRASP"

RESET_UNSTABLE = "RESET_UNSTABLE"
INITIAL_PENETRATION = "INITIAL_PENETRATION"
NO_CONTACT = "NO_CONTACT"
ONE_FINGER_CONTACT_ONLY = "ONE_FINGER_CONTACT_ONLY"
GRIP_FORCE_INSUFFICIENT = "GRIP_FORCE_INSUFFICIENT"
BOTTLE_SLIPPED = "BOTTLE_SLIPPED"
BOTTLE_DROPPED = "BOTTLE_DROPPED"
COLLIDER_PENETRATION = "COLLIDER_PENETRATION"
JOINT_LIMIT = "JOINT_LIMIT"
CONTROL_TIMEOUT = "CONTROL_TIMEOUT"
UNKNOWN = "UNKNOWN"

FAIL_RESET = "FAIL_RESET"
FAIL_COLLIDER = "FAIL_COLLIDER"
FAIL_CONTACT = "FAIL_CONTACT"
FAIL_GRIPPER_CONTROL = "FAIL_GRIPPER_CONTROL"
FAIL_SOLVER_STABILITY = "FAIL_SOLVER_STABILITY"
BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class TrialClassification:
    success: bool
    reason: str
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "reason": self.reason,
            "metrics": dict(self.metrics),
        }


def _bool(metrics: dict[str, Any], key: str, default: bool = False) -> bool:
    return bool(metrics.get(key, default))


def _float(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(metrics.get(key, default))
    except (TypeError, ValueError):
        return float("nan")
    return value


def _finite_metric_values(metrics: dict[str, Any]) -> bool:
    for value in metrics.values():
        if isinstance(value, bool | str) or value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not isfinite(numeric):
            return False
    return True


def classify_trial(
    metrics: dict[str, Any],
    *,
    min_lift_height_m: float = 0.08,
    max_slip_m: float = 0.01,
    min_contact_force_n: float = 1e-6,
) -> TrialClassification:
    """Classify one deterministic fixed-bottle grasp trial.

    The order is intentionally conservative: reset and penetration failures
    outrank contact and lift failures, so one bad reset cannot be counted as a
    weak gripper or weak friction result.
    """

    metrics = dict(metrics)
    if not _bool(metrics, "reset_stable", True):
        return TrialClassification(False, RESET_UNSTABLE, metrics)
    if _bool(metrics, "initial_penetration"):
        return TrialClassification(False, INITIAL_PENETRATION, metrics)
    if _bool(metrics, "control_timeout"):
        return TrialClassification(False, CONTROL_TIMEOUT, metrics)
    if _bool(metrics, "nan_or_inf") or not _finite_metric_values(metrics):
        return TrialClassification(False, UNKNOWN, metrics)
    if _bool(metrics, "joint_limit_or_effort_violation"):
        return TrialClassification(False, JOINT_LIMIT, metrics)
    if _bool(metrics, "collider_penetration"):
        return TrialClassification(False, COLLIDER_PENETRATION, metrics)

    left_contact = _bool(metrics, "left_contact")
    right_contact = _bool(metrics, "right_contact")
    if not left_contact and not right_contact:
        return TrialClassification(False, NO_CONTACT, metrics)
    if left_contact != right_contact:
        return TrialClassification(False, ONE_FINGER_CONTACT_ONLY, metrics)
    if _float(metrics, "max_contact_force_n") <= float(min_contact_force_n):
        return TrialClassification(False, GRIP_FORCE_INSUFFICIENT, metrics)
    if _float(metrics, "lift_height_m") < float(min_lift_height_m):
        return TrialClassification(False, BOTTLE_DROPPED, metrics)
    if not _bool(metrics, "left_table_during_hold", True) or _bool(metrics, "touched_table_during_hold"):
        return TrialClassification(False, BOTTLE_DROPPED, metrics)
    if _float(metrics, "max_slip_m") > float(max_slip_m):
        return TrialClassification(False, BOTTLE_SLIPPED, metrics)
    return TrialClassification(True, PASS_FIXED_BOTTLE_GRASP, metrics)


def _failure_conclusion(reason_counts: Counter[str]) -> str:
    if reason_counts[RESET_UNSTABLE]:
        return FAIL_RESET
    if reason_counts[INITIAL_PENETRATION] or reason_counts[COLLIDER_PENETRATION]:
        return FAIL_COLLIDER
    if reason_counts[JOINT_LIMIT] or reason_counts[CONTROL_TIMEOUT]:
        return FAIL_GRIPPER_CONTROL
    if reason_counts[NO_CONTACT] or reason_counts[ONE_FINGER_CONTACT_ONLY] or reason_counts[GRIP_FORCE_INSUFFICIENT]:
        return FAIL_CONTACT
    if reason_counts[BOTTLE_SLIPPED] or reason_counts[BOTTLE_DROPPED]:
        return FAIL_CONTACT
    if reason_counts[UNKNOWN]:
        return BLOCKED
    return FAIL_SOLVER_STABILITY


def summarize_trials(trials: list[dict[str, Any]], *, required_successes: int = 19) -> dict[str, Any]:
    success_count = sum(1 for trial in trials if bool(trial.get("success")))
    failure_reasons = [str(trial.get("reason") or UNKNOWN) for trial in trials if not bool(trial.get("success"))]
    reason_counts = Counter(failure_reasons)
    final_conclusion = PASS_FIXED_BOTTLE_GRASP if success_count >= int(required_successes) else _failure_conclusion(reason_counts)
    max_slip = max(
        (
            _float(dict(trial.get("metrics") or {}), "max_slip_m", float("nan"))
            for trial in trials
            if trial.get("metrics")
        ),
        default=float("nan"),
    )
    return {
        "trial_count": len(trials),
        "success_count": success_count,
        "failure_count": len(trials) - success_count,
        "failure_reason_counts": dict(reason_counts),
        "max_slip_m": max_slip,
        "required_successes": int(required_successes),
        "final_conclusion": final_conclusion,
    }
