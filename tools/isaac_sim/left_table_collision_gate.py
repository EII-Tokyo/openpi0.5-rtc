"""Pure acceptance rules for the follower-left/table collision gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


TABLE_PATH = "/World/environment/worldBody/user_confirmed_table"
ALLOWED_TIP_ROOTS = (
    "/World/follower_left/vx300s_left/follower_left_wrist_link",
    "/World/follower_left/vx300s_left/follower_left_gripper_link",
    "/World/follower_left/vx300s_left/follower_left_gripper_prop_link",
    "/World/follower_left/vx300s_left/follower_left_gripper_bar_link",
    "/World/follower_left/vx300s_left/follower_left_ee_gripper_link",
    "/World/follower_left/vx300s_left/follower_left_left_finger_link",
    "/World/follower_left/vx300s_left/follower_left_right_finger_link",
)
MAX_CONTACT_SEPARATION_M = 0.0005
MAX_TABLE_TOP_PENETRATION_M = 0.0005
MAX_VISUAL_COLLISION_ERROR_M = 0.0001
MIN_TARGET_ERROR_RAD = math.radians(2.0)
MIN_PERSISTENT_CONTACT_STEPS = 180
REQUIRED_TRIALS = 3


@dataclass
class TrialMetrics:
    """Measurements needed to decide one independent collision trial."""

    contact_pairs: list[tuple[str, str]]
    minimum_target_separation_m: float
    minimum_table_local_finger_z_m: float
    maximum_visual_collision_error_m: float
    final_target_error_rad: float
    persistent_contact_steps: int
    finite: bool
    within_joint_limits: bool
    ccd_effective: bool
    disallowed_tip_contacts: list[tuple[str, str]]
    physx_errors: list[str]


def _is_same_or_descendant(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def _is_allowed_tip(path: str) -> bool:
    return any(_is_same_or_descendant(path, root) for root in ALLOWED_TIP_ROOTS)


def _is_target_pair(pair: tuple[str, str]) -> bool:
    first, second = pair
    return (
        _is_same_or_descendant(first, TABLE_PATH) and _is_allowed_tip(second)
    ) or (
        _is_same_or_descendant(second, TABLE_PATH) and _is_allowed_tip(first)
    )


def evaluate_trial(metrics: TrialMetrics) -> dict[str, Any]:
    """Evaluate one trial without importing Isaac Sim or USD modules."""

    target_contact_found = (
        any(_is_target_pair(pair) for pair in metrics.contact_pairs)
        and math.isfinite(metrics.minimum_target_separation_m)
        and metrics.minimum_target_separation_m <= MAX_CONTACT_SEPARATION_M
    )
    tabletop_penetrated = (
        not math.isfinite(metrics.minimum_table_local_finger_z_m)
        or metrics.minimum_table_local_finger_z_m
        < -MAX_TABLE_TOP_PENETRATION_M
    )
    visual_collision_match = (
        math.isfinite(metrics.maximum_visual_collision_error_m)
        and metrics.maximum_visual_collision_error_m
        <= MAX_VISUAL_COLLISION_ERROR_M
    )
    infeasible_target_blocked = (
        math.isfinite(metrics.final_target_error_rad)
        and abs(metrics.final_target_error_rad) >= MIN_TARGET_ERROR_RAD
    )
    persistent_contact = (
        metrics.persistent_contact_steps >= MIN_PERSISTENT_CONTACT_STEPS
    )

    failure_reasons: list[str] = []
    if not target_contact_found:
        failure_reasons.append("missing_physical_table_tip_contact")
    if tabletop_penetrated:
        failure_reasons.append("finger_penetrated_table_top")
    if not visual_collision_match:
        failure_reasons.append("visual_collision_mismatch")
    if not infeasible_target_blocked:
        failure_reasons.append("infeasible_target_not_blocked")
    if not persistent_contact:
        failure_reasons.append("insufficient_persistent_contact")
    if not metrics.finite:
        failure_reasons.append("non_finite_state")
    if not metrics.within_joint_limits:
        failure_reasons.append("joint_limit_violation")
    if not metrics.ccd_effective:
        failure_reasons.append("ccd_not_effective")
    if metrics.disallowed_tip_contacts:
        failure_reasons.append("disallowed_tip_environment_contact")
    if metrics.physx_errors:
        failure_reasons.append("physx_errors")

    return {
        "status": "PASS" if not failure_reasons else "FAIL",
        "target_contact_found": target_contact_found,
        "tabletop_penetrated": tabletop_penetrated,
        "visual_collision_match": visual_collision_match,
        "infeasible_target_blocked": infeasible_target_blocked,
        "persistent_contact_ok": persistent_contact,
        "failure_reasons": failure_reasons,
        "metrics": asdict(metrics),
    }


def aggregate_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Require exactly three individually passing trial decisions."""

    passed = len(trials) == REQUIRED_TRIALS and all(
        row.get("status") == "PASS" for row in trials
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "trial_count": len(trials),
        "required_trial_count": REQUIRED_TRIALS,
        "failure_reasons": [] if passed else ["exact_three_trial_gate_failed"],
        "trials": trials,
    }
