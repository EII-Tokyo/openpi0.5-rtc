"""Pure contracts for the ALOHA1 Task 7 PhysicsRules root-cause matrix."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any

FROZEN_STAGE_SHA256 = (
    "327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9"
)

REQUIRED_RULE_COUNTS = {
    "JointHasCorrectTransformAndState": 10,
    "MimicAPICheck": 2,
    "RigidBodyHasCollider": 8,
}

ALLOWED_HELPER_CLASSES = {
    "PHYSICAL_LINK_REQUIRES_COLLIDER",
    "VIRTUAL_HELPER_SHOULD_NOT_BE_RIGID_BODY",
    "INCONCLUSIVE",
}


def classify_collider_finding(
    *,
    visual_count: int,
    collision_count: int,
    incoming_joint_types: Sequence[str],
) -> str:
    """Classify one missing-collider finding from source geometry semantics."""

    if collision_count > 0:
        return "PHYSICAL_LINK_REQUIRES_COLLIDER"
    if visual_count > 0:
        return "INCONCLUSIVE"
    if list(incoming_joint_types) == ["fixed"]:
        return "VIRTUAL_HELPER_SHOULD_NOT_BE_RIGID_BODY"
    return "INCONCLUSIVE"


def mapped_mimic_interval(
    *,
    reference_lower: float,
    reference_upper: float,
    gearing: float,
    offset: float = 0.0,
) -> tuple[float, float]:
    """Map a reference interval through q_mimic = gearing*q_ref + offset."""

    endpoints = (
        gearing * reference_lower + offset,
        gearing * reference_upper + offset,
    )
    return min(endpoints), max(endpoints)


def mapped_physx_mimic_interval(
    *,
    reference_lower: float,
    reference_upper: float,
    gearing: float,
    offset: float = 0.0,
) -> tuple[float, float]:
    """Map the local 107.3 PhysX equation q + gearing*q_ref + offset = 0."""

    endpoints = (
        -(gearing * reference_lower + offset),
        -(gearing * reference_upper + offset),
    )
    return min(endpoints), max(endpoints)


def _prim_path(location: object) -> str:
    value = str(location)
    if value.startswith("Prim <") and value.endswith(">"):
        return value[6:-1]
    raise ValueError(f"finding location is not a prim path: {value}")


def build_finding_inventory(
    issues: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and normalize the exact scoped follower finding set."""

    keys = [(str(item.get("rule")), str(item.get("at"))) for item in issues]
    duplicate_keys = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicate_keys:
        raise ValueError(f"duplicate finding: {duplicate_keys}")
    counts = dict(sorted(Counter(rule for rule, _ in keys).items()))
    if counts != REQUIRED_RULE_COUNTS:
        raise ValueError(
            f"unexpected rule counts: {counts} != {REQUIRED_RULE_COUNTS}"
        )
    paths = sorted(_prim_path(item.get("at")) for item in issues)
    return {
        "finding_count": len(issues),
        "rule_counts": counts,
        "prim_paths": paths,
    }


def build_hypothesis_signature(contract: Mapping[str, Any]) -> str:
    """Hash a one-variable hypothesis independent of target ordering."""

    normalized = dict(contract)
    normalized["target_prims"] = sorted(str(item) for item in contract["target_prims"])
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def should_escalate_screenshot(
    fresh_results: Sequence[Mapping[str, Any]],
) -> bool:
    """Require visual failure evidence after two identical fresh failures."""

    if len(fresh_results) < 2:
        return False
    first, second = fresh_results[-2:]
    return (
        first.get("status") == second.get("status") == "FAIL"
        and first.get("signature") == second.get("signature")
    )


def summarize_runtime_trace(
    *,
    dof_names: Sequence[str],
    expected_dof_names: Sequence[str],
    samples: Sequence[Mapping[str, Any]],
    first_frame_arm_gate_rad: float,
) -> dict[str, Any]:
    """Summarize a candidate's uncommanded reset/step trace.

    The only numerical gate is the existing Task 7 six-arm-DOF first-frame
    gate. Finger motion remains diagnostic because the project has no frozen
    source-backed tolerance for an uncommanded first-step finger displacement.
    """

    if len(samples) < 2:
        raise ValueError("runtime trace requires at least reset and first-frame samples")
    names = [str(name) for name in dof_names]
    expected = [str(name) for name in expected_dof_names]
    width = len(names)
    rows = [[float(value) for value in sample["positions"]] for sample in samples]
    if any(len(row) != width for row in rows):
        raise ValueError("runtime trace position width does not match DOF order")
    finite = all(math.isfinite(value) for row in rows for value in row)
    first_delta = [abs(after - before) for before, after in zip(rows[0], rows[1], strict=True)]
    hold_delta = [abs(end - start) for start, end in zip(rows[1], rows[-1], strict=True)]
    arm_first = max(first_delta[:6], default=0.0)
    arm_hold = max(hold_delta[:6], default=0.0)
    finger_indices = [names.index(name) for name in ("left_finger", "right_finger") if name in names]
    finger_first = max((first_delta[index] for index in finger_indices), default=0.0)
    finger_hold = max((hold_delta[index] for index in finger_indices), default=0.0)
    reasons = []
    if names != expected:
        reasons.append("DOF_ORDER_MISMATCH")
    if not finite:
        reasons.append("NONFINITE_RUNTIME_READBACK")
    if arm_first > first_frame_arm_gate_rad:
        reasons.append("FIRST_FRAME_ARM_JUMP_EXCEEDS_FROZEN_GATE")
    if arm_hold > first_frame_arm_gate_rad:
        reasons.append("STATIC_ARM_DRIFT_EXCEEDS_FROZEN_GATE")
    return {
        "status": "FAIL" if reasons else "PASS",
        "failure_reasons": reasons,
        "dof_order_matches": names == expected,
        "finite": finite,
        "sample_count": len(samples),
        "first_frame_arm_jump_max_abs_rad": arm_first,
        "static_arm_drift_max_abs_rad": arm_hold,
        "first_frame_arm_gate_rad": first_frame_arm_gate_rad,
        "first_frame_finger_jump_max_abs_m": finger_first,
        "static_finger_drift_max_abs_m": finger_hold,
        "finger_jump_gate": "RECORDED_NOT_GATED_NO_FROZEN_TOLERANCE",
    }
