"""Pure data contracts for ALOHA1 Task 7A swept-collision validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
from typing import Any

ARM_JOINTS = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
)
ROBOTS = ("follower_left", "follower_right")
DIRECTIONS = ("negative", "positive")


def canonical_pair(first: str, second: str) -> tuple[str, str]:
    """Return one stable unordered path pair."""
    return tuple(sorted((str(first), str(second))))


def build_sweep_cases(
    limits_by_robot: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    limit_margin_fraction: float = 0.02,
    minimum_margin: float = 0.02,
) -> list[dict[str, Any]]:
    """Build exactly six-DOF, two-direction cases for both followers."""
    if not 0.0 < limit_margin_fraction < 0.5:
        raise ValueError("limit_margin_fraction must be between 0 and 0.5")
    cases: list[dict[str, Any]] = []
    for robot in ROBOTS:
        records = {
            str(item["name"]): item for item in limits_by_robot[robot]
        }
        if set(records) != set(ARM_JOINTS):
            raise ValueError(f"{robot} does not have exactly six arm limits")
        for joint_index, joint in enumerate(ARM_JOINTS):
            record = records[joint]
            lower = float(record["lower"])
            upper = float(record["upper"])
            home = float(record["home"])
            if not all(math.isfinite(value) for value in (lower, upper, home)):
                raise ValueError(f"non-finite limit for {robot}/{joint}")
            if not lower < home < upper:
                raise ValueError(f"home outside limits for {robot}/{joint}")
            span = upper - lower
            margin = min(
                span * 0.25,
                max(minimum_margin, span * limit_margin_fraction),
            )
            targets = {
                "negative": lower + margin,
                "positive": upper - margin,
            }
            for direction in DIRECTIONS:
                target = targets[direction]
                if not lower < target < upper or target == home:
                    raise ValueError(
                        f"invalid target for {robot}/{joint}/{direction}"
                    )
                cases.append(
                    {
                        "case_id": f"{robot}:{joint}:{direction}",
                        "robot": robot,
                        "joint": joint,
                        "joint_index": joint_index,
                        "direction": direction,
                        "unit": "rad",
                        "lower": lower,
                        "upper": upper,
                        "home": home,
                        "target": target,
                        "limit_margin": margin,
                    }
                )
    return cases


def _robot_for_path(path: str) -> str | None:
    for robot in ROBOTS:
        if path == f"/World/{robot}" or path.startswith(
            f"/World/{robot}/"
        ):
            return robot
    return None


def classify_contact_pair(
    actor0: str,
    actor1: str,
    adjacent_body_pairs: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    """Classify one actor pair under the current authored collision semantics."""
    pair = canonical_pair(actor0, actor1)
    adjacency = {
        canonical_pair(first, second)
        for first, second in adjacent_body_pairs
    }
    robot0 = _robot_for_path(actor0)
    robot1 = _robot_for_path(actor1)
    if not actor0 or not actor1 or actor0 == "/" or actor1 == "/":
        classification = "UNRESOLVED_CONTACT_PATH"
        allowed = False
    elif actor0 == actor1:
        classification = "SAME_RIGID_BODY"
        allowed = True
    elif pair in adjacency:
        classification = "ADJACENT_BODY_CONTACT"
        allowed = True
    elif robot0 is not None and robot0 == robot1:
        classification = "NON_ADJACENT_SELF_CONTACT"
        allowed = False
    elif robot0 is not None and robot1 is not None:
        classification = "CROSS_FOLLOWER_CONTACT"
        allowed = False
    elif (robot0 is None) != (robot1 is None):
        robot_path = actor0 if robot0 is not None else actor1
        environment_path = actor1 if robot0 is not None else actor0
        if (
            environment_path.endswith("/user_confirmed_table")
            and robot_path.endswith("_finger_link")
        ):
            classification = (
                "USER_CONFIRMED_ALLOWED_FINGER_TABLE_CONTACT"
            )
            allowed = True
        else:
            classification = "ROBOT_ENVIRONMENT_CONTACT"
            allowed = False
    else:
        classification = "NON_ROBOT_CONTACT"
        allowed = True
    result = {
        "actor_pair": list(pair),
        "classification": classification,
        "allowed": allowed,
        "robot0": robot0,
        "robot1": robot1,
    }
    if classification == "USER_CONFIRMED_ALLOWED_FINGER_TABLE_CONTACT":
        result["policy_evidence"] = "USER_CONFIRMATION_2026_07_29"
    return result


def classify_sweep_case(
    *,
    direction_pass: bool,
    target_reached: bool,
    non_target_drift_pass: bool,
    legal: bool,
    finite: bool,
    unexpected_contact_count: int,
    allowed_workspace_contact_count: int,
) -> dict[str, str]:
    """Separate control correctness from contact-limited workcell reach."""
    if unexpected_contact_count:
        return {
            "status": "FAIL",
            "motion_status": "BLOCKED_BY_FORBIDDEN_CONTACT",
            "control_direction_status": (
                "PASS" if direction_pass else "FAIL"
            ),
            "collision_policy_status": "FAIL",
            "target_reachability_status": (
                "BLOCKED_BY_FORBIDDEN_CONTACT"
            ),
        }
    if not (direction_pass and non_target_drift_pass and legal and finite):
        return {
            "status": "FAIL",
            "motion_status": "CONTROL_OR_READBACK_GATE_FAILED",
            "control_direction_status": (
                "PASS" if direction_pass else "FAIL"
            ),
            "collision_policy_status": "PASS",
            "target_reachability_status": (
                "NOT_EVALUATED_CONTROL_OR_READBACK_FAILURE"
            ),
        }
    if target_reached:
        return {
            "status": "PASS",
            "motion_status": "TARGET_REACHED",
            "control_direction_status": "PASS",
            "collision_policy_status": "PASS",
            "target_reachability_status": "REACHED",
        }
    if allowed_workspace_contact_count:
        return {
            "status": "PASS",
            "motion_status": "CONTACT_LIMITED_WORKCELL_REACHABILITY",
            "control_direction_status": "PASS",
            "collision_policy_status": "PASS",
            "target_reachability_status": (
                "CONTACT_LIMITED_BY_ALLOWED_WORKCELL_CONTACT"
            ),
        }
    return {
        "status": "FAIL",
        "motion_status": "UNEXPLAINED_TARGET_SHORTFALL",
        "control_direction_status": "PASS",
        "collision_policy_status": "PASS",
        "target_reachability_status": "UNEXPLAINED_SHORTFALL",
    }


def classify_contact_observation(
    *,
    base_classification: str,
    base_allowed: bool,
    minimum_separation_m: float | None,
    maximum_impulse_norm_n_s: float,
) -> dict[str, Any]:
    """Separate a broadphase contact envelope from physical contact."""
    penetration = (
        minimum_separation_m is not None
        and math.isfinite(float(minimum_separation_m))
        and float(minimum_separation_m) <= 0.0
    )
    finite_impulse = (
        math.isfinite(float(maximum_impulse_norm_n_s))
        and float(maximum_impulse_norm_n_s) > 0.0
    )
    physical = penetration or finite_impulse
    if not physical:
        return {
            "classification": "CONTACT_ENVELOPE_ONLY",
            "geometric_classification": base_classification,
            "physical_contact": False,
            "allowed": True,
        }
    return {
        "classification": base_classification,
        "geometric_classification": base_classification,
        "physical_contact": True,
        "allowed": bool(base_allowed),
    }


def _round_float(value: Any) -> float:
    return round(float(value), 8)


def deterministic_case_signature(case: Mapping[str, Any]) -> str:
    """Hash deterministic case evidence while ignoring record ordering."""
    contacts = [
            {
                "actor_pair": list(
                    canonical_pair(*item.get("actor_pair", ["", ""]))
                ),
                "collider_pair": list(
                    canonical_pair(*item.get("collider_pair", ["", ""]))
                ),
                "classification": item.get("classification"),
                "maximum_penetration_m": _round_float(
                    item.get("maximum_penetration_m", 0.0)
                ),
            }
        for item in case.get("contact_pairs", [])
    ]
    contacts.sort(
        key=lambda item: (
            item["classification"] or "",
            item["actor_pair"],
            item["collider_pair"],
        )
    )
    payload = {
        "robot": case.get("robot"),
        "joint": case.get("joint"),
        "direction": case.get("direction"),
        "status": case.get("status"),
        "target": _round_float(case.get("target", 0.0)),
        "final_readback": _round_float(
            case.get("final_readback", 0.0)
        ),
        "maximum_non_target_drift": _round_float(
            case.get("maximum_non_target_drift", 0.0)
        ),
        "contact_pairs": contacts,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def summarize_sweep_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    repeat_count: int,
) -> dict[str, Any]:
    """Enforce exact case coverage, pass state, and repeat determinism."""
    expected = {
        f"{robot}:{joint}:{direction}"
        for robot in ROBOTS
        for joint in ARM_JOINTS
        for direction in DIRECTIONS
    }
    count_per_repeat: list[int] = []
    repeat_signatures: list[str] = []
    repeat_coverage: list[bool] = []
    for repeat in range(repeat_count):
        repeat_cases = [
            item for item in cases if int(item["repeat"]) == repeat
        ]
        count_per_repeat.append(len(repeat_cases))
        identifiers = {
            str(
                item.get(
                    "case_id",
                    (
                        f"{item['robot']}:{item['joint']}:"
                        f"{item['direction']}"
                    ),
                )
            )
            for item in repeat_cases
        }
        repeat_coverage.append(
            len(repeat_cases) == len(expected) and identifiers == expected
        )
        signature_payload = sorted(
            deterministic_case_signature(item) for item in repeat_cases
        )
        repeat_signatures.append(
            hashlib.sha256(
                json.dumps(
                    signature_payload,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        )
    coverage = (
        len(cases) == repeat_count * len(expected)
        and all(repeat_coverage)
    )
    deterministic = (
        len(repeat_signatures) == repeat_count
        and len(set(repeat_signatures)) == 1
    )
    statuses = [item.get("status") for item in cases]
    if not coverage or not deterministic or "FAIL" in statuses:
        status = "FAIL"
    elif "PARTIAL" in statuses:
        status = "PARTIAL"
    else:
        status = "PASS"
    return {
        "status": status,
        "case_count": len(cases),
        "expected_case_count": repeat_count * len(expected),
        "case_count_per_repeat": count_per_repeat,
        "coverage_status": "PASS" if coverage else "FAIL",
        "failed_case_count": sum(
            item.get("status") == "FAIL" for item in cases
        ),
        "partial_case_count": sum(
            item.get("status") == "PARTIAL" for item in cases
        ),
        "contact_limited_case_count": sum(
            item.get("motion_status")
            == "CONTACT_LIMITED_WORKCELL_REACHABILITY"
            for item in cases
        ),
        "determinism": {
            "status": "PASS" if deterministic else "FAIL",
            "repeat_count": repeat_count,
            "signatures": repeat_signatures,
        },
    }
