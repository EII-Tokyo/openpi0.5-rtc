"""Pure planning and result classification for ALOHA 1 gripper validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import yaml

REQUIRED_TRIAL_CHECKS = (
    "solve_articulation_contact_last_ok",
    "open_direction_ok",
    "close_direction_ok",
    "limits_ok",
    "readback_ok",
    "mimic_ok",
    "aperture_monotonic",
    "left_finger_contact",
    "right_finger_contact",
    "bilateral_contact_before_release",
    "impulses_finite",
    "persistent_penetration",
    "unexpected_gripper_collision",
    "released_without_constraint",
    "held_for_required_steps",
    "finite_state",
)

_INVERTED_BOOLEAN_CHECKS = {
    "persistent_penetration",
    "unexpected_gripper_collision",
}


def canonicalize_contact_events(
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return contact events in a stable order without changing values."""

    normalized = []
    for event in events:
        item = dict(event)
        item["contacts"] = sorted(
            (dict(contact) for contact in event.get("contacts", [])),
            key=lambda contact: json.dumps(
                contact,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        normalized.append(item)
    return sorted(
        normalized,
        key=lambda item: json.dumps(
            item,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def classify_repeat_determinism(
    previous_signature: str | None,
    current_signature: str,
) -> dict[str, Any]:
    if previous_signature is None:
        status = "PARTIAL"
        deterministic = False
    else:
        deterministic = previous_signature == current_signature
        status = "PASS" if deterministic else "FAIL"
    return {
        "status": status,
        "deterministic": deterministic,
        "previous_signature": previous_signature,
        "current_signature": current_signature,
    }


def _check_passed(name: str, metrics: Mapping[str, Any]) -> bool:
    value = bool(metrics.get(name, False))
    return not value if name in _INVERTED_BOOLEAN_CHECKS else value


def build_gripper_validation_plan(project_root: Path) -> dict[str, Any]:
    """Build an explicit test plan from the generated joint map."""

    root = project_root.resolve(strict=True)
    joint_map = yaml.safe_load((root / "configs/aloha1_joint_map.yaml").read_text(encoding="utf-8"))
    robots = []
    for name in ("follower_left", "follower_right"):
        robot_map = joint_map["robots"][name]
        dofs = {item["name"]: item for item in robot_map["dofs"]}
        left = dofs["left_finger"]
        right = dofs["right_finger"]
        left_limits = left["position_limit"]
        mimic = right["mimic"]
        robots.append(
            {
                "name": name,
                "asset": str(
                    (
                        root
                        / "assets/Trossen/ALOHA1/1.0/follower_vx300s"
                        / name
                        / "configuration"
                        / f"{name}_debug_acceleration_drive.usd"
                    ).resolve()
                ),
                "dof_order": list(robot_map["isaac_dof_order"]),
                "open_left_finger_m": float(left_limits["upper"]),
                "closed_left_finger_m": float(left_limits["lower"]),
                "mimic": {
                    "target": "right_finger",
                    "reference": "left_finger",
                    "multiplier": float(mimic["multiplier"]),
                    "offset": float(mimic["offset"]),
                },
            }
        )

    return {
        "schema_version": 1,
        "status": "PARTIAL",
        "robots": robots,
        "physics": {
            "physics_dt_s": 1.0 / 60.0,
            "solve_articulation_contact_last": True,
            "self_collision": False,
            "author_contact_rest_offsets": False,
            "offset_policy": (
                "preserve Isaac Sim 5.1 defaults; require contactOffset > "
                "restOffset before any evidence-based adjustment"
            ),
        },
        "motion": {
            "settle_steps": 60,
            "open_steps": 90,
            "close_steps": 120,
            "fixed_contact_steps": 60,
            "readback_tolerance_m": 0.002,
            "mimic_tolerance_m": 0.001,
        },
        "fingertip_material": {
            "status": "TEMPORARY_UNCALIBRATED",
            "friction_scan": [0.3, 0.5, 0.7],
            "restitution": 0.0,
            "source": "engineering scan; no measured fingertip/bottle data",
        },
        "bottle_proxy": {
            "status": "PARTIAL_MEASURED_BODY_PROXY",
            "shape": "cylinder",
            "diameter_m": 0.065,
            "height_m": 0.210,
            "mass_kg": 0.020,
            "measurement_source": (
                "user-supplied real bottle body diameter, total height, mass; recorded in .codex/TASK_STATE.md"
            ),
            "inertia_status": "ENGINEERING_DERIVED_UNCALIBRATED",
            "profile_limitation": ("neck, shoulder, base profile and measured inertia unavailable"),
        },
        "released_hold": {
            "hold_steps": 120,
            "hold_time_s": 2.0,
            "max_drop_m": 0.010,
            "surface_gripper_allowed": False,
            "fixed_constraint_allowed": False,
            "threshold_status": "ENGINEERING_ACCEPTANCE_THRESHOLD",
        },
        "penetration": {
            "maximum_persistent_depth_m": 0.002,
            "persistence_steps": 5,
            "threshold_status": "ENGINEERING_ACCEPTANCE_THRESHOLD",
        },
        "hard_blockers": [
            "measured fingertip/bottle friction",
            "complete measured bottle collision profile and inertia",
            "real gripper motor-angle/aperture calibration",
            "calibrated force-drive response",
        ],
    }


def classify_gripper_trial(
    metrics: Mapping[str, Any],
    *,
    hard_blockers: Sequence[str],
) -> dict[str, Any]:
    """Classify a trial with exact PASS/FAIL/PARTIAL semantics."""

    failed = [name for name in REQUIRED_TRIAL_CHECKS if not _check_passed(name, metrics)]
    interface_pass = not failed
    if failed:
        status = "FAIL"
    elif hard_blockers:
        status = "PARTIAL"
    else:
        status = "PASS"
    return {
        "schema_version": 1,
        "status": status,
        "passed_interface_gate": interface_pass,
        "failed_checks": failed,
        "hard_blockers": list(hard_blockers),
        "metrics": dict(metrics),
    }


def _longest_consecutive_run(values: set[int]) -> int:
    longest = 0
    current = 0
    previous = None
    for value in sorted(values):
        if previous is not None and value == previous + 1:
            current += 1
        else:
            current = 1
        longest = max(longest, current)
        previous = value
    return longest


def summarize_contact_events(
    events: Sequence[Mapping[str, Any]],
    *,
    bottle_path_token: str,
    penetration_limit_m: float,
    persistence_steps: int,
) -> dict[str, Any]:
    """Summarize serialized Isaac 5.1 contact reports."""

    finger_contact = {"left": False, "right": False}
    contact_records: list[Mapping[str, Any]] = []
    excessive_penetration_frames: set[int] = set()
    unexpected = []

    for event in events:
        collider0 = str(event.get("collider0", ""))
        collider1 = str(event.get("collider1", ""))
        bottle_pair = bottle_path_token in collider0 or bottle_path_token in collider1
        pair_text = f"{collider0}\n{collider1}"
        if bottle_pair:
            for side in ("left", "right"):
                if f"{side}_finger_link" in pair_text:
                    finger_contact[side] = True
            for contact in event.get("contacts", []):
                contact_records.append(contact)
                separation = float(contact.get("separation", math.nan))
                if math.isfinite(separation) and separation < -abs(penetration_limit_m):
                    excessive_penetration_frames.add(int(event.get("frame", -1)))
        else:
            gripper_tokens = (
                "_finger_link",
                "gripper_bar_link",
                "gripper_link",
            )
            if any(token in collider0 for token in gripper_tokens) and any(
                token in collider1 for token in gripper_tokens
            ):
                unexpected.append(
                    {
                        "frame": event.get("frame"),
                        "collider0": collider0,
                        "collider1": collider1,
                    }
                )

    impulses_finite = bool(contact_records) and all(
        len(contact.get("impulse", [])) == 3
        and all(math.isfinite(float(component)) for component in contact["impulse"])
        for contact in contact_records
    )
    longest_penetration_run = _longest_consecutive_run(excessive_penetration_frames)
    finite_separations = [
        float(contact["separation"])
        for contact in contact_records
        if math.isfinite(float(contact.get("separation", math.nan)))
    ]
    minimum_separation = min(finite_separations) if finite_separations else None
    return {
        "left_finger_contact": finger_contact["left"],
        "right_finger_contact": finger_contact["right"],
        "impulses_finite": impulses_finite,
        "persistent_penetration": (longest_penetration_run >= persistence_steps),
        "maximum_excessive_penetration_run_steps": longest_penetration_run,
        "minimum_separation_m": minimum_separation,
        "maximum_penetration_depth_m": (max(0.0, -minimum_separation) if minimum_separation is not None else None),
        "unexpected_gripper_collision": bool(unexpected),
        "unexpected_gripper_collision_events": unexpected,
        "contact_point_count": len(contact_records),
    }
