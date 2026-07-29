"""Pure classification and reproducibility helpers for supplier-CAD Task 7."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any


def classify_rigid_body_scope(
    *,
    path: str,
    rigid_body_enabled: bool,
    joint_body_target: bool,
    only_fixed_joint_targets: bool,
    collider_count: int,
) -> str:
    """Classify a rigid body without inventing mass, inertia, or colliders."""

    if not rigid_body_enabled:
        return "DISABLED_RIGID_BODY"
    if joint_body_target and collider_count > 0:
        return "ROBOT_RIGID_BODY"
    if joint_body_target and only_fixed_joint_targets:
        return "FIXED_REFERENCE_HELPER_EXCLUDE_FROM_ROBOT_DIAGNOSTIC"
    if joint_body_target:
        return "PARTICIPATING_BODY_MISSING_COLLIDER_HARD_BLOCKER"
    if collider_count == 0:
        return (
            "NONPHYSICAL_HELPER_REMOVE_RIGID_BODY_API_IN_DIAGNOSTIC_LAYER"
        )
    return "NON_ROBOT_COLLIDER_EXCLUDE_FROM_ROBOT_SCOPE"


def classify_task7(
    checks: Sequence[Mapping[str, Any]],
    hard_blockers: Sequence[str],
) -> str:
    """Return a literal Task 7 status without hiding failed checks."""

    if any(check.get("status") == "FAIL" for check in checks):
        return "FAIL"
    if hard_blockers or any(
        check.get("status") == "PARTIAL" for check in checks
    ):
        return "PARTIAL"
    return "PASS"


def deterministic_signature(report: Mapping[str, Any]) -> str:
    """Hash stable validation content, excluding only the repeat wrapper."""

    normalized = {
        key: value
        for key, value in report.items()
        if key != "repeat_determinism"
    }
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
