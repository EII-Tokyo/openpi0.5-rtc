"""Pure classification contracts for ALOHA1 Task 7 acceptance layers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

RUNTIME_GATES = (
    "structure_and_runtime_order",
    "joint_mapping",
    "follower_left_one_joint",
    "follower_right_one_joint",
    "small_up_down",
    "drive_mimic_structure",
    "initial_target_readback_first_frame",
)


def _combined_status(statuses: list[str]) -> str:
    if not statuses or any(status == "FAIL" for status in statuses):
        return "FAIL"
    if any(status in {"PARTIAL", "NOT_RUN"} for status in statuses):
        return "PARTIAL"
    if all(status == "PASS" for status in statuses):
        return "PASS"
    return "FAIL"


def classify_runtime_control(task7a: dict[str, Any]) -> dict[str, Any]:
    """Classify measured control behavior without package-only rules."""
    gates = {name: str(task7a.get(name, "NOT_RUN")) for name in RUNTIME_GATES}
    return {
        "status": _combined_status(list(gates.values())),
        "gates": gates,
        "failed_gates": [name for name, status in gates.items() if status == "FAIL"],
        "partial_or_not_run_gates": [name for name, status in gates.items() if status in {"PARTIAL", "NOT_RUN"}],
        "official_asset_rules_included": False,
        "acceptance_boundary": (
            "Runtime motion, order, target/readback, drive/mimic, and "
            "determinism only. Asset packaging is reported separately."
        ),
    }


def classify_workcell_physics(swept: dict[str, Any]) -> dict[str, Any]:
    """Classify swept workcell behavior using the frozen policy-v2 report."""
    summary = swept.get("summary", {})
    determinism = summary.get("determinism", {})
    policy = swept.get("contact_policy", {})
    checks = {
        "swept_report": str(swept.get("status", "NOT_RUN")),
        "coverage": str(summary.get("coverage_status", "NOT_RUN")),
        "determinism": str(determinism.get("status", "NOT_RUN")),
        "case_count": ("PASS" if summary.get("case_count") == summary.get("expected_case_count") else "FAIL"),
        "forbidden_contact_cases": ("PASS" if int(summary.get("failed_case_count", -1)) == 0 else "FAIL"),
        "unclassified_partial_cases": ("PASS" if int(summary.get("partial_case_count", -1)) == 0 else "FAIL"),
        "policy_revision": ("PASS" if policy.get("revision") == 2 else "FAIL"),
    }
    return {
        "status": _combined_status(list(checks.values())),
        "checks": checks,
        "contact_limited_case_count": int(summary.get("contact_limited_case_count", 0)),
        "allowed_pair": policy.get("allowed_pair"),
        "allowed_contact_meaning": policy.get("allowed_contact_meaning"),
        "forbidden_contact_policy": {
            "generic_robot_environment_contact": policy.get("generic_robot_environment_contact"),
            "non_adjacent_self_contact": policy.get("non_adjacent_self_contact"),
            "cross_follower_contact": policy.get("cross_follower_contact"),
        },
        "acceptance_boundary": (
            "A user-confirmed supplier-CAD finger/table contact is a "
            "workcell reachability boundary, not a controller failure. "
            "All other robot/environment, non-adjacent self, and "
            "cross-follower contacts remain forbidden."
        ),
    }


def classify_asset_promotion_readiness(
    *,
    official: dict[str, Any],
    triage: dict[str, Any],
    helper_audit: dict[str, Any],
) -> dict[str, Any]:
    """Preserve literal official status and separately classify readiness."""
    official_status = str(official.get("official_status", "NOT_RUN"))
    suppressed = bool(triage.get("official_status_suppressed", False))
    unclassified = int(triage.get("unclassified_issue_count", -1))
    helper_status = str(helper_audit.get("status", "NOT_RUN"))
    if suppressed or unclassified != 0 or helper_status == "FAIL":
        status = "FAIL"
    elif official_status == "PASS" and helper_status == "PASS":
        status = "PASS"
    elif official_status == "FAIL" and helper_status == "PASS":
        status = "PARTIAL"
    else:
        status = "FAIL"
    return {
        "status": status,
        "ready_for_promotion": status == "PASS",
        "official_status": official_status,
        "official_status_suppressed": suppressed,
        "unclassified_issue_count": unclassified,
        "classification_counts": dict(triage.get("classification_counts", {})),
        "helper_link_audit_status": helper_status,
        "helper_link_decision": helper_audit.get("decision"),
        "acceptance_boundary": (
            "PARTIAL means the runtime asset is not ready for promotion. "
            "Literal NVIDIA rule failures remain visible and must be closed "
            "in an isolated, regression-tested package."
        ),
    }


def combine_task7a_layers(
    *,
    runtime_status: str,
    workcell_status: str,
    promotion_status: str,
) -> str:
    """Combine layers while preserving direct runtime/physics failures."""
    if runtime_status == "FAIL" or workcell_status == "FAIL":
        return "FAIL"
    return _combined_status([runtime_status, workcell_status, promotion_status])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_sha256(path: Path, expected: str) -> str:
    resolved = path.resolve(strict=True)
    actual = file_sha256(resolved)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {resolved}: {actual} != {expected}")
    return actual
