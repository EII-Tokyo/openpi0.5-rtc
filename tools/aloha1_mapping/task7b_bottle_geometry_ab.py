"""Pure gates for the ALOHA1 Task 7B bottle-geometry A/B experiment."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

REQUIRED_TRIALS = 20
ALLOWED_CONCLUSIONS = (
    "PROJECT_BOTTLE_MATCHES_BASELINE",
    "PROJECT_BOTTLE_WORSENS_HOLD",
    "PROJECT_BOTTLE_IMPROVES_HOLD",
    "INCONCLUSIVE",
)


def flatten_mapping(
    value: Mapping[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten nested mappings while preserving lists as causal values."""

    flattened: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(flatten_mapping(item, path))
        else:
            flattened[path] = item
    return flattened


def validate_single_geometry_variable(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    allowed_differences: Collection[str],
) -> dict[str, Any]:
    """Require the two profiles to differ in exactly the allowed fields."""

    left = flatten_mapping(baseline)
    right = flatten_mapping(candidate)
    differences = sorted(
        path
        for path in left.keys() | right.keys()
        if left.get(path) != right.get(path)
    )
    allowed = sorted(set(allowed_differences))
    unexpected = [path for path in differences if path not in allowed]
    missing = [path for path in allowed if path not in differences]
    return {
        "status": (
            "PASS" if not unexpected and not missing else "FAIL"
        ),
        "differences": differences,
        "allowed_differences": allowed,
        "unexpected_differences": unexpected,
        "missing_expected_differences": missing,
    }


def _group_is_complete(summary: Mapping[str, Any]) -> bool:
    trial_count = int(summary.get("trial_count", 0))
    pass_count = int(summary.get("pass_count", 0))
    status = str(summary.get("status"))
    if trial_count != REQUIRED_TRIALS:
        return False
    if not 0 <= pass_count <= trial_count:
        return False
    if status == "PASS":
        return (
            pass_count == REQUIRED_TRIALS
            and bool(summary.get("deterministic"))
            and int(summary.get("unique_signature_count", 0)) == 1
        )
    return status == "FAIL"


def compare_geometry_groups(
    baseline: Mapping[str, Any],
    project: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare complete A/B groups without weakening the Task 5 gate."""

    baseline_complete = _group_is_complete(baseline)
    project_complete = _group_is_complete(project)
    baseline_status = str(baseline.get("status"))
    project_status = str(project.get("status"))

    if not baseline_complete or not project_complete:
        conclusion = "INCONCLUSIVE"
    elif baseline_status == "PASS" and project_status == "PASS":
        conclusion = "PROJECT_BOTTLE_MATCHES_BASELINE"
    elif baseline_status == "PASS" and project_status == "FAIL":
        conclusion = "PROJECT_BOTTLE_WORSENS_HOLD"
    elif baseline_status == "FAIL" and project_status == "PASS":
        conclusion = "PROJECT_BOTTLE_IMPROVES_HOLD"
    else:
        conclusion = "INCONCLUSIVE"

    keys = (
        "status",
        "pass_count",
        "trial_count",
        "deterministic",
        "unique_signature_count",
        "minimum_drop_m",
        "maximum_drop_m",
        "mean_drop_m",
        "failure_modes",
    )
    groups = {}
    for name, summary, complete in (
        ("procedural_cylinder", baseline, baseline_complete),
        ("project_bottle500", project, project_complete),
    ):
        groups[name] = {key: summary.get(key) for key in keys}
        groups[name]["complete_acceptance_group"] = complete

    status = (
        "PASS"
        if (
            baseline_complete
            and project_complete
            and baseline_status == "PASS"
            and project_status == "PASS"
        )
        else "FAIL"
    )
    return {
        "status": status,
        "conclusion": conclusion,
        "groups": groups,
        "acceptance_boundary": (
            "STATIC_FREE_BOTTLE_HOLD_ONLY_NOT_SUPPORT_TO_LIFT_PICKUP"
        ),
        "task8": "NOT_RUN",
    }


def render_comparison_markdown(report: Mapping[str, Any]) -> str:
    """Render a bounded summary without converting static hold to pickup."""

    groups = report["groups"]
    rows = [
        "# ALOHA1 Task 7B Bottle Geometry A/B",
        "",
        f"- Status: `{report['status']}`",
        f"- Conclusion: `{report['conclusion']}`",
        f"- Boundary: `{report['acceptance_boundary']}`",
        f"- Task 8: `{report['task8']}`",
        "",
        "| Profile | Status | Passes | Deterministic | Max drop (m) |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for name in ("procedural_cylinder", "project_bottle500"):
        group = groups[name]
        rows.append(
            f"| {name} | {group['status']} | "
            f"{group['pass_count']}/{group['trial_count']} | "
            f"{group['deterministic']} | {group['maximum_drop_m']} |"
        )
    rows.extend(
        [
            "",
            "This experiment evaluates static free-bottle hold only. It does "
            "not prove support-to-lift pickup.",
        ]
    )
    return "\n".join(rows) + "\n"
