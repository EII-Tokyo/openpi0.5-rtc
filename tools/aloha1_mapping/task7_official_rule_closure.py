"""Pure closure policy for literal ALOHA1 Task 7 validator findings."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

CLASSIFICATION_ACTIONS = {
    "LAYER_PACKAGING_DEFECT": "CREATE_ISOLATED_PACKAGING_CANDIDATE",
    "MISSING_SOURCE_EVIDENCE": "HARD_BLOCKER_NO_SOURCE_GEOMETRY",
    "ISAAC_5_1_VALIDATOR_SCHEMA_CONFLICT": (
        "KEEP_UNSUPPRESSED_VERSION_CONFLICT"
    ),
    "NON_APPLICABLE_FALSE_POSITIVE": "RECORD_NON_BLOCKING_INFORMATION",
}


def classify_official_rule_closure(
    issues: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Partition findings without changing their literal official result."""

    classifications = Counter(
        str(issue.get("classification", "")) for issue in issues
    )
    unsupported = sorted(set(classifications) - set(CLASSIFICATION_ACTIONS))
    if unsupported:
        raise ValueError(f"unsupported classifications: {unsupported}")

    action_counts = Counter(
        CLASSIFICATION_ACTIONS[str(issue["classification"])]
        for issue in issues
    )
    packaging = [
        issue
        for issue in issues
        if issue["classification"] == "LAYER_PACKAGING_DEFECT"
    ]
    packaging_rule_counts = Counter(str(issue["rule"]) for issue in packaging)

    return {
        "issue_count": len(issues),
        "unclassified_issue_count": 0,
        "official_status": "FAIL",
        "official_status_suppressed": False,
        "classification_counts": dict(sorted(classifications.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "packaging_rule_counts": dict(sorted(packaging_rule_counts.items())),
        "candidate_mutation_issue_count": len(packaging),
        "source_or_runtime_mutation_issue_count": 0,
        "candidate_boundary": {
            "directory": (
                "assets/Trossen/ALOHA1/1.0/diagnostics/"
                "task7_promotion_candidate/"
            ),
            "isolated_only": True,
            "geometry_change_allowed": False,
            "joint_or_drive_change_allowed": False,
            "collision_change_allowed": False,
            "final_or_default_asset_change_allowed": False,
        },
        "task8": "NOT_RUN",
    }
