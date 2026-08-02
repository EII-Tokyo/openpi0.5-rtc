"""Pure comparison helpers for Task 7 validator controls."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any


def issue_signature(report: Mapping[str, Any]) -> str:
    """Hash the rule result independently of output path and run metadata."""

    payload = {
        "category": report["category"],
        "official_status": report["official_status"],
        "rules": sorted(report["rules"]),
        "issues": sorted(
            report["issues"],
            key=lambda item: (
                item.get("severity") or "",
                item.get("rule") or "",
                item.get("at") or "",
                item.get("message") or "",
            ),
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def fresh_runs_match(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    """Return whether two process reports contain the same rule result."""

    return issue_signature(first) == issue_signature(second)


def issue_counter(report: Mapping[str, Any]) -> Counter[tuple[str, str]]:
    """Count literal findings by severity and rule."""

    return Counter(
        (str(issue.get("severity")), str(issue.get("rule")))
        for issue in report.get("issues", [])
    )


def validate_negative_delta(
    *,
    baseline: Mapping[str, Any],
    negative: Mapping[str, Any],
    expected_rule: str,
    expected_target_fragment: str,
) -> dict[str, Any]:
    """Require a negative control to add the expected literal defect."""

    baseline_messages = {
        (issue.get("severity"), issue.get("rule"), issue.get("at"), issue.get("message"))
        for issue in baseline.get("issues", [])
    }
    added = [
        issue
        for issue in negative.get("issues", [])
        if (issue.get("severity"), issue.get("rule"), issue.get("at"), issue.get("message"))
        not in baseline_messages
    ]
    matching = [
        issue
        for issue in added
        if issue.get("severity") in {"ERROR", "FAILURE"}
        and issue.get("rule") == expected_rule
        and expected_target_fragment in str(issue.get("at") or issue.get("message") or "")
    ]
    if not matching:
        raise ValueError(
            f"negative control did not add {expected_rule} at {expected_target_fragment}"
        )
    return {
        "status": "PASS",
        "expected_rule": expected_rule,
        "expected_target_fragment": expected_target_fragment,
        "added_issue_count": len(added),
        "matching_added_issues": matching,
    }


def summarize_two_runs(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require exactly two fresh reports and summarize determinism."""

    if len(reports) != 2:
        raise ValueError("expected exactly two fresh-process reports")
    return {
        "fresh_process_count": 2,
        "consistent": fresh_runs_match(reports[0], reports[1]),
        "signatures": [issue_signature(report) for report in reports],
        "statuses": [report["official_status"] for report in reports],
        "blocking_counts": [report["blocking_issue_count"] for report in reports],
        "warning_counts": [report["warning_count"] for report in reports],
    }
