"""Pure classification and reproducibility helpers for supplier-CAD Task 7."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any


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
