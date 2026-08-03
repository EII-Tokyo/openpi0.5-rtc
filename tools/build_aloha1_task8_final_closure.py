#!/usr/bin/env python3
"""Close ALOHA1 Task 8 without promoting diagnostic candidates."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/aloha1_mapping"
DEFAULT_VISUAL = REPORT_ROOT / "aloha1_task8_benchmark_comparison.json"
DEFAULT_VISUAL_CANDIDATE = REPORT_ROOT / "aloha1_task8_visual_material_candidate.json"
DEFAULT_COLLIDER = REPORT_ROOT / "aloha1_task8_comparison.json"
DEFAULT_JSON = REPORT_ROOT / "aloha1_task8_final_closure.json"
DEFAULT_MARKDOWN = REPORT_ROOT / "aloha1_task8_final_closure.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_closure(
    *, visual: Mapping[str, Any], collider: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the literal no-promotion Task 8 closure after strict checks."""

    visual_result = visual.get("status", visual.get("classification"))
    if visual_result != "NO_MEASURABLE_IMPROVEMENT":
        raise ValueError(f"unexpected visual candidate result: {visual_result!r}")
    visual_candidate = visual.get("candidate")
    visual_promoted = (
        visual_candidate.get("candidate_promoted")
        if isinstance(visual_candidate, Mapping)
        else visual.get("candidate_promoted")
    )
    if visual_promoted is not False:
        raise ValueError("visual candidate promotion boundary is not false")
    if collider.get("status") != "NO_MEASURABLE_IMPROVEMENT":
        raise ValueError(f"unexpected collider result: {collider.get('status')!r}")
    if collider.get("candidate_promoted") is not False:
        raise ValueError("collider candidate promotion boundary is not false")
    if collider.get("final_or_default_asset_modified") is not False:
        raise ValueError("collider report does not preserve final/default assets")

    return {
        "schema_version": 1,
        "status": "PASS",
        "classification": "ALOHA1_TASK8_FINAL_CLOSURE",
        "task8_status": "COMPLETE",
        "task8_result": "COMPLETE_WITH_NO_PROMOTION",
        "visual_material_candidate": "NO_MEASURABLE_IMPROVEMENT",
        "collider_lod_candidate": "NO_MEASURABLE_IMPROVEMENT",
        "candidate_promoted": False,
        "final_default_asset_modified": False,
        "reopen_policy": (
            "EXPLICIT_USER_REQUEST_OR_PROFILER_IDENTIFIED_NEW_BOTTLENECK"
        ),
        "next_scope": "ALOHA1_REAL_SIM_HOME_SIGNAL_CORRESPONDENCE",
    }


def _markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# ALOHA1 Task 8 final closure",
            "",
            f"- Status: `{report['status']}`",
            f"- Task 8: `{report['task8_status']}`",
            f"- Result: `{report['task8_result']}`",
            f"- Visual material candidate: `{report['visual_material_candidate']}`",
            f"- Collider LOD candidate: `{report['collider_lod_candidate']}`",
            f"- Candidate promoted: `{str(report['candidate_promoted']).lower()}`",
            (
                "- Final/default asset modified: "
                f"`{str(report['final_default_asset_modified']).lower()}`"
            ),
            f"- Deterministic signature: `{report['deterministic_signature']}`",
            "",
            "Task 8 is complete with no promoted optimization. The next scope is "
            "real-versus-digital Home/Sleep signal correspondence; this report does "
            "not authorize access to or motion of the real robot.",
            "",
        ]
    )


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument(
        "--visual-candidate", type=Path, default=DEFAULT_VISUAL_CANDIDATE
    )
    parser.add_argument("--collider", type=Path, default=DEFAULT_COLLIDER)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    visual_path = args.visual.resolve(strict=True)
    visual_candidate_path = args.visual_candidate.resolve(strict=True)
    collider_path = args.collider.resolve(strict=True)
    visual = _load(visual_path)
    visual_candidate = _load(visual_candidate_path)
    visual_for_closure = {
        **visual,
        "candidate_promoted": False,
    }
    candidate_classification = str(visual.get("classification", ""))
    if "NOT_PROMOTED" not in candidate_classification:
        raise ValueError("visual benchmark report is not an unpromoted candidate")
    if visual_candidate.get("status") != "PASS_STATIC_CANDIDATE_NOT_PROMOTED":
        raise ValueError("visual static candidate report does not preserve boundary")

    report = build_closure(
        visual=visual_for_closure,
        collider=_load(collider_path),
    )
    report["inputs"] = {
        "visual_benchmark": {
            "absolute_path": str(visual_path),
            "sha256": _sha256(visual_path),
        },
        "visual_candidate": {
            "absolute_path": str(visual_candidate_path),
            "sha256": _sha256(visual_candidate_path),
        },
        "collider_comparison": {
            "absolute_path": str(collider_path),
            "sha256": _sha256(collider_path),
        },
    }
    report["deterministic_signature"] = _canonical_signature(report)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "task8_status": report["task8_status"],
                "task8_result": report["task8_result"],
                "deterministic_signature": report["deterministic_signature"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
