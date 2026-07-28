#!/usr/bin/env python3
"""Finalize the explicit visual review of eight Isaac CAD finger captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.aloha1_mapping.isaac_screenshot_review import build_review_report
from tools.aloha1_mapping.isaac_screenshot_review import render_markdown


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-report", type=Path, required=True)
    parser.add_argument("--annotation-metadata", type=Path, required=True)
    parser.add_argument("--visual-decisions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    raw_report = json.loads(
        args.raw_report.resolve(strict=True).read_text(encoding="utf-8")
    )
    annotation_metadata = json.loads(
        args.annotation_metadata.resolve(strict=True).read_text(encoding="utf-8")
    )
    decision_manifest = json.loads(
        args.visual_decisions.resolve(strict=True).read_text(encoding="utf-8")
    )
    report = build_review_report(
        raw_report=raw_report,
        annotation_metadata=annotation_metadata,
        decisions=decision_manifest["decisions"],
        retake_history=decision_manifest["retake_history"],
        approved_source_stage=decision_manifest["approved_source_stage"],
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"status={report['status']}")
    print(f"capture_count={report['capture_count']}")
    print(f"json={args.output_json.resolve()}")
    print(f"markdown={args.output_md.resolve()}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
