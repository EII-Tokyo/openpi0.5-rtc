#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.aloha1_mapping.actuator_drive_source_boundary import build_report
from tools.aloha1_mapping.actuator_drive_source_boundary import render_markdown

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = ROOT / "configs/aloha1_official_parameter_sources.yaml"
DEFAULT_JSON = ROOT / "reports/aloha1_mapping/aloha1_actuator_drive_source_boundary.json"
DEFAULT_MARKDOWN = ROOT / "reports/aloha1_mapping/aloha1_actuator_drive_source_boundary.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_report(args.root, args.sources)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "continuous_envelope": report["continuous_envelope"]["status"],
                "physx_drive_mapping": report["physx_drive_mapping"]["status"],
            }
        )
    )
    return 0 if report["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
