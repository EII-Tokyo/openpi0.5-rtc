#!/usr/bin/env python3
"""Generate the frozen Stationary ALOHA 1 baseline manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.aloha1_mapping.signal_correspondence_baseline import write_baseline_reports
from tools.aloha1_mapping.signal_correspondence_baseline import write_workcell_layers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    report_root = root / "reports/aloha1_mapping"
    write_baseline_reports(
        root,
        json_path=report_root / "aloha1_stationary_user_confirmed_baseline_v1.json",
        markdown_path=report_root / "aloha1_stationary_user_confirmed_baseline_v1.md",
    )
    stage_manifest = write_workcell_layers(root)
    (report_root / "aloha1_signal_correspondence_stage_manifest.json").write_text(
        json.dumps(
            stage_manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
