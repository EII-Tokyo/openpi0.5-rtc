#!/usr/bin/env python3
"""Build the isolated world-zero tabletop alignment diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.aloha1_mapping.table_support_alignment import build_alignment_diagnostic

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/"
    "1.0/aloha1_signal_correspondence_workcell.usda"
)
DEFAULT_OUTPUT = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "table_support_alignment/1.0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-stage", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_alignment_diagnostic(
        source_stage=args.source_stage,
        output_dir=args.output_dir,
        table_prim_path=(
            "/World/environment/worldBody/user_confirmed_table"
        ),
        table_dimensions_m=(1.1, 0.6, 0.015),
        target_table_top_z_m=0.0,
        support_contact_z_m=0.0,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
