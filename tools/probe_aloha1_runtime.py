#!/usr/bin/env python3
"""Probe ALOHA 1 articulation DOFs using the Isaac Sim 5.1 runtime."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path

from tools.aloha1_mapping.runtime_probe import build_probe_targets
from tools.aloha1_mapping.runtime_probe import probe_runtime


def _environment_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--enable-leaders", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/aloha1_mapping/usd_dof_inventory.json"),
    )
    arguments = parser.parse_args(argv)
    report_path = (
        arguments.report
        if arguments.report.is_absolute()
        else arguments.project_root / arguments.report
    ).resolve()
    targets = build_probe_targets(
        arguments.project_root,
        enable_leaders=(
            arguments.enable_leaders or _environment_flag("ENABLE_LEADERS")
        ),
    )
    probe_runtime(targets, report_path=report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
