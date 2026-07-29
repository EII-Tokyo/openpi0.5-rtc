#!/usr/bin/env python3
"""Generate the explicit Stationary ALOHA 1 joint map YAML."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import yaml

from tools.aloha1_mapping.joint_map import build_joint_map


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/aloha1_joint_map.yaml"),
    )
    parser.add_argument(
        "--runtime-report",
        type=Path,
        default=Path("reports/aloha1_mapping/usd_dof_inventory.json"),
        help="Runtime DOF inventory relative to --project-root.",
    )
    arguments = parser.parse_args(argv)
    output = (
        arguments.output if arguments.output.is_absolute() else arguments.project_root / arguments.output
    ).resolve()
    mapping = build_joint_map(
        arguments.project_root,
        runtime_report_relative=arguments.runtime_report,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
