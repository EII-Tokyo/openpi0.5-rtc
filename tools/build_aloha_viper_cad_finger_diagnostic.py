#!/usr/bin/env python3
"""Create the isolated supplier-CAD finger visual diagnostic USD."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.aloha1_mapping.cad_finger_diagnostic import (
    create_diagnostic_asset,
)
from tools.aloha1_mapping.cad_finger_diagnostic import (
    write_diagnostic_report,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_STAGE = (
    ROOT / "local_eval_assets/aloha_isaac_assets/aloha_viperx.usd"
)
DEFAULT_MESH_ROOT = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "viper_gripper/tessellation_angular_controlled/run_a"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_finger_installation"
)
DEFAULT_REPORT_JSON = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_diagnostic_asset.json"
)
DEFAULT_REPORT_MARKDOWN = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_diagnostic_asset.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-stage",
        type=Path,
        default=DEFAULT_SOURCE_STAGE,
    )
    parser.add_argument("--mesh-root", type=Path, default=DEFAULT_MESH_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DEFAULT_REPORT_JSON,
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=DEFAULT_REPORT_MARKDOWN,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = create_diagnostic_asset(
        source_stage_path=args.source_stage,
        left_obj_path=args.mesh_root / "left_finger.obj",
        right_obj_path=args.mesh_root / "right_finger.obj",
        output_root=args.output_root,
    )
    write_diagnostic_report(
        report,
        args.report_json,
        args.report_markdown,
    )
    print(f"status={report['status']}")
    print(
        "root_usd="
        f"{report['diagnostic_outputs']['root_usd']['absolute_path']}"
    )
    print(f"report_json={args.report_json.resolve()}")
    print(f"report_markdown={args.report_markdown.resolve()}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
