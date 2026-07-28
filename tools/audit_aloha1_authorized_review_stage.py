#!/usr/bin/env python3
"""Audit the user-approved ALOHA Viper review Stage without saving it."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.aloha1_mapping.authorized_stage_audit import collect_stage_snapshot
from tools.aloha1_mapping.authorized_stage_audit import evaluate_stage_snapshot
from tools.aloha1_mapping.authorized_stage_audit import write_stage_audit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE = (
    ROOT / "local_eval_assets/aloha_isaac_assets/aloha_viperx.usd"
)
DEFAULT_JSON = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_authorized_stage_audit.json"
)
DEFAULT_MARKDOWN = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_authorized_stage_audit.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = collect_stage_snapshot(args.stage)
    report = evaluate_stage_snapshot(snapshot)
    write_stage_audit(report, args.json_output, args.markdown_output)
    print(f"status={report['status']}")
    print(f"source_immutable={report['source_immutable_gate']}")
    print(f"required_prims={report['required_key_prims_status']}")
    print(f"layer_stack={report['layer_stack_status']}")
    print(f"json={args.json_output.resolve()}")
    print(f"markdown={args.markdown_output.resolve()}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
