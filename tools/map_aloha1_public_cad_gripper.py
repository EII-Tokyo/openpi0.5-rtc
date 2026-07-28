#!/usr/bin/env python3
"""Build the public-CAD installed-finger to follower-link mapping report."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.aloha1_mapping.public_cad_gripper_mapping import build_gripper_mapping_report
from tools.aloha1_mapping.public_cad_gripper_mapping import write_gripper_mapping_reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--freecad-audit",
        type=Path,
        default=Path(
            "reports/aloha1_mapping/aloha_public_cad_assembly_audit.json"
        ),
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("generated/urdf/follower_left.urdf"),
    )
    parser.add_argument(
        "--widow-audit",
        type=Path,
        default=Path(
            "reports/aloha1_mapping/aloha_widow_gripper_assembly_audit.json"
        ),
    )
    parser.add_argument(
        "--widow-tessellation",
        type=Path,
        default=Path(
            ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
            "widow_gripper/widow_gripper_tessellation.json"
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "reports/aloha1_mapping/aloha_public_cad_gripper_mapping.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path(
            "reports/aloha1_mapping/aloha_public_cad_gripper_mapping.md"
        ),
    )
    args = parser.parse_args()
    report = build_gripper_mapping_report(
        args.freecad_audit,
        args.urdf,
        args.widow_audit,
        args.widow_tessellation,
    )
    write_gripper_mapping_reports(
        report,
        args.json_output,
        args.markdown_output,
    )
    print(f"status={report['status']}")
    print(f"json={args.json_output.resolve()}")
    print(f"markdown={args.markdown_output.resolve()}")
    return 0 if report["orientation_mapping_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
