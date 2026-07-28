#!/usr/bin/env python3
"""Compare the purchased ALOHA drawing against public STEP candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.aloha1_mapping.purchased_aloha_model_identification import build_model_identification_report
from tools.aloha1_mapping.purchased_aloha_model_identification import write_model_identification_reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drawing", type=Path, required=True)
    parser.add_argument(
        "--public-cad-audit",
        type=Path,
        default=Path(
            "reports/aloha1_mapping/aloha_public_cad_assembly_audit.json"
        ),
    )
    parser.add_argument(
        "--widow-audit",
        type=Path,
        default=Path(
            "reports/aloha1_mapping/aloha_widow_gripper_assembly_audit.json"
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "reports/aloha1_mapping/aloha_purchased_model_identification.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path(
            "reports/aloha1_mapping/aloha_purchased_model_identification.md"
        ),
    )
    args = parser.parse_args()
    report = build_model_identification_report(
        args.drawing,
        args.public_cad_audit,
        args.widow_audit,
    )
    write_model_identification_reports(
        report,
        args.json_output,
        args.markdown_output,
    )
    print(f"status={report['status']}")
    print(f"classification={report['classification']}")
    print(f"json={args.json_output.resolve()}")
    print(f"markdown={args.markdown_output.resolve()}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
