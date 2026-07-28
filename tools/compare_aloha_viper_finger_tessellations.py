#!/usr/bin/env python3
"""Write the two-run ALOHA Viper finger tessellation comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.aloha1_mapping.compare_finger_tessellations import build_comparison
from tools.aloha1_mapping.compare_finger_tessellations import write_comparison


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "reports/aloha1_mapping/aloha_viper_finger_tessellation.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path(
            "reports/aloha1_mapping/aloha_viper_finger_tessellation.md"
        ),
    )
    args = parser.parse_args()
    report = build_comparison(args.run_a, args.run_b)
    write_comparison(report, args.json_output, args.markdown_output)
    print(f"status={report['status']}")
    print(f"determinism_gate={report['determinism_gate']}")
    print(f"json={args.json_output.resolve()}")
    print(f"markdown={args.markdown_output.resolve()}")
    return 0 if report["determinism_gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
