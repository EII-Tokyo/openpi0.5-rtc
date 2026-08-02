#!/usr/bin/env python3
"""Write the supplier-CAD versus legacy cooked-source identity report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.aloha1_mapping.finger_cooked_source_identity import build_source_identity_boundary
from tools.aloha1_mapping.finger_cooked_source_identity import render_markdown


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "reports/aloha1_mapping/aloha1_finger_cooked_source_identity_boundary.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path(
            "reports/aloha1_mapping/aloha1_finger_cooked_source_identity_boundary.md"
        ),
    )
    args = parser.parse_args()
    report = build_source_identity_boundary(args.root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.json_output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
