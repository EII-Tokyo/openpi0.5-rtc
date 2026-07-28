#!/usr/bin/env python3
"""Generate the immutable manifest for the public ALOHA Drive CAD cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.aloha1_mapping.drive_cad_sources import build_public_cad_manifest
from tools.aloha1_mapping.drive_cad_sources import write_public_cad_reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "reports/aloha1_mapping/aloha_public_cad_source_manifest.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path(
            "reports/aloha1_mapping/aloha_public_cad_source_manifest.md"
        ),
    )
    args = parser.parse_args()
    manifest = build_public_cad_manifest(args.source_root)
    write_public_cad_reports(
        manifest,
        args.json_output,
        args.markdown_output,
    )
    print(f"status={manifest['status']}")
    print(
        "files="
        f"{manifest['present_file_count']}/{manifest['expected_file_count']}"
    )
    print(f"json={args.json_output.resolve()}")
    print(f"markdown={args.markdown_output.resolve()}")
    return 0 if manifest["inventory_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
