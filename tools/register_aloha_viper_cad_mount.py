#!/usr/bin/env python3
"""Generate the supplier-CAD to Stage mounting datum registration report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.aloha1_mapping.cad_mount_registration import build_mount_registration_report
from tools.aloha1_mapping.cad_mount_registration import render_mount_registration_markdown


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--cad-shell-obj", type=Path, required=True)
    parser.add_argument("--follower-urdf", type=Path, required=True)
    parser.add_argument("--gripper-stl", type=Path, required=True)
    parser.add_argument("--gripper-bar-stl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    report = build_mount_registration_report(
        probe_manifest_path=args.probe_manifest.resolve(strict=True),
        cad_shell_obj_path=args.cad_shell_obj.resolve(strict=True),
        follower_urdf_path=args.follower_urdf.resolve(strict=True),
        gripper_stl_path=args.gripper_stl.resolve(strict=True),
        gripper_bar_stl_path=args.gripper_bar_stl.resolve(strict=True),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(
        render_mount_registration_markdown(report),
        encoding="utf-8",
    )
    print(f"status={report['status']}")
    print(f"json={args.output_json.resolve()}")
    print(f"markdown={args.output_md.resolve()}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
