#!/usr/bin/env python3
"""Synchronized dual-follower publisher; fail-closed by default.

This entry point is suitable for deployment into the approved project path on
103.  It performs no ROS import or network mutation unless both
``--execute-real`` and ``--allow-dual-real-motion`` are supplied.  The current
workflow intentionally uses its dry-run path until a fresh physical-motion
authorization is granted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.dual_real_publisher import build_dual_dry_run_report
from tools.aloha1_mapping.dual_real_publisher import validate_dual_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, report: dict[str, Any]) -> None:
    path.resolve().parent.mkdir(parents=True, exist_ok=True)
    path.resolve().write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-manifest", type=Path, required=True)
    parser.add_argument("--right-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-delay-s", type=float, default=3.0)
    parser.add_argument("--left-role", default="puppet_left")
    parser.add_argument("--right-role", default="puppet_right")
    parser.add_argument("--execute-real", action="store_true")
    parser.add_argument("--allow-dual-real-motion", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    left_path = args.left_manifest.resolve(strict=True)
    right_path = args.right_manifest.resolve(strict=True)
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    validated = validate_dual_manifest(left, right)
    report = build_dual_dry_run_report(
        left_sha256=_sha256(left_path),
        right_sha256=_sha256(right_path),
        sample_count=validated["sample_count"],
    )
    report["left_manifest"] = str(left_path)
    report["right_manifest"] = str(right_path)
    report["validation"] = validated
    report["execute_real_requested"] = bool(args.execute_real)
    report["dual_motion_flag_present"] = bool(args.allow_dual_real_motion)
    report["roles"] = {"left": args.left_role, "right": args.right_role}
    report["status_reason"] = "A separate reviewed live implementation is required before any dual publish"
    # Deliberately do not implement a live path in this first landing.  A
    # future change must add a reviewed ROS adapter and a fresh authorization.
    _write(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
