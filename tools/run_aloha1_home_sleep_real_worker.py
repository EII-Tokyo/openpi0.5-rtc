#!/usr/bin/env python3
"""Fail-closed entry point for the synchronized ALOHA real worker."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT / "reports/aloha1_mapping/aloha1_home_sleep_command_manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/aloha1_mapping/aloha1_home_sleep_sync_real_worker_dry_run.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dry_run_report(*, manifest_sha256: str, sample_count: int) -> dict[str, Any]:
    """Return a report that proves the real transport was not instantiated."""

    return {
        "schema_version": 1,
        "mode": "DRY_RUN",
        "status": "NOT_RUN_AUTHORIZATION_REQUIRED",
        "manifest_sha256": manifest_sha256,
        "planned_samples": int(sample_count),
        "network_access_performed": False,
        "ssh_connection_opened": False,
        "ros_transport_instantiated": False,
        "serial_device_opened": False,
        "commands_published": 0,
        "torque_changed": False,
        "real_access_authorized": False,
        "real_motion_authorized": False,
        "present_current_required": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--transport", choices=("fake",), default="fake")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = build_dry_run_report(
        manifest_sha256=_sha256(manifest_path),
        sample_count=int(manifest["sample_count"]),
    )
    report["manifest"] = {
        "absolute_path": str(manifest_path),
        "command_signature": manifest["command_signature"],
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "commands_published": 0,
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
