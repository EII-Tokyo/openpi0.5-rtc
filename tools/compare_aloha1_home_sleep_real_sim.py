#!/usr/bin/env python3
"""Compare immutable ALOHA Home/Sleep real and digital traces when both exist."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.home_sleep_correspondence import ARM_JOINT_ORDER
from tools.aloha1_mapping.home_sleep_correspondence import compare_aligned_joint_rows

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/aloha1_mapping"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(path: Path, q_field: str) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            {
                "command_index": int(row["command_index"]),
                "q": json.loads(row[q_field])[:6],
            }
            for row in csv.DictReader(stream)
        ]
    # Physics telemetry repeats held command indices; retain the final readback per command.
    deduplicated = {int(row["command_index"]): row for row in rows}
    return [deduplicated[index] for index in sorted(deduplicated)]


def build_missing_real_report(
    *, digital_path: Path, digital_sha256: str, command_signature: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "NOT_RUN_REAL_EVIDENCE_MISSING",
        "classification": "REAL_DIGITAL_COMPARISON_NOT_RUN",
        "digital_telemetry": {
            "absolute_path": str(digital_path.resolve()),
            "sha256": digital_sha256,
        },
        "real_telemetry": None,
        "command_signature": command_signature,
        "layers": {
            "COMMAND_IDENTITY": "NOT_RUN_REAL_EVIDENCE_MISSING",
            "JOINT_SEMANTICS": "NOT_RUN_REAL_EVIDENCE_MISSING",
            "KINEMATIC_ENDPOINT_CORRESPONDENCE": "NOT_RUN_REAL_EVIDENCE_MISSING",
            "DYNAMIC_TRAJECTORY_CORRESPONDENCE": "NOT_RUN_REAL_EVIDENCE_MISSING",
        },
        "raw_inputs_modified": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--digital",
        type=Path,
        default=REPORT_ROOT / "aloha1_home_sleep_digital_telemetry_run_01.csv",
    )
    parser.add_argument("--real", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPORT_ROOT / "aloha1_home_sleep_command_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_ROOT / "aloha1_home_sleep_real_sim_comparison.json",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.real is None or not args.real.is_file():
        report = build_missing_real_report(
            digital_path=args.digital,
            digital_sha256=_sha256(args.digital),
            command_signature=manifest["command_signature"],
        )
    else:
        comparison = compare_aligned_joint_rows(
            _load_rows(args.digital, "left_q"),
            _load_rows(args.real, "q"),
            joint_names=ARM_JOINT_ORDER,
        )
        report = {
            "schema_version": 1,
            "status": "PARTIAL_UNCALIBRATED_THRESHOLDS",
            "classification": "REAL_DIGITAL_NUMERIC_COMPARISON",
            "digital_telemetry": {
                "absolute_path": str(args.digital.resolve()),
                "sha256": _sha256(args.digital),
            },
            "real_telemetry": {
                "absolute_path": str(args.real.resolve()),
                "sha256": _sha256(args.real),
            },
            "command_signature": manifest["command_signature"],
            "metrics": comparison,
            "layers": {
                "COMMAND_IDENTITY": "PARTIAL",
                "JOINT_SEMANTICS": "PARTIAL",
                "KINEMATIC_ENDPOINT_CORRESPONDENCE": "PARTIAL",
                "DYNAMIC_TRAJECTORY_CORRESPONDENCE": "PARTIAL",
            },
            "raw_inputs_modified": False,
        }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
