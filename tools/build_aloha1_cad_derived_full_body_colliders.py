#!/usr/bin/env python3
"""Build the isolated ALOHA CAD-derived link-collider geometry report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.cad_derived_link_colliders import build_candidate

ROOT = Path(__file__).resolve().parents[1]
SOURCE_STEP = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "gdrive_source_readonly/Simple Aloha Viper 2024-5-13.step"
)
SOURCE_STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/"
    "aloha1_signal_correspondence_workcell.usda"
)
OUTPUT_ROOT = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_derived_full_body_colliders/1.0"
)
REPORT_JSON = (
    ROOT / "reports/aloha1_mapping/aloha1_cad_derived_collider_geometry.json"
)
REPORT_MD = (
    ROOT / "reports/aloha1_mapping/aloha1_cad_derived_collider_geometry.md"
)
PROFILE = "CAD_SUBPART_COMPOUND_CONVEX_HULL"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 CAD-derived collider geometry",
        "",
        f"- Status: `{report['status']}`",
        f"- Profile: `{report['profile']}`",
        f"- Two-run determinism: `{report['two_fresh_directory_determinism']}`",
        "- Final/default collider modified: `false`",
        "- Task 8: `NOT_RUN`",
        "",
        "| Robot/link | Source | Result | Pieces | Triangles | Registration |",
        "|---|---|---|---:|---:|---|",
    ]
    lines.extend(
        (
            f"| {record['urdf_link_name']} | {record['source_object']} | "
            f"{record['status']} | {record['convex_piece_count']} | "
            f"{record['triangle_count']} | {record['registration_method']} |"
        )
        for record in report["physical_link_records"]
    )
    lines.extend(
        [
            "",
            "`wrist_link` remains blocked because supplier `Part__Feature005` "
            "fails the B-Rep validity gate. No repair was applied. "
            "`gripper_prop_link` and `gripper_bar_link` remain identity blockers; "
            "no collider was invented for them.",
            "",
            "Surface-distance values compare different supplier/URDF revisions "
            "and are diagnostic only; they do not select orientation or hide "
            "registration failures.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    report = build_candidate(
        source_step=SOURCE_STEP,
        source_stage=SOURCE_STAGE,
        output_root=OUTPUT_ROOT,
        profile=PROFILE,
    )
    REPORT_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    REPORT_MD.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "physical_link_records": len(report["physical_link_records"]),
                "invalid_brep_blockers": report["invalid_brep_blockers"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
