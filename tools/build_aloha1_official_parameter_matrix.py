#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.official_parameter_contract import build_parameter_matrix
from tools.aloha1_mapping.official_parameter_sources import load_source_manifest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs/aloha1_official_parameter_sources.yaml"
DEFAULT_JSON = ROOT / "reports/aloha1_mapping/aloha1_official_parameter_matrix.json"
DEFAULT_MARKDOWN = ROOT / "reports/aloha1_mapping/aloha1_official_parameter_matrix.md"


def _markdown(matrix: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 official parameter coverage matrix",
        "",
        f"- Matrix status: **{matrix['status']}**",
        f"- Formal candidate gate: **{matrix['formal_parameter_candidate_gate']['status']}**",
        f"- Records: `{matrix['record_count']}`",
        f"- Narrow hard blockers: `{matrix['hard_blocker_count']}`",
        f"- Deterministic signature: `{matrix['deterministic_signature']}`",
        "",
        "A matrix `PASS` means all required parameter groups are explicitly inventoried and "
        "schema-valid. It does **not** mean the formal USD candidate may be authored. The "
        "candidate gate remains blocked wherever an exact physical mapping is absent.",
        "",
        "## Coverage",
        "",
        "| Group | Records | Hard blockers |",
        "|---|---:|---:|",
    ]
    for group, coverage in matrix["coverage"].items():
        lines.append(f"| `{group}` | {coverage['record_count']} | {coverage['hard_blocker_count']} |")
    lines.extend(["", "## Hard blockers", ""])
    for item in matrix["formal_parameter_candidate_gate"]["blocking_records"]:
        blocker = item.get("blocker") or {}
        lines.append(f"- `{blocker.get('id', item['id'])}`: {blocker.get('missing_definition', item['id'])}")
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "- No value from machine `192.168.1.103` is used.",
            "- No experimental fit, historical convenient value, or related robot model is used.",
            "- DYNAMIXEL stall torque is retained as a momentary manufacturer rating, not a continuous torque limit.",
            "- Hardware PID/register values are not copied into PhysX stiffness or damping.",
            "- Contact friction and solver policy remain blocked rather than filled with defaults.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    source_manifest = load_source_manifest(args.manifest)
    matrix = build_parameter_matrix(source_manifest, repository_root=ROOT)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(_markdown(matrix), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": matrix["status"],
                "candidate_gate": matrix["formal_parameter_candidate_gate"]["status"],
                "hard_blockers": matrix["hard_blocker_count"],
            }
        )
    )
    return 0 if matrix["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
