#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.official_parameter_sources import build_source_audit
from tools.aloha1_mapping.official_parameter_sources import load_source_manifest
from tools.aloha1_mapping.official_parameter_sources import validate_source_manifest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs/aloha1_official_parameter_sources.yaml"
DEFAULT_JSON = ROOT / "reports/aloha1_mapping/aloha1_official_parameter_source_audit.json"
DEFAULT_MARKDOWN = ROOT / "reports/aloha1_mapping/aloha1_official_parameter_source_audit.md"


def _markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 official parameter source audit",
        "",
        f"- Status: **{audit['status']}**",
        f"- Product: `{audit['product']['product']}` / `{audit['product']['project_model']}`",
        f"- Frozen required sources: `{audit['source_count']}`",
        f"- Formal parameter candidate gate: **{audit['formal_parameter_candidate_gate']}**",
        f"- Deterministic signature: `{audit['deterministic_signature']}`",
        "",
        "## Frozen source chain",
        "",
        "| ID | Authority | Evidence | Commit / SHA-256 |",
        "|---|---|---|---|",
    ]
    for source in audit["sources"]:
        pinned = source.get("commit") or source.get("sha256")
        lines.append(f"| `{source['id']}` | {source['authority']} | `{source['evidence_class']}` | `{pinned}` |")
    lines.extend(
        [
            "",
            "## Retained official-source conflict",
            "",
            "Trossen's ViperX-300 page contains contradictory ID 6/7 joint-name tables. "
            "The conflict is retained. The pinned official motor configurations and Xacro "
            "support `ID6=forearm_roll` and `ID7=wrist_angle`; this resolution does not "
            "rewrite or hide the contradictory webpage row.",
            "",
            "## Local mirror boundary",
            "",
            "`external/ros2-essentials` is a third-party aggregate mirror, not the Interbotix "
            "authority. Its local `aloha_vx300s.yaml` sleep positions differ from the pinned "
            "upstream file, so that local sleep pose is not labeled official.",
            "",
            "## License boundary",
            "",
            "The supplier STEP is user-confirmed public vendor material, but no formal "
            "redistribution license text was found. It remains local read-only evidence and "
            "is not committed or redistributed (`UNKNOWN_HARD_BLOCKER` only for redistribution).",
            "",
            "## Findings",
            "",
        ]
    )
    if audit["findings"]:
        lines.extend(f"- `{item['code']}`: {item['message']}" for item in audit["findings"])
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    manifest = load_source_manifest(args.manifest)
    findings = validate_source_manifest(manifest, repository_root=ROOT, verify_files=True)
    audit = build_source_audit(manifest, findings)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(_markdown(audit), encoding="utf-8")
    print(json.dumps({"status": audit["status"], "findings": len(findings)}))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
