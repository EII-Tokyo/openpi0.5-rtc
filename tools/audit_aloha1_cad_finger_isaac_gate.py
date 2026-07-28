#!/usr/bin/env python3
"""Write the supplier-CAD finger Isaac Stage authorization gate."""

from __future__ import annotations

from pathlib import Path

from tools.aloha1_mapping.cad_finger_isaac_gate import build_gate_report
from tools.aloha1_mapping.cad_finger_isaac_gate import write_gate_report

ROOT = Path(__file__).resolve().parents[1]
ISAAC_ROOT = ROOT / ".venv_issac/lib/python3.11/site-packages/isaacsim"
IMPORTER_ROOT = ISAAC_ROOT / "exts/isaacsim.asset.importer.urdf"


def main() -> int:
    report = build_gate_report(
        mapping_path=ROOT
        / "reports/aloha1_mapping/aloha_public_cad_gripper_mapping.json",
        tessellation_path=ROOT
        / "reports/aloha1_mapping/aloha_viper_finger_tessellation.json",
        source_manifest_path=ROOT
        / "reports/aloha1_mapping/aloha_public_cad_source_manifest.json",
        candidate_stage_path=ROOT
        / "local_eval_assets/aloha_isaac_assets/aloha_viperx.usd",
        importer_api_path=IMPORTER_ROOT / "docs/api.rst",
        importer_manifest_path=IMPORTER_ROOT / "config/extension.toml",
    )
    json_path = (
        ROOT
        / "reports/aloha1_mapping/aloha_viper_cad_finger_isaac_stage_gate.json"
    )
    markdown_path = (
        ROOT
        / "reports/aloha1_mapping/aloha_viper_cad_finger_isaac_stage_gate.md"
    )
    write_gate_report(report, json_path, markdown_path)
    print(f"status={report['status']}")
    print(f"stage_selection={report['stage_selection']['status']}")
    print(f"json={json_path.resolve()}")
    print(f"markdown={markdown_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
