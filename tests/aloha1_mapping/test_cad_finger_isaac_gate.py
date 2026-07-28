from __future__ import annotations

import json
from pathlib import Path

from tools.aloha1_mapping.cad_finger_isaac_gate import build_gate_report

ROOT = Path(__file__).resolve().parents[2]
ISAAC_ROOT = ROOT / ".venv_issac/lib/python3.11/site-packages/isaacsim"
IMPORTER_ROOT = ISAAC_ROOT / "exts/isaacsim.asset.importer.urdf"


def _build() -> dict:
    return build_gate_report(
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


def test_stage_gate_blocks_unapproved_historical_candidate() -> None:
    report = _build()
    assert report["status"] == "PARTIAL"
    assert report["stage_selection"]["status"] == "HARD_BLOCKER"
    candidate = report["stage_selection"]["historical_candidate"]
    assert candidate["classification"] == (
        "HISTORICAL_CANDIDATE_NOT_AUTHORIZED_CURRENT_TASK"
    )
    assert candidate["root_prim"] == "UNVERIFIED_NOT_LOADED"
    assert report["execution_status"] == {
        "isolated_diagnostic_usd": "NOT_RUN",
        "isaac_open_closed_screenshots": "NOT_RUN",
        "task_5_correct_cad_finger": "NOT_RUN",
        "task_7": "NOT_RUN",
        "task_8": "NOT_RUN",
    }


def test_saved_stage_gate_matches_recomputed() -> None:
    expected = _build()
    saved = json.loads(
        (
            ROOT
            / "reports/aloha1_mapping/"
            "aloha_viper_cad_finger_isaac_stage_gate.json"
        ).read_text(encoding="utf-8")
    )
    assert saved == expected
