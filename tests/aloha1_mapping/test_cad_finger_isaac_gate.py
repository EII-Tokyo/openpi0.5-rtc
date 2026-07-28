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
        authorized_stage_audit_path=ROOT
        / "reports/aloha1_mapping/"
        "aloha_viper_cad_finger_authorized_stage_audit.json",
    )


def test_stage_gate_accepts_user_approved_read_only_candidate() -> None:
    report = _build()
    assert report["status"] == "PARTIAL"
    assert report["stage_selection"]["status"] == "PASS"
    candidate = report["stage_selection"]["approved_review_stage"]
    assert candidate["classification"] == (
        "USER_APPROVED_ISOLATED_DIAGNOSTIC_REVIEW_STAGE"
    )
    assert candidate["root_prim"] == "/workcell"
    assert candidate["read_only"] is True
    assert candidate["source_sha256_before"] == candidate[
        "source_sha256_after"
    ]
    assert candidate["required_key_prims_status"] == "PASS"
    assert candidate["layer_stack_status"] == "PASS"
    assert report["execution_status"] == {
        "isolated_diagnostic_usd": "PASS",
        "isaac_open_closed_screenshots": "PASS",
        "task_5_correct_cad_finger": "NOT_RUN",
        "task_7": "NOT_RUN",
        "task_8": "NOT_RUN",
    }
    assert report["input_gates"]["isolated_diagnostic_asset_pass"] is True
    assert report["input_gates"]["isaac_screenshot_review_pass"] is True
    blocker_ids = {entry["id"] for entry in report["hard_blockers"]}
    assert "ISAAC_REVIEW_STAGE_NOT_USER_APPROVED" not in blocker_ids
    assert "ANGULAR_TESSELLATION_CONTROL_UNAVAILABLE" not in blocker_ids
    assert report["input_gates"][
        "production_angular_tessellation_pass"
    ] is True


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
