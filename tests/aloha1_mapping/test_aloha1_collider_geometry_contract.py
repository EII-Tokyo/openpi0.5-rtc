from __future__ import annotations

import json
from pathlib import Path

from tools.audit_aloha1_collider_geometry_contract import build_contract

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/aloha1_mapping/aloha1_collider_geometry_contract.json"


def test_collider_contract_preserves_cad_and_pinned_tessellation_identity() -> None:
    contract = build_contract(ROOT)

    assert contract["source_cad"]["sha256"] == "337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571"
    assert contract["toolchain"] == {
        "freecad_version": "1.1.1",
        "opencascade_version": "7.8.1",
        "linear_deflection_mm": 0.2,
        "angular_deflection_deg": 20.0,
    }
    assert contract["two_fresh_directory_determinism"] == "PASS"
    assert contract["final_or_default_asset_modified"] is False


def test_existing_swept_checks_do_not_erase_unmapped_link_blockers() -> None:
    contract = build_contract(ROOT)

    assert contract["existing_swept_collision_gate"] == "PASS"
    assert contract["status"] == "PARTIAL"
    assert contract["formal_candidate_gate"] == "BLOCKED"
    assert contract["unresolved_identity_blocker_count"] == 6
    assert "wrist_link" in contract["unresolved_link_suffixes"]
    assert "gripper_bar_link" in contract["unresolved_link_suffixes"]
    assert "gripper_prop_link" in contract["unresolved_link_suffixes"]


def test_report_matches_deterministic_contract() -> None:
    contract = build_contract(ROOT)
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["deterministic_signature"] == contract["deterministic_signature"]
