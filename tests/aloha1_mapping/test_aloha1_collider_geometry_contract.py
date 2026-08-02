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


def test_explicit_source_boundaries_remove_false_cad_identity_blockers() -> None:
    contract = build_contract(ROOT)

    assert contract["existing_swept_collision_gate"] == "PASS"
    assert contract["status"] == "PARTIAL"
    assert contract["formal_candidate_gate"] == "BLOCKED"
    assert contract["unresolved_identity_blocker_count"] == 0
    assert contract["unresolved_link_suffixes"] == []
    assert contract["link_identity_resolution"] == "PASS"
    assert contract["surface_error_certificate"] == "COMPLETE_NUMERICAL"
    assert contract["surface_error_acceptance"] == ("HARD_BLOCKER_ERROR_BUDGET_NOT_DEFINED")
    assert contract["surface_certificate_link_count"] == 11
    assert contract["supplier_finger_exact_brep_gate"] == ("ALL_PROFILES_CROSS_INWARD_CAD_SURFACE")
    assert contract["supplier_finger_decomposition_comparison"] == ("DECOMPOSITION_MIXED_OR_WORSE")
    assert contract["supplier_finger_exact_asset_decision"] == ("REJECTED_EXACT_CAD_CONTACT_GATE")
    assert contract["supplier_finger_task_local_acceptance"] == ("HARD_BLOCKER_NOT_DERIVED_OR_MEASURED")
    assert contract["supplier_finger_task_contact_band_status"] == (
        "FAIL_CENTRAL_TANGENCY_OUTSIDE_COMPOUND_PATCH"
    )
    assert contract["supplier_finger_task_contact_band_decision"] == (
        "REJECTED_TASK_CONTACT_BAND_NOT_PROMOTED"
    )


def test_report_matches_deterministic_contract() -> None:
    contract = build_contract(ROOT)
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["deterministic_signature"] == contract["deterministic_signature"]


def test_compound_contact_candidate_is_recorded_without_promotion() -> None:
    contract = build_contract(ROOT)

    assert contract["supplier_finger_compound_runtime_status"] == (
        "PASS_RUNTIME_COOKED_CONTACT_REGION_GEOMETRY_NOT_PROMOTED"
    )
    assert contract["supplier_finger_compound_coordinate_frame"] == ("FINGER_LINK_LOCAL_METRES")
    assert contract["supplier_finger_compound_contact_region_gates"] == {
        "left": "PASS",
        "right": "PASS",
    }
    assert contract["supplier_finger_compound_full_face_scope"] == ("PARTIAL_CONTACT_REGION_ONLY")
    assert contract["supplier_finger_compound_usd_status"] == ("PASS_GEOMETRY_ONLY_DIAGNOSTIC_USD")
    assert contract["supplier_finger_compound_usd_piece_count"] == 68
    assert contract["supplier_finger_task_contact_band_minimum_patch_miss_m"] > 0.0015
    assert contract["formal_candidate_gate"] == "BLOCKED"
    assert contract["final_or_default_asset_modified"] is False
