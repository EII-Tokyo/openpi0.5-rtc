from __future__ import annotations

import json
from pathlib import Path

from tools.derive_aloha1_gripper_geometry_contract import build_contract

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/aloha1_mapping/aloha1_gripper_geometry_contract.json"


def test_linkage_formula_is_monotonic_symmetric_and_source_bound() -> None:
    contract = build_contract(ROOT)

    assert contract["formula_validation"]["status"] == "PASS"
    assert contract["formula_validation"]["sample_count"] == 1001
    assert contract["formula_validation"]["monotonic"] is True
    assert contract["formula_validation"]["right_is_negative_left"] is True
    assert contract["source_cad"]["sha256"] == "337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571"


def test_aperture_conflict_is_defined_and_retained_not_fitted_away() -> None:
    contract = build_contract(ROOT)

    assert contract["aperture"]["urdf_carriage_center_range_m"] == [0.042, 0.114]
    assert contract["aperture"]["trossen_exact_product_claim_m"] == [0.042, 0.116]
    assert contract["aperture"]["status"] == (
        "PASS_WITH_DOCUMENTED_OFFICIAL_SOURCE_CONFLICT"
    )
    assert contract["aperture"]["contact_surface_gap_is_single_scalar"] is False
    assert contract["aperture"]["implemented_joint_range_source"] == (
        "PINNED_OFFICIAL_URDF_AND_CAD_CARRIAGE_DATUM"
    )
    assert contract["status"] == "PASS"
    assert contract["formal_candidate_gate"] == "PASS"


def test_report_matches_deterministic_contract() -> None:
    contract = build_contract(ROOT)
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["deterministic_signature"] == contract["deterministic_signature"]
