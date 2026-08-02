from __future__ import annotations

import json
from pathlib import Path

from tools.aloha1_mapping.finger_cooked_source_identity import build_source_identity_boundary

ROOT = Path(__file__).resolve().parents[2]
REPORT = (
    ROOT
    / "reports/aloha1_mapping/aloha1_finger_cooked_source_identity_boundary.json"
)


def test_legacy_cooked_report_is_rejected_for_supplier_cad_proof() -> None:
    report = build_source_identity_boundary(ROOT)

    assert report["status"] == "PASS_SOURCE_MISMATCH_DETECTED"
    assert report["classification"] == (
        "LEGACY_COOKED_SOURCE_NOT_CURRENT_SUPPLIER_SOURCE"
    )
    assert report["legacy_cooked_geometry_reusable_for_supplier_cad"] is False
    assert report["next_gate"] == "REQUIRES_SUPPLIER_CAD_COOKED_READBACK"


def test_each_handed_source_retains_exact_provenance_and_geometry_metrics() -> None:
    report = build_source_identity_boundary(ROOT)

    expected_supplier_hashes = {
        "left": "c6710d0fe5b2030a32722d9df5c0b553c771c9d61d92b8ddaec36c94c5963488",
        "right": "b0979c5d55fee448dab512dc75b1251bab17d94892decd01de9a6e76c01482d1",
    }
    expected_legacy_hashes = {
        "left": "df73ae5b9058e5d50a6409ac2ab687dade75053a86591bb5e23ab051dbf2d659",
        "right": "56fb3cc1236d4193106038adf8e457c7252ae9e86c7cee6dabf0578c53666358",
    }
    for side, record in report["records"].items():
        assert record["supplier_cad"]["sha256"] == expected_supplier_hashes[side]
        assert record["legacy_cooked_source"]["sha256"] == expected_legacy_hashes[side]
        assert record["exact_source_hash_match"] is False
        assert record["supplier_cad"]["face_count"] == 1662
        assert record["legacy_cooked_source"]["face_count"] == 1666
        assert record["signed_volume_abs_m3_difference"] > 0.0
        assert record["sorted_aabb_extent_m_difference"] != [0.0, 0.0, 0.0]


def test_repository_report_matches_deterministic_builder() -> None:
    generated = build_source_identity_boundary(ROOT)
    frozen = json.loads(REPORT.read_text(encoding="utf-8"))

    assert frozen == generated
