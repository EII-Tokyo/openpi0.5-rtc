from __future__ import annotations

import json
from pathlib import Path

from tools.aloha1_mapping.cad_source_geometry_resolution import build_aperture_resolution
from tools.aloha1_mapping.cad_source_geometry_resolution import build_link_identity_resolution

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "reports/aloha1_mapping/aloha1_cad_source_geometry_probe.json"
LINK_REPORT = (
    ROOT / "reports/aloha1_mapping/aloha1_cad_link_identity_resolution.json"
)
APERTURE_REPORT = (
    ROOT
    / "reports/aloha1_mapping/aloha1_gripper_aperture_definition_resolution.json"
)


def test_link_resolution_preserves_supplier_cad_source_boundaries() -> None:
    report = build_link_identity_resolution(ROOT, PROBE)

    assert report["status"] == "PASS"
    assert report["source_cad"]["sha256"] == (
        "337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571"
    )
    assert report["toolchain"] == {
        "freecad_version": "1.1.1",
        "opencascade_version": "7.8.1",
        "linear_deflection_mm": 0.2,
        "angular_deflection_deg": 20.0,
    }
    assert report["two_fresh_process_determinism"] == "PASS"
    assert report["mirror_used"] is False
    assert report["source_brep_repaired"] is False


def test_combined_gripper_is_not_falsely_split_into_cad_products() -> None:
    report = build_link_identity_resolution(ROOT, PROBE)
    records = {record["link_suffix"]: record for record in report["records"]}

    for suffix in ("gripper_bar_link", "gripper_prop_link"):
        record = records[suffix]
        assert record["supplier_cad_identity"] == (
            "COMBINED_GRIPPER_SOLID_NO_INDEPENDENT_PRODUCT"
        )
        assert record["authoritative_link_geometry_source"] == (
            "PINNED_OFFICIAL_URDF_MESH"
        )
        assert record["resolution_status"] == (
            "RESOLVED_WITH_EXPLICIT_SOURCE_BOUNDARY"
        )
        assert record["cad_subgeometry_claim"] == "NOT_CLAIMED"


def test_invalid_supplier_wrist_brep_is_preserved_not_repaired() -> None:
    report = build_link_identity_resolution(ROOT, PROBE)
    record = next(
        item for item in report["records"] if item["link_suffix"] == "wrist_link"
    )

    assert record["supplier_cad_identity"] == "EXPOSED_INVALID_BREP"
    assert "Self-intersecting wire" in record["cad_brep_diagnostics"]
    assert "Unorientable shape" in record["cad_brep_diagnostics"]
    assert record["authoritative_link_geometry_source"] == (
        "PINNED_OFFICIAL_URDF_MESH"
    )
    assert record["source_brep_repaired"] is False


def test_aperture_resolution_keeps_geometry_definitions_separate() -> None:
    report = build_aperture_resolution(ROOT, PROBE)

    assert report["status"] == "PASS"
    assert report["cad_carriage_center_distance_m"] == {
        "closed_reference": 0.042,
        "open_derived": 0.114,
    }
    assert report["urdf_carriage_center_distance_m"] == [0.042, 0.114]
    assert report["trossen_product_table_range_m"] == [0.042, 0.116]
    assert report["contact_surface_gap_is_single_scalar"] is False
    assert report["source_conflict"]["classification"] == (
        "VERIFIED_OFFICIAL_SOURCE_CONFLICT_PRODUCT_PAGE_NOT_CAD_SUPPORTED"
    )
    assert report["implemented_joint_range_source"] == (
        "PINNED_OFFICIAL_URDF_AND_CAD_CARRIAGE_DATUM"
    )
    assert report["fitted_endpoint_used"] is False


def test_committed_resolution_reports_match_builders() -> None:
    link = build_link_identity_resolution(ROOT, PROBE)
    aperture = build_aperture_resolution(ROOT, PROBE)

    assert json.loads(LINK_REPORT.read_text(encoding="utf-8")) == link
    assert json.loads(APERTURE_REPORT.read_text(encoding="utf-8")) == aperture
