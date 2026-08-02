from __future__ import annotations

import json
from pathlib import Path

from tools.aloha1_mapping.collider_surface_certificate import build_certificate

ROOT = Path(__file__).resolve().parents[2]
REPORT = (
    ROOT
    / "reports/aloha1_mapping/aloha1_official_collider_surface_certificate.json"
)


def test_certificate_covers_every_physical_link_without_mirroring() -> None:
    report = build_certificate(ROOT)

    assert report["source_completeness"] == "PASS"
    assert report["link_suffixes"] == [
        "base_link",
        "shoulder_link",
        "upper_arm_link",
        "upper_forearm_link",
        "lower_forearm_link",
        "wrist_link",
        "gripper_link",
        "gripper_prop_link",
        "gripper_bar_link",
        "left_finger_link",
        "right_finger_link",
    ]
    assert all(record["mirror_used"] is False for record in report["records"])
    assert all(record["source_contained_by_hulls"] for record in report["records"])


def test_certificate_reports_geometry_error_without_inventing_tolerance() -> None:
    report = build_certificate(ROOT)

    assert report["surface_error_certificate"] == "COMPLETE_NUMERICAL"
    assert report["acceptance_tolerance"] is None
    assert report["acceptance_status"] == "HARD_BLOCKER_ERROR_BUDGET_NOT_DEFINED"
    for record in report["records"]:
        assert record["source_sample_count"] > 0
        assert record["hull_sample_count"] > 0
        assert record["source_to_hull_sample_max_m"] >= 0.0
        assert record["hull_to_source_sample_max_m"] >= 0.0
        assert record["hull_volume_m3"] >= record["source_signed_volume_abs_m3"]
        assert record["degenerate_triangle_count"] >= 0
        assert record["source_mesh_quality_status"] in {
            "PASS_NO_DEGENERATE_TRIANGLES",
            "SOURCE_HAS_DEGENERATE_TRIANGLES_RECORDED_NOT_REPAIRED",
        }


def test_handed_supplier_fingers_remain_distinct_sources() -> None:
    report = build_certificate(ROOT)
    records = {record["link_suffix"]: record for record in report["records"]}

    assert records["left_finger_link"]["source_sha256"] == (
        "c6710d0fe5b2030a32722d9df5c0b553c771c9d61d92b8ddaec36c94c5963488"
    )
    assert records["right_finger_link"]["source_sha256"] == (
        "b0979c5d55fee448dab512dc75b1251bab17d94892decd01de9a6e76c01482d1"
    )
    assert records["left_finger_link"]["source_sha256"] != records[
        "right_finger_link"
    ]["source_sha256"]
    for suffix in ("left_finger_link", "right_finger_link"):
        contact = records[suffix]["inward_contact_surface"]
        assert contact["sample_count"] > 0
        assert contact["tessellation_error_budget_m"] == 0.0002
        assert contact["source_to_hull_boundary_max_m"] > 0.0002
        assert contact["status"] == "FAIL_SINGLE_HULL_RECESSES_CONTACT_SURFACE"


def test_repository_report_matches_deterministic_certificate() -> None:
    generated = build_certificate(ROOT)
    frozen = json.loads(REPORT.read_text(encoding="utf-8"))

    assert frozen["deterministic_signature"] == generated["deterministic_signature"]
