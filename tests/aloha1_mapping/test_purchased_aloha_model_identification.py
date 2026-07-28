from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.aloha1_mapping.purchased_aloha_model_identification import build_model_identification_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF = (
    PROJECT_ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "purchased_aloha_reference/Aloha VX300 6DOF Drawing 2024-5-13.pdf"
)


def _report() -> dict:
    return build_model_identification_report(
        PDF,
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_public_cad_assembly_audit.json",
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_widow_gripper_assembly_audit.json",
    )


def test_purchased_drawing_directly_identifies_vx300s_follower() -> None:
    report = _report()

    assert report["status"] == "PASS"
    assert report["drawing_identity"]["project"] == "Aloha ViperX 6DOF"
    assert report["drawing_identity"]["title"] == (
        "Aloha VX300S Follower Robot Arm"
    )
    assert report["classification"] == (
        "SIMPLE_ALOHA_VIPER_2024_5_13_STEP"
    )
    assert report["classification_confidence"] == (
        "DIRECT_MODEL_AND_DIMENSION_MATCH"
    )


def test_viper_base_dimensions_match_drawing_and_widow_does_not() -> None:
    report = _report()
    viper = report["candidate_comparison"]["simple_aloha_viper"]
    widow = report["candidate_comparison"]["aloha_widow_with_gripper"]

    assert viper["cad_model_family"] == "VX"
    assert viper["base_dimensions_mm"]["x"] == pytest.approx(204.0)
    assert viper["base_dimensions_mm"]["y"] == pytest.approx(299.4629868)
    assert viper["drawing_base_absolute_error_mm"]["x"] < 1.0e-6
    assert viper["drawing_base_absolute_error_mm"]["y"] < 0.0031
    assert widow["cad_model_family"] == "WX"
    assert widow["base_dimensions_mm"]["x"] == pytest.approx(153.072)
    assert widow["base_dimensions_mm"]["y"] == pytest.approx(233.536)
    assert widow["drawing_base_absolute_error_mm"]["x"] > 50.0
    assert widow["drawing_base_absolute_error_mm"]["y"] > 65.0


def test_visual_similarity_is_explained_by_shared_finger_assembly() -> None:
    report = _report()
    shared = report["shared_gripper_explanation"]

    assert shared["same_finger_labels"] is True
    assert shared["same_finger_topology"] is True
    assert shared["same_finger_volumes"] is True
    assert shared["same_finger_pair_dimensions"] is True
    assert shared["conclusion"] == (
        "gripper/finger visual similarity is expected and does not identify "
        "the arm model"
    )


def test_standalone_finger_dimension_matches_drawing_81_71_callout() -> None:
    report = _report()
    match = report["finger_dimension_cross_check"]

    assert match["drawing_callout_mm"] == 81.71
    assert match["standalone_vx_finger_bbox_dimension_mm"] == pytest.approx(
        81.7075881151
    )
    assert match["absolute_error_mm"] < 0.0025
    assert match["role"] == (
        "supports finger-family identity but does not replace assembly "
        "installation transforms"
    )


def test_saved_identification_report_matches_recomputed_report() -> None:
    expected = _report()
    saved = json.loads(
        (
            PROJECT_ROOT
            / "reports/aloha1_mapping/aloha_purchased_model_identification.json"
        ).read_text(encoding="utf-8")
    )
    assert saved == expected


def test_first_party_source_chain_records_sales_page_and_all_downloads() -> None:
    report = _report()
    sources = report["first_party_source_chain"]

    assert sources["sales_page"]["url"] == (
        "https://idminer.com.tw/product/aloha-viperx/"
    )
    assert sources["sales_catalog"]["google_drive_file_id"] == (
        "11KcnA49dhTiOD_MxmmC_SG75Cs97-JKh"
    )
    assert sources["technical_drawing"]["google_drive_file_id"] == (
        "11M96-4JDw0y31OZMTQQ3Nqz1qCIqk_DU"
    )
    assert sources["public_3d_cad"]["google_drive_folder_id"] == (
        "1mhJuhzT4lBnvZ9VE57UgT6vmJDFPVsBf"
    )
    assert sources["trossen_manual"]["url"] == (
        "https://docs.trossenrobotics.com/aloha_docs/"
    )
    assert Path(sources["sales_page"]["local_snapshot_path"]).is_file()
    assert Path(sources["sales_catalog"]["local_path"]).is_file()
    assert sources["sales_catalog"]["sha256"] == (
        "d06346070022f300b8fb73176fbeeaf4eb096300238a1142cde5a2399c3f3888"
    )
