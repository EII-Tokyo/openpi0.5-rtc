from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/aloha1_bottle_asset.yaml"
REFERENCE_MANIFEST = ROOT / "reports/aloha1_mapping/aloha_bottle_cad_source_manifest.json"
PROJECT_AUDIT = ROOT / "reports/aloha1_mapping/aloha_project_bottle_cad_audit.json"
COMPARISON = ROOT / "reports/aloha1_mapping/aloha_bottle_cad_comparison.json"
SCREENSHOTS = ROOT / "reports/aloha1_mapping/aloha_bottle_cad_screenshot_review.json"
TROSSEN = ROOT / "reports/aloha1_mapping/aloha_vx300s_official_reference_manifest.json"
README = ROOT / "README_ALOHA1_ISAACSIM_5_1.md"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_project_bottle_is_primary_and_reference_step_is_not_grasp_scope() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    reference = _load_json(REFERENCE_MANIFEST)

    assert config["selection_status"] == "ACTIVE_PRIMARY_FOR_FUTURE_DIGITAL_GRASP_TESTS"
    assert reference["selection"]["status"] == "GEOMETRY_REFERENCE_ONLY_NOT_DEFAULT_FOR_GRASP"
    assert reference["selection"]["scope"] == [
        "geometry_comparison",
        "dimensional_cross_check",
        "visual_reference",
    ]
    assert "follower_left_bottle_grasp" in config["selection_scope"]
    assert "follower_left_bottle_grasp" not in reference["selection"]["scope"]


def test_primary_bottle_sources_match_recorded_hashes() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    primary = config["primary_cad"]

    for path_key, hash_key in (
        ("build_script_absolute_path", "build_script_sha256"),
        ("fcstd_absolute_path", "fcstd_sha256"),
        ("exported_step_absolute_path", "exported_step_sha256"),
    ):
        source = Path(primary[path_key])
        assert source.is_file()
        assert _sha256(source) == primary[hash_key]


def test_project_bottle_cad_audit_and_comparison_preserve_evidence_boundary() -> None:
    audit = _load_json(PROJECT_AUDIT)
    comparison = _load_json(COMPARISON)

    assert audit["status"] == "PASS"
    assert "BottleMaster" in audit["fcstd"]["objects"]
    assert audit["bottle_master"]["solid_count"] == 1
    assert audit["bottle_master"]["optimal_bbox_mm"] == {
        "x": 68.0,
        "y": 68.0,
        "z": 206.0,
        "api": "Part.Shape.optimalBoundingBox()",
    }
    assert audit["fcstd_to_step_comparison"]["status"] == "PASS"
    assert comparison["status"] == "PASS"
    assert comparison["selection_decision"]["primary_status"] == "ACTIVE_PRIMARY_FOR_FUTURE_DIGITAL_GRASP_TESTS"
    assert comparison["selection_decision"]["reference_status"] == "GEOMETRY_REFERENCE_ONLY_NOT_DEFAULT_FOR_GRASP"
    assert comparison["physics_status"] == "NOT_RUN_THIS_AUDIT"
    assert comparison["task8_status"] == "NOT_RUN"


def test_screenshot_review_records_six_raw_and_annotated_pairs() -> None:
    report = _load_json(SCREENSHOTS)

    assert report["status"] == "PASS"
    assert report["visual_model_self_review"] == "PASS"
    assert report["user_review"] == "PENDING"
    assert len(report["captures"]) == 6
    for capture in report["captures"]:
        assert capture["status"] == "PASS"
        assert capture["visual_self_review"] == "PASS"
        for path_key, hash_key in (
            ("raw_path", "raw_sha256"),
            ("annotated_path", "annotated_sha256"),
        ):
            image = Path(capture[path_key])
            assert image.is_absolute()
            assert image.is_file()
            assert _sha256(image) == capture[hash_key]


def test_trossen_vx300s_first_party_reference_is_registered() -> None:
    report = _load_json(TROSSEN)

    assert report["status"] == "PASS"
    assert report["source"]["publisher"] == "Trossen Robotics"
    assert report["source"]["url"].startswith("https://docs.trossenrobotics.com/")
    assert report["official_facts"]["degrees_of_freedom"] == 6
    assert report["official_facts"]["reach_mm"] == 750
    assert report["official_facts"]["working_payload_g"] == 750
    assert report["official_facts"]["total_servos"] == 9


def test_readme_records_primary_bottle_and_reference_limits() -> None:
    text = README.read_text(encoding="utf-8")

    assert "### Primary bottle CAD and downloaded geometry reference" in text
    assert "assets/bottle_500ml/cad/bottle_500ml.FCStd" in text
    assert "3594f60200e54181bc8480a229484293a0d386c146d3f235b32e31a0c16bbf8a" in text
    assert "geometry reference, not the default grasp bottle" in text
    assert "UNKNOWN_HARD_BLOCKER" in text
    assert "Task 8 optimization | **COMPLETE / NO_PROMOTION**" in text
    assert "The candidate remains diagnostic and is not promoted." in text
