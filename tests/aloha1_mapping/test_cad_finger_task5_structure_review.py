from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from tools.aloha1_mapping.cad_finger_task5_structure_review import REQUIRED_CHECKS
from tools.aloha1_mapping.cad_finger_task5_structure_review import build_review_report


def _image(path: Path) -> str:
    Image.new("RGB", (8, 6), (10, 20, 30)).save(path)
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixtures(tmp_path: Path) -> tuple[dict, dict, dict]:
    states = ("closed", "partial", "maximum_legal_aperture")
    views = ("true_top", "true_bottom", "tip_end", "base_oblique")
    raw_records = []
    annotation_records = []
    decisions = {}
    gaps = {"closed": 0.004, "partial": 0.040, "maximum_legal_aperture": 0.076}
    for state in states:
        for view in views:
            name = f"{state}_{view}"
            raw_path = tmp_path / f"{name}_raw.png"
            ann_path = tmp_path / f"{name}_annotated.png"
            raw_hash = _image(raw_path)
            ann_hash = _image(ann_path)
            camera = {
                "actual_position_world_m": [1.0, 2.0, 3.0],
                "actual_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                "target_world_m": [0.0, 0.0, 0.0],
                "resolution": [8, 6],
                "view": view,
            }
            simulation = {
                "state": state,
                "surface_gap_m": gaps[state],
            }
            raw_records.append(
                {
                    "capture_name": name,
                    "absolute_path": str(raw_path),
                    "file_sha256": raw_hash,
                    "resolution": [8, 6],
                    "camera": camera,
                    "simulation": simulation,
                }
            )
            annotation_records.append(
                {
                    "capture_name": name,
                    "raw_sha256": raw_hash,
                    "annotated_absolute_path": str(ann_path),
                    "annotated_sha256": ann_hash,
                    "annotated_resolution": [8, 6],
                    "camera": camera,
                    "simulation": simulation,
                }
            )
            decisions[name] = {
                "raw": "PASS",
                "annotated": "PASS",
                "conclusion": "PASS",
                "checks": dict.fromkeys(REQUIRED_CHECKS, True),
                "notes": "individually reviewed",
            }
    protected = {
        "approved_source_stage": "source-hash",
        "root_usd": "diagnostic-hash",
    }
    raw_report = {
        "status": "FAIL",
        "screenshot_manifest": {"status": "PASS"},
        "gates": {
            "post_step_drive_tracking": False,
            "physx_mimic_or_controller_coupling": False,
        },
        "captures": raw_records,
        "protected_hashes_before": protected,
        "protected_hashes_after": protected,
        "stage_absolute_path": "/diagnostic.usda",
    }
    annotation_metadata = {
        "physics_report_status": "FAIL",
        "captures": annotation_records,
    }
    return raw_report, annotation_metadata, decisions


def test_review_pass_is_separate_from_preserved_physics_fail(
    tmp_path: Path,
) -> None:
    raw_report, metadata, decisions = _fixtures(tmp_path)
    report = build_review_report(
        raw_report=raw_report,
        annotation_metadata=metadata,
        decisions=decisions,
        retake_history=[],
        approved_source_stage={
            "absolute_path": "/approved.usd",
            "sha256_before": "source-hash",
            "sha256_after": "source-hash",
        },
    )

    assert report["status"] == "PASS"
    assert report["capture_count"] == 12
    assert report["separate_gate_status"]["dynamic_drive_tracking"] == "FAIL"
    assert report["scope_boundaries"]["screenshot_pass_unblocks_bottle_test"] is False


def test_review_rejects_missing_visual_check(tmp_path: Path) -> None:
    raw_report, metadata, decisions = _fixtures(tmp_path)
    decisions["closed_true_top"]["checks"]["labels_do_not_overlap"] = False

    with pytest.raises(RuntimeError, match="visual checks did not pass"):
        build_review_report(
            raw_report=raw_report,
            annotation_metadata=metadata,
            decisions=decisions,
            retake_history=[],
            approved_source_stage={
                "absolute_path": "/approved.usd",
                "sha256_before": "source-hash",
                "sha256_after": "source-hash",
            },
        )


def test_review_rejects_hidden_dynamic_pass(tmp_path: Path) -> None:
    raw_report, metadata, decisions = _fixtures(tmp_path)
    raw_report["gates"]["post_step_drive_tracking"] = True

    with pytest.raises(RuntimeError, match="failure was not preserved"):
        build_review_report(
            raw_report=raw_report,
            annotation_metadata=metadata,
            decisions=decisions,
            retake_history=[],
            approved_source_stage={
                "absolute_path": "/approved.usd",
                "sha256_before": "source-hash",
                "sha256_after": "source-hash",
            },
        )
