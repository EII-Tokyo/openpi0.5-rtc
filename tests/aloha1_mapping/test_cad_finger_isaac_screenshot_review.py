from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from tools.aloha1_mapping.isaac_screenshot_review import build_review_report
from tools.aloha1_mapping.isaac_screenshot_review import render_markdown


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_review_requires_explicit_visual_pass_for_all_eight_pairs(
    tmp_path: Path,
) -> None:
    captures = []
    annotations = []
    decisions = {}
    for state in ("closed", "open"):
        for view in ("true_top", "true_bottom", "tip_end", "base_oblique"):
            name = f"{state}_{view}"
            raw = tmp_path / f"{name}_raw.png"
            annotated = tmp_path / f"{name}_annotated.png"
            Image.new("RGB", (32, 24), (20, 30, 40)).save(raw)
            Image.new("RGB", (48, 24), (20, 30, 40)).save(annotated)
            camera = {
                "view": view,
                "position_world_m": [1.0, 2.0, 3.0],
                "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                "target_world_m": [0.0, 0.0, 0.0],
            }
            simulation = {
                "state": state,
                "robot": "follower_left",
                "frame": 2,
                "time_s": 1.0 / 30.0,
                "finger_targets_m": (
                    [0.021, -0.021]
                    if state == "closed"
                    else [0.057, -0.057]
                ),
                "finger_readback_m": (
                    [0.021, -0.021]
                    if state == "closed"
                    else [0.057, -0.057]
                ),
                "surface_gap_m": (
                    0.00446 if state == "closed" else 0.07648
                ),
                "visual_type": "SUPPLIER_CAD_V2_VISUAL_ONLY_DIAGNOSTIC",
                "collider_type": "SOURCE_COLLIDER_UNCHANGED",
            }
            captures.append(
                {
                    "capture_name": name,
                    "absolute_path": str(raw),
                    "file_sha256": _sha256(raw),
                    "resolution": [32, 24],
                    "camera": camera,
                    "simulation": simulation,
                }
            )
            annotations.append(
                {
                    "capture_name": name,
                    "raw_absolute_path": str(raw),
                    "raw_sha256": _sha256(raw),
                    "raw_resolution": [32, 24],
                    "annotated_absolute_path": str(annotated),
                    "annotated_sha256": _sha256(annotated),
                    "annotated_resolution": [48, 24],
                    "camera": camera,
                    "simulation": simulation,
                }
            )
            decisions[name] = {
                "raw": "PASS",
                "annotated": "PASS",
                "checks": {
                    "both_fingers_fully_visible": True,
                    "blue_orange_mapping_correct": True,
                    "inward_surfaces_opposed": True,
                    "no_critical_crop": True,
                    "no_critical_occlusion": True,
                    "labels_do_not_overlap": True,
                    "annotations_do_not_cover_key_geometry": True,
                    "visual_gate_only_wording": True,
                },
                "conclusion": "PASS",
                "retake_reason": None,
            }

    report = build_review_report(
        raw_report={
            "status": "PASS",
            "stage_absolute_path": "/diagnostic.usda",
            "stage_sha256_before": "d" * 64,
            "stage_sha256_after": "d" * 64,
            "captures": captures,
        },
        annotation_metadata={
            "status": "PENDING_VISUAL_MODEL_REVIEW",
            "captures": annotations,
        },
        decisions=decisions,
        retake_history=[{"attempt": "v2_attempt1", "status": "REJECTED"}],
        approved_source_stage={
            "absolute_path": "/source.usd",
            "sha256_before": "s" * 64,
            "sha256_after": "s" * 64,
        },
    )

    assert report["status"] == "PASS"
    assert report["capture_count"] == 8
    assert all(
        record["visual_self_review"]["status"] == "PASS"
        for record in report["captures"]
    )
    assert report["gates"]["paired_camera_pose_exact"] is True
    assert report["gates"]["open_closed_visually_distinct"] is True
    assert report["scope_boundaries"]["physics_acceptance"] == "NOT_RUN"
    markdown = render_markdown(report)
    assert "ISAAC_CAD_INSTALLATION_VISUAL_EVIDENCE_ONLY" in markdown
    assert "NO collision/contact/grasp acceptance" in markdown
    assert str(tmp_path / "closed_true_top_raw.png") in markdown
