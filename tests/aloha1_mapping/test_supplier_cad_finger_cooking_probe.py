from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from tools.cook_aloha1_supplier_cad_finger_geometry import _geometry_signature
from tools.render_aloha1_supplier_cad_finger_cooked_failure import _write_review_reports


def test_geometry_signature_ignores_runtime_but_detects_vertex_change() -> None:
    cooked = {
        "source_sha256": "abc",
        "approximation_readback": "convexHull",
        "runtime_s": 1.5,
        "pieces": [
            {
                "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "indices": [0, 1],
                "polygons": [],
            }
        ],
    }
    changed_runtime = deepcopy(cooked)
    changed_runtime["runtime_s"] = 99.0
    changed_vertex = deepcopy(cooked)
    changed_vertex["pieces"][0]["vertices"][1][0] = 1.001

    assert _geometry_signature(cooked) == _geometry_signature(changed_runtime)
    assert _geometry_signature(cooked) != _geometry_signature(changed_vertex)


def test_screenshot_review_report_preserves_visual_and_geometry_boundaries(
    tmp_path: Path,
) -> None:
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "OFFLINE_COOKED_GEOMETRY_FAILURE_SCREENSHOTS_NOT_RUNTIME_CONTACT",
        "capture_count": 1,
        "captures": [
            {
                "side": "right",
                "approximation": "convexDecomposition",
                "visual_review_status": "PASS",
                "raw_absolute_path": "/tmp/raw.png",
                "raw_sha256": "a" * 64,
                "annotated_absolute_path": "/tmp/annotated.png",
                "annotated_sha256": "b" * 64,
                "visual_review_note": "markers visible",
            }
        ],
        "retake_history": [
            {"attempt": 1, "status": "REJECTED_VECTOR_NOT_LEGIBLE"}
        ],
        "timeline_started": False,
        "runtime_video_required": False,
        "runtime_video_reason": "static cooked geometry only",
        "final_or_default_collider_modified": False,
    }
    geometry_certificate = {
        "status": "PASS_COOKING_DETERMINISTIC",
        "classification": "DECOMPOSITION_MIXED_OR_WORSE",
        "profiles_by_side": {
            "right": {
                "convexDecomposition": {
                    "contact_envelope": {
                        "status": "FAIL_EXCEEDS_TESSELLATION_ERROR_BUDGET",
                        "maximum_contact_surface_deviation_m": 0.001328,
                        "tessellation_error_budget_m": 0.0002,
                    }
                }
            }
        },
    }
    output_json = tmp_path / "review.json"
    output_md = tmp_path / "review.md"

    _write_review_reports(
        manifest,
        geometry_certificate,
        output_json=output_json,
        output_md=output_md,
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["screenshot_evidence_status"] == "PASS"
    assert report["geometry_gate_status"] == "FAIL"
    assert report["final_or_default_collider_modified"] is False
    markdown = output_md.read_text(encoding="utf-8")
    assert "截图证据质量: `PASS`" in markdown
    assert "几何门: `FAIL`" in markdown
    assert "不代表 collider 几何通过" in markdown
