from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.aloha1_mapping.bottle_swept_contact_band_certificate import build_certificate
from tools.aloha1_mapping.bottle_swept_contact_band_certificate import classify_task_contact_band
from tools.aloha1_mapping.bottle_swept_contact_band_certificate import rectangle_point_metrics

ROOT = Path(__file__).resolve().parents[2]
FROZEN_REPORT = ROOT / "reports/aloha1_mapping/aloha1_bottle_swept_contact_band_collider_certificate.json"


def test_rectangle_point_metrics_distinguishes_plane_from_finite_patch() -> None:
    rectangle = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )

    inside = rectangle_point_metrics(
        rectangle,
        np.asarray([0.25, 0.0, 0.5]),
        numeric_tolerance_m=1.0e-9,
    )
    outside = rectangle_point_metrics(
        rectangle,
        np.asarray([-0.2, 0.0, 0.5]),
        numeric_tolerance_m=1.0e-9,
    )

    assert inside["point_on_plane"] is True
    assert inside["inside_finite_rectangle"] is True
    assert inside["minimum_in_plane_distance_to_rectangle_m"] == pytest.approx(0.0)
    assert outside["point_on_plane"] is True
    assert outside["inside_finite_rectangle"] is False
    assert outside["minimum_in_plane_distance_to_rectangle_m"] == pytest.approx(0.2)


def test_task_contact_band_rejects_plane_only_coverage() -> None:
    result = classify_task_contact_band(
        {
            "left": {
                "center_tangent_point_on_cad_plane": True,
                "center_tangent_point_inside_cooked_patch": False,
            },
            "right": {
                "center_tangent_point_on_cad_plane": True,
                "center_tangent_point_inside_cooked_patch": False,
            },
        }
    )

    assert result["status"] == "FAIL_CENTRAL_TANGENCY_OUTSIDE_COMPOUND_PATCH"
    assert result["candidate_decision"] == "REJECTED_TASK_CONTACT_BAND_NOT_PROMOTED"


def test_real_certificate_rejects_current_compound_without_using_grasp_success() -> None:
    report = build_certificate(ROOT)

    assert report["status"] == "PASS_DETERMINISTIC_REJECTION"
    assert report["task_contact_band"]["status"] == ("FAIL_CENTRAL_TANGENCY_OUTSIDE_COMPOUND_PATCH")
    assert report["candidate_decision"] == ("REJECTED_TASK_CONTACT_BAND_NOT_PROMOTED")
    assert report["grasp_success_used_to_set_tolerance"] is False
    assert report["final_or_default_collider_modified"] is False
    assert report["timeline_started"] is False
    for side in ("left", "right"):
        finger = report["fingers"][side]
        assert finger["center_tangent_point_on_cad_plane"] is True
        assert finger["center_tangent_point_inside_cooked_patch"] is False
        assert finger["minimum_in_plane_distance_to_rectangle_m"] > 0.0015
        assert finger["minimum_in_plane_distance_to_rectangle_m"] < 0.0017
        assert finger["cooked_contact_normal_max_error_deg"] < 1.0e-4
        assert finger["maximum_outward_crossing_m"] < finger["numeric_tolerance_m"]
    assert report["known_numerical_error_budget"]["known_sum_m"] < 0.00021
    assert report["known_numerical_error_budget"]["contact_offset_readback"] == (
        "NOT_EXPOSED_BY_LOCAL_107_3_USD_READBACK"
    )
    assert report["known_numerical_error_budget"]["physical_bottle_geometry"] == (
        "OUT_OF_SCOPE_PROJECT_CAD_IS_DIGITAL_GEOMETRY_AUTHORITY"
    )


def test_frozen_report_matches_deterministic_build() -> None:
    generated = build_certificate(ROOT)
    frozen = json.loads(FROZEN_REPORT.read_text(encoding="utf-8"))

    assert frozen == generated
