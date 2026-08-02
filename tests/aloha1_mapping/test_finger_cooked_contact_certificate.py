from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.aloha1_mapping.finger_cooked_contact_certificate import classify_exact_brep_profiles
from tools.aloha1_mapping.finger_cooked_contact_certificate import classify_profile_comparison
from tools.aloha1_mapping.finger_cooked_contact_certificate import derive_cooked_brep_numeric_tolerance
from tools.aloha1_mapping.finger_cooked_contact_certificate import load_exact_brep_contact_surface
from tools.aloha1_mapping.finger_cooked_contact_certificate import load_supplier_contact_surface
from tools.aloha1_mapping.finger_cooked_contact_certificate import positive_union_exit_distance
from tools.aloha1_mapping.finger_cooked_contact_certificate import summarize_contact_envelope

ROOT = Path(__file__).resolve().parents[2]


def _box(x_min: float, x_max: float) -> dict[str, list[list[float]]]:
    return {
        "vertices": [
            [x, y, z]
            for x in (x_min, x_max)
            for y in (-0.01, 0.01)
            for z in (-0.01, 0.01)
        ]
    }


def test_positive_exit_measures_inward_over_envelope() -> None:
    result = positive_union_exit_distance(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        [_box(-0.01, 0.001)],
    )

    assert result["source_point_covered"] is True
    assert result["positive_exit_distance_m"] == pytest.approx(0.001)


def test_overlapping_convex_pieces_are_measured_as_one_union() -> None:
    result = positive_union_exit_distance(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        [_box(-0.01, 0.0001), _box(0.00005, 0.0003)],
    )

    assert result["source_point_covered"] is True
    assert result["positive_exit_distance_m"] == pytest.approx(0.0003)
    assert result["contributing_piece_count"] == 2


def test_uncovered_source_surface_is_not_reported_as_zero_error() -> None:
    result = positive_union_exit_distance(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        [_box(0.001, 0.002)],
    )

    assert result["source_point_covered"] is False
    assert result["positive_exit_distance_m"] is None
    assert result["nearest_positive_entry_m"] == pytest.approx(0.001)


def test_small_uncovered_gap_is_measured_against_existing_budget() -> None:
    result = summarize_contact_envelope(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([1.0, 0.0, 0.0]),
        [_box(0.0001, 0.01)],
        tessellation_budget_m=0.0002,
    )

    assert result["source_point_coverage_ratio"] == 0.0
    assert result["uncovered_nearest_surface_max_m"] == pytest.approx(0.0001)
    assert result["maximum_contact_surface_deviation_m"] == pytest.approx(0.0001)
    assert result["status"] == "PASS_WITHIN_TESSELLATION_ERROR_BUDGET"
    assert result["maximum_deviation_kind"] == "UNCOVERED_NEAREST_SURFACE"
    assert result["maximum_deviation_sample_index"] == 0
    assert result["maximum_deviation_source_point_m"] == [0.0, 0.0, 0.0]
    assert result["maximum_deviation_target_point_m"] == pytest.approx(
        [0.0001, 0.0, 0.0]
    )


def test_summary_applies_frozen_tessellation_budget_without_fitting() -> None:
    samples = np.array([[0.0, 0.0, 0.0], [0.0, 0.002, 0.0]])
    result = summarize_contact_envelope(
        samples,
        np.array([1.0, 0.0, 0.0]),
        [_box(-0.01, 0.0001)],
        tessellation_budget_m=0.0002,
    )

    assert result["status"] == "PASS_WITHIN_TESSELLATION_ERROR_BUDGET"
    assert result["source_point_coverage_ratio"] == 1.0
    assert result["positive_exit_distance_max_m"] == pytest.approx(0.0001)
    assert result["maximum_inward_crossing_sample_index"] == 0
    assert result["maximum_inward_crossing_source_point_m"] == [0.0, 0.0, 0.0]
    assert result["tessellation_error_budget_m"] == 0.0002


@pytest.mark.parametrize(
    ("side", "expected_hash", "normal_x"),
    [
        (
            "left",
            "c6710d0fe5b2030a32722d9df5c0b553c771c9d61d92b8ddaec36c94c5963488",
            -0.9945218953682733,
        ),
        (
            "right",
            "b0979c5d55fee448dab512dc75b1251bab17d94892decd01de9a6e76c01482d1",
            0.9945218953682733,
        ),
    ],
)
def test_supplier_contact_samples_are_bound_to_freecad_faces(
    side: str, expected_hash: str, normal_x: float
) -> None:
    result = load_supplier_contact_surface(ROOT, side)

    assert result["source_sha256"] == expected_hash
    assert result["sample_count"] == 124
    assert result["tessellation_error_budget_m"] == 0.0002
    assert result["mirror_used"] is False
    assert result["normal"][0] == pytest.approx(normal_x)
    assert result["samples"].shape == (124, 3)


def test_profile_classification_does_not_promote_improved_decomposition() -> None:
    profiles = {
        side: {
            "convexHull": {
                "status": "FAIL_EXCEEDS_TESSELLATION_ERROR_BUDGET",
                "maximum_contact_surface_deviation_m": 0.0008,
            },
            "convexDecomposition": {
                "status": "PASS_WITHIN_TESSELLATION_ERROR_BUDGET",
                "maximum_contact_surface_deviation_m": 0.00015,
            },
        }
        for side in ("left", "right")
    }

    result = classify_profile_comparison(profiles)

    assert result["classification"] == (
        "DECOMPOSITION_GEOMETRY_IMPROVES_WITHIN_BUDGET_NOT_PROMOTED"
    )
    assert result["final_or_default_collider_modified"] is False
    assert result["runtime_hold_claim"] == "NOT_MADE"


def test_exact_brep_loader_requires_two_matching_fresh_reports(
    tmp_path: Path,
) -> None:
    report = {
        "status": "PASS",
        "classification": "EXACT_OCCT_BREP_CONTACT_FACE_SAMPLES",
        "deterministic_signature": "same",
        "source": {"sha256": "337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571"},
        "toolchain": {
            "required_freecad": "1.1.1",
            "required_opencascade": "7.8.1",
            "opencascade": "7.8.1",
        },
        "sampling": {"no_tessellation_used_for_points": True},
        "fingers": {
            "left": {
                "face_index_1_based": 117,
                "normal": [-1.0, 0.0, 0.0],
                "samples_mm": [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]],
                "sample_count": 2,
                "uv_grid": {"membership_tolerance_mm": 1.0e-7},
            }
        },
    }
    paths = [tmp_path / "run1.json", tmp_path / "run2.json"]
    for process_id, path in enumerate(paths, start=10):
        run = dict(report)
        run["process_id"] = process_id
        path.write_text(json.dumps(run), encoding="utf-8")

    result = load_exact_brep_contact_surface(paths, "left")

    np.testing.assert_allclose(
        result["samples_m"],
        [[0.001, 0.002, 0.003], [0.002, 0.002, 0.003]],
    )
    assert result["fresh_process_count"] == 2
    assert result["source_geometry"] == "trimmed OCCT B-Rep face"


def test_numeric_tolerance_is_derived_from_brep_and_float32_precision() -> None:
    result = derive_cooked_brep_numeric_tolerance(
        np.array([[0.5, -0.2, 0.1]]),
        brep_membership_tolerance_m=1.0e-10,
    )

    assert result["numeric_tolerance_m"] >= 1.0e-10
    assert result["numeric_tolerance_m"] == pytest.approx(
        result["float32_quantization_allowance_m"]
    )
    assert result["derivation"] == "MAX(BREP_MEMBERSHIP_TOLERANCE,8_FLOAT32_ULP)"


def test_exact_brep_classification_rejects_crossing_without_promoting() -> None:
    profiles = {
        "left": {
            "convexHull": {"maximum_inward_crossing_m": 0.00068},
            "convexDecomposition": {"maximum_inward_crossing_m": 0.00055},
        },
        "right": {
            "convexHull": {"maximum_inward_crossing_m": 0.00068},
            "convexDecomposition": {"maximum_inward_crossing_m": 0.00135},
        },
    }

    result = classify_exact_brep_profiles(
        profiles,
        numeric_tolerance_m=5.0e-7,
    )

    assert result["exact_surface_status"] == (
        "ALL_PROFILES_CROSS_INWARD_CAD_SURFACE"
    )
    assert result["decomposition_comparison"] == "DECOMPOSITION_MIXED_OR_WORSE"
    assert result["asset_decision"] == "REJECTED_EXACT_CAD_CONTACT_GATE"
    assert result["final_or_default_collider_modified"] is False
