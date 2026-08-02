from __future__ import annotations

import numpy as np
import pytest

from tools.aloha1_mapping.cad_compound_contact_candidate import build_contact_preserving_candidate
from tools.aloha1_mapping.cad_compound_contact_candidate import canonical_runtime_cooking_signature
from tools.aloha1_mapping.cad_compound_contact_candidate import classify_fresh_runtime_runs
from tools.aloha1_mapping.cad_compound_contact_candidate import clip_convex_piece_to_halfspace
from tools.aloha1_mapping.cad_compound_contact_candidate import compound_piece_prim_path
from tools.aloha1_mapping.cad_compound_contact_candidate import convex_triangle_topology
from tools.aloha1_mapping.cad_compound_contact_candidate import runtime_contact_region_status
from tools.aloha1_mapping.cad_compound_contact_candidate import tolerance_adjusted_contact_coverage
from tools.aloha1_mapping.cad_compound_contact_candidate import transform_contact_candidate
from tools.aloha1_mapping.cad_compound_contact_candidate import triangular_contact_prism
from tools.aloha1_mapping.finger_cooked_contact_certificate import positive_union_exit_distance


def _crossing_box() -> dict[str, list[list[float]]]:
    return {"vertices": [[x, y, z] for x in (-0.002, 0.001) for y in (-0.01, 0.01) for z in (-0.01, 0.01)]}


def test_halfspace_clip_removes_inward_contact_surface_crossing() -> None:
    clipped = clip_convex_piece_to_halfspace(
        _crossing_box(),
        plane_point=np.zeros(3),
        outward_normal=np.array([1.0, 0.0, 0.0]),
        numeric_tolerance_m=1.0e-9,
    )

    assert clipped is not None
    vertices = np.asarray(clipped["vertices"])
    assert vertices[:, 0].max() == pytest.approx(0.0, abs=1.0e-12)
    crossing = positive_union_exit_distance(np.zeros(3), np.array([1.0, 0.0, 0.0]), [clipped])
    assert crossing["source_point_covered"] is True
    assert crossing["positive_exit_distance_m"] == pytest.approx(0.0)


def test_triangular_contact_prism_extrudes_only_into_finger_body() -> None:
    prism = triangular_contact_prism(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [0.0, 0.0, 0.01],
            ]
        ),
        outward_normal=np.array([1.0, 0.0, 0.0]),
        depth_m=0.002,
    )

    vertices = np.asarray(prism["vertices"])
    assert vertices[:, 0].max() == pytest.approx(0.0)
    assert vertices[:, 0].min() == pytest.approx(-0.002)


def test_compound_candidate_combines_clipped_body_and_contact_prism() -> None:
    triangle = np.array([[0.0, -0.01, -0.01], [0.0, 0.01, -0.01], [0.0, 0.0, 0.01]])
    result = build_contact_preserving_candidate(
        cooked_pieces=[_crossing_box()],
        contact_triangles=np.array([triangle]),
        plane_point=np.zeros(3),
        outward_normal=np.array([1.0, 0.0, 0.0]),
        contact_prism_depth_m=0.002,
        numeric_tolerance_m=1.0e-9,
    )

    assert result["clipped_body_piece_count"] == 1
    assert result["contact_prism_piece_count"] == 1
    crossing = positive_union_exit_distance(
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        result["pieces"],
    )
    assert crossing["source_point_covered"] is True
    assert crossing["positive_exit_distance_m"] == pytest.approx(0.0)


def test_convex_triangle_topology_authors_closed_cube_faces() -> None:
    vertices = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])

    topology = convex_triangle_topology(vertices)

    assert topology["face_vertex_counts"] == [3] * 12
    assert len(topology["face_vertex_indices"]) == 36
    assert set(topology["face_vertex_indices"]) == set(range(8))
    assert topology["volume_m3"] == pytest.approx(8.0)


def test_runtime_cooking_signature_ignores_runtime_and_process_metadata() -> None:
    base = {
        "runtime_s": 1.0,
        "process_id": 10,
        "fingers": {
            "left": {
                "pieces": [
                    {
                        "source_piece_index": 0,
                        "approximation_readback": "convexHull",
                        "cooked": {"vertices": [[0.0, 0.0, 0.0]]},
                    }
                ]
            }
        },
    }
    changed_metadata = {
        **base,
        "runtime_s": 9.0,
        "process_id": 99,
    }

    assert canonical_runtime_cooking_signature(base) == canonical_runtime_cooking_signature(changed_metadata)


def test_runtime_contact_gate_requires_coverage_and_no_outward_crossing() -> None:
    passing = {
        "source_point_coverage_ratio": 1.0,
        "positive_exit_distance_max_m": 2.0e-7,
    }
    crossing = {
        "source_point_coverage_ratio": 1.0,
        "positive_exit_distance_max_m": 2.0e-5,
    }

    assert runtime_contact_region_status(passing, 5.0e-7) == "PASS"
    assert runtime_contact_region_status(crossing, 5.0e-7) == "FAIL"


def test_runtime_contact_gate_accepts_float32_surface_quantization() -> None:
    exact_ray_classification = {
        "contact_sample_count": 100,
        "source_point_covered_count": 30,
        "source_point_coverage_ratio": 0.3,
        "uncovered_count": 70,
        "uncovered_nearest_surface_max_m": 2.2e-8,
        "uncovered_nearest_surface_normal_projection_min_m": -2.5e-9,
        "uncovered_nearest_surface_normal_projection_max_m": 1.3e-14,
        "positive_exit_distance_max_m": 1.3e-9,
    }

    adjusted = tolerance_adjusted_contact_coverage(exact_ray_classification, numeric_tolerance_m=4.8e-7)

    assert adjusted["exact_ray_coverage_ratio"] == pytest.approx(0.3)
    assert adjusted["tolerance_adjusted_coverage_ratio"] == pytest.approx(1.0)
    assert adjusted["quantization_boundary_sample_count"] == 70
    assert runtime_contact_region_status(adjusted, 4.8e-7) == "PASS"


def test_two_fresh_runtime_runs_require_matching_geometry_signatures() -> None:
    first = {
        "process_id": 101,
        "status": "PASS_RUNTIME_COOKED_CONTACT_REGION_GEOMETRY_NOT_PROMOTED",
        "deterministic_signature": "same",
    }
    second = {**first, "process_id": 202}

    result = classify_fresh_runtime_runs([first, second])

    assert result["status"] == "PASS_DETERMINISTIC_FRESH_PROCESS_COOKING"
    assert result["fresh_processes"] is True
    assert result["matching_geometry_signatures"] is True


def test_contact_candidate_rigid_transform_preserves_depth_and_piece_volume() -> None:
    candidate = {
        "pieces": [
            triangular_contact_prism(
                np.array([[0.0, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]),
                outward_normal=np.array([1.0, 0.0, 0.0]),
                depth_m=0.002,
            )
        ],
        "outward_normal": [1.0, 0.0, 0.0],
        "plane_point_m": [0.0, 0.0, 0.0],
        "contact_rectangle_vertices_m": [
            [0.0, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.01, 0.01],
            [0.0, 0.0, 0.01],
        ],
        "contact_prism_depth_m": 0.002,
    }
    matrix = np.array(
        [
            [0.0, -1.0, 0.0, 0.5],
            [1.0, 0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    transformed = transform_contact_candidate(candidate, matrix)

    assert transformed["outward_normal"] == pytest.approx([0.0, 1.0, 0.0])
    assert transformed["plane_point_m"] == pytest.approx([0.5, -0.2, 0.1])
    assert transformed["pieces"][0]["volume_m3"] == pytest.approx(candidate["pieces"][0]["volume_m3"])
    assert transformed["rigid_transform_determinant"] == pytest.approx(1.0)


def test_compound_piece_prim_paths_are_explicit_and_side_scoped() -> None:
    assert compound_piece_prim_path("left", 7) == "/CadFingerCompoundContactCandidate/left_finger/piece_007"
    with pytest.raises(ValueError, match="unsupported finger side"):
        compound_piece_prim_path("middle", 0)
