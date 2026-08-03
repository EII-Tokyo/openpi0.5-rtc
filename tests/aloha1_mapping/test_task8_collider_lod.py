from __future__ import annotations

import numpy as np
import pytest

from tools.aloha1_mapping.task8_collider_lod import build_containment_pruning_certificate
from tools.aloha1_mapping.task8_collider_lod import build_single_hull_geometry
from tools.aloha1_mapping.task8_collider_lod import classify_benchmark_improvement
from tools.aloha1_mapping.task8_collider_lod import classify_link_role
from tools.aloha1_mapping.task8_collider_lod import compare_compound_to_single_hull
from tools.aloha1_mapping.task8_collider_lod import compare_profile_inventories
from tools.aloha1_mapping.task8_collider_lod import ordered_mesh_components
from tools.aloha1_mapping.task8_collider_lod import rank_pair_merge_candidates
from tools.aloha1_mapping.task8_collider_lod import select_throughput_links
from tools.aloha1_mapping.task8_collider_lod import split_face_components
from tools.aloha1_mapping.task8_collider_lod import summarize_hold_contact_telemetry
from tools.aloha1_mapping.task8_collider_lod import summarize_profile_runs
from tools.benchmark_aloha1_task8_physics import _normalized_cooking
from tools.build_aloha1_task8_benchmark_stages import _derive_spacing
from tools.validate_aloha1_cad_derived_collision_static import _expected_source_counts


def test_task_contact_links_are_never_selected_for_collider_lod() -> None:
    for suffix in (
        "gripper_link",
        "gripper_bar_link",
        "gripper_prop_link",
        "left_finger_link",
        "right_finger_link",
    ):
        assert classify_link_role(suffix, has_collider=True) == "task_contact_critical"


def test_arm_links_remain_environment_clearance_critical() -> None:
    for suffix in (
        "base_link",
        "shoulder_link",
        "upper_arm_link",
        "upper_forearm_link",
        "lower_forearm_link",
        "wrist_link",
    ):
        assert (
            classify_link_role(suffix, has_collider=True)
            == "environment_clearance_critical"
        )


def test_throughput_selection_requires_geometry_and_swept_evidence() -> None:
    records = [
        {
            "link_suffix": "upper_arm_link",
            "role": "environment_clearance_critical",
            "source_convex_piece_count": 4,
            "source_brep_valid": True,
            "baseline_static_audit": "PASS",
            "baseline_swept_audit": "PASS",
        },
        {
            "link_suffix": "gripper_link",
            "role": "task_contact_critical",
            "source_convex_piece_count": 9,
            "source_brep_valid": True,
            "baseline_static_audit": "PASS",
            "baseline_swept_audit": "PASS",
        },
        {
            "link_suffix": "shoulder_link",
            "role": "environment_clearance_critical",
            "source_convex_piece_count": 1,
            "source_brep_valid": True,
            "baseline_static_audit": "PASS",
            "baseline_swept_audit": "PASS",
        },
    ]

    assert select_throughput_links(records) == ["upper_arm_link"]

    records[0]["baseline_swept_audit"] = "PARTIAL"
    assert select_throughput_links(records) == []


def test_single_hull_geometry_is_deterministic_for_disconnected_source() -> None:
    first = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    second = first + np.asarray([2.0, 0.0, 0.0])

    result_a = build_single_hull_geometry(np.vstack((first, second)))
    result_b = build_single_hull_geometry(np.vstack((second, first)))

    assert result_a["canonical_signature"] == result_b["canonical_signature"]
    assert result_a["volume_m3"] > 0.0
    assert result_a["vertex_count"] >= 4
    assert result_a["face_count"] >= 4
    assert result_a["aabb_m"] == {
        "minimum": [0.0, 0.0, 0.0],
        "maximum": [3.0, 1.0, 1.0],
        "extent": [3.0, 1.0, 1.0],
    }


def test_compound_to_single_hull_reports_added_envelope_without_losing_points() -> None:
    first = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    second = first + np.asarray([2.0, 0.0, 0.0])
    vertices = np.vstack((first, second))
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
            [4, 5, 6],
            [4, 5, 7],
            [4, 6, 7],
            [5, 6, 7],
        ]
    )

    components = split_face_components(faces)
    comparison = compare_compound_to_single_hull(vertices, faces)

    assert len(components) == 2
    assert comparison["source_component_count"] == 2
    assert comparison["candidate_piece_count"] == 1
    assert comparison["candidate_volume_m3"] > comparison["source_union_volume_estimate_m3"]
    assert comparison["source_vertex_outside_candidate_count"] == 0
    assert comparison["inward_vertex_deviation_max_m"] == pytest.approx(0.0)


def test_pair_merge_ranking_prefers_smallest_geometry_gap() -> None:
    tetra = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    vertices = np.vstack(
        (
            tetra,
            tetra + np.asarray([1.1, 0.0, 0.0]),
            tetra + np.asarray([4.0, 0.0, 0.0]),
        )
    )
    faces = np.vstack(
        [
            np.asarray(
                [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
            )
            + offset
            for offset in (0, 4, 8)
        ]
    )

    candidates = rank_pair_merge_candidates(vertices, faces)

    assert candidates[0]["merged_component_indices"] == [0, 1]
    assert candidates[0]["piece_reduction"] == 1
    assert candidates[0]["source_vertex_outside_candidate_count"] == 0


def test_component_order_matches_cad_collider_authoring_order() -> None:
    first = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
        ]
    )
    second = np.asarray(
        [
            [-2.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-2.0, 1.0, 0.0],
            [-2.0, 0.0, 1.0],
        ]
    )
    vertices = np.vstack((first, second))
    faces = np.vstack(
        (
            np.asarray([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
            np.asarray([[4, 5, 6], [4, 5, 7], [4, 6, 7], [5, 6, 7]]),
        )
    )

    components = ordered_mesh_components(vertices, faces)

    assert len(components) == 2
    assert components[0]["minimum_point"] == [-2.0, 0.0, 0.0]
    assert components[1]["minimum_point"] == [2.0, 0.0, 0.0]
    assert components[0]["vertex_count"] == 4
    assert components[0]["face_count"] == 4


def test_containment_certificate_selects_existing_outer_piece() -> None:
    outer = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    inner = np.asarray(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.1, 0.1],
            [0.1, 0.2, 0.1],
            [0.1, 0.1, 0.2],
        ]
    )
    vertices = np.vstack((outer, inner))
    faces = np.vstack(
        (
            np.asarray([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
            np.asarray([[4, 5, 6], [4, 5, 7], [4, 6, 7], [5, 6, 7]]),
        )
    )

    certificate = build_containment_pruning_certificate(vertices, faces)

    assert certificate["status"] == "VERIFIED_EXISTING_PIECE_CONTAINS_ALL_OTHERS"
    assert certificate["retained_piece_index"] == 0
    assert certificate["removed_piece_indices"] == [1]
    assert certificate["full_hull_matches_retained_hull"] is True
    assert certificate["maximum_outside_distance_m"] <= certificate["tolerance_m"]


def test_profile_inventory_comparison_allows_only_declared_removed_colliders() -> None:
    common = {
        "articulations": [{"path": "/World/robot"}],
        "joints": [{"path": "/World/robot/joint"}],
        "rigid_bodies": [{"path": "/World/robot/link"}],
        "visuals": [{"path": "/World/robot/visual", "geometry": "same"}],
    }
    baseline = {
        **common,
        "colliders": [
            {"path": "/World/robot/piece_000", "geometry": "outer"},
            {"path": "/World/robot/piece_001", "geometry": "inner"},
        ],
    }
    candidate = {
        **common,
        "colliders": [
            {"path": "/World/robot/piece_000", "geometry": "outer"},
        ],
    }

    comparison = compare_profile_inventories(
        baseline, candidate, removed_collider_paths=["/World/robot/piece_001"]
    )

    assert comparison["status"] == "PASS"
    assert comparison["unexpected_removed_collider_paths"] == []
    assert comparison["unexpected_added_collider_paths"] == []
    assert comparison["retained_collider_drift_paths"] == []

    candidate["colliders"][0]["geometry"] = "changed"
    assert (
        compare_profile_inventories(
            baseline, candidate, removed_collider_paths=["/World/robot/piece_001"]
        )["status"]
        == "FAIL"
    )


def test_task8_static_coverage_expects_only_six_declared_piece_removals() -> None:
    assert _expected_source_counts("gripper_decomposition") == {
        "CAD_DERIVED": 34,
        "SUPPLIER_CAD_FINGER": 4,
        "IMPORTER_BASELINE_FALLBACK": 4,
    }
    assert _expected_source_counts("task8_throughput") == {
        "CAD_DERIVED": 28,
        "SUPPLIER_CAD_FINGER": 4,
        "IMPORTER_BASELINE_FALLBACK": 4,
    }
    assert _expected_source_counts("task8_fidelity") == _expected_source_counts(
        "gripper_decomposition"
    )


def test_benchmark_environment_spacing_is_derived_from_frozen_bounds() -> None:
    assert _derive_spacing({"xmin": -0.75, "xmax": 0.75}) == pytest.approx(3.0)


def test_profile_summary_requires_two_fresh_runs_per_scale() -> None:
    runs = [
        {"profile": "fidelity_profile", "environment_count": 1, "physics_ms": 1.0},
        {"profile": "fidelity_profile", "environment_count": 1, "physics_ms": 1.2},
        {"profile": "throughput_profile", "environment_count": 1, "physics_ms": 0.8},
        {"profile": "throughput_profile", "environment_count": 1, "physics_ms": 0.9},
    ]

    summary = summarize_profile_runs(runs)

    assert summary["fidelity_profile"]["1"]["physics_ms"]["count"] == 2
    assert summary["throughput_profile"]["1"]["physics_ms"]["mean"] == pytest.approx(
        0.85
    )


def test_benchmark_requires_non_overlapping_improvement_at_every_scale() -> None:
    summary = {
        "fidelity_profile": {
            "1": {"physics_step_ms": {"min": 0.70, "mean": 0.72, "max": 0.75}},
            "4": {"physics_step_ms": {"min": 1.62, "mean": 1.68, "max": 1.73}},
        },
        "throughput_profile": {
            "1": {"physics_step_ms": {"min": 0.66, "mean": 0.68, "max": 0.71}},
            "4": {"physics_step_ms": {"min": 1.58, "mean": 1.61, "max": 1.63}},
        },
    }

    result = classify_benchmark_improvement(summary)

    assert result["classification"] == "NO_MEASURABLE_IMPROVEMENT"
    assert result["all_scales_non_overlapping_improvement"] is False


def test_hold_contact_summary_keeps_signed_separation_and_solver_impulse() -> None:
    rows = [
        {
            "phase": "HOLD",
            "contacts": [
                {
                    "actor0_path": "/robot/left_finger_link",
                    "actor1_path": "/Bottle500",
                    "collider0_path": "/robot/left",
                    "collider1_path": "/Bottle500/body",
                    "separation_m": 2.0e-7,
                    "impulse_ns": 0.004,
                },
                {
                    "actor0_path": "/robot/right_finger_link",
                    "actor1_path": "/Bottle500",
                    "collider0_path": "/robot/right",
                    "collider1_path": "/Bottle500/body",
                    "separation_m": -3.0e-6,
                    "impulse_ns": 0.005,
                },
            ],
            "bottle": {"position_world_m": [0.0, 0.0, 0.2]},
            "observation": {"hold_drop_m": 0.0002},
        }
    ]

    summary = summarize_hold_contact_telemetry(rows)

    assert summary["left_finger"]["minimum_separation_m"]["mean"] == pytest.approx(
        2.0e-7
    )
    assert summary["left_finger"]["geometric_contact_frame_count"] == 0
    assert summary["left_finger"]["solver_active_frame_count"] == 1
    assert summary["right_finger"]["geometric_contact_frame_count"] == 1


def test_cooking_determinism_excludes_runtime_but_not_geometry() -> None:
    first = {"colliders": [{"path": "/mesh", "runtime_s": 0.1, "signature": "a"}]}
    second = {"colliders": [{"path": "/mesh", "runtime_s": 0.2, "signature": "a"}]}

    assert _normalized_cooking(first) == _normalized_cooking(second)
    second["colliders"][0]["signature"] = "b"
    assert _normalized_cooking(first) != _normalized_cooking(second)
