from __future__ import annotations

import json
import math

import numpy as np

from tools.aloha1_mapping.bottle_com_velocity import analyze_samples
from tools.aloha1_mapping.bottle_com_velocity import build_sample
from tools.aloha1_mapping.bottle_com_velocity import build_velocity_diagnosis
from tools.aloha1_mapping.bottle_com_velocity import classify_velocity_result
from tools.aloha1_mapping.bottle_com_velocity import derive_baseline_tolerance
from tools.aloha1_mapping.bottle_com_velocity import evaluate_post_step_alignment
from tools.aloha1_mapping.bottle_com_velocity import isolated_control_profile
from tools.build_aloha1_bottle_com_velocity_diagnosis import load_control_metrics
from tools.diagnose_aloha1_bottle_com_velocity import normalize_extension_version


def _translation_samples(*, dt: float, count: int) -> list[dict[str, object]]:
    velocity = np.asarray([0.12, -0.08, 0.05], dtype=np.float64)
    return [
        build_sample(
            step_index=index,
            dt_s=dt,
            actor_prim_path="/World/Bottle500",
            tensor_index=0,
            actor_position_world_m=(velocity * dt * index),
            actor_orientation_world_wxyz=[1.0, 0.0, 0.0, 0.0],
            center_of_mass_local_m=[0.0, 0.0, 0.088455],
            linear_velocity_com_world_m_s=velocity,
            angular_velocity_world_rad_s=[0.0, 0.0, 0.0],
        )
        for index in range(count)
    ]


def test_pure_translation_preserves_signed_components_and_integral() -> None:
    metrics = analyze_samples(_translation_samples(dt=1 / 60, count=121))

    assert metrics["sample_count"] == 121
    assert metrics["transition_count"] == 120
    assert np.allclose(metrics["com_delta_m"], [0.24, -0.16, 0.1])
    assert math.isclose(metrics["vz_min_m_s"], 0.05)
    assert math.isclose(metrics["vz_max_m_s"], 0.05)
    assert math.isclose(metrics["signed_vz_mean_m_s"], 0.05)
    assert math.isclose(metrics["signed_velocity_integral_m"], 0.1)
    assert metrics["com_forward_fd_vs_velocity"]["max_error_m_s"] < 1e-12


def test_pure_rotation_keeps_com_fixed_and_moves_actor_origin() -> None:
    dt = 1 / 6000
    omega = 2.0
    radius = 0.088455
    samples = []
    for index in range(121):
        angle = omega * dt * index
        orientation = [math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0]
        rotated_offset = np.asarray(
            [radius * math.sin(angle), 0.0, radius * math.cos(angle)]
        )
        samples.append(
            build_sample(
                step_index=index,
                dt_s=dt,
                actor_prim_path="/World/Bottle500",
                tensor_index=0,
                actor_position_world_m=-rotated_offset,
                actor_orientation_world_wxyz=orientation,
                center_of_mass_local_m=[0.0, 0.0, radius],
                linear_velocity_com_world_m_s=[0.0, 0.0, 0.0],
                angular_velocity_world_rad_s=[0.0, omega, 0.0],
            )
        )
    metrics = analyze_samples(samples)

    assert np.linalg.norm(metrics["com_delta_m"]) < 1e-12
    assert np.linalg.norm(metrics["actor_origin_delta_m"]) > 0.003
    assert metrics["com_forward_fd_vs_velocity"]["max_error_m_s"] < 1e-12
    assert metrics["actor_origin_forward_fd_vs_prediction"]["max_error_m_s"] < 1e-4


def test_tolerance_is_derived_from_baselines_and_float32_dt_floor() -> None:
    baseline = analyze_samples(_translation_samples(dt=1 / 60, count=121))
    tolerance = derive_baseline_tolerance(
        v1=baseline,
        v2=baseline,
        dt_s=1 / 60,
    )

    assert tolerance["velocity_tolerance_m_s"] > 0.0
    assert tolerance["velocity_tolerance_m_s"] >= tolerance[
        "measured_baseline_max_error_m_s"
    ]
    assert tolerance["source"] == "V1_V2_BASELINE_PLUS_FLOAT32_POSITION_ULP_PER_DT"


def test_sampling_time_mismatch_requires_shifted_alignment_only() -> None:
    assert (
        classify_velocity_result(
            v1_pass=True,
            v2_pass=True,
            v3_current_alignment=False,
            v3_shifted_alignment=True,
            v3_com_frame_explains=False,
        )
        == "SAMPLING_TIME_MISMATCH"
    )


def test_local_disagreement_requires_valid_controls_and_both_alignments_fail() -> None:
    assert (
        classify_velocity_result(
            v1_pass=True,
            v2_pass=True,
            v3_current_alignment=False,
            v3_shifted_alignment=False,
            v3_com_frame_explains=False,
        )
        == "VERIFIED_LOCAL_PHYSX_VELOCITY_TRANSFORM_DISAGREEMENT"
    )
    assert (
        classify_velocity_result(
            v1_pass=False,
            v2_pass=True,
            v3_current_alignment=False,
            v3_shifted_alignment=False,
            v3_com_frame_explains=False,
        )
        == "INCONCLUSIVE"
    )


def test_com_frame_explanation_has_priority_over_sampling() -> None:
    assert (
        classify_velocity_result(
            v1_pass=True,
            v2_pass=True,
            v3_current_alignment=False,
            v3_shifted_alignment=True,
            v3_com_frame_explains=True,
        )
        == "COM_FRAME_SEMANTICS_EXPLAINS_DISAGREEMENT"
    )


def test_sample_preserves_explicit_post_step_sampling_contract() -> None:
    sample = build_sample(
        step_index=7,
        state_boundary_index=7,
        dt_s=1.0 / 60.0,
        sampling_phase="POST_PHYSICS_STEP",
        actor_prim_path="/World/Bottle500",
        tensor_index=0,
        actor_position_world_m=[0.0, 0.0, 0.0],
        actor_orientation_world_wxyz=[1.0, 0.0, 0.0, 0.0],
        center_of_mass_local_m=[0.0, 0.0, 0.088455],
        linear_velocity_com_world_m_s=[0.0, 0.0, -0.1],
        angular_velocity_world_rad_s=[0.0, 0.0, 0.0],
    )

    assert sample["sampling_phase"] == "POST_PHYSICS_STEP"
    assert sample["state_boundary_index"] == 7


def test_isolated_controls_change_only_commanded_motion() -> None:
    translation = isolated_control_profile("V1")
    rotation = isolated_control_profile("V2")

    assert translation["gravity_enabled"] is False
    assert translation["collisions_enabled"] is False
    assert translation["linear_velocity_com_world_m_s"] == [
        0.12,
        -0.08,
        0.05,
    ]
    assert translation["angular_velocity_world_rad_s"] == [0.0, 0.0, 0.0]
    assert rotation["linear_velocity_com_world_m_s"] == [0.0, 0.0, 0.0]
    assert rotation["angular_velocity_world_rad_s"] == [0.0, 2.0, 0.0]
    assert rotation["preserve_authored_center_of_mass"] is True


def test_missing_extension_metadata_falls_back_without_crashing() -> None:
    assert normalize_extension_version(None, fallback="107.3.26") == (
        "107.3.26"
    )
    assert normalize_extension_version(
        {"package": {"version": "107.3.26+107.3.3"}},
        fallback="107.3.26",
    ) == "107.3.26+107.3.3"


def test_post_step_alignment_evaluates_com_frame_and_shifted_time() -> None:
    translation = analyze_samples(_translation_samples(dt=1 / 60, count=121))
    alignment = evaluate_post_step_alignment(
        translation,
        velocity_tolerance_m_s=1.0e-6,
    )

    assert alignment["declared_post_step_backward_alignment"] is True
    assert alignment["shifted_forward_alignment"] is True
    assert alignment["com_frame_explains_origin_disagreement"] is False


def test_diagnosis_retains_local_disagreement_only_after_valid_controls() -> None:
    v1_metrics = analyze_samples(_translation_samples(dt=1 / 60, count=121))
    v2_metrics = analyze_samples(_translation_samples(dt=1 / 60, count=121))
    v3_metrics = dict(v1_metrics)
    for key in (
        "com_forward_fd_vs_velocity",
        "com_backward_fd_vs_velocity",
        "com_midpoint_fd_vs_velocity",
    ):
        v3_metrics[key] = {
            **v3_metrics[key],
            "max_error_m_s": 0.1,
        }

    diagnosis = build_velocity_diagnosis(
        v1_metrics=v1_metrics,
        v2_metrics=v2_metrics,
        v3_metrics=v3_metrics,
        v1_runtime_valid=True,
        v2_runtime_valid=True,
        v3_signature_unchanged=True,
        dt_s=1.0 / 60.0,
    )

    assert diagnosis["conclusion"] == (
        "VERIFIED_LOCAL_PHYSX_VELOCITY_TRANSFORM_DISAGREEMENT"
    )
    assert diagnosis["status"] == "PASS"


def test_control_metrics_are_recomputed_from_frozen_samples(
    tmp_path,
) -> None:
    samples = _translation_samples(dt=1 / 60, count=3)
    sample_path = tmp_path / "samples.jsonl"
    sample_path.write_text(
        "".join(json.dumps(row) + "\n" for row in samples),
        encoding="utf-8",
    )
    report = {
        "samples": {"absolute_path": str(sample_path)},
        "metrics": {"schema": "OLD"},
    }

    metrics = load_control_metrics(report)

    assert metrics["sample_count"] == 3
    assert "actor_origin_backward_fd_vs_com_velocity_uncorrected" in metrics
