from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from tools.aloha1_mapping.physics_numerical_convergence import apply_articulation_solver_override
from tools.aloha1_mapping.physics_numerical_convergence import build_predeclared_convergence_plan
from tools.aloha1_mapping.physics_numerical_convergence import build_runtime_cell_command
from tools.aloha1_mapping.physics_numerical_convergence import compare_runtime_cells
from tools.aloha1_mapping.physics_numerical_convergence import extract_runtime_cell_metrics
from tools.aloha1_mapping.physics_numerical_convergence import physical_model_signature
from tools.aloha1_mapping.physics_numerical_convergence import scaled_frame_count
from tools.aloha1_mapping.physics_numerical_convergence import select_coarsest_converged_value
from tools.aloha1_mapping.physics_numerical_convergence import should_continue_solver_sweeps
from tools.aloha1_mapping.physics_numerical_convergence import validate_numerical_override
from tools.run_aloha1_physics_numerical_convergence import _stable_report_signature


class _FakeArticulation:
    def __init__(self, *, mismatch: bool = False) -> None:
        self.position = 32
        self.velocity = 1
        self.mismatch = mismatch

    def set_solver_position_iteration_count(self, count: int) -> None:
        self.position = count

    def get_solver_position_iteration_count(self) -> int:
        return self.position + int(self.mismatch)

    def set_solver_velocity_iteration_count(self, count: int) -> None:
        self.velocity = count

    def get_solver_velocity_iteration_count(self) -> int:
        return self.velocity


def test_predeclared_convergence_plan_changes_one_numerical_axis_at_a_time() -> None:
    plan = build_predeclared_convergence_plan()

    assert plan["frequency_sweep"] == [
        {"frequency_hz": 60, "position_iterations": 64, "velocity_iterations": 8},
        {"frequency_hz": 120, "position_iterations": 64, "velocity_iterations": 8},
        {"frequency_hz": 240, "position_iterations": 64, "velocity_iterations": 8},
        {"frequency_hz": 480, "position_iterations": 64, "velocity_iterations": 8},
    ]


def test_convergence_config_matches_predeclared_matrix_and_frozen_hashes() -> None:
    config = yaml.safe_load(
        Path("configs/aloha1_physics_numerical_convergence.yaml").read_text(
            encoding="utf-8"
        )
    )
    plan = build_predeclared_convergence_plan()

    assert config["matrix"]["frequency_hz"] == [
        cell["frequency_hz"] for cell in plan["frequency_sweep"]
    ]
    assert config["matrix"]["position_iterations"] == plan[
        "position_iteration_sweep"
    ]
    assert config["matrix"]["velocity_iterations"] == plan[
        "velocity_iteration_sweep"
    ]
    for record in config["frozen_inputs"].values():
        path = Path(record["path"])
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    assert plan["position_iteration_sweep"] == [4, 8, 16, 32, 64]
    assert plan["velocity_iteration_sweep"] == [1, 2, 4, 8]
    assert plan["optional_frequency_hz"] == 960
    assert plan["selection_order"] == [
        "frequency_sweep",
        "position_iteration_sweep",
        "velocity_iteration_sweep",
    ]


@pytest.mark.parametrize(
    ("frequency_hz", "position_iterations", "velocity_iterations"),
    [(60.0, 64, 8), (480.0, 4, 1)],
)
def test_numerical_override_accepts_only_positive_finite_values(
    frequency_hz: float,
    position_iterations: int,
    velocity_iterations: int,
) -> None:
    assert validate_numerical_override(
        frequency_hz=frequency_hz,
        position_iterations=position_iterations,
        velocity_iterations=velocity_iterations,
    ) == {
        "frequency_hz": frequency_hz,
        "physics_dt_s": 1.0 / frequency_hz,
        "position_iterations": position_iterations,
        "velocity_iterations": velocity_iterations,
    }


@pytest.mark.parametrize(
    ("frequency_hz", "position_iterations", "velocity_iterations"),
    [(0.0, 64, 8), (float("nan"), 64, 8), (60.0, 0, 8), (60.0, 64, -1)],
)
def test_numerical_override_rejects_invalid_values(
    frequency_hz: float,
    position_iterations: int,
    velocity_iterations: int,
) -> None:
    with pytest.raises(ValueError, match="must be"):
        validate_numerical_override(
            frequency_hz=frequency_hz,
            position_iterations=position_iterations,
            velocity_iterations=velocity_iterations,
        )


@pytest.mark.parametrize(
    ("base_frames", "frequency_hz", "expected"),
    [(10, 60.0, 10), (10, 120.0, 20), (5, 480.0, 40)],
)
def test_scaled_frame_count_preserves_physical_duration(
    base_frames: int,
    frequency_hz: float,
    expected: int,
) -> None:
    assert scaled_frame_count(
        base_frames=base_frames,
        frequency_hz=frequency_hz,
        baseline_frequency_hz=60.0,
    ) == expected


def test_physical_model_signature_excludes_only_frequency() -> None:
    config = {
        "physics": {
            "frequency_hz": 60,
            "mass_kg": 0.020,
            "friction": 0.7,
            "restitution": 0.0,
        },
        "robot": {"finger_drive_type": "force"},
    }
    changed_dt = copy.deepcopy(config)
    changed_dt["physics"]["frequency_hz"] = 240
    changed_friction = copy.deepcopy(config)
    changed_friction["physics"]["friction"] = 1.0

    assert physical_model_signature(config) == physical_model_signature(changed_dt)
    assert physical_model_signature(config) != physical_model_signature(changed_friction)


def test_articulation_solver_override_requires_exact_runtime_readback() -> None:
    articulation = _FakeArticulation()

    assert apply_articulation_solver_override(
        articulation,
        position_iterations=64,
        velocity_iterations=8,
    ) == {
        "requested_position_iterations": 64,
        "requested_velocity_iterations": 8,
        "effective_position_iterations": 64,
        "effective_velocity_iterations": 8,
        "readback_status": "PASS",
    }


def test_articulation_solver_override_rejects_readback_mismatch() -> None:
    with pytest.raises(RuntimeError, match="solver iteration readback mismatch"):
        apply_articulation_solver_override(
            _FakeArticulation(mismatch=True),
            position_iterations=64,
            velocity_iterations=8,
        )


def test_grasp_launcher_exposes_session_only_numerical_overrides() -> None:
    source = Path("tools/run_aloha1_grasp_20cm_gui.py").read_text(encoding="utf-8")

    assert '"--physics-frequency-hz"' in source
    assert '"--solver-position-iterations"' in source
    assert '"--solver-velocity-iterations"' in source
    assert '"--enable-solver-residual-reporting"' in source
    assert "numerical_override=numerical_override" in source


def test_grasp_bindings_records_effective_numerical_readback() -> None:
    source = Path(
        "tools/aloha1_mapping/grasp_20cm_isaac_bindings.py"
    ).read_text(encoding="utf-8")

    reset = source.index("self.world.reset()")
    apply_override = source.index("apply_articulation_solver_override", reset)
    report = source.index('"numerical_convergence":', apply_override)
    assert reset < apply_override < report
    assert "get_physics_dt()" in source[apply_override:report]
    assert "preload_stable_frames_required" in source[apply_override:report]
    assert "phase_timeout_frames" in source[apply_override:report]


def test_grasp_telemetry_records_effort_work_and_solver_residuals() -> None:
    source = Path(
        "tools/aloha1_mapping/grasp_20cm_isaac_bindings.py"
    ).read_text(encoding="utf-8")

    assert "get_measured_joint_efforts()" in source
    assert '"joint_effort":' in source
    assert '"joint_power_w":' in source
    assert '"drive_work_increment_j":' in source
    assert '"solver_residuals":' in source
    assert "get_solver_position_residual" in source
    assert "get_solver_velocity_residual" in source


def _telemetry_row(*, frame: int, time_s: float, phase: str) -> dict:
    return {
        "frame": frame,
        "time_s": time_s,
        "phase": phase,
        "joint_target": [time_s, -time_s],
        "joint_readback": [time_s + 0.001, -time_s],
        "joint_velocity": [1.0, -1.0],
        "joint_effort": [2.0, -2.0],
        "joint_power_w": [2.0, 2.0],
        "drive_work_increment_j": 0.04,
        "solver_residuals": {
            "status": "OBSERVED",
            "position_max": 0.02 * time_s,
            "position_rms": 0.01 * time_s,
            "velocity_max": 0.03 * time_s,
            "velocity_rms": 0.01 * time_s,
        },
        "bottle": {
            "position_world_m": [time_s, 0.0, 0.1],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "pose_finite_difference_velocity": {
                "linear_velocity_world_m_s": [1.0, 0.0, 0.0],
            },
        },
        "contact_semantics": {
            "bilateral_geometric_contact": phase != "SETTLE",
            "bilateral_solver_active_contact": phase != "SETTLE",
        },
        "contacts": [],
        "observation": {"hold_drop_m": 0.001 * time_s},
    }


def test_extract_runtime_cell_metrics_preserves_signed_contact_and_work() -> None:
    rows = [
        _telemetry_row(frame=1, time_s=0.1, phase="SETTLE"),
        _telemetry_row(frame=2, time_s=0.2, phase="BILATERAL_CONTACT"),
    ]
    rows[1]["contacts"] = [
        {
            "collider0_path": "/left_finger",
            "collider1_path": "/Bottle500",
            "impulse_ns": 0.2,
            "impulse_vector_ns": [0.2, 0.0, 0.0],
            "normal_world": [1.0, 0.0, 0.0],
            "separation_m": -0.001,
        },
        {
            "collider0_path": "/right_finger",
            "collider1_path": "/Bottle500",
            "impulse_ns": 0.3,
            "impulse_vector_ns": [-0.3, 0.0, 0.0],
            "normal_world": [1.0, 0.0, 0.0],
            "separation_m": -0.0005,
        },
    ]
    report = {
        "status": "PASS",
        "reason": "stable_20cm_hold",
        "metrics": {"hold_drop_m": 0.0002},
        "runtime": {
            "numerical_convergence": {
                "physical_model_signature": "same-model",
                "readback": {
                    "effective_physics_dt_s": 0.1,
                    "effective_position_iterations": 64,
                    "effective_velocity_iterations": 8,
                },
            }
        },
    }

    result = extract_runtime_cell_metrics(report=report, telemetry=rows)

    assert result["first_bilateral_solver_contact_time_s"] == 0.2
    assert result["contact"]["left_signed_normal_impulse_ns"] == pytest.approx(0.2)
    assert result["contact"]["right_signed_normal_impulse_ns"] == pytest.approx(-0.3)
    assert result["contact"]["maximum_penetration_m"] == pytest.approx(0.001)
    assert result["drive"]["signed_work_j"] == pytest.approx(0.08)
    assert result["drive"]["maximum_joint_target_error"] == pytest.approx(0.001)
    assert result["solver_residuals"]["position_max"] == pytest.approx(0.004)


def test_compare_runtime_cells_interpolates_equivalent_phase_trajectories() -> None:
    coarse = [
        _telemetry_row(frame=i + 1, time_s=t, phase="HOLD")
        for i, t in enumerate((0.0, 0.5, 1.0))
    ]
    fine = [
        _telemetry_row(frame=i + 1, time_s=t, phase="HOLD")
        for i, t in enumerate((0.0, 0.25, 0.5, 0.75, 1.0))
    ]

    result = compare_runtime_cells(coarse=coarse, fine=fine)

    assert result["joint_position_max_abs_difference"] == pytest.approx(0.0)
    assert result["joint_velocity_max_abs_difference"] == pytest.approx(0.0)
    assert result["bottle_position_max_norm_difference_m"] == pytest.approx(0.0)
    assert result["common_phases"] == ["HOLD"]


def test_select_coarsest_converged_value_requires_all_finer_pairs() -> None:
    comparisons = [
        {"coarse": 60, "fine": 120, "gates": {"position": False}},
        {"coarse": 120, "fine": 240, "gates": {"position": True}},
        {"coarse": 240, "fine": 480, "gates": {"position": True}},
    ]

    assert select_coarsest_converged_value(
        ordered_values=[60, 120, 240, 480],
        comparisons=comparisons,
    ) == 120


def test_select_coarsest_converged_value_returns_none_if_finest_pair_fails() -> None:
    assert select_coarsest_converged_value(
        ordered_values=[120, 240, 480],
        comparisons=[
            {"coarse": 120, "fine": 240, "gates": {"position": True}},
            {"coarse": 240, "fine": 480, "gates": {"position": False}},
        ],
    ) is None


def test_solver_sweeps_require_a_selected_timestep() -> None:
    assert should_continue_solver_sweeps(selected_frequency_hz=240) is True
    assert should_continue_solver_sweeps(selected_frequency_hz=None) is False


def test_optional_frequency_also_runs_free_motion_reference() -> None:
    source = Path(
        "tools/run_aloha1_physics_numerical_convergence.py"
    ).read_text(encoding="utf-8")
    optional_branch = source.index("if selected_frequency is None:")
    optional_runtime = source.index("frequency_cells.append(", optional_branch)

    assert "free_motion.append(" in source[optional_branch:optional_runtime]


def test_report_signature_ignores_process_runtime_metadata() -> None:
    first = {
        "status": "PARTIAL",
        "cells": [{"value": 1.0, "process": {"process_id": 1}}],
        "boundaries": {},
    }
    second = {
        "status": "PARTIAL",
        "cells": [
            {
                "value": 1.0,
                "process": {"process_id": 999, "runtime_seconds": 4.2},
            }
        ],
        "boundaries": {
            "extra_solver_cells_from_superseded_driver": {
                "status": "EXCLUDED_FROM_DECISION_IF_PRESENT",
                "absolute_paths": ["/volatile/old-driver-cell"],
            }
        },
    }

    assert _stable_report_signature(first) == _stable_report_signature(second)


def test_runtime_cell_command_scales_frame_gates_and_changes_only_numerics() -> None:
    command = build_runtime_cell_command(
        isaac_python=Path(".venv_issac/bin/python"),
        launcher=Path("tools/run_aloha1_grasp_20cm_gui.py"),
        runtime_config=Path("configs/runtime.yaml"),
        artifact_root=Path("artifacts/cell"),
        bottle_transform_path=Path("artifacts/pose.json"),
        initial_arm_q_rad=[0.0] * 6,
        frequency_hz=240,
        position_iterations=32,
        velocity_iterations=4,
        capture_failure_evidence=False,
    )

    assert command[0:2] == [
        ".venv_issac/bin/python",
        "tools/run_aloha1_grasp_20cm_gui.py",
    ]
    assert command[command.index("--initial-pose-hold-frames") + 1] == "240"
    assert command[command.index("--physics-frequency-hz") + 1] == "240"
    assert command[command.index("--solver-position-iterations") + 1] == "32"
    assert command[command.index("--solver-velocity-iterations") + 1] == "4"
    assert "--skip-video-capture" in command
    assert "--skip-collider-evidence" in command
    assert "--enable-solver-residual-reporting" in command
