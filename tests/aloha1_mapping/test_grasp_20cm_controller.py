from __future__ import annotations

from dataclasses import replace

import pytest

from tools.aloha1_mapping.grasp_20cm_controller import ACTIVE_PHASES
from tools.aloha1_mapping.grasp_20cm_controller import Grasp20cmController
from tools.aloha1_mapping.grasp_20cm_controller import Grasp20cmThresholds
from tools.aloha1_mapping.grasp_20cm_controller import Phase
from tools.aloha1_mapping.grasp_20cm_controller import RunObservation
from tools.aloha1_mapping.grasp_20cm_controller import canonical_run_signature
from tools.aloha1_mapping.grasp_20cm_controller import evaluate_terminal_run
from tools.aloha1_mapping.grasp_20cm_controller import measured_clearance_m


def _terminal_metrics(**updates: object) -> dict[str, object]:
    metrics: dict[str, object] = {
        "height_reached": True,
        "maximum_clearance_m": 0.204,
        "ee_vertical_displacement_m": 0.205,
        "dynamic_during_formal_phases": True,
        "bilateral_contact_before_lift": True,
        "bilateral_contact_through_hold": True,
        "hold_duration_s": 2.0,
        "hold_drop_m": 0.001,
        "finite_state": True,
        "forbidden_constraint": False,
        "persistent_penetration": False,
        "numerical_ejection": False,
        "aborted": False,
    }
    metrics.update(updates)
    return metrics


def _observation(**updates: object) -> RunObservation:
    values: dict[str, object] = {
        "frame": 1,
        "time_s": 1.0 / 60.0,
        "clearance_m": 0.0,
        "bottle_dynamic": True,
        "support_contact": True,
        "bottle_linear_speed_m_s": 0.0,
        "bottle_angular_speed_rad_s": 0.0,
        "stage_contract_valid": True,
        "setup_complete": True,
        "open_target_reached": True,
        "descent_complete": True,
        "bilateral_contact": True,
        "preload_complete": True,
        "lift_waypoint_exhausted": False,
        "hold_drop_m": 0.0,
        "finite_state": True,
        "persistent_penetration": False,
        "numerical_ejection": False,
        "forbidden_constraint": False,
        "phase_timed_out": False,
        "ee_vertical_displacement_m": 0.0,
    }
    values.update(updates)
    return RunObservation(**values)


def _controller_at(
    phase: Phase,
    *,
    thresholds: Grasp20cmThresholds | None = None,
) -> Grasp20cmController:
    controller = Grasp20cmController(
        thresholds or Grasp20cmThresholds(settle_consecutive_frames=1)
    )
    controller.restore_for_test(phase)
    return controller


def test_clearance_uses_bottle_collision_minimum_and_table_top() -> None:
    assert measured_clearance_m(
        bottle_collision_min_world_z_m=0.247,
        table_top_world_z_m=0.047,
    ) == pytest.approx(0.200)


def test_clearance_rejects_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        measured_clearance_m(
            bottle_collision_min_world_z_m=float("nan"),
            table_top_world_z_m=0.0,
        )


def test_ee_motion_without_bottle_motion_cannot_pass() -> None:
    result = evaluate_terminal_run(
        _terminal_metrics(
            height_reached=False,
            maximum_clearance_m=0.004,
            ee_vertical_displacement_m=0.205,
        ),
        Grasp20cmThresholds(),
    )
    assert result == {
        "status": "FAIL",
        "failure_mode": "gripper_moved_without_bottle_lift",
        "task8": "NOT_RUN",
    }


def test_stable_bilateral_20cm_hold_passes() -> None:
    result = evaluate_terminal_run(
        _terminal_metrics(),
        Grasp20cmThresholds(),
    )
    assert result == {
        "status": "PASS",
        "failure_mode": "stable_20cm_hold",
        "task8": "NOT_RUN",
    }


@pytest.mark.parametrize(
    ("updates", "failure_mode"),
    [
        ({"aborted": True}, "aborted"),
        ({"forbidden_constraint": True}, "forbidden_constraint"),
        ({"finite_state": False}, "non_finite_state"),
        ({"persistent_penetration": True}, "numerical_penetration_or_ejection"),
        ({"numerical_ejection": True}, "numerical_penetration_or_ejection"),
        (
            {"dynamic_during_formal_phases": False},
            "bottle_not_dynamic_during_formal_phases",
        ),
        (
            {"bilateral_contact_before_lift": False},
            "bilateral_contact_not_established",
        ),
        ({"height_reached": False}, "height_target_not_reached"),
        (
            {"bilateral_contact_through_hold": False},
            "bilateral_contact_lost",
        ),
        ({"hold_duration_s": 1.99}, "hold_interval_incomplete"),
        ({"hold_drop_m": 0.011}, "hold_drop_exceeded"),
    ],
)
def test_terminal_failure_precedence(
    updates: dict[str, object],
    failure_mode: str,
) -> None:
    assert evaluate_terminal_run(
        _terminal_metrics(**updates),
        Grasp20cmThresholds(),
    )["failure_mode"] == failure_mode


def test_height_transition_uses_measured_bottle_clearance() -> None:
    controller = _controller_at(Phase.VERTICAL_LIFT)
    transition = controller.observe(_observation(clearance_m=0.1999))
    assert transition.current is Phase.VERTICAL_LIFT

    transition = controller.observe(
        replace(_observation(clearance_m=0.2000), frame=2)
    )
    assert transition.current is Phase.HEIGHT_REACHED


def test_descent_completes_before_bilateral_contact_is_required() -> None:
    controller = _controller_at(Phase.VERTICAL_DESCENT)
    transition = controller.observe(
        _observation(
            descent_complete=True,
            bilateral_contact=False,
        )
    )
    assert transition.current is Phase.BILATERAL_CONTACT

    transition = controller.observe(
        replace(
            _observation(bilateral_contact=False),
            frame=2,
        )
    )
    assert transition.current is Phase.BILATERAL_CONTACT

    transition = controller.observe(
        replace(
            _observation(bilateral_contact=True),
            frame=3,
        )
    )
    assert transition.current is Phase.CLOSE_PRELOAD


def test_lift_waypoint_exhaustion_fails_below_target() -> None:
    controller = _controller_at(Phase.VERTICAL_LIFT)
    transition = controller.observe(
        _observation(
            clearance_m=0.006,
            lift_waypoint_exhausted=True,
            ee_vertical_displacement_m=0.205,
        )
    )
    assert transition.current is Phase.FAIL
    assert transition.reason == "gripper_moved_without_bottle_lift"


def test_lift_waits_after_last_waypoint_while_bottle_is_still_moving() -> None:
    controller = _controller_at(Phase.VERTICAL_LIFT)
    transition = controller.observe(
        _observation(
            clearance_m=0.198,
            lift_waypoint_exhausted=True,
            ee_vertical_displacement_m=0.141,
            bottle_linear_speed_m_s=0.302,
            bottle_angular_speed_rad_s=3.75,
        )
    )
    assert transition.current is Phase.VERTICAL_LIFT
    assert transition.reason == "phase_in_progress"


def test_phase_timeout_fails_closed() -> None:
    controller = _controller_at(Phase.BILATERAL_CONTACT)
    transition = controller.observe(
        _observation(
            bilateral_contact=False,
            phase_timed_out=True,
        )
    )
    assert transition.current is Phase.FAIL
    assert transition.reason == "bilateral_contact_timeout"


def test_close_preload_waits_for_bilateral_contact_to_restabilize() -> None:
    controller = _controller_at(Phase.CLOSE_PRELOAD)
    transition = controller.observe(
        _observation(
            bilateral_contact=False,
            preload_complete=False,
            phase_timed_out=False,
        )
    )
    assert transition.current is Phase.CLOSE_PRELOAD
    assert transition.reason == "phase_in_progress"


def test_hold_records_transient_contact_loss_until_interval_end() -> None:
    controller = _controller_at(Phase.HEIGHT_REACHED)
    transition = controller.observe(_observation(frame=1, time_s=1.0))
    assert transition.current is Phase.HOLD

    transition = controller.observe(
        _observation(
            frame=2,
            time_s=1.0 + 1.0 / 60.0,
            bilateral_contact=False,
            hold_drop_m=0.006,
        )
    )
    assert transition.current is Phase.HOLD

    transition = controller.observe(
        _observation(
            frame=3,
            time_s=3.0,
            bilateral_contact=True,
            hold_drop_m=0.006,
        )
    )
    assert transition.current is Phase.FAIL
    assert transition.reason == "bilateral_contact_lost"


def test_abort_is_reachable_from_every_active_phase() -> None:
    controller = Grasp20cmController()
    for phase in ACTIVE_PHASES:
        controller.restore_for_test(phase)
        transition = controller.request_abort()
        assert transition.previous is phase
        assert transition.current is Phase.ABORTED
        assert transition.reason == "user_abort"
        controller.reset()


def test_reset_rejects_active_phase() -> None:
    controller = _controller_at(Phase.VERTICAL_DESCENT)
    with pytest.raises(RuntimeError, match="active"):
        controller.reset()


def test_signature_ignores_artifact_paths_and_wall_clock_runtime() -> None:
    observations = [
        _observation(frame=1),
        replace(_observation(clearance_m=0.2), frame=2),
    ]
    first = canonical_run_signature(
        observations,
        {
            **_terminal_metrics(),
            "artifact_absolute_path": "/tmp/run-a",
            "runtime_seconds": 2.0,
        },
    )
    second = canonical_run_signature(
        observations,
        {
            **_terminal_metrics(),
            "artifact_absolute_path": "/tmp/run-b",
            "runtime_seconds": 999.0,
        },
    )
    assert first == second
    assert len(first) == 64
