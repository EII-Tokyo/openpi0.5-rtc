import math

import pytest

from aloha.current_pose_rearm import (
    DualGripperRearmDetector,
    RearmState,
    evaluate_joint_alignment,
    hold_leader_arms_at_current_pose,
    wait_for_safe_current_pose_rearm,
)


def test_rearm_requires_both_grippers_to_open_then_both_to_close():
    detector = DualGripperRearmDetector(debounce_samples=1)

    assert not detector.update({"left": -0.05, "right": -0.05})
    assert detector.state is RearmState.WAITING_FOR_OPEN
    assert not detector.update({"left": 0.4, "right": -0.05})
    assert detector.state is RearmState.WAITING_FOR_OPEN
    assert not detector.update({"left": 0.4, "right": 0.4})
    assert detector.state is RearmState.WAITING_FOR_CLOSE
    assert not detector.update({"left": -0.05, "right": 0.4})

    assert detector.update({"left": -0.05, "right": -0.05})
    assert detector.state is RearmState.READY


def test_rearm_debounces_open_and_close_samples():
    detector = DualGripperRearmDetector(debounce_samples=3)

    for _ in range(2):
        assert not detector.update({"left": 0.4, "right": 0.4})
    assert detector.state is RearmState.WAITING_FOR_OPEN
    assert not detector.update({"left": -0.05, "right": -0.05})

    for _ in range(3):
        assert not detector.update({"left": 0.4, "right": 0.4})
    assert detector.state is RearmState.WAITING_FOR_CLOSE

    for _ in range(2):
        assert not detector.update({"left": -0.05, "right": -0.05})
    assert detector.update({"left": -0.05, "right": -0.05})


def test_rejected_alignment_can_reset_for_another_deliberate_gesture():
    detector = DualGripperRearmDetector(debounce_samples=1)
    detector.update({"left": 0.4, "right": 0.4})
    assert detector.update({"left": -0.05, "right": -0.05})

    detector.reset()

    assert detector.state is RearmState.WAITING_FOR_OPEN
    assert not detector.update({"left": -0.05, "right": -0.05})


def test_alignment_uses_shortest_distance_for_continuous_joints():
    leader = {
        "left": [0.0, 0.1, 0.2, math.pi - 0.01, 0.4, -math.pi + 0.02],
    }
    follower = {
        "left": [0.0, 0.1, 0.2, -math.pi + 0.01, 0.4, math.pi - 0.02],
    }

    report = evaluate_joint_alignment(
        leader,
        follower,
        max_joint_error_rad=0.05,
        continuous_joint_indices=(3, 5),
    )

    assert report.safe
    assert report.max_error_rad == pytest.approx(0.04)
    assert report.pair_errors_rad["left"] == pytest.approx(0.04)


def test_alignment_rejects_missing_pair_and_excess_error():
    with pytest.raises(ValueError, match="unmatched.*right"):
        evaluate_joint_alignment(
            {"left": [0.0] * 6, "right": [0.0] * 6},
            {"left": [0.0] * 6},
            max_joint_error_rad=0.1,
        )

    report = evaluate_joint_alignment(
        {"left": [0.0] * 6},
        {"left": [0.0, 0.0, 0.0, 0.0, 0.0, 0.2]},
        max_joint_error_rad=0.1,
    )
    assert not report.safe
    assert report.max_error_rad == pytest.approx(0.2)


def test_hold_leader_arms_commands_current_pose_before_enabling_torque():
    leader = object()
    calls = []

    hold_leader_arms_at_current_pose(
        {"leader_left": leader},
        gravity_compensation=True,
        read_positions=lambda robot: (
            calls.append(("read", robot)) or [0.1, 0.2]
        ),
        disable_gravity_compensation=lambda robot: calls.append(
            ("disable_gravity", robot)
        ),
        set_position_mode=lambda robot: calls.append(
            ("position_mode", robot)
        ),
        command_positions=lambda robot, positions: (
            calls.append(("command", robot, positions)) or True
        ),
        torque_enable=lambda robot, cmd_type, name, enable: calls.append(
            ("torque", robot, cmd_type, name, enable)
        ),
    )

    assert calls == [
        ("read", leader),
        ("disable_gravity", leader),
        ("position_mode", leader),
        ("command", leader, (0.1, 0.2)),
        ("torque", leader, "group", "arm", True),
        ("torque", leader, "single", "gripper", False),
    ]


def test_hold_leader_arms_fails_before_torque_when_goal_is_refused():
    torque_calls = []

    with pytest.raises(RuntimeError, match="leader_left.*refused"):
        hold_leader_arms_at_current_pose(
            {"leader_left": object()},
            gravity_compensation=False,
            read_positions=lambda _robot: [0.1, 0.2],
            disable_gravity_compensation=lambda _robot: pytest.fail(
                "gravity compensation is disabled"
            ),
            set_position_mode=lambda _robot: None,
            command_positions=lambda _robot, _positions: False,
            torque_enable=lambda *_args: torque_calls.append(_args),
        )

    assert torque_calls == []


def test_hold_leader_arms_reports_rejected_measured_positions():
    measured = [0.0, -0.96, 1.16, 1.57, 0.0, -1.57]

    with pytest.raises(RuntimeError) as error:
        hold_leader_arms_at_current_pose(
            {"leader_left": object()},
            gravity_compensation=False,
            read_positions=lambda _robot: measured,
            disable_gravity_compensation=lambda _robot: None,
            set_position_mode=lambda _robot: None,
            command_positions=lambda _robot, _positions: False,
            torque_enable=lambda *_args: pytest.fail(
                "torque must not be enabled"
            ),
        )

    message = str(error.value)
    assert "leader_left" in message
    assert repr(tuple(measured)) in message


def test_wait_warns_about_misalignment_but_accepts_the_gesture():
    samples = iter(
        [
            {"left": 0.4, "right": 0.4},
            {"left": -0.05, "right": -0.05},
        ]
    )
    restored = []
    logs = []

    accepted = wait_for_safe_current_pose_rearm(
        read_grippers=lambda: next(samples),
        read_leader_positions=lambda: {
            "left": [0.0] * 6,
            "right": [0.0] * 6,
        },
        read_follower_positions=lambda: {
            "left": [0.4] * 6,
            "right": [0.0] * 6,
        },
        restore_teleop=lambda: restored.append(True),
        stop_requested=lambda: False,
        max_joint_error_rad=0.1,
        debounce_samples=1,
        sleep=lambda _seconds: None,
        logger=logs.append,
    )

    assert accepted
    assert restored == [True]
    assert any("warning" in message and "0.4000" in message for message in logs)


def test_wait_returns_without_restoring_when_stop_is_requested():
    restored = []

    accepted = wait_for_safe_current_pose_rearm(
        read_grippers=lambda: pytest.fail("must not read after stop"),
        read_leader_positions=lambda: {},
        read_follower_positions=lambda: {},
        restore_teleop=lambda: restored.append(True),
        stop_requested=lambda: True,
        max_joint_error_rad=0.1,
        sleep=lambda _seconds: None,
    )

    assert not accepted
    assert restored == []


def test_wait_rechecks_health_until_restore_and_rejects_new_staleness():
    grippers = iter(
        [
            {"left": 0.4, "right": 0.4},
            {"left": -0.05, "right": -0.05},
        ]
    )
    health_checks = []
    restored = []

    def health_check():
        health_checks.append(len(health_checks) + 1)
        if len(health_checks) == 2:
            raise RuntimeError("leader state became stale")

    with pytest.raises(RuntimeError, match="became stale"):
        wait_for_safe_current_pose_rearm(
            read_grippers=lambda: next(grippers),
            read_leader_positions=lambda: {
                "left": [0.0] * 6,
                "right": [0.0] * 6,
            },
            read_follower_positions=lambda: {
                "left": [0.0] * 6,
                "right": [0.0] * 6,
            },
            restore_teleop=lambda: restored.append(True),
            stop_requested=lambda: False,
            health_check=health_check,
            max_joint_error_rad=0.1,
            debounce_samples=1,
            sleep=lambda _seconds: None,
        )

    assert health_checks == [1, 2]
    assert restored == []
