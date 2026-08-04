import math
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

from aloha.robot_health import (
    RobotHealthMonitor,
    RobotHealthUnavailable,
    attach_joint_state_subscriptions,
)


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def joint_state(*positions, names=("waist", "shoulder", "gripper")):
    return SimpleNamespace(name=list(names), position=list(positions))


def make_monitor():
    clock = FakeClock()
    monitor = RobotHealthMonitor(clock=clock, watchdog_rate_hz=10.0)
    monitor.register_robot("leader_left", {"waist", "shoulder", "gripper"})
    monitor.register_robot("leader_right", {"waist", "shoulder", "gripper"})
    return clock, monitor


@pytest.fixture
def sensor_msgs_stub(monkeypatch):
    sensor_msgs = ModuleType("sensor_msgs")
    sensor_msgs_msg = ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.JointState = object
    sensor_msgs.msg = sensor_msgs_msg
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_msgs)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msgs_msg)


def test_valid_samples_require_consecutive_sequence_updates():
    _, monitor = make_monitor()

    monitor.accept("leader_left", joint_state(0.1, 0.2, 0.3))
    monitor.accept("leader_left", joint_state(0.1, 0.2, 0.3))
    monitor.accept("leader_left", joint_state(0.1, 0.2, 0.3))

    snapshot = monitor.snapshot("leader_left")
    assert snapshot.sequence == 3
    assert snapshot.consecutive_valid == 3
    assert snapshot.valid


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        (joint_state(0.1, 0.2, names=("waist", "shoulder")), "missing_expected_joints"),
        (joint_state(0.1, math.nan, 0.3), "non_finite_position"),
        (
            joint_state(-math.pi, -math.pi, -math.pi),
            "invalid_all_minus_pi",
        ),
    ],
)
def test_invalid_sample_resets_count_and_latches_first_fault(message, reason):
    _, monitor = make_monitor()
    monitor.accept("leader_left", joint_state(0.1, 0.2, 0.3))

    monitor.accept("leader_left", message)

    snapshot = monitor.snapshot("leader_left")
    assert snapshot.sequence == 2
    assert snapshot.consecutive_valid == 0
    assert not snapshot.valid
    assert snapshot.reason == reason
    assert monitor.fault_event.is_set()
    assert monitor.first_fault.robot_name == "leader_left"
    assert monitor.first_fault.reason == reason


def test_first_fault_is_permanent():
    _, monitor = make_monitor()
    monitor.accept(
        "leader_left",
        joint_state(-math.pi, -math.pi, -math.pi),
    )
    monitor.accept(
        "leader_right",
        joint_state(0.1, math.inf, 0.3),
    )

    assert monitor.first_fault.robot_name == "leader_left"
    assert monitor.first_fault.reason == "invalid_all_minus_pi"


def test_watchdog_marks_only_armed_stale_robot():
    clock, monitor = make_monitor()
    monitor.accept("leader_left", joint_state(0.1, 0.2, 0.3))

    with monitor.arm_scope(
        {"leader_left"},
        phase="opening_home",
        max_age=0.30,
        latch_global=True,
    ) as scope:
        clock.advance(0.31)
        monitor.check_once()

        assert scope.fault_event.is_set()
        assert monitor.first_fault.robot_name == "leader_left"
        assert monitor.first_fault.reason == "joint_state_stale"


def test_unarmed_stale_robot_does_not_fault_scope():
    clock, monitor = make_monitor()
    monitor.accept("leader_left", joint_state(0.1, 0.2, 0.3))
    monitor.accept("leader_right", joint_state(0.1, 0.2, 0.3))

    with monitor.arm_scope(
        {"leader_right"},
        phase="safe_sleep",
        max_age=0.30,
        latch_global=False,
    ) as scope:
        clock.advance(0.20)
        monitor.accept("leader_right", joint_state(0.1, 0.2, 0.3))
        clock.advance(0.20)
        monitor.check_once()

        assert not scope.fault_event.is_set()


def test_recovery_scope_is_not_poisoned_by_original_robot_fault():
    _, monitor = make_monitor()
    monitor.latch_fault("leader_left", "opening_home", "joint_state_stale")

    with monitor.arm_scope(
        {"leader_right"},
        phase="safe_sleep",
        max_age=0.30,
        latch_global=False,
    ) as scope:
        monitor.accept("leader_right", joint_state(0.1, 0.2, 0.3))
        monitor.check_once()

        assert not scope.fault_event.is_set()
        assert monitor.first_fault.robot_name == "leader_left"


def test_wait_for_fresh_honors_stop_request():
    _, monitor = make_monitor()

    with pytest.raises(RobotHealthUnavailable, match="stop requested"):
        monitor.wait_for_fresh(
            {"leader_left"},
            consecutive=3,
            max_age=0.30,
            timeout=1.0,
            stop_requested=lambda: True,
        )


def test_wait_for_fresh_requires_new_samples_after_gate_entry():
    clock = FakeClock()
    monitor = None

    def deliver_next_sample(_seconds):
        clock.advance(0.01)
        monitor.accept("leader_left", joint_state(0.1, 0.2, 0.3))

    monitor = RobotHealthMonitor(
        clock=clock,
        sleeper=deliver_next_sample,
    )
    monitor.register_robot(
        "leader_left",
        {"waist", "shoulder", "gripper"},
    )
    for _ in range(5):
        monitor.accept("leader_left", joint_state(0.1, 0.2, 0.3))

    snapshots = monitor.wait_for_fresh(
        {"leader_left"},
        consecutive=3,
        max_age=0.30,
        timeout=1.0,
        stop_requested=lambda: False,
    )

    assert snapshots["leader_left"].sequence == 8


def test_wait_for_fresh_requires_new_samples_from_every_rearmed_robot():
    clock = FakeClock()
    robot_names = (
        "leader_left",
        "leader_right",
        "follower_left",
        "follower_right",
    )
    monitor = None

    def deliver_next_samples(_seconds):
        clock.advance(0.01)
        for robot_name in robot_names:
            monitor.accept(robot_name, joint_state(0.1, 0.2, 0.3))

    monitor = RobotHealthMonitor(
        clock=clock,
        sleeper=deliver_next_samples,
    )
    for robot_name in robot_names:
        monitor.register_robot(
            robot_name,
            {"waist", "shoulder", "gripper"},
        )
        for _ in range(5):
            monitor.accept(robot_name, joint_state(0.1, 0.2, 0.3))

    snapshots = monitor.wait_for_fresh(
        robot_names,
        consecutive=3,
        max_age=0.30,
        timeout=1.0,
        stop_requested=lambda: False,
    )

    assert {
        robot_name: snapshot.sequence
        for robot_name, snapshot in snapshots.items()
    } == {robot_name: 8 for robot_name in robot_names}


def test_require_fresh_returns_current_valid_snapshots():
    _, monitor = make_monitor()
    monitor.accept("leader_left", joint_state(0.1, 0.2, 0.3))

    snapshots = monitor.require_fresh(
        {"leader_left"},
        max_age=0.10,
        phase="teleop_wait",
    )

    assert snapshots["leader_left"].message_age == 0.0
    assert snapshots["leader_left"].valid


def test_require_fresh_reports_stale_robot_phase_and_age():
    clock, monitor = make_monitor()
    monitor.accept("leader_left", joint_state(0.1, 0.2, 0.3))
    clock.advance(0.11)

    with pytest.raises(
        RobotHealthUnavailable,
        match=r"leader_left.*teleop_wait.*0\.110",
    ):
        monitor.require_fresh(
            {"leader_left"},
            max_age=0.10,
            phase="teleop_wait",
        )


def test_require_fresh_rejects_invalid_current_sample():
    _, monitor = make_monitor()
    monitor.accept(
        "leader_left",
        joint_state(-math.pi, -math.pi, -math.pi),
    )

    with pytest.raises(
        RobotHealthUnavailable,
        match="invalid_all_minus_pi",
    ):
        monitor.require_fresh(
            {"leader_left"},
            max_age=0.10,
            phase="episode_collection",
        )


def test_require_fresh_rejects_latched_malformed_then_valid_sample():
    _, monitor = make_monitor()
    monitor.accept(
        "leader_left",
        joint_state(-math.pi, -math.pi, -math.pi),
    )
    monitor.accept("leader_left", joint_state(0.1, 0.2, 0.3))
    assert monitor.snapshot("leader_left").valid

    with pytest.raises(
        RobotHealthUnavailable,
        match="invalid_all_minus_pi",
    ):
        monitor.require_fresh(
            {"leader_left"},
            max_age=0.10,
            phase="episode_collection",
        )


def test_duplicate_registration_is_rejected():
    _, monitor = make_monitor()

    with pytest.raises(ValueError, match="already registered"):
        monitor.register_robot("leader_left", {"waist"})


def test_subscription_wiring_uses_joint_state_topics_without_bus_reads(
    sensor_msgs_stub,
):
    class FakeNode:
        def __init__(self):
            self.calls = []

        def create_subscription(self, message_type, topic, callback, qos):
            subscription = SimpleNamespace(
                message_type=message_type,
                topic=topic,
                callback=callback,
                qos=qos,
            )
            self.calls.append(subscription)
            return subscription

    clock = FakeClock()
    monitor = RobotHealthMonitor(clock=clock)
    node = FakeNode()

    subscriptions = attach_joint_state_subscriptions(
        node,
        monitor,
        {"leader_left": {"waist", "shoulder", "gripper"}},
    )
    subscriptions[0].callback(joint_state(0.1, 0.2, 0.3))

    assert [call.topic for call in node.calls] == ["/leader_left/joint_states"]
    assert monitor.snapshot("leader_left").consecutive_valid == 1


def test_subscription_wiring_rolls_back_and_can_retry_after_failure(
    sensor_msgs_stub,
):
    class FakeNode:
        def __init__(self):
            self.fail = True
            self.created = []
            self.destroyed = []

        def create_subscription(
            self,
            message_type,
            topic,
            callback,
            qos,
        ):
            if self.fail and self.created:
                raise RuntimeError("subscription unavailable")
            subscription = SimpleNamespace(
                message_type=message_type,
                topic=topic,
                callback=callback,
                qos=qos,
            )
            self.created.append(subscription)
            return subscription

        def destroy_subscription(self, subscription):
            self.destroyed.append(subscription)

    monitor = RobotHealthMonitor(clock=FakeClock())
    node = FakeNode()
    expected = {
        "leader_left": {"waist", "shoulder", "gripper"},
        "leader_right": {"waist", "shoulder", "gripper"},
    }

    with pytest.raises(RuntimeError, match="subscription unavailable"):
        attach_joint_state_subscriptions(node, monitor, expected)
    with pytest.raises(KeyError, match="leader_left"):
        monitor.snapshot("leader_left")
    with pytest.raises(KeyError, match="leader_right"):
        monitor.snapshot("leader_right")
    assert len(node.destroyed) == 1

    node.fail = False
    node.created.clear()
    subscriptions = attach_joint_state_subscriptions(
        node,
        monitor,
        expected,
    )
    assert len(subscriptions) == 2
    subscriptions[0].callback(joint_state(0.1, 0.2, 0.3))
    assert monitor.snapshot("leader_left").consecutive_valid == 1


def test_disjoint_health_scopes_can_overlap_and_fault_independently():
    clock, monitor = make_monitor()
    monitor.accept("leader_left", joint_state(0.1, 0.2, 0.3))
    monitor.accept("leader_right", joint_state(0.1, 0.2, 0.3))

    with monitor.arm_scope(
        {"leader_left"},
        phase="left_recovery",
        max_age=0.30,
        latch_global=False,
    ) as left_scope, monitor.arm_scope(
        {"leader_right"},
        phase="right_recovery",
        max_age=0.30,
        latch_global=False,
    ) as right_scope:
        clock.advance(0.20)
        monitor.accept("leader_right", joint_state(0.1, 0.2, 0.3))
        clock.advance(0.20)
        monitor.check_once()

        assert left_scope.fault_event.is_set()
        assert left_scope.fault.phase == "left_recovery"
        assert not right_scope.fault_event.is_set()
        assert monitor.first_fault is None


def test_overlapping_health_scopes_are_rejected():
    _, monitor = make_monitor()

    with monitor.arm_scope(
        {"leader_left"},
        phase="left",
        max_age=0.30,
        latch_global=False,
    ):
        with pytest.raises(RuntimeError, match="overlapping"):
            with monitor.arm_scope(
                {"leader_left", "leader_right"},
                phase="both",
                max_age=0.30,
                latch_global=False,
            ):
                pass


def test_malformed_feedback_faults_only_owning_disjoint_scope():
    _, monitor = make_monitor()
    monitor.accept("leader_left", joint_state(0.1, 0.2, 0.3))
    monitor.accept("leader_right", joint_state(0.1, 0.2, 0.3))

    with monitor.arm_scope(
        {"leader_left"},
        phase="left_recovery",
        max_age=0.30,
        latch_global=False,
    ) as left_scope, monitor.arm_scope(
        {"leader_right"},
        phase="right_recovery",
        max_age=0.30,
        latch_global=False,
    ) as right_scope:
        monitor.accept(
            "leader_left",
            joint_state(-math.pi, -math.pi, -math.pi),
        )

        assert left_scope.fault_event.is_set()
        assert left_scope.fault.reason == "invalid_all_minus_pi"
        assert not right_scope.fault_event.is_set()
        assert monitor.first_fault is None


def test_deactivating_one_scope_leaves_other_scope_active():
    clock, monitor = make_monitor()
    monitor.accept("leader_left", joint_state(0.1, 0.2, 0.3))
    monitor.accept("leader_right", joint_state(0.1, 0.2, 0.3))

    with monitor.arm_scope(
        {"leader_right"},
        phase="right_recovery",
        max_age=0.30,
        latch_global=False,
    ) as right_scope:
        with monitor.arm_scope(
            {"leader_left"},
            phase="left_recovery",
            max_age=0.30,
            latch_global=False,
        ):
            pass
        clock.advance(0.31)
        monitor.check_once()

        assert right_scope.fault_event.is_set()
        assert right_scope.fault.phase == "right_recovery"


def test_unregister_rejects_robot_in_any_active_scope():
    _, monitor = make_monitor()

    with monitor.arm_scope(
        {"leader_right"},
        phase="right_recovery",
        max_age=0.30,
        latch_global=False,
    ):
        with pytest.raises(RuntimeError, match="active robot"):
            monitor.unregister_robot("leader_right")
        monitor.unregister_robot("leader_left")

    with pytest.raises(KeyError, match="leader_left"):
        monitor.snapshot("leader_left")
