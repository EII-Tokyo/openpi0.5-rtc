import threading
from types import SimpleNamespace

import pytest

from aloha.safe_motion import (
    GuardedMotionAborted,
    move_robots_guarded,
    plan_motion_duration,
)


def fake_robot(name, publishes, current=(0.0,)):
    class FakeArm:
        def get_joint_positions(self):
            return list(current)

        def set_joint_positions(self, positions, *, blocking):
            assert blocking is False
            publishes.append((name, tuple(positions)))

    return SimpleNamespace(arm=FakeArm())


def test_duration_uses_caller_selected_two_second_floor():
    assert plan_motion_duration(
        current=[0.0, 0.0],
        target=[0.1, 0.2],
        minimum_seconds=2.0,
        max_joint_speed=0.4,
    ) == 2.0


def test_duration_default_remains_four_seconds():
    assert plan_motion_duration(
        current=[0.0],
        target=[0.1],
    ) == 4.0


def test_duration_expands_for_large_joint_delta():
    assert plan_motion_duration(
        current=[0.0],
        target=[2.0],
        minimum_seconds=2.0,
        max_joint_speed=0.4,
    ) == 5.0


@pytest.mark.parametrize(
    "minimum_seconds",
    [0.0, -0.1, float("nan"), float("inf"), True],
)
def test_duration_rejects_invalid_minimum(minimum_seconds):
    with pytest.raises(ValueError, match="minimum_seconds"):
        plan_motion_duration(
            [0.0],
            [0.1],
            minimum_seconds=minimum_seconds,
        )


def test_duration_rejects_invalid_speed_and_pose_lengths():
    with pytest.raises(ValueError, match="max_joint_speed"):
        plan_motion_duration([0.0], [0.1], max_joint_speed=0.0)
    with pytest.raises(ValueError, match="same non-zero length"):
        plan_motion_duration([0.0], [0.1, 0.2])


def test_guarded_pair_motion_publishes_synchronized_cycles_and_target():
    publishes = []
    sleeps = []

    move_robots_guarded(
        robots={
            "leader_left": fake_robot("leader_left", publishes, current=(0.0,)),
            "follower_left": fake_robot(
                "follower_left",
                publishes,
                current=(1.0,),
            ),
        },
        targets={
            "leader_left": [0.2],
            "follower_left": [0.6],
        },
        dt=1.0,
        duration=4.0,
        fault_event=threading.Event(),
        health_check=lambda: None,
        sleep=sleeps.append,
    )

    assert len(publishes) == 10
    assert [name for name, _ in publishes[:2]] == [
        "leader_left",
        "follower_left",
    ]
    assert publishes[-2:] == [
        ("leader_left", (0.2,)),
        ("follower_left", (0.6,)),
    ]
    assert sleeps == [1.0, 1.0, 1.0, 1.0]


def test_guarded_pair_motion_stops_before_next_cycle_after_fault():
    fault = threading.Event()
    publishes = []
    checks = 0

    def health_check():
        nonlocal checks
        checks += 1
        if checks == 2:
            fault.set()

    with pytest.raises(GuardedMotionAborted):
        move_robots_guarded(
            robots={
                "leader_left": fake_robot("leader_left", publishes),
                "follower_left": fake_robot("follower_left", publishes),
            },
            targets={
                "leader_left": [0.2],
                "follower_left": [0.2],
            },
            dt=0.02,
            duration=4.0,
            fault_event=fault,
            health_check=health_check,
            sleep=lambda _: None,
        )

    assert [name for name, _positions in publishes] == [
        "leader_left",
        "follower_left",
    ]


def test_guarded_motion_checks_fault_before_first_publish():
    fault = threading.Event()
    fault.set()
    publishes = []

    with pytest.raises(GuardedMotionAborted):
        move_robots_guarded(
            robots={"leader_left": fake_robot("leader_left", publishes)},
            targets={"leader_left": [0.2]},
            dt=0.02,
            duration=4.0,
            fault_event=fault,
            health_check=lambda: None,
            sleep=lambda _: None,
        )

    assert publishes == []
