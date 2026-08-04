import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from aloha.robot_health import RobotHealthSnapshot
from aloha.safe_sleep import (
    RobotSleepResult,
    SafeSleepReport,
    SleepStatus,
    recover_robots_to_sleep,
    resolve_safe_sleep_target,
)


ROOT = Path(__file__).resolve().parents[1]
SLEEP_SCRIPT = ROOT / "scripts" / "sleep.py"


def load_standalone_sleep(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "aloha.robot_utils",
        SimpleNamespace(
            get_arm_joint_positions=lambda _robot: [],
            load_yaml_file=lambda *_args, **_kwargs: {},
        ),
    )
    for package_name in (
        "interbotix_common_modules",
        "interbotix_common_modules.common_robot",
        "interbotix_xs_modules",
        "interbotix_xs_modules.xs_robot",
    ):
        package = ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)
    robot_module = ModuleType(
        "interbotix_common_modules.common_robot.robot"
    )
    robot_module.create_interbotix_global_node = lambda *_args: None
    robot_module.robot_shutdown = lambda *_args: None
    robot_module.robot_startup = lambda *_args: None
    monkeypatch.setitem(
        sys.modules,
        "interbotix_common_modules.common_robot.robot",
        robot_module,
    )
    arm_module = ModuleType("interbotix_xs_modules.xs_robot.arm")
    arm_module.InterbotixManipulatorXS = object
    monkeypatch.setitem(
        sys.modules,
        "interbotix_xs_modules.xs_robot.arm",
        arm_module,
    )
    spec = importlib.util.spec_from_file_location(
        "standalone_sleep_for_test",
        SLEEP_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(name, *, valid=True, age=0.01):
    return RobotHealthSnapshot(
        robot_name=name,
        sequence=10,
        consecutive_valid=10 if valid else 0,
        message_age=age,
        valid=valid,
        reason=None if valid else "joint_state_stale",
    )


class SnapshotHealth:
    def __init__(self, snapshots):
        self.snapshots = snapshots

    def snapshot(self, name):
        return self.snapshots[name]


def four_robots():
    return {
        "follower_right": object(),
        "leader_right": object(),
        "follower_left": object(),
        "leader_left": object(),
    }


def slept(name):
    return RobotSleepResult(
        robot_name=name,
        status=SleepStatus.SLEPT_VERIFIED,
        max_error_rad=0.01,
        reason="verified",
        phase="complete",
        torque_off_verified=True,
    )


def group_info(
    *,
    names=("waist", "shoulder"),
    sleep=(0.2, -1.0),
    lower=(-2.0, -1.85),
    upper=(2.0, 1.85),
):
    return SimpleNamespace(
        joint_names=list(names),
        joint_sleep_positions=list(sleep),
        joint_lower_limits=list(lower),
        joint_upper_limits=list(upper),
    )


def robot_with_group(**kwargs):
    return SimpleNamespace(
        arm=SimpleNamespace(group_info=group_info(**kwargs))
    )


def test_named_pose_is_resolved_in_group_joint_order():
    assert resolve_safe_sleep_target(
        "follower_left",
        group_info(),
        {
            "follower_left": {
                "shoulder": -1.84,
                "waist": 0.0,
            }
        },
        limit_margin_rad=0.01,
    ) == (0.0, -1.84)


def test_model_pose_is_used_when_no_machine_override_exists():
    assert resolve_safe_sleep_target(
        "leader_left",
        group_info(),
        {},
        limit_margin_rad=0.01,
    ) == (0.2, -1.0)


def test_model_pose_inside_true_limit_but_inside_margin_is_accepted():
    info = group_info(
        names=("shoulder",),
        sleep=(-1.880,),
        lower=(-1.885,),
        upper=(1.990,),
    )

    assert resolve_safe_sleep_target(
        "leader_left",
        info,
        {},
        limit_margin_rad=0.010,
    ) == (-1.880,)


def test_configured_pose_inside_margin_is_rejected():
    info = group_info(
        names=("shoulder",),
        sleep=(-1.880,),
        lower=(-1.885,),
        upper=(1.990,),
    )

    with pytest.raises(ValueError, match="inside limits"):
        resolve_safe_sleep_target(
            "leader_left",
            info,
            {"leader_left": {"shoulder": -1.880}},
            limit_margin_rad=0.010,
        )


@pytest.mark.parametrize(
    "lower,upper,sleep,match",
    [
        ((1.0,), (-1.0,), (0.0,), "ordered"),
        ((float("nan"),), (1.0,), (0.0,), "finite"),
        ((-1.0,), (float("inf"),), (0.0,), "finite"),
        ((-1.0,), (1.0,), (1.1,), "inside limits"),
    ],
)
def test_invalid_model_limit_metadata_or_pose_is_rejected(
    lower,
    upper,
    sleep,
    match,
):
    info = group_info(
        names=("shoulder",),
        sleep=sleep,
        lower=lower,
        upper=upper,
    )

    with pytest.raises(ValueError, match=match):
        resolve_safe_sleep_target("leader_left", info, {})


@pytest.mark.parametrize(
    "configured,match",
    [
        ({"waist": 0.0}, "joint names"),
        (
            {"waist": 0.0, "shoulder": -1.0, "extra": 0.0},
            "joint names",
        ),
        (
            {"waist": 0.0, "shoulder": float("nan")},
            "finite",
        ),
        (
            {"waist": 0.0, "shoulder": -1.85},
            "inside limits",
        ),
    ],
)
def test_invalid_named_pose_fails_before_prepare(configured, match):
    calls = []
    report = recover_robots_to_sleep(
        robots={"follower_left": robot_with_group()},
        health=SnapshotHealth(
            {"follower_left": snapshot("follower_left")}
        ),
        safe_sleep_positions={"follower_left": configured},
        prepare_robot=lambda *_args: calls.append("prepare"),
        read_positions=lambda _robot: [0.0, 0.0],
        home_positions=[0.0, 0.0],
    )

    result = report.results["follower_left"]
    assert result.status is SleepStatus.FAILED
    assert match in result.reason
    assert calls == []


def test_unresponsive_robot_does_not_block_other_sleep_attempts():
    robots = four_robots()
    health = SnapshotHealth(
        {
            name: snapshot(
                name,
                valid=name != "leader_left",
                age=1.0 if name == "leader_left" else 0.01,
            )
            for name in robots
        }
    )
    attempted = []

    def sleep_one(name, _robot):
        attempted.append(name)
        return slept(name)

    report = recover_robots_to_sleep(
        robots=robots,
        health=health,
        sleep_one=sleep_one,
    )

    assert (
        report.results["leader_left"].status
        is SleepStatus.UNRESPONSIVE
    )
    assert set(attempted) == {
        "leader_right",
        "follower_left",
        "follower_right",
    }


def test_one_sleep_failure_does_not_block_later_robots():
    robots = four_robots()
    health = SnapshotHealth(
        {name: snapshot(name) for name in robots}
    )
    attempted = set()
    attempted_lock = threading.Lock()
    all_entered = threading.Event()
    worker_threads = {}

    def sleep_one(name, _robot):
        with attempted_lock:
            attempted.add(name)
            worker_threads[name] = threading.current_thread()
            if attempted == set(robots):
                all_entered.set()
        if not all_entered.wait(0.5):
            raise AssertionError("safe-sleep workers did not overlap")
        if name == "leader_right":
            raise RuntimeError("injected motion failure")
        return slept(name)

    report = recover_robots_to_sleep(
        robots=robots,
        health=health,
        sleep_one=sleep_one,
    )

    assert attempted == set(robots)
    assert all(
        thread.name == f"aloha-safe-sleep-{name}"
        and not thread.daemon
        for name, thread in worker_threads.items()
    )
    assert (
        report.results["leader_right"].status
        is SleepStatus.FAILED
    )
    assert {
        name
        for name, result in report.results.items()
        if result.status is SleepStatus.SLEPT_VERIFIED
    } == {
        "leader_left",
        "follower_left",
        "follower_right",
    }
    assert (
        "injected motion failure"
        in report.results["leader_right"].reason
    )


def test_report_is_safe_only_when_every_robot_is_verified():
    partial_report = SafeSleepReport(
        results={
            "leader_left": slept("leader_left"),
            "leader_right": RobotSleepResult(
                robot_name="leader_right",
                status=SleepStatus.UNRESPONSIVE,
                max_error_rad=None,
                reason="stale",
            ),
        }
    )
    complete_report = SafeSleepReport(
        results={
            "leader_left": slept("leader_left"),
            "leader_right": slept("leader_right"),
        }
    )

    assert not partial_report.safe_to_stop
    assert complete_report.safe_to_stop
    assert not SafeSleepReport(results={}).safe_to_stop


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"max_error_rad": float("nan")}, "max_error_rad"),
        ({"settle_error_rad": 0.0}, "settle_error_rad"),
        ({"settle_seconds": -0.1}, "settle_seconds"),
        ({"verification_samples": 0}, "verification_samples"),
    ],
)
def test_invalid_verification_limits_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        recover_robots_to_sleep(
            robots={"leader_left": object()},
            health=object(),
            sleep_one=lambda *_args: slept("leader_left"),
            **kwargs,
        )


def test_default_sleep_uses_independent_scope_and_verifies_target():
    calls = []
    robot = SimpleNamespace(
        arm=SimpleNamespace(
            group_info=group_info(
                sleep=(0.2, 0.3),
            ),
        )
    )

    class Scope:
        def __init__(self):
            import threading

            self.fault_event = threading.Event()

        def __enter__(self):
            calls.append("scope_enter")
            return self

        def __exit__(self, *_args):
            calls.append("scope_exit")

        def raise_if_faulted(self):
            calls.append("health_check")

    class Health(SnapshotHealth):
        def arm_scope(self, names, *, phase, max_age, latch_global):
            calls.append(
                ("scope", names, phase, max_age, latch_global)
            )
            return Scope()

        def wait_for_fresh(self, names, **kwargs):
            calls.append(("fresh", names, kwargs["consecutive"]))
            return {name: snapshot(name) for name in names}

    health = Health({"leader_left": snapshot("leader_left")})
    positions = iter(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.3],
            [0.2, 0.3],
            [0.2, 0.3],
            [0.45, 0.55],
            [0.46, 0.54],
            [0.45, 0.55],
            [0.46, 0.54],
        ]
    )

    report = recover_robots_to_sleep(
        robots={"leader_left": robot},
        health=health,
        prepare_robot=lambda name, _robot: calls.append(
            ("prepare", name)
        ),
        torque_off_robot=lambda name, _robot: calls.append(
            ("torque_off", name)
        ),
        read_positions=lambda _robot: next(positions),
        home_positions=[0.1, 0.1],
        settle_sleep=lambda seconds: calls.append(
            ("settle", seconds)
        ),
        move_guarded=lambda **kwargs: calls.append(
            ("move", tuple(kwargs["targets"]["leader_left"]))
        ),
        plan_duration=lambda *_args, **_kwargs: 4.0,
    )

    assert report.safe_to_stop
    assert (
        "scope",
        {"leader_left"},
        "safe_sleep:leader_left",
        0.3,
        False,
    ) in calls
    assert ("move", (0.1, 0.1)) in calls
    assert ("move", (0.2, 0.3)) in calls
    assert calls.count(("fresh", {"leader_left"}, 1)) == 7
    assert ("torque_off", "leader_left") in calls
    assert ("settle", 0.5) in calls
    result = report.results["leader_left"]
    assert result.torque_off_verified
    assert result.phase == "complete"
    assert result.max_error_rad == 0.01
    assert "target displacement=0.260 rad" in result.reason


def test_safe_sleep_prepares_then_requires_fresh_feedback_before_scope():
    events = []
    robot = robot_with_group(sleep=(0.2, 0.3))

    class Scope:
        def __init__(self):
            import threading

            self.fault_event = threading.Event()

        def __enter__(self):
            events.append("scope_enter")
            return self

        def __exit__(self, *_args):
            return None

        def raise_if_faulted(self):
            return None

    class Health(SnapshotHealth):
        def arm_scope(self, *_args, **_kwargs):
            return Scope()

        def wait_for_fresh(self, names, **kwargs):
            events.append(
                (
                    "fresh",
                    names,
                    kwargs["consecutive"],
                    kwargs["max_age"],
                    kwargs["timeout"],
                )
            )
            return {name: snapshot(name) for name in names}

    positions = iter(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.3],
            [0.2, 0.3],
            [0.2, 0.3],
            [0.45, 0.55],
            [0.46, 0.54],
            [0.45, 0.55],
            [0.46, 0.54],
        ]
    )

    def read_positions(_robot):
        events.append("read_positions")
        return next(positions)

    report = recover_robots_to_sleep(
        robots={"leader_left": robot},
        health=Health({"leader_left": snapshot("leader_left")}),
        prepare_robot=lambda *_args: events.append("prepare"),
        torque_off_robot=lambda *_args: None,
        read_positions=read_positions,
        home_positions=[0.1, 0.1],
        settle_sleep=lambda _seconds: None,
        move_guarded=lambda **_kwargs: None,
        plan_duration=lambda *_args, **_kwargs: 2.0,
    )

    post_prepare_gate = ("fresh", {"leader_left"}, 3, 0.3, 2.0)
    assert report.safe_to_stop
    assert events.index("prepare") < events.index(post_prepare_gate)
    assert events.index(post_prepare_gate) < events.index("scope_enter")
    assert events.index("scope_enter") < events.index("read_positions")


def test_safe_sleep_does_not_move_when_post_prepare_feedback_fails():
    calls = []

    class Scope:
        def __init__(self):
            import threading

            self.fault_event = threading.Event()

        def __enter__(self):
            calls.append("scope_enter")
            return self

        def __exit__(self, *_args):
            return None

        def raise_if_faulted(self):
            return None

    class Health(SnapshotHealth):
        def arm_scope(self, *_args, **_kwargs):
            return Scope()

        def wait_for_fresh(self, _names, **kwargs):
            assert kwargs["consecutive"] == 3
            raise RuntimeError("feedback did not recover")

    report = recover_robots_to_sleep(
        robots={"leader_left": robot_with_group()},
        health=Health({"leader_left": snapshot("leader_left")}),
        prepare_robot=lambda *_args: calls.append("prepare"),
        torque_off_robot=lambda *_args: pytest.fail(
            "torque-off verification must not run"
        ),
        read_positions=lambda _robot: pytest.fail(
            "positions must not be read without fresh feedback"
        ),
        home_positions=[0.0, 0.0],
        move_guarded=lambda **_kwargs: pytest.fail(
            "motion must not start without fresh feedback"
        ),
    )

    result = report.results["leader_left"]
    assert calls == ["prepare"]
    assert result.status is SleepStatus.FAILED
    assert "feedback did not recover" in result.reason


def test_safe_sleep_default_plans_home_and_sleep_with_one_second():
    robot = robot_with_group(
        names=("waist",),
        sleep=(0.2,),
        lower=(-2.0,),
        upper=(2.0,),
    )
    planned = []

    class Scope:
        def __init__(self):
            import threading

            self.fault_event = threading.Event()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_if_faulted(self):
            return None

    class Health(SnapshotHealth):
        def arm_scope(self, *_args, **_kwargs):
            return Scope()

        def wait_for_fresh(self, names, **_kwargs):
            return {name: snapshot(name) for name in names}

    positions = iter(
        [
            [0.0],
            [0.1],
            [0.2],
            [0.2],
            [0.2],
            [0.3],
            [0.3],
            [0.3],
            [0.3],
        ]
    )

    def plan_duration(_current, _target, **kwargs):
        planned.append(
            (
                kwargs["minimum_seconds"],
                kwargs["max_joint_speed"],
            )
        )
        return kwargs["minimum_seconds"]

    report = recover_robots_to_sleep(
        robots={"leader_left": robot},
        health=Health({"leader_left": snapshot("leader_left")}),
        prepare_robot=lambda *_args: None,
        torque_off_robot=lambda *_args: None,
        read_positions=lambda _robot: next(positions),
        home_positions=[0.1],
        settle_sleep=lambda _seconds: None,
        move_guarded=lambda **_kwargs: None,
        plan_duration=plan_duration,
    )

    assert report.safe_to_stop
    assert planned == [(1.0, 0.4), (1.0, 0.4)]


def test_default_sleep_rejects_position_outside_tolerance():
    robot = SimpleNamespace(
        arm=SimpleNamespace(
            group_info=group_info(
                names=("waist",),
                sleep=(0.2,),
                lower=(-2.0,),
                upper=(2.0,),
            ),
        )
    )

    class Scope:
        def __init__(self):
            import threading

            self.fault_event = threading.Event()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_if_faulted(self):
            return None

    class Health(SnapshotHealth):
        def arm_scope(self, *_args, **_kwargs):
            return Scope()

        def wait_for_fresh(self, names, **_kwargs):
            return {name: snapshot(name) for name in names}

    positions = iter([[0.0], [0.1], [0.35]])
    report = recover_robots_to_sleep(
        robots={"leader_left": robot},
        health=Health({"leader_left": snapshot("leader_left")}),
        prepare_robot=lambda *_args: None,
        torque_off_robot=lambda *_args: pytest.fail(
            "out-of-tolerance pose must retain torque"
        ),
        read_positions=lambda _robot: next(positions),
        home_positions=[0.1],
        move_guarded=lambda **_kwargs: None,
        plan_duration=lambda *_args, **_kwargs: 4.0,
        max_error_rad=0.10,
    )

    result = report.results["leader_left"]
    assert result.status is SleepStatus.FAILED
    assert result.max_error_rad == 0.15
    assert not report.safe_to_stop


def test_large_torque_off_displacement_then_stability_is_verified():
    robot = robot_with_group(
        names=("waist",),
        sleep=(0.2,),
        lower=(-2.0,),
        upper=(2.0,),
    )

    class Scope:
        def __init__(self):
            import threading

            self.fault_event = threading.Event()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_if_faulted(self):
            return None

    class Health(SnapshotHealth):
        def arm_scope(self, *_args, **_kwargs):
            return Scope()

        def wait_for_fresh(self, names, **_kwargs):
            return {name: snapshot(name) for name in names}

    positions = iter(
        [
            [0.0],
            [0.1],
            [0.2],
            [0.2],
            [0.2],
            [0.55],
            [0.56],
            [0.55],
            [0.56],
        ]
    )
    report = recover_robots_to_sleep(
        robots={"leader_left": robot},
        health=Health({"leader_left": snapshot("leader_left")}),
        prepare_robot=lambda *_args: None,
        torque_off_robot=lambda *_args: None,
        read_positions=lambda _robot: next(positions),
        home_positions=[0.1],
        settle_sleep=lambda _seconds: None,
        move_guarded=lambda **_kwargs: None,
        plan_duration=lambda *_args, **_kwargs: 2.0,
        settle_error_rad=0.02,
    )

    result = report.results["leader_left"]
    assert result.status is SleepStatus.SLEPT_VERIFIED
    assert result.torque_off_verified
    assert result.max_error_rad == 0.01
    assert "target displacement=0.360 rad" in result.reason
    assert report.safe_to_stop


def test_continued_post_torque_motion_fails_after_one_window():
    robot = robot_with_group(
        names=("waist",),
        sleep=(0.2,),
        lower=(-2.0,),
        upper=(2.0,),
    )

    class Scope:
        def __init__(self):
            import threading

            self.fault_event = threading.Event()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_if_faulted(self):
            return None

    class Health(SnapshotHealth):
        def arm_scope(self, *_args, **_kwargs):
            return Scope()

        def wait_for_fresh(self, names, **_kwargs):
            return {name: snapshot(name) for name in names}

    clock_values = iter([0.0, 0.0, 0.25, 0.50, 0.75, 1.10])
    positions = iter(
        [
            [0.0],
            [0.1],
            [0.2],
            [0.2],
            [0.2],
            [0.35],
            [0.40],
            [0.45],
            [0.50],
        ]
    )
    report = recover_robots_to_sleep(
        robots={"leader_left": robot},
        health=Health({"leader_left": snapshot("leader_left")}),
        prepare_robot=lambda *_args: None,
        torque_off_robot=lambda *_args: None,
        read_positions=lambda _robot: next(positions),
        home_positions=[0.1],
        settle_sleep=lambda _seconds: None,
        move_guarded=lambda **_kwargs: None,
        plan_duration=lambda *_args, **_kwargs: 2.0,
        settle_error_rad=0.02,
        gate_timeout=1.0,
        verification_clock=lambda: next(clock_values),
        allow_pose_deviation=True,
    )

    result = report.results["leader_left"]
    assert result.status is SleepStatus.FAILED
    assert result.phase == "verify_settle"
    assert not result.torque_off_verified
    assert result.max_error_rad == 0.05
    assert "did not stabilize" in result.reason
    assert not report.safe_to_stop


def test_non_finite_post_torque_sample_fails_closed():
    robot = robot_with_group(
        names=("waist",),
        sleep=(0.2,),
        lower=(-2.0,),
        upper=(2.0,),
    )

    class Scope:
        def __init__(self):
            import threading

            self.fault_event = threading.Event()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_if_faulted(self):
            return None

    class Health(SnapshotHealth):
        def arm_scope(self, *_args, **_kwargs):
            return Scope()

        def wait_for_fresh(self, names, **_kwargs):
            return {name: snapshot(name) for name in names}

    positions = iter(
        [
            [0.0],
            [0.1],
            [0.2],
            [0.2],
            [0.2],
            [0.4],
            [float("nan")],
        ]
    )
    report = recover_robots_to_sleep(
        robots={"leader_left": robot},
        health=Health({"leader_left": snapshot("leader_left")}),
        prepare_robot=lambda *_args: None,
        torque_off_robot=lambda *_args: None,
        read_positions=lambda _robot: next(positions),
        home_positions=[0.1],
        settle_sleep=lambda _seconds: None,
        move_guarded=lambda **_kwargs: None,
        plan_duration=lambda *_args, **_kwargs: 2.0,
    )

    result = report.results["leader_left"]
    assert result.status is SleepStatus.FAILED
    assert result.phase == "verify_settle"
    assert not result.torque_off_verified
    assert "finite" in result.reason


def test_middle_pre_torque_sample_outside_tolerance_fails_closed():
    robot = robot_with_group(
        names=("waist",),
        sleep=(0.2,),
        lower=(-2.0,),
        upper=(2.0,),
    )

    class Scope:
        def __init__(self):
            import threading

            self.fault_event = threading.Event()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_if_faulted(self):
            return None

    class Health(SnapshotHealth):
        def arm_scope(self, *_args, **_kwargs):
            return Scope()

        def wait_for_fresh(self, names, **_kwargs):
            return {name: snapshot(name) for name in names}

    positions = iter([[0.0], [0.1], [0.2], [0.35], [0.2]])
    report = recover_robots_to_sleep(
        robots={"leader_left": robot},
        health=Health({"leader_left": snapshot("leader_left")}),
        prepare_robot=lambda *_args: None,
        torque_off_robot=lambda *_args: pytest.fail(
            "one bad sample must retain torque"
        ),
        read_positions=lambda _robot: next(positions),
        home_positions=[0.1],
        settle_sleep=lambda _seconds: None,
        move_guarded=lambda **_kwargs: None,
        plan_duration=lambda *_args, **_kwargs: 4.0,
        max_error_rad=0.10,
    )

    result = report.results["leader_left"]
    assert result.status is SleepStatus.FAILED
    assert result.phase == "verify_sleep_pose"
    assert result.max_error_rad == 0.15


def test_s_policy_records_finite_pose_deviation_then_verifies_torque_off():
    robot = robot_with_group(
        names=("waist",),
        sleep=(0.2,),
        lower=(-2.0,),
        upper=(2.0,),
    )

    class Scope:
        def __init__(self):
            import threading

            self.fault_event = threading.Event()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_if_faulted(self):
            return None

    class Health(SnapshotHealth):
        def arm_scope(self, *_args, **_kwargs):
            return Scope()

        def wait_for_fresh(self, names, **_kwargs):
            return {name: snapshot(name) for name in names}

    torque_off_calls = []
    positions = iter(
        [[0.0], [0.1], [0.2], [0.35], [0.35], [0.35], [0.35], [0.35]]
    )
    report = recover_robots_to_sleep(
        robots={"leader_left": robot},
        health=Health({"leader_left": snapshot("leader_left")}),
        prepare_robot=lambda *_args: None,
        torque_off_robot=lambda *args: torque_off_calls.append(args),
        read_positions=lambda _robot: next(positions),
        home_positions=[0.1],
        settle_sleep=lambda _seconds: None,
        move_guarded=lambda **_kwargs: None,
        plan_duration=lambda *_args, **_kwargs: 4.0,
        max_error_rad=0.10,
        allow_pose_deviation=True,
    )

    result = report.results["leader_left"]
    assert result.status is SleepStatus.SLEPT_VERIFIED
    assert result.torque_off_verified
    assert result.max_error_rad == pytest.approx(0.15)
    assert "pose deviation accepted for s exit" in result.reason
    assert len(torque_off_calls) == 1
    assert report.safe_to_stop


def test_non_finite_pose_sample_fails_closed():
    robot = robot_with_group(
        names=("waist",),
        sleep=(0.2,),
        lower=(-2.0,),
        upper=(2.0,),
    )

    class Scope:
        def __init__(self):
            import threading

            self.fault_event = threading.Event()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_if_faulted(self):
            return None

    class Health(SnapshotHealth):
        def arm_scope(self, *_args, **_kwargs):
            return Scope()

        def wait_for_fresh(self, names, **_kwargs):
            return {name: snapshot(name) for name in names}

    positions = iter([[0.0], [0.1], [float("nan")]])
    report = recover_robots_to_sleep(
        robots={"leader_left": robot},
        health=Health({"leader_left": snapshot("leader_left")}),
        prepare_robot=lambda *_args: None,
        torque_off_robot=lambda *_args: pytest.fail(
            "non-finite sample must retain torque"
        ),
        read_positions=lambda _robot: next(positions),
        home_positions=[0.1],
        settle_sleep=lambda _seconds: None,
        move_guarded=lambda **_kwargs: None,
        plan_duration=lambda *_args, **_kwargs: 4.0,
        allow_pose_deviation=True,
    )

    result = report.results["leader_left"]
    assert result.status is SleepStatus.FAILED
    assert result.phase == "verify_sleep_pose"
    assert "finite" in result.reason


def test_torque_off_failure_is_reported_and_not_verified():
    robot = robot_with_group(
        names=("waist",),
        sleep=(0.2,),
        lower=(-2.0,),
        upper=(2.0,),
    )

    class Scope:
        def __init__(self):
            import threading

            self.fault_event = threading.Event()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_if_faulted(self):
            return None

    class Health(SnapshotHealth):
        def arm_scope(self, *_args, **_kwargs):
            return Scope()

        def wait_for_fresh(self, names, **_kwargs):
            return {name: snapshot(name) for name in names}

    positions = iter(
        [[0.0], [0.1], [0.2], [0.2], [0.2]]
    )
    report = recover_robots_to_sleep(
        robots={"leader_left": robot},
        health=Health({"leader_left": snapshot("leader_left")}),
        prepare_robot=lambda *_args: None,
        torque_off_robot=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("torque service failed")
        ),
        read_positions=lambda _robot: next(positions),
        home_positions=[0.1],
        settle_sleep=lambda _seconds: None,
        move_guarded=lambda **_kwargs: None,
        plan_duration=lambda *_args, **_kwargs: 4.0,
        allow_pose_deviation=True,
    )

    result = report.results["leader_left"]
    assert result.status is SleepStatus.FAILED
    assert result.phase == "torque_off"
    assert not result.torque_off_verified
    assert "torque service failed" in result.reason


def test_gripper_restore_forwards_follower_profiles_with_one_timeout(
    monkeypatch,
):
    sleep_script = load_standalone_sleep(monkeypatch)
    robots = {
        "follower_left": object(),
        "follower_right": object(),
    }
    mode_calls = []
    profiles = {
        "follower_left": ("velocity", 50, 10),
        "follower_right": ("velocity", 0, 0),
    }

    def configure_follower(
        robot_name,
        robot,
        *,
        set_operating_modes,
    ):
        profile_type, velocity, acceleration = profiles[robot_name]
        set_operating_modes(
            robot,
            "single",
            "gripper",
            "current_based_position",
            profile_type=profile_type,
            profile_velocity=velocity,
            profile_acceleration=acceleration,
        )

    def restore_idle(
        robots,
        *,
        configure_follower_gripper,
        torque_enable,
        logger,
    ):
        del torque_enable, logger
        for robot_name, robot in robots.items():
            configure_follower_gripper(robot_name, robot)

    sleep_script._restore_post_session_grippers(
        robots,
        logger=lambda _message: None,
        configure_follower=configure_follower,
        restore_idle=restore_idle,
        set_modes=lambda *args, **kwargs: mode_calls.append(
            (args, kwargs)
        ),
    )

    assert mode_calls == [
        (
            (
                robots["follower_left"],
                "single",
                "gripper",
                "current_based_position",
            ),
            {
                "timeout_sec": sleep_script.SERVICE_TIMEOUT_SECONDS,
                "profile_type": "velocity",
                "profile_velocity": 50,
                "profile_acceleration": 10,
            },
        ),
        (
            (
                robots["follower_right"],
                "single",
                "gripper",
                "current_based_position",
            ),
            {
                "timeout_sec": sleep_script.SERVICE_TIMEOUT_SECONDS,
                "profile_type": "velocity",
                "profile_velocity": 0,
                "profile_acceleration": 0,
            },
        ),
    ]


def test_gripper_restore_uses_real_idle_policy_and_torque_order(
    monkeypatch,
):
    sleep_script = load_standalone_sleep(monkeypatch)
    robots = {
        "leader_left": object(),
        "follower_left": object(),
        "leader_right": object(),
        "follower_right": object(),
    }
    events = []

    def configure_follower(
        robot_name,
        _robot,
        *,
        set_operating_modes,
    ):
        del set_operating_modes
        events.append(("configure", robot_name))

    sleep_script._restore_post_session_grippers(
        robots,
        logger=lambda message: events.append(("log", message)),
        configure_follower=configure_follower,
        set_torque=lambda robot, command_type, name, enabled, **kwargs: (
            events.append(
                (
                    "torque",
                    robot,
                    command_type,
                    name,
                    enabled,
                    kwargs,
                )
            )
        ),
    )

    timeout = {"timeout_sec": sleep_script.SERVICE_TIMEOUT_SECONDS}
    assert events == [
        (
            "torque",
            robots["leader_left"],
            "single",
            "gripper",
            False,
            timeout,
        ),
        ("configure", "follower_left"),
        (
            "torque",
            robots["follower_left"],
            "single",
            "gripper",
            True,
            timeout,
        ),
        (
            "torque",
            robots["leader_right"],
            "single",
            "gripper",
            False,
            timeout,
        ),
        ("configure", "follower_right"),
        (
            "torque",
            robots["follower_right"],
            "single",
            "gripper",
            True,
            timeout,
        ),
    ]


@pytest.mark.parametrize("failure", ["mode", "torque"])
def test_gripper_restore_failure_logs_before_safe_publish(
    monkeypatch,
    failure,
):
    sleep_script = load_standalone_sleep(monkeypatch)
    events = []
    report = object()
    robots = {
        "follower_left" if failure == "mode" else "leader_left": (
            object()
        )
    }

    def set_modes(*_args, **_kwargs):
        events.append("restore")
        raise RuntimeError("mode unavailable")

    def set_torque(*_args, **_kwargs):
        events.append("restore")
        raise RuntimeError("torque unavailable")

    def restore_grippers(robots, logger):
        sleep_script._restore_post_session_grippers(
            robots,
            logger,
            set_modes=set_modes,
            set_torque=set_torque,
        )

    sleep_script._publish_safe_recovery(
        report,
        robots,
        lambda message: events.append(("log", message)),
        lambda state, *, report: events.append(
            ("publish", state, report)
        ),
        restore_grippers=restore_grippers,
    )

    assert events[0] == "restore"
    assert events[1][0] == "log"
    assert failure in events[1][1]
    assert events[2] == ("publish", "SAFE_TO_STOP", report)


def test_manual_sleep_uses_isolated_recovery_and_fails_closed():
    source = SLEEP_SCRIPT.read_text(encoding="utf-8")

    assert "initialize_ros_context(" in source
    assert "SignalHandlerOptions.NO" in source
    assert "RecoveryLease.acquire(" in source
    assert "RecoveryIdentity(" in source
    assert "initialize_robots_independently(" in source
    assert "merge_initialization_failures(" in source
    assert "retry_failed_initializations(" not in source
    assert "wait_for_safety_retry(" not in source
    assert "gravity_compensation_active=" in source
    assert '"safe_sleep_positions"' in source
    assert "torque_off_robot=" in source
    assert "_restore_post_session_grippers(" in source
    publish_helper = source[
        source.index("def _publish_safe_recovery("):
        source.index("def _expected_joint_state_names(")
    ]
    restore_idle = publish_helper.index("restore_grippers(robots, logger)")
    safe_publish = publish_helper.index(
        'publish_state("SAFE_TO_STOP"'
    )
    assert restore_idle < safe_publish
    safe_handoff = source.index(
        "_publish_safe_recovery(",
        source.index("def main("),
    )
    owner_grace = source.index(
        "SAFE_STATE_OWNER_GRACE_SECONDS",
        safe_handoff,
    )
    shutdown = source.index("robot_shutdown(node)", owner_grace)
    assert safe_handoff < owner_grace < shutdown
    assert "recover_robots_to_sleep" in source
    assert "输入 s 回车" not in source
    assert "等待已核验的 SIGUSR1 " not in source
    assert "失败臂不重试" in source
    assert '"safe-sleep 退出。"' in source
    assert "sleep_arms" not in source
    assert "for robot_name, result in report.results.items()" in source
    assert "if not report.safe_to_stop:" in source
    assert "UNSAFE_HOLD" in source
    assert "robot_shutdown" in source
