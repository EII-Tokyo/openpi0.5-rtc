import importlib.util
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from aloha.episode_attempt import AttemptDecision
from aloha.external_recovery import ExternalRecoveryError
from aloha.safe_motion import GuardedMotionAborted
from aloha.safe_sleep import (
    RobotSleepResult,
    SafeSleepReport,
    SleepStatus,
)


ROOT = Path(__file__).resolve().parents[1]
RECORDER = ROOT / "scripts" / "record_episodes_copy.py"


def load_recorder():
    spec = importlib.util.spec_from_file_location(
        "record_episodes_current_pose_test",
        RECORDER,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeHealthScope:
    def __init__(self):
        self.fault_event = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def raise_if_faulted(self):
        if self.fault_event.is_set():
            raise RuntimeError("faulted")


class FakeOpeningHealth:
    def arm_scope(self, *_args, **_kwargs):
        return FakeHealthScope()


def fake_opening_robot(name, positions=(0.0,) * 6):
    return SimpleNamespace(
        core=SimpleNamespace(robot_name=name),
        arm=SimpleNamespace(
            get_joint_positions=lambda: list(positions),
        ),
    )


def test_opening_requires_fresh_samples_before_mode_torque_and_motion(
    monkeypatch,
):
    recorder = load_recorder()
    calls = []
    leader = fake_opening_robot("leader_left")
    follower = fake_opening_robot("follower_left")

    monkeypatch.setattr(
        recorder,
        "_wait_for_health_gate",
        lambda *_args, phase, **_kwargs: calls.append(f"fresh:{phase}"),
    )
    monkeypatch.setattr(
        recorder,
        "_configure_opening_pair_modes",
        lambda *_args, **_kwargs: calls.append("set_mode"),
    )
    monkeypatch.setattr(
        recorder,
        "_torque_on_opening_pair",
        lambda *_args, **_kwargs: calls.append("torque_on"),
    )
    monkeypatch.setattr(
        recorder,
        "move_robots_guarded",
        lambda **_kwargs: calls.append("guarded_home"),
    )
    monkeypatch.setattr(recorder, "move_grippers", lambda *_args, **_kwargs: None)

    recorder._prepare_opening_pair(
        suffix="left",
        leader_bot=leader,
        follower_bot=follower,
        health=FakeOpeningHealth(),
        dt=0.02,
        start_arm_qpos=[0.0] * 6,
        leader_gripper_qpos=0.1,
        follower_gripper_qpos=0.2,
        continuous_roll_joints=True,
        opening_home_min_seconds=4.0,
        opening_max_joint_speed=0.4,
        joint_state_idle_timeout=0.75,
        joint_state_moving_timeout=0.30,
        stop_requested=lambda: False,
    )

    assert calls == [
        "fresh:pre_mode:left",
        "set_mode",
        "fresh:post_mode:left",
        "torque_on",
        "fresh:post_torque:left",
        "guarded_home",
    ]


def test_opening_pairs_run_concurrently_and_report_after_both_finish(
    monkeypatch,
):
    recorder = load_recorder()
    entered = set()
    completed = []
    entered_lock = threading.Lock()
    both_entered = threading.Event()

    def prepare_pair(*, suffix, **_kwargs):
        with entered_lock:
            entered.add(suffix)
            if entered == {"left", "right"}:
                both_entered.set()
        assert both_entered.wait(0.5), "opening pairs did not overlap"
        if suffix == "left":
            raise GuardedMotionAborted("injected left fault")
        completed.append(suffix)

    monkeypatch.setattr(recorder, "_prepare_opening_pair", prepare_pair)

    with pytest.raises(RuntimeError, match=r"opening pair.*left"):
        recorder._prepare_opening_pairs(
            pairs=[
                (
                    "left",
                    fake_opening_robot("leader_left"),
                    fake_opening_robot("follower_left"),
                ),
                (
                    "right",
                    fake_opening_robot("leader_right"),
                    fake_opening_robot("follower_right"),
                ),
            ],
            health=FakeOpeningHealth(),
            dt=0.02,
            selected_start_pose={
                "left_arm": [0.0] * 6,
                "right_arm": [0.0] * 6,
                "left_gripper": 0.5,
                "right_gripper": 0.5,
            },
            continuous_roll_joints=True,
            opening_home_min_seconds=4.0,
            opening_max_joint_speed=0.4,
            joint_state_idle_timeout=0.75,
            joint_state_moving_timeout=0.30,
            stop_requested=lambda: False,
        )

    assert entered == {"left", "right"}
    assert completed == ["right"]


def test_opening_defaults_use_one_second_floor_and_speed_limit():
    recorder = load_recorder()

    assert recorder._OPENING_HOME_MIN_SECONDS == 1.0
    assert recorder._OPENING_MAX_JOINT_SPEED == 0.4


def test_return_home_duration_uses_slowest_leader_at_speed_limit():
    recorder = load_recorder()
    left = object()
    right = object()
    positions = {
        left: [0.0] * 6,
        right: [0.0] * 6,
    }
    recorder.get_arm_joint_positions = lambda robot: positions[robot]

    duration = recorder._plan_synchronized_return_duration(
        {"leader_left": left, "leader_right": right},
        {
            "left_arm": [0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
            "right_arm": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    )

    assert duration == pytest.approx(2.5)


def test_return_home_duration_requires_both_leaders():
    recorder = load_recorder()

    with pytest.raises(ValueError, match="leader_left.*leader_right"):
        recorder._plan_synchronized_return_duration(
            {"leader_left": object()},
            {
                "left_arm": [0.0] * 6,
                "right_arm": [0.0] * 6,
            },
        )


def test_verify_acquisition_home_arrival_checks_all_four_arms():
    recorder = load_recorder()
    robots = {
        "leader_left": object(),
        "leader_right": object(),
        "follower_left": object(),
        "follower_right": object(),
    }
    left = [0.0, -0.96, 1.16, 1.57, 0.0, -1.57]
    right = [0.0, -0.96, 1.16, 0.0, 0.0, 0.0]
    positions = {
        robots["leader_left"]: left,
        robots["follower_left"]: [value + 0.01 for value in left],
        robots["leader_right"]: right,
        robots["follower_right"]: [value - 0.02 for value in right],
    }

    errors = recorder.verify_acquisition_home_arrival(
        robots,
        {"left_arm": left, "right_arm": right},
        read_positions=lambda robot: positions[robot],
        tolerance=0.10,
    )

    assert set(errors) == set(robots)
    assert errors["follower_right"] == pytest.approx(0.02)


def test_verify_acquisition_home_arrival_names_failed_robot():
    recorder = load_recorder()
    robots = {
        "leader_left": object(),
        "leader_right": object(),
        "follower_left": object(),
        "follower_right": object(),
    }
    positions = {robot: [0.0] * 6 for robot in robots.values()}
    positions[robots["follower_right"]] = [0.11] + [0.0] * 5

    with pytest.raises(
        RuntimeError,
        match=r"follower_right.*0\.110.*0\.100",
    ):
        recorder.verify_acquisition_home_arrival(
            robots,
            {"left_arm": [0.0] * 6, "right_arm": [0.0] * 6},
            read_positions=lambda robot: positions[robot],
            tolerance=0.10,
        )


def test_lock_robots_commands_side_specific_home_for_all_four_arms():
    recorder = load_recorder()
    robots = {
        "leader_left": object(),
        "leader_right": object(),
        "follower_left": object(),
        "follower_right": object(),
    }
    left = [1.0] * 6
    right = [2.0] * 6
    calls = []

    recorder.lock_robots_at_acquisition_home(
        robots,
        {"left_arm": left, "right_arm": right},
        set_modes=lambda robot, *args: calls.append(
            ("mode", robot, args)
        ),
        torque_enable=lambda robot, *args: calls.append(
            ("torque", robot, args)
        ),
        command_positions=lambda robot, target: calls.append(
            ("command", robot, tuple(target))
        ),
    )

    for name, robot in robots.items():
        target = left if name.endswith("left") else right
        assert ("mode", robot, ("group", "arm", "position")) in calls
        assert ("torque", robot, ("group", "arm", True)) in calls
        assert ("command", robot, tuple(target)) in calls


def test_saved_b_return_stops_diagnostics_asynchronously_and_records_tail():
    source = RECORDER.read_text(encoding="utf-8")
    capture = source.split("def capture_one_episode(", 1)[1].split(
        "def get_auto_index(",
        1,
    )[0]
    return_path = capture.split("def return_attempt_to_start(", 1)[1].split(
        "def discard_attempt(",
        1,
    )[0]

    request = return_path.index("request_diagnostic_stop(attempt)")
    motion = return_path.index("_return_to_start_position(")
    join = return_path.index("stop_attempt_diagnostics(attempt)", motion)
    assert request < motion < join
    assert "return_dt_history" in return_path
    assert "attempt.dt_history.extend(return_dt_history)" in return_path

    motion_source = source.split("def _return_to_start_position(", 1)[1].split(
        "def _wait_for_health_gate(",
        1,
    )[0]
    assert "verify_acquisition_home_arrival(" in motion_source
    assert "lock_robots_at_acquisition_home(" in motion_source
    assert "_RETURN_HOME_STABLE_SAMPLES" in motion_source


def test_opening_gate_rejects_a_previously_latched_health_fault():
    recorder = load_recorder()
    health = SimpleNamespace(
        fault_event=SimpleNamespace(is_set=lambda: True),
        first_fault=SimpleNamespace(
            robot_name="leader_left",
            phase="joint_state_callback",
            reason="invalid_all_minus_pi",
        ),
        wait_for_fresh=lambda *_args, **_kwargs: pytest.fail(
            "latched fault must fail before waiting"
        ),
    )

    with pytest.raises(
        recorder.RobotHealthUnavailable,
        match="invalid_all_minus_pi",
    ):
        recorder._wait_for_health_gate(
            health,
            {"leader_left"},
            phase="pre_mode:left",
            max_age=0.75,
            stop_requested=lambda: False,
        )


def test_health_expected_names_come_from_initialized_robot_interfaces():
    recorder = load_recorder()
    robot = SimpleNamespace(
        core=SimpleNamespace(robot_name="leader_left"),
        arm=SimpleNamespace(
            group_info=SimpleNamespace(
                joint_names=["waist", "shoulder"],
            )
        ),
        gripper=SimpleNamespace(
            gripper_info=SimpleNamespace(
                joint_names=["left_finger"],
            )
        ),
    )

    assert recorder._expected_joint_state_names(robot) == {
        "waist",
        "shoulder",
        "left_finger",
    }


def sleep_report(*, safe):
    status = (
        SleepStatus.SLEPT_VERIFIED
        if safe
        else SleepStatus.UNRESPONSIVE
    )
    return SafeSleepReport(
        results={
            "leader_left": RobotSleepResult(
                robot_name="leader_left",
                status=status,
                max_error_rad=0.01 if safe else None,
                reason="test",
                phase="complete" if safe else "health_gate",
                torque_off_verified=safe,
            )
        }
    )


def fake_finalizer_runtime():
    class Health:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    return SimpleNamespace(
        env=SimpleNamespace(robots={"leader_left": object()}),
        health=Health(),
        camera_runtime=None,
        dt=0.02,
    )


def test_runtime_creation_injects_isolated_image_recorder(monkeypatch):
    recorder = load_recorder()
    calls = []
    node = SimpleNamespace(context=object())
    camera = SimpleNamespace(
        image_recorder=object(),
        close=lambda: calls.append("camera_close"),
    )
    robot = SimpleNamespace(
        core=SimpleNamespace(robot_name="leader_left"),
        arm=SimpleNamespace(
            group_info=SimpleNamespace(joint_names=["waist"]),
        ),
        gripper=SimpleNamespace(
            gripper_info=SimpleNamespace(joint_names=["gripper"]),
        ),
    )
    env = SimpleNamespace(robots={"leader_left": robot})

    class Health:
        fault_event = threading.Event()
        first_fault = None

        def start(self):
            calls.append("health_start")

        def stop(self):
            calls.append("health_stop")

    monkeypatch.setattr(
        recorder,
        "create_interbotix_global_node",
        lambda name: calls.append(("create_node", name)) or node,
    )
    monkeypatch.setattr(
        recorder,
        "CameraRuntime",
        SimpleNamespace(
            create=lambda **kwargs: (
                calls.append(("camera_create", kwargs)) or camera
            ),
        ),
        raising=False,
    )
    captured = {}
    monkeypatch.setattr(
        recorder,
        "make_real_env",
        lambda **kwargs: captured.update(kwargs) or env,
    )
    monkeypatch.setattr(
        recorder,
        "robot_startup",
        lambda value: calls.append(("robot_startup", value)),
    )
    monkeypatch.setattr(
        recorder,
        "RobotHealthMonitor",
        lambda **_kwargs: Health(),
    )
    monkeypatch.setattr(
        recorder,
        "attach_joint_state_subscriptions",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        recorder,
        "publish_safety_state",
        lambda *_args, **_kwargs: None,
    )

    runtime = recorder.create_recorder_runtime(
        config={"fps": 50, "base": False},
        torque_base=False,
        continuous_roll_joints=True,
    )

    assert captured["image_recorder"] is camera.image_recorder
    assert runtime.camera_runtime is camera
    assert ("camera_create", {
        "config": {"fps": 50, "base": False},
        "context": node.context,
    }) in calls


def test_prepare_episode_start_moves_home_once_then_rearms_current_pose(
    monkeypatch,
):
    recorder = load_recorder()
    monkeypatch.setattr(
        recorder,
        "_wait_for_health_gate",
        lambda *_args, **_kwargs: None,
    )
    runtime = recorder.RecorderRuntime(
        node=object(),
        env=object(),
        health=object(),
        health_subscriptions=[],
    )
    calls = []

    assert recorder.prepare_episode_start(
        runtime,
        robots={},
        gravity_compensation=False,
        dt=0.02,
        start_arm_pose={},
        continuous_roll_joints=True,
        return_home_between_episodes=False,
        opening=lambda **_kwargs: calls.append("home"),
        rearm=lambda **_kwargs: calls.append("rearm") or True,
    )
    assert recorder.prepare_episode_start(
        runtime,
        robots={},
        gravity_compensation=False,
        dt=0.02,
        start_arm_pose={},
        continuous_roll_joints=True,
        return_home_between_episodes=False,
        opening=lambda **_kwargs: calls.append("home"),
        rearm=lambda **_kwargs: calls.append("rearm") or True,
    )

    assert calls == ["home", "rearm"]
    assert runtime.home_initialized


def test_current_pose_rearm_requires_post_pause_joint_states(monkeypatch):
    recorder = load_recorder()
    robot_names = {
        "leader_left",
        "leader_right",
        "follower_left",
        "follower_right",
    }
    runtime = recorder.RecorderRuntime(
        node=object(),
        env=SimpleNamespace(
            robots={name: object() for name in robot_names},
        ),
        health=object(),
        health_subscriptions=[],
        home_initialized=True,
    )
    calls = []
    monkeypatch.setattr(
        recorder,
        "_wait_for_health_gate",
        lambda health, names, **kwargs: calls.append(
            (
                "fresh",
                health,
                frozenset(names),
                kwargs["phase"],
                kwargs["max_age"],
            )
        ),
    )

    def rearm(**kwargs):
        calls.append(("restore",))
        kwargs["post_restore_health_gate"]()
        return True

    assert recorder.prepare_episode_start(
        runtime,
        robots=runtime.env.robots,
        gravity_compensation=False,
        dt=0.02,
        start_arm_pose={},
        continuous_roll_joints=True,
        return_home_between_episodes=False,
        rearm=rearm,
    )

    assert calls == [
        (
            "fresh",
            runtime.health,
            frozenset(robot_names),
            "current_pose_rearm",
            recorder._TELEOP_LEADER_MAX_AGE_SECONDS,
        ),
        ("restore",),
        (
            "fresh",
            runtime.health,
            frozenset(robot_names),
            "current_pose_rearm_post_restore",
            recorder._TELEOP_LEADER_MAX_AGE_SECONDS,
        ),
    ]


def test_recorder_source_wires_post_restore_health_gate():
    source = RECORDER.read_text(encoding="utf-8")
    prepare = source.split("def prepare_episode_start(", 1)[1].split(
        "def _require_fresh_leaders(",
        1,
    )[0]

    assert "post_restore_health_gate=lambda: _wait_for_health_gate(" in prepare
    assert 'phase="current_pose_rearm_post_restore"' in prepare
    assert "set(robots)" in prepare


def test_require_fresh_leaders_uses_initialized_leader_interfaces():
    recorder = load_recorder()
    calls = []
    health = SimpleNamespace(
        require_fresh=lambda names, **kwargs: calls.append(
            (frozenset(names), kwargs)
        ),
    )
    runtime = recorder.RecorderRuntime(
        node=object(),
        env=SimpleNamespace(
            robots={
                "leader_left": object(),
                "leader_right": object(),
                "follower_left": object(),
            },
        ),
        health=health,
        health_subscriptions=[],
    )

    recorder._require_fresh_leaders(
        runtime,
        phase="episode_collection",
    )

    assert calls == [
        (
            frozenset({"leader_left", "leader_right"}),
            {
                "max_age": recorder._TELEOP_LEADER_MAX_AGE_SECONDS,
                "phase": "episode_collection",
            },
        )
    ]


def test_wait_and_collection_loops_use_guarded_fresh_actions():
    source = RECORDER.read_text(encoding="utf-8")
    capture = source.split("def capture_one_episode(", 1)[1].split(
        "def get_auto_index(",
        1,
    )[0]
    wait_loop = capture.split(
        "def wait_for_attempt_start(",
        1,
    )[1].split("def stop_attempt_diagnostics(", 1)[0]
    collection_loop = capture.split(
        "def collect_attempt(",
        1,
    )[1].split("def return_attempt_to_start(", 1)[0]

    assert "guarded_teleop_step(" in wait_loop
    assert 'phase="teleop_wait"' in wait_loop
    assert "guarded_teleop_step(" in collection_loop
    assert 'phase="episode_collection"' in collection_loop


def test_compatibility_flag_returns_home_between_saved_episodes():
    recorder = load_recorder()
    runtime = recorder.RecorderRuntime(
        node=object(),
        env=object(),
        health=object(),
        health_subscriptions=[],
        home_initialized=True,
    )
    calls = []

    recorder.prepare_episode_start(
        runtime,
        robots={},
        gravity_compensation=False,
        dt=0.02,
        start_arm_pose={},
        continuous_roll_joints=True,
        return_home_between_episodes=True,
        opening=lambda **_kwargs: calls.append("home"),
        rearm=lambda **_kwargs: calls.append("rearm") or True,
    )

    assert calls == ["home"]


def test_successful_save_holds_current_pose_by_default_but_discard_returns_home():
    recorder = load_recorder()

    assert not recorder.should_return_attempt_to_home(
        AttemptDecision.SAVE,
        return_home_between_episodes=False,
    )
    assert recorder.should_return_attempt_to_home(
        AttemptDecision.SAVE,
        return_home_between_episodes=True,
    )
    assert recorder.should_return_attempt_to_home(
        AttemptDecision.DISCARD,
        return_home_between_episodes=False,
    )


def test_save_handoff_queues_owned_payload_before_trigger_reset():
    recorder = load_recorder()
    calls = []
    queued = []

    class Worker:
        def submit(self, job):
            calls.append("submit")
            queued.append(job)

    class Trigger:
        def complete_save_handoff(self):
            calls.append("complete_handoff")
            return True

    payload = SimpleNamespace(dataset_name="episode_9")
    recorder.handoff_episode_save(
        Worker(),
        payload,
        Trigger(),
    )

    assert calls == ["submit", "complete_handoff"]
    assert queued[0].name == "episode_9"
    assert queued[0].payload is payload


def test_leader_hold_adapter_publishes_current_pose_before_arm_torque():
    recorder = load_recorder()
    calls = []

    class Arm:
        def set_joint_positions(self, positions, *, blocking):
            calls.append(("command", tuple(positions), blocking))
            return True

    leader = SimpleNamespace(arm=Arm())
    recorder.get_arm_joint_positions = lambda robot: (
        calls.append(("read", robot)) or [0.1, 0.2]
    )
    recorder._set_operating_modes_bounded = (
        lambda robot, cmd_type, name, mode: calls.append(
            ("mode", robot, cmd_type, name, mode)
        )
    )
    recorder._torque_enable_bounded = (
        lambda robot, cmd_type, name, enable: calls.append(
            ("torque", robot, cmd_type, name, enable)
        )
    )

    recorder.hold_leaders_for_current_pose_rearm(
        {"leader_left": leader},
        gravity_compensation=False,
    )

    assert calls == [
        ("read", leader),
        ("mode", leader, "group", "arm", "position"),
        ("command", (0.1, 0.2), False),
        ("torque", leader, "group", "arm", True),
        ("torque", leader, "single", "gripper", False),
    ]


def test_default_save_stop_holds_leaders_before_stopping_diagnostics():
    recorder = load_recorder()
    calls = []
    attempt = object()

    prepared = recorder.prepare_current_pose_save_stop(
        attempt,
        robots={"leader_left": object()},
        gravity_compensation=False,
        hold_leaders=lambda _robots, _gravity: calls.append("hold"),
        stop_diagnostics=lambda seen_attempt: calls.append(
            ("stop_diagnostics", seen_attempt)
        ),
        force_no_save=lambda reason: calls.append(("force_no_save", reason)),
        logger=lambda message: calls.append(("log", message)),
    )

    assert prepared
    assert calls[0] == "hold"
    assert calls[1] == ("stop_diagnostics", attempt)
    assert "保持当前位置" in calls[2][1]


def test_strict_hold_policy_fails_closed():
    recorder = load_recorder()
    calls = []

    def fail_hold(_robots, _gravity):
        raise RuntimeError("injected hold failure")

    prepared = recorder.prepare_current_pose_save_stop(
        object(),
        robots={"leader_left": object()},
        gravity_compensation=False,
        leader_hold_policy="strict",
        hold_leaders=fail_hold,
        stop_diagnostics=lambda _attempt: calls.append("stop_diagnostics"),
        force_no_save=lambda reason: calls.append(("force_no_save", reason)),
        logger=lambda message: calls.append(("log", message)),
    )

    assert not prepared
    assert "stop_diagnostics" not in calls
    assert ("force_no_save", "leader stop-pose hold failed") in calls
    assert any(
        item[0] == "log"
        and "strict" in item[1]
        and "injected hold failure" in item[1]
        for item in calls
    )


def test_best_effort_hold_policy_saves_after_hold_failure():
    recorder = load_recorder()
    calls = []
    attempt = object()

    def fail_hold(_robots, _gravity):
        raise RuntimeError("injected hold failure")

    prepared = recorder.prepare_current_pose_save_stop(
        attempt,
        robots={"leader_left": object()},
        gravity_compensation=False,
        leader_hold_policy="best-effort",
        hold_leaders=fail_hold,
        stop_diagnostics=lambda seen: calls.append(
            ("stop_diagnostics", seen)
        ),
        force_no_save=lambda reason: calls.append(("force_no_save", reason)),
        logger=lambda message: calls.append(("log", message)),
    )

    assert prepared
    assert ("stop_diagnostics", attempt) in calls
    assert not any(
        item[0] == "force_no_save"
        for item in calls
        if isinstance(item, tuple)
    )
    assert any(
        item[0] == "log"
        and "仍将保存" in item[1]
        and "未被机械锁定" in item[1]
        for item in calls
    )


def test_off_hold_policy_skips_hold_and_saves():
    recorder = load_recorder()
    calls = []
    attempt = object()

    prepared = recorder.prepare_current_pose_save_stop(
        attempt,
        robots={"leader_left": object()},
        gravity_compensation=False,
        leader_hold_policy="off",
        hold_leaders=lambda *_args: calls.append("hold"),
        stop_diagnostics=lambda seen: calls.append(
            ("stop_diagnostics", seen)
        ),
        force_no_save=lambda reason: calls.append(("force_no_save", reason)),
        logger=lambda message: calls.append(("log", message)),
    )

    assert prepared
    assert "hold" not in calls
    assert ("stop_diagnostics", attempt) in calls
    assert not any(
        item[0] == "force_no_save"
        for item in calls
        if isinstance(item, tuple)
    )
    assert any(
        item[0] == "log"
        and "off" in item[1]
        and "仍将保存" in item[1]
        for item in calls
    )


@pytest.mark.parametrize("policy", ["", "unknown", None])
def test_current_pose_save_stop_rejects_invalid_hold_policy(policy):
    recorder = load_recorder()

    with pytest.raises(ValueError, match="leader_hold_policy"):
        recorder.prepare_current_pose_save_stop(
            object(),
            robots={},
            gravity_compensation=False,
            leader_hold_policy=policy,
            force_no_save=lambda _reason: None,
        )


def test_default_b_stop_message_describes_current_pose_hold(capsys):
    recorder = load_recorder()
    recorder._RETURN_HOME_BETWEEN_EPISODES = False
    recorder._COMMAND_COORDINATOR = SimpleNamespace(
        handle_b=lambda: recorder.TriggerResult.STOPPED
    )

    recorder._handle_b_trigger("foot-pedal")

    output = capsys.readouterr().out
    assert "保持当前位置" in output
    assert "回到初始位置并保存" not in output


def test_r_message_matches_fail_closed_standalone_sleep_policy(capsys):
    recorder = load_recorder()
    recorder._COMMAND_COORDINATOR = SimpleNamespace(
        request_save=lambda **_kwargs: True
    )

    recorder._handle_r_trigger()

    output = capsys.readouterr().out
    assert "独立 safe-sleep" in output
    assert "不回 sleep" not in output
    assert "不 sleep" not in output


@pytest.mark.parametrize(
    ("result_name", "expected"),
    [
        ("NOT_READY", "尚未准备完成"),
        ("NO_SAMPLES", "首个数据时间步"),
    ],
)
def test_b_handler_explains_nonfatal_trigger_rejection(
    result_name,
    expected,
    capsys,
):
    recorder = load_recorder()
    recorder._COMMAND_COORDINATOR = SimpleNamespace(
        handle_b=lambda: getattr(recorder.TriggerResult, result_name)
    )

    recorder._handle_b_trigger("foot-pedal")

    assert expected in capsys.readouterr().out


def test_recorder_wires_leader_hold_policy_option():
    source = RECORDER.read_text(encoding="utf-8")

    assert '"--leader-hold-policy"' in source
    assert 'choices=("strict", "best-effort", "off")' in source
    assert 'args.get("leader_hold_policy", "best-effort")' in source
    assert "leader_hold_policy=leader_hold_policy" in source


def test_recorder_wires_positive_health_and_opening_motion_options():
    source = RECORDER.read_text(encoding="utf-8")

    for option in (
        "--opening-home-min-seconds",
        "--opening-max-joint-speed",
        "--joint-state-moving-timeout",
        "--joint-state-idle-timeout",
        "--health-watchdog-rate-hz",
    ):
        assert f'"{option}"' in source
    assert "opening_home_min_seconds must be at least 1.0" in source
    assert "opening_max_joint_speed must be positive" in source
    assert "joint_state_moving_timeout must be positive" in source
    assert "joint_state_idle_timeout must be positive" in source
    assert "health_watchdog_rate_hz must be positive" in source


class FakeRecoveryLease:
    def __init__(self, calls, recovery_id):
        self.calls = calls
        self.metadata = SimpleNamespace(
            recovery_id=recovery_id,
            owner_pid=4321,
            source="recorder",
        )

    def release(self):
        self.calls.append(("lease_release", self.metadata.recovery_id))


def external_finalizer_fakes(
    recorder,
    calls,
    *,
    drain_error=None,
    allow_pose_deviation=False,
):
    class Worker:
        def drain(self, *, timeout=None):
            calls.append("save_drain")
            calls.append(("drain_timeout", timeout))
            if drain_error is not None:
                raise drain_error

        def abort(self, *, timeout=None):
            calls.append(("save_abort", timeout))

    runtime = recorder.RecorderRuntime(
        node=object(),
        env=SimpleNamespace(robots={}),
        health=SimpleNamespace(stop=lambda: calls.append("health_stop")),
        health_subscriptions=[object()],
        camera_runtime=SimpleNamespace(close=lambda: calls.append("camera_close")),
    )
    controller = SimpleNamespace(
        allow_pose_deviation=allow_pose_deviation,
        begin_cleanup=lambda: calls.append("begin_cleanup"),
        enter_unsafe_hold=lambda: calls.append("enter_unsafe"),
        leave_unsafe_hold=lambda: calls.append("leave_unsafe"),
        wait_for_safety_retry=lambda timeout: True,
    )
    return runtime, Worker(), controller


@pytest.mark.parametrize(
    "outcome_name",
    [
        "EXIT_SAVE_AND_SLEEP",
        "EXIT_SAVE_WITHOUT_SLEEP",
    ],
)
def test_external_finalizer_drains_quiesces_then_supervises(
    monkeypatch, outcome_name
):
    recorder = load_recorder()
    calls = []
    runtime, worker, controller = external_finalizer_fakes(recorder, calls)
    expected = sleep_report(safe=True)
    monkeypatch.setattr(recorder, "_SAFE_STOP_CONTROLLER", controller)
    monkeypatch.setattr(
        recorder,
        "robot_shutdown",
        lambda node: calls.append(("robot_shutdown", node)),
    )

    report = recorder.finalize_recorder_runtime(
        runtime,
        outcome=getattr(recorder.SessionOutcome, outcome_name),
        save_worker=worker,
        robot_name="aloha_stationary",
        supervise_recovery=lambda **_kwargs: (
            calls.append("external_recovery") or expected
        ),
        publish_state=lambda *_args, **_kwargs: None,
        lease_factory=lambda **kwargs: FakeRecoveryLease(
            calls, kwargs["recovery_id"]
        ),
    )

    assert report is expected
    assert calls.index("save_drain") < calls.index("camera_close")
    assert calls.index(("robot_shutdown", runtime.node)) < calls.index(
        "external_recovery"
    )


def test_finalizer_logs_authoritative_save_handoff_and_safe_transitions(
    monkeypatch,
):
    recorder = load_recorder()
    calls = []
    messages = []
    runtime, worker, controller = external_finalizer_fakes(recorder, calls)
    runtime.last_saved_episode_name = "episode_42"
    runtime.terminal_save_source = "m"
    monkeypatch.setattr(recorder, "_SAFE_STOP_CONTROLLER", controller)
    monkeypatch.setattr(
        recorder,
        "robot_shutdown",
        lambda _node: calls.append("robot_shutdown"),
    )

    recorder.finalize_recorder_runtime(
        runtime,
        outcome=recorder.SessionOutcome.EXIT_SAVE_AND_SLEEP,
        save_worker=worker,
        robot_name="aloha_stationary",
        supervise_recovery=lambda **_kwargs: (
            calls.append("external_recovery") or sleep_report(safe=True)
        ),
        publish_state=lambda *_args, **_kwargs: None,
        lease_factory=lambda **kwargs: FakeRecoveryLease(
            calls, kwargs["recovery_id"]
        ),
        logger=messages.append,
    )

    assert messages[0] == (
        "[m] episode_42 已保存完成，开始独立 safe-sleep。"
    )
    assert "[handoff] recorder ROS runtime 已关闭。" in messages
    assert any(
        "leader_left" in message
        and "slept_verified" in message
        and "torque_off_verified=true" in message
        for message in messages
    )
    assert messages[-1] == "[SAFE_TO_STOP] 四臂均已归位并验证扭矩关闭。"
    assert calls.index("save_drain") < calls.index("camera_close")
    assert calls.index("robot_shutdown") < calls.index("external_recovery")


@pytest.mark.parametrize(
    (
        "outcome_name",
        "controller_policy",
        "expected_policy",
        "expected_message",
    ),
    [
        (
            "EXIT_DISCARD_AND_SLEEP",
            True,
            True,
            "[SAFE_TO_STOP] s 退出：四臂扭矩关闭已验证；"
            "姿态仅作诊断并已记录。",
        ),
        (
            "EXIT_DISCARD_AND_SLEEP",
            False,
            False,
            "[SAFE_TO_STOP] 四臂均已归位并验证扭矩关闭。",
        ),
        (
            "EXIT_SAVE_AND_SLEEP",
            True,
            False,
            "[SAFE_TO_STOP] 四臂均已归位并验证扭矩关闭。",
        ),
    ],
)
def test_finalizer_relaxes_pose_only_for_s_discard(
    monkeypatch,
    outcome_name,
    controller_policy,
    expected_policy,
    expected_message,
):
    recorder = load_recorder()
    calls = []
    messages = []
    forwarded = []
    runtime, worker, controller = external_finalizer_fakes(
        recorder,
        calls,
        allow_pose_deviation=controller_policy,
    )
    monkeypatch.setattr(recorder, "_SAFE_STOP_CONTROLLER", controller)
    monkeypatch.setattr(recorder, "robot_shutdown", lambda _node: None)

    recorder.finalize_recorder_runtime(
        runtime,
        outcome=getattr(recorder.SessionOutcome, outcome_name),
        save_worker=worker,
        robot_name="aloha_stationary",
        supervise_recovery=lambda **kwargs: (
            forwarded.append(kwargs) or sleep_report(safe=True)
        ),
        publish_state=lambda *_args, **_kwargs: None,
        lease_factory=lambda **kwargs: FakeRecoveryLease(
            calls, kwargs["recovery_id"]
        ),
        logger=messages.append,
    )

    assert forwarded[0]["allow_pose_deviation"] is expected_policy
    assert messages[-1] == expected_message


def test_stage_logger_base_exception_is_delayed_until_recovery_is_safe(
    monkeypatch,
):
    recorder = load_recorder()
    calls = []
    runtime, worker, controller = external_finalizer_fakes(recorder, calls)
    error = KeyboardInterrupt("stage logger interrupted")
    monkeypatch.setattr(recorder, "_SAFE_STOP_CONTROLLER", controller)
    monkeypatch.setattr(recorder, "robot_shutdown", lambda _node: None)

    with pytest.raises(KeyboardInterrupt) as raised:
        recorder.finalize_recorder_runtime(
            runtime,
            outcome=recorder.SessionOutcome.EXIT_DISCARD_AND_SLEEP,
            save_worker=worker,
            robot_name="aloha_stationary",
            supervise_recovery=lambda **_kwargs: (
                calls.append("child_safe") or sleep_report(safe=True)
            ),
            publish_state=lambda *_args, **_kwargs: None,
            lease_factory=lambda **kwargs: FakeRecoveryLease(
                calls, kwargs["recovery_id"]
            ),
            logger=lambda _message: (_ for _ in ()).throw(error),
        )

    assert raised.value is error
    assert calls.index("child_safe") < calls.index("leave_unsafe")


def test_non_tty_finalizer_unsafe_wait_has_no_keyboard_retry_prompt(monkeypatch):
    recorder = load_recorder()
    calls = []
    messages = []
    runtime, worker, _controller = external_finalizer_fakes(recorder, calls)
    controller = recorder.SafeStopController(
        threading.Event(),
        threading.Event(),
        threading.Event(),
        interrupt_main=lambda: pytest.fail("must not interrupt"),
        logger=messages.append,
        retry_input_available=False,
    )
    retries = iter((False, True))
    controller.wait_for_safety_retry = lambda timeout: next(retries)
    monkeypatch.setattr(recorder, "_SAFE_STOP_CONTROLLER", controller)
    monkeypatch.setattr(recorder, "robot_shutdown", lambda _node: None)

    def supervise(**kwargs):
        kwargs["wait_for_restart"](
            "attempt-1",
            RuntimeError("child exited"),
        )
        return sleep_report(safe=True)

    recorder.finalize_recorder_runtime(
        runtime,
        outcome=recorder.SessionOutcome.EXIT_DISCARD_AND_SLEEP,
        save_worker=worker,
        robot_name="aloha_stationary",
        supervise_recovery=supervise,
        publish_state=lambda *_args, **_kwargs: None,
        lease_factory=lambda **kwargs: FakeRecoveryLease(
            calls, kwargs["recovery_id"]
        ),
        logger=messages.append,
        clock=lambda: 1.0,
        log_interval_seconds=0.0,
    )

    assert all("按 s" not in message for message in messages)
    assert all("Press s" not in message for message in messages)
    assert any("交互终端" in message for message in messages)
    assert any("独立恢复" in message for message in messages)


def test_external_finalizer_defers_save_error_until_child_is_safe(monkeypatch):
    recorder = load_recorder()
    calls = []
    save_error = RuntimeError("save drain failed")
    runtime, worker, controller = external_finalizer_fakes(
        recorder, calls, drain_error=save_error
    )
    monkeypatch.setattr(recorder, "_SAFE_STOP_CONTROLLER", controller)
    monkeypatch.setattr(recorder, "robot_shutdown", lambda _node: None)

    with pytest.raises(RuntimeError, match="save drain failed") as raised:
        recorder.finalize_recorder_runtime(
            runtime,
            outcome=recorder.SessionOutcome.EXIT_SAVE_AND_SLEEP,
            save_worker=worker,
            robot_name="aloha_stationary",
            supervise_recovery=lambda **_kwargs: (
                calls.append("child_safe") or sleep_report(safe=True)
            ),
            publish_state=lambda *_args, **_kwargs: None,
            lease_factory=lambda **kwargs: FakeRecoveryLease(
                calls, kwargs["recovery_id"]
            ),
        )

    assert raised.value is save_error
    assert "health_stop" in calls
    assert "child_safe" in calls


def test_quiesce_retries_only_failed_stages_until_fully_quiesced(monkeypatch):
    recorder = load_recorder()
    calls = []

    camera_attempts = iter((RuntimeError("camera failed"), None))

    def close_camera():
        calls.append("camera_close")
        error = next(camera_attempts)
        if error is not None:
            raise error

    runtime = recorder.RecorderRuntime(
        node=object(),
        env=object(),
        health=SimpleNamespace(stop=lambda: calls.append("health_stop")),
        health_subscriptions=[object()],
        camera_runtime=SimpleNamespace(close=close_camera),
    )
    monkeypatch.setattr(
        recorder,
        "robot_shutdown",
        lambda node: calls.append(("robot_shutdown", node)),
    )

    with pytest.raises(
        recorder.RecorderRuntimeShutdownError, match="camera.close"
    ):
        recorder._quiesce_recorder_runtime(runtime)
    recorder._quiesce_recorder_runtime(runtime)

    assert calls == [
        "camera_close",
        "health_stop",
        ("robot_shutdown", runtime.node),
        "camera_close",
    ]
    assert runtime.health_subscriptions == []
    assert runtime.camera_runtime is None
    assert runtime.quiesced


def test_drain_timeout_aborts_before_quiesce_and_external_recovery(monkeypatch):
    recorder = load_recorder()
    calls = []
    drain_error = TimeoutError("save stuck")
    runtime, worker, controller = external_finalizer_fakes(
        recorder, calls, drain_error=drain_error
    )
    monkeypatch.setattr(recorder, "_SAFE_STOP_CONTROLLER", controller)
    monkeypatch.setattr(
        recorder,
        "robot_shutdown",
        lambda node: calls.append(("robot_shutdown", node)),
    )

    with pytest.raises(TimeoutError, match="save stuck") as raised:
        recorder.finalize_recorder_runtime(
            runtime,
            outcome=recorder.SessionOutcome.EXIT_SAVE_AND_SLEEP,
            save_worker=worker,
            robot_name="aloha_stationary",
            save_drain_timeout_seconds=0.25,
            save_abort_timeout_seconds=0.5,
            supervise_recovery=lambda **_kwargs: (
                calls.append("external_recovery") or sleep_report(safe=True)
            ),
            publish_state=lambda *_args, **_kwargs: None,
            lease_factory=lambda **kwargs: FakeRecoveryLease(
                calls, kwargs["recovery_id"]
            ),
        )

    assert raised.value is drain_error
    assert calls.index(("drain_timeout", 0.25)) < calls.index(
        ("save_abort", 0.5)
    )
    assert calls.index(("save_abort", 0.5)) < calls.index("camera_close")
    assert calls.index("camera_close") < calls.index("external_recovery")


def test_ros_shutdown_failure_requires_explicit_retry_before_child(monkeypatch):
    recorder = load_recorder()
    calls = []
    runtime, worker, controller = external_finalizer_fakes(recorder, calls)
    controller.wait_for_safety_retry = lambda timeout: (
        calls.append(("retry_wait", timeout)) or True
    )
    shutdown_attempts = iter((RuntimeError("shutdown failed"), None))

    def shutdown(node):
        calls.append(("robot_shutdown", node))
        error = next(shutdown_attempts)
        if error is not None:
            raise error

    monkeypatch.setattr(recorder, "_SAFE_STOP_CONTROLLER", controller)
    monkeypatch.setattr(recorder, "robot_shutdown", shutdown)

    with pytest.raises(
        recorder.RecorderRuntimeShutdownError, match="robot_shutdown"
    ):
        recorder.finalize_recorder_runtime(
            runtime,
            outcome=recorder.SessionOutcome.EXIT_SAVE_AND_SLEEP,
            save_worker=worker,
            robot_name="aloha_stationary",
            supervise_recovery=lambda **_kwargs: (
                calls.append("external_recovery") or sleep_report(safe=True)
            ),
            publish_state=lambda *_args, **_kwargs: None,
            lease_factory=lambda **kwargs: FakeRecoveryLease(
                calls, kwargs["recovery_id"]
            ),
        )

    shutdown_calls = [
        index
        for index, item in enumerate(calls)
        if item == ("robot_shutdown", runtime.node)
    ]
    assert len(shutdown_calls) == 2
    assert shutdown_calls[1] < calls.index("external_recovery")
    assert ("retry_wait", 1.0) in calls


def test_finalizer_defers_quiesce_error_until_external_safe(monkeypatch):
    recorder = load_recorder()
    calls = []
    runtime, worker, controller = external_finalizer_fakes(recorder, calls)

    def close_camera():
        calls.append("camera_close")
        raise RuntimeError("camera failed")

    runtime.camera_runtime.close = close_camera
    monkeypatch.setattr(recorder, "_SAFE_STOP_CONTROLLER", controller)
    monkeypatch.setattr(
        recorder,
        "robot_shutdown",
        lambda node: calls.append(("robot_shutdown", node)),
    )

    with pytest.raises(
        recorder.RecorderRuntimeShutdownError, match="camera.close"
    ):
        recorder.finalize_recorder_runtime(
            runtime,
            outcome=recorder.SessionOutcome.EXIT_SAVE_AND_SLEEP,
            save_worker=worker,
            robot_name="aloha_stationary",
            supervise_recovery=lambda **_kwargs: (
                calls.append("child_safe") or sleep_report(safe=True)
            ),
            publish_state=lambda *_args, **_kwargs: None,
            lease_factory=lambda **kwargs: FakeRecoveryLease(
                calls, kwargs["recovery_id"]
            ),
        )

    assert "health_stop" in calls
    assert runtime.health_subscriptions == []
    assert ("robot_shutdown", runtime.node) in calls
    assert "child_safe" in calls


def test_finalizer_handoff_callbacks_hold_lease_until_explicit_retry(
    monkeypatch,
):
    recorder = load_recorder()
    calls = []
    runtime, worker, controller = external_finalizer_fakes(recorder, calls)
    retries = iter((False, False, True))
    controller.wait_for_safety_retry = lambda timeout: (
        calls.append(("retry_wait", timeout)) or next(retries)
    )
    monkeypatch.setattr(recorder, "_SAFE_STOP_CONTROLLER", controller)
    monkeypatch.setattr(recorder, "robot_shutdown", lambda _node: None)

    def lease_factory(**kwargs):
        calls.append(("lease_acquire", kwargs.copy()))
        return FakeRecoveryLease(calls, kwargs["recovery_id"])

    def supervise(**kwargs):
        kwargs["prepare_attempt"]("attempt-1")
        kwargs["wait_for_restart"]("attempt-1", RuntimeError("early exit"))
        calls.append("respawn_allowed")
        return sleep_report(safe=True)

    recorder.finalize_recorder_runtime(
        runtime,
        outcome=recorder.SessionOutcome.EXIT_SAVE_AND_SLEEP,
        save_worker=worker,
        robot_name="aloha_stationary",
        supervise_recovery=supervise,
        publish_state=lambda state, **kwargs: calls.append(
            ("state", state, kwargs)
        ),
        lease_factory=lease_factory,
        logger=lambda message: calls.append(("log", message)),
        clock=iter((0.0, 0.2, 2.0)).__next__,
    )

    acquired = [item for item in calls if item[0] == "lease_acquire"]
    assert [item[1]["recovery_id"] for item in acquired] == [
        "attempt-1",
        "attempt-1",
    ]
    states = [item for item in calls if item[0] == "state"]
    assert states[0][1] == "EXTERNAL_RECOVERY_REQUIRED"
    assert states[0][2]["context_ok"] is False
    assert states[1][1] == "UNSAFE_HOLD"
    assert calls.count(("retry_wait", 1.0)) == 3
    unsafe_wait_logs = [
        item
        for item in calls
        if item[0] == "log" and "standalone recovery exited" in item[1]
    ]
    assert len(unsafe_wait_logs) <= 2
    wait_release = max(
        index
        for index, item in enumerate(calls)
        if item == ("lease_release", "attempt-1")
    )
    assert wait_release < calls.index("respawn_allowed")


def test_finalizer_latches_unsafe_controller_around_supervision(monkeypatch):
    recorder = load_recorder()
    calls = []
    runtime, worker, _controller = external_finalizer_fakes(recorder, calls)
    stop_no_save = threading.Event()
    controller = recorder.SafeStopController(
        stop_no_save,
        threading.Event(),
        threading.Event(),
        interrupt_main=lambda: pytest.fail("must not interrupt supervision"),
        logger=lambda message: calls.append(("signal_log", message)),
    )
    monkeypatch.setattr(recorder, "_SAFE_STOP_CONTROLLER", controller)
    monkeypatch.setattr(recorder, "robot_shutdown", lambda _node: None)

    def supervise(**_kwargs):
        calls.append("supervising")
        controller.handle_sigint()
        controller.handle_sigterm()
        calls.append("signals_ignored")
        return sleep_report(safe=True)

    recorder.finalize_recorder_runtime(
        runtime,
        outcome=recorder.SessionOutcome.EXIT_SAVE_AND_SLEEP,
        save_worker=worker,
        robot_name="aloha_stationary",
        supervise_recovery=supervise,
        publish_state=lambda *_args, **_kwargs: None,
        lease_factory=lambda **kwargs: FakeRecoveryLease(
            calls, kwargs["recovery_id"]
        ),
    )

    assert calls.index("supervising") < calls.index("signals_ignored")
    assert not stop_no_save.is_set()
    assert len([item for item in calls if item[0] == "signal_log"]) == 2
    assert controller._unsafe_hold is False


def test_unexpected_supervisor_base_exception_exits_without_retry(monkeypatch):
    recorder = load_recorder()
    calls = []
    runtime, worker, controller = external_finalizer_fakes(recorder, calls)
    error = KeyboardInterrupt("injected supervisor interrupt")
    monkeypatch.setattr(recorder, "_SAFE_STOP_CONTROLLER", controller)
    monkeypatch.setattr(recorder, "robot_shutdown", lambda _node: None)

    def supervise(**_kwargs):
        calls.append("supervise")
        raise error

    with pytest.raises(KeyboardInterrupt) as raised:
        recorder.finalize_recorder_runtime(
            runtime,
            outcome=recorder.SessionOutcome.EXIT_SAVE_AND_SLEEP,
            save_worker=worker,
            robot_name="aloha_stationary",
            supervise_recovery=supervise,
            publish_state=lambda *_args, **_kwargs: None,
            lease_factory=lambda **kwargs: FakeRecoveryLease(
                calls, kwargs["recovery_id"]
            ),
            logger=lambda _message: (_ for _ in ()).throw(
                RuntimeError("logger failed")
            ),
        )

    assert raised.value is error
    assert calls.count("supervise") == 1
    assert "child_safe" not in calls
    assert "leave_unsafe" not in calls


def test_finalizer_does_not_retry_an_active_recovery_session(monkeypatch):
    recorder = load_recorder()
    calls = []
    runtime, worker, controller = external_finalizer_fakes(recorder, calls)
    retry_waits = []
    controller.wait_for_safety_retry = lambda timeout: (
        retry_waits.append(timeout) or True
    )
    monkeypatch.setattr(recorder, "_SAFE_STOP_CONTROLLER", controller)
    monkeypatch.setattr(recorder, "robot_shutdown", lambda _node: None)
    sessions = []

    def supervise(**kwargs):
        session = kwargs["session"]
        sessions.append(session)
        if len(sessions) == 1:
            session.start_process(
                recovery_id="attempt-1",
                stop_timeout_seconds=0.0,
                process_factory=lambda *_args, **_kwargs: SimpleNamespace(
                    pid=901
                ),
                command=["sleep.py"],
            )
            raise KeyboardInterrupt("logger interrupted")
        assert session.active is not None
        session.mark_reaped(session.active.token)
        return sleep_report(safe=True)

    with pytest.raises(KeyboardInterrupt, match="logger interrupted"):
        recorder.finalize_recorder_runtime(
            runtime,
            outcome=recorder.SessionOutcome.EXIT_SAVE_AND_SLEEP,
            save_worker=worker,
            robot_name="aloha_stationary",
            supervise_recovery=supervise,
            publish_state=lambda *_args, **_kwargs: None,
            lease_factory=lambda **kwargs: FakeRecoveryLease(
                calls, kwargs["recovery_id"]
            ),
        )

    assert len(sessions) == 1
    assert sessions[0].active is not None
    assert retry_waits == []


def test_real_supervisor_releases_lease_then_does_not_respawn(monkeypatch):
    recorder = load_recorder()
    calls = []
    runtime, worker, controller = external_finalizer_fakes(recorder, calls)
    active = {"lease": False}
    ids = []
    attempts = {"count": 0}

    class SingleOwnerLease(FakeRecoveryLease):
        def release(self):
            assert active["lease"]
            calls.append(("lease_release", self.metadata.recovery_id))
            active["lease"] = False

    def lease_factory(**kwargs):
        assert not active["lease"]
        active["lease"] = True
        calls.append(("lease_acquire", kwargs["recovery_id"]))
        return SingleOwnerLease(calls, kwargs["recovery_id"])

    def run_attempt(**kwargs):
        assert not active["lease"]
        calls.append(("child", kwargs["recovery_id"]))
        attempts["count"] += 1
        raise ExternalRecoveryError("early exit")

    def recovery_id_factory():
        if ids:
            pytest.fail("external supervisor must not create a second ID")
        ids.append("attempt-1")
        return "attempt-1"

    def supervise(**kwargs):
        return recorder.supervise_external_recovery(
            **kwargs,
            recovery_id_factory=recovery_id_factory,
            run_attempt=run_attempt,
        )

    monkeypatch.setattr(recorder, "_SAFE_STOP_CONTROLLER", controller)
    monkeypatch.setattr(recorder, "robot_shutdown", lambda _node: None)
    with pytest.raises(ExternalRecoveryError, match="early exit"):
        recorder.finalize_recorder_runtime(
            runtime,
            outcome=recorder.SessionOutcome.EXIT_SAVE_AND_SLEEP,
            save_worker=worker,
            robot_name="aloha_stationary",
            supervise_recovery=supervise,
            publish_state=lambda state, **_kwargs: calls.append(
                ("state", state)
            ),
            lease_factory=lease_factory,
        )

    assert calls.index(("lease_release", "attempt-1")) < calls.index(
        ("child", "attempt-1")
    )
    assert ids == ["attempt-1"]
    assert attempts["count"] == 1
    assert ("lease_acquire", "attempt-2") not in calls
    assert "leave_unsafe" not in calls


@pytest.mark.parametrize("failure_stage", ["lease_acquire", "publish", "release"])
def test_real_supervisor_callback_failure_exits_without_retry(
    monkeypatch, failure_stage
):
    recorder = load_recorder()
    calls = []
    runtime, worker, controller = external_finalizer_fakes(recorder, calls)
    ids = iter(("attempt-1", "attempt-2"))
    error = RuntimeError(f"{failure_stage} failed")
    failed = {"value": False}

    class Lease(FakeRecoveryLease):
        def release(self):
            calls.append(("lease_release", self.metadata.recovery_id))
            if failure_stage == "release" and not failed["value"]:
                failed["value"] = True
                raise error

    def lease_factory(**kwargs):
        calls.append(("lease_acquire", kwargs["recovery_id"]))
        if failure_stage == "lease_acquire" and not failed["value"]:
            failed["value"] = True
            raise error
        return Lease(calls, kwargs["recovery_id"])

    def publish(state, **_kwargs):
        calls.append(("state", state))
        if failure_stage == "publish" and not failed["value"]:
            failed["value"] = True
            raise error

    def supervise(**kwargs):
        return recorder.supervise_external_recovery(
            **kwargs,
            recovery_id_factory=lambda: next(ids),
            run_attempt=lambda **attempt: (
                calls.append(("child", attempt["recovery_id"]))
                or sleep_report(safe=True)
            ),
        )

    monkeypatch.setattr(recorder, "_SAFE_STOP_CONTROLLER", controller)
    monkeypatch.setattr(recorder, "robot_shutdown", lambda _node: None)
    with pytest.raises(RuntimeError) as raised:
        recorder.finalize_recorder_runtime(
            runtime,
            outcome=recorder.SessionOutcome.EXIT_SAVE_AND_SLEEP,
            save_worker=worker,
            robot_name="aloha_stationary",
            supervise_recovery=supervise,
            publish_state=publish,
            lease_factory=lease_factory,
        )

    assert raised.value is error
    assert ("child", "attempt-1") not in calls
    assert ("child", "attempt-2") not in calls
    assert "leave_unsafe" not in calls


def test_repeated_retry_dependency_errors_are_logged_and_stored_once(
    monkeypatch,
):
    recorder = load_recorder()
    calls = []
    messages = []
    runtime, worker, controller = external_finalizer_fakes(recorder, calls)
    shutdown_attempts = iter((RuntimeError("shutdown failed"), None))
    wait_attempts = {"count": 0}

    def shutdown(_node):
        error = next(shutdown_attempts)
        if error is not None:
            raise error

    def broken_clock():
        raise RuntimeError("clock failed")

    def retry_wait(timeout):
        wait_attempts["count"] += 1
        if wait_attempts["count"] <= 3:
            raise RuntimeError("wait failed")
        return True

    controller.wait_for_safety_retry = retry_wait
    monkeypatch.setattr(recorder, "_SAFE_STOP_CONTROLLER", controller)
    monkeypatch.setattr(recorder, "robot_shutdown", shutdown)
    monkeypatch.setattr(recorder.time, "sleep", lambda _seconds: None)

    with pytest.raises(recorder.RecorderRuntimeShutdownError):
        recorder.finalize_recorder_runtime(
            runtime,
            outcome=recorder.SessionOutcome.EXIT_SAVE_AND_SLEEP,
            save_worker=worker,
            robot_name="aloha_stationary",
            supervise_recovery=lambda **_kwargs: sleep_report(safe=True),
            publish_state=lambda *_args, **_kwargs: None,
            lease_factory=lambda **kwargs: FakeRecoveryLease(
                calls, kwargs["recovery_id"]
            ),
            clock=broken_clock,
            logger=messages.append,
        )

    clock_logs = [message for message in messages if "clock failed" in message]
    wait_logs = [message for message in messages if "wait failed" in message]
    assert len(clock_logs) == 1
    assert len(wait_logs) == 1
    assert all("\n" not in message and len(message) <= 512 for message in messages)


def test_finalizer_source_has_no_parent_arm_recovery_path():
    source = RECORDER.read_text(encoding="utf-8")
    finalizer = source.split("def finalize_recorder_runtime(", 1)[1].split(
        "def capture_one_episode(", 1
    )[0]

    assert "_recover_runtime_to_sleep" not in finalizer
    assert "recover_robots_to_sleep" not in finalizer
    assert "_restore_post_session_gripper_idle_modes" not in finalizer
