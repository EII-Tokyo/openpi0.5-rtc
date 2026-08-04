from pathlib import Path

import pytest

from aloha.safe_sleep import (
    RobotSleepResult,
    SafeSleepReport,
    SleepStatus,
)
from aloha.safe_sleep_runtime import (
    build_prepare_robot,
    initialize_robots_independently,
    initialize_ros_context,
    install_recovery_signal_handlers,
    merge_initialization_failures,
    parse_sleep_args,
    retry_failed_initializations,
)


def test_gravity_compensation_is_inactive_by_default():
    args = parse_sleep_args(["--robot", "aloha_stationary"])

    assert args.gravity_compensation_active is False
    assert args.skip_gravity_compensation is False
    assert args.allow_pose_deviation is False


def test_pose_deviation_policy_requires_explicit_cli_flag():
    args = parse_sleep_args(
        [
            "--robot",
            "aloha_stationary",
            "--allow-pose-deviation",
        ]
    )

    assert args.allow_pose_deviation is True


def test_deprecated_skip_alias_keeps_gravity_inactive():
    args = parse_sleep_args(
        [
            "--robot",
            "aloha_stationary",
            "--skip-gravity-compensation",
        ]
    )

    assert args.gravity_compensation_active is False


def test_conflicting_gravity_flags_are_rejected():
    with pytest.raises(SystemExit):
        parse_sleep_args(
            [
                "--robot",
                "aloha_stationary",
                "--gravity-compensation-active",
                "--skip-gravity-compensation",
            ]
        )


def test_fast_sleep_help_does_not_claim_a_four_second_floor(capsys):
    with pytest.raises(SystemExit) as exc_info:
        parse_sleep_args(["--help"])

    assert exc_info.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "4 second safety floor" not in help_text


def test_manual_sleep_uses_dedicated_safe_sleep_floor():
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "sleep.py"
    ).read_text(encoding="utf-8")

    assert "SAFE_SLEEP_MIN_SECONDS" in source
    assert "minimum_seconds=SAFE_SLEEP_MIN_SECONDS" in source
    assert "allow_pose_deviation=args.allow_pose_deviation" in source


def test_manual_sleep_exits_after_one_unsafe_report_without_retry_wait():
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "sleep.py"
    ).read_text(encoding="utf-8")
    main = source.split("def main(argv=None)", 1)[1].split(
        'if __name__ == "__main__":',
        1,
    )[0]

    assert "hold_unsafe_until_safe" not in main
    assert "safe-sleep-retry-input" not in main
    assert "等待已核验的 SIGUSR1" not in main
    unsafe_result = main.split("if not report.safe_to_stop:", 1)[1]
    assert 'publish_state("UNSAFE_HOLD", report=report)' in unsafe_result
    assert "失败臂不重试" in unsafe_result
    assert "return 2" in unsafe_result


def test_inactive_gravity_never_calls_service():
    calls = []
    prepare = build_prepare_robot(
        gravity_compensation_active=False,
        timeout_seconds=2.0,
        set_gravity=lambda *_args, **_kwargs: pytest.fail(
            "inactive gravity must not create a service client"
        ),
        set_mode=lambda *args, **kwargs: calls.append(
            ("mode", args, kwargs)
        ),
        set_torque=lambda *args, **kwargs: calls.append(
            ("torque", args, kwargs)
        ),
    )

    prepare("leader_left", object())

    assert [call[0] for call in calls] == ["mode", "torque"]


def test_active_gravity_is_disabled_before_mode_and_torque():
    calls = []
    robot = object()
    prepare = build_prepare_robot(
        gravity_compensation_active=True,
        timeout_seconds=2.0,
        set_gravity=lambda *args, **kwargs: calls.append(
            ("gravity", args, kwargs)
        ),
        set_mode=lambda *args, **kwargs: calls.append(
            ("mode", args, kwargs)
        ),
        set_torque=lambda *args, **kwargs: calls.append(
            ("torque", args, kwargs)
        ),
    )

    prepare("leader_left", robot)

    assert [call[0] for call in calls] == [
        "gravity",
        "mode",
        "torque",
    ]
    assert calls[0][1] == (robot, False)


def test_active_gravity_does_not_apply_to_followers():
    prepare = build_prepare_robot(
        gravity_compensation_active=True,
        timeout_seconds=2.0,
        set_gravity=lambda *_args, **_kwargs: pytest.fail(
            "followers do not expose gravity compensation"
        ),
        set_mode=lambda *_args, **_kwargs: None,
        set_torque=lambda *_args, **_kwargs: None,
    )

    prepare("follower_left", object())


def test_ros_context_is_initialized_without_rclpy_signal_handlers():
    calls = []

    initialize_ros_context(
        ok=lambda: False,
        init=lambda **kwargs: calls.append(kwargs),
        no_signal_handlers="NO",
    )

    assert calls == [{"signal_handler_options": "NO"}]


def test_existing_ros_context_is_not_reinitialized():
    initialize_ros_context(
        ok=lambda: True,
        init=lambda **_kwargs: pytest.fail("context already initialized"),
        no_signal_handlers="NO",
    )


def test_recovery_signal_handlers_are_installed_for_all_safe_stop_paths():
    calls = []

    class Controller:
        def handle_sigint(self):
            calls.append("sigint")

        def handle_sigterm(self):
            calls.append("sigterm")

        def request_from_s(self, *, wake_main):
            calls.append(("sigusr1", wake_main))

    class Signals:
        SIGINT = 2
        SIGTERM = 15
        SIGUSR1 = 10

        def __init__(self):
            self.handlers = {}

        def signal(self, number, callback):
            self.handlers[number] = callback

    signals = Signals()
    install_recovery_signal_handlers(
        signal_module=signals,
        controller=Controller(),
    )

    signals.handlers[signals.SIGINT](None, None)
    signals.handlers[signals.SIGTERM](None, None)
    signals.handlers[signals.SIGUSR1](None, None)
    assert calls == ["sigint", "sigterm", ("sigusr1", False)]


def test_manual_sleep_owns_signals_before_creating_robot_runtime():
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "sleep.py"
    ).read_text(encoding="utf-8")
    main = source.split("def main(argv=None)", 1)[1].split(
        'if __name__ == "__main__":',
        1,
    )[0]

    ros_init = main.index("initialize_ros_context(")
    install_signals = main.index(
        "install_recovery_signal_handlers("
    )
    create_node = main.index("create_interbotix_global_node(")
    create_robots = main.index("initialize_robots_independently(")

    assert ros_init < install_signals < create_node < create_robots
    assert "except BaseException as exc:" in main
    unsafe_branch = main.split("except BaseException as exc:", 1)[1]
    assert (
        'publish_safety_state(\n                "UNSAFE_HOLD"'
        in unsafe_branch
    )
    assert "robot_shutdown(node)" not in unsafe_branch


def test_one_initialization_failure_does_not_block_later_arms():
    configs = [
        {"name": "leader_left", "model": "leader"},
        {"name": "leader_right", "model": "leader"},
        {"name": "follower_left", "model": "follower"},
    ]
    attempted = []

    def create_robot(config):
        attempted.append(config["name"])
        if config["name"] == "leader_left":
            raise RuntimeError("no status packet")
        return object()

    robots, failures = initialize_robots_independently(
        configs,
        create_robot=create_robot,
        logger=lambda _message: None,
    )

    assert attempted == [
        "leader_left",
        "leader_right",
        "follower_left",
    ]
    assert set(robots) == {"leader_right", "follower_left"}
    assert "no status packet" in failures["leader_left"]


def test_initialization_failure_keeps_complete_recovery_unsafe():
    safe_result = RobotSleepResult(
        robot_name="leader_right",
        status=SleepStatus.SLEPT_VERIFIED,
        max_error_rad=0.01,
        reason="verified",
        phase="complete",
        torque_off_verified=True,
    )

    report = merge_initialization_failures(
        SafeSleepReport({"leader_right": safe_result}),
        {"leader_left": "RuntimeError: no status packet"},
    )

    assert not report.safe_to_stop
    assert (
        report.results["leader_left"].status
        is SleepStatus.UNRESPONSIVE
    )
    assert report.results["leader_left"].phase == "initialize"


def test_explicit_retry_stages_a_previously_missing_arm_until_commit():
    configs = [
        {"name": "leader_left", "model": "leader"},
        {"name": "leader_right", "model": "leader"},
    ]
    existing_right = object()
    recovered_left = object()
    robots = {"leader_right": existing_right}
    failures = {"leader_left": "RuntimeError: no status packet"}
    attempted = []

    newly_initialized, current_failures = retry_failed_initializations(
        configs,
        robots=robots,
        failures=failures,
        create_robot=lambda config: (
            attempted.append(config["name"]) or recovered_left
        ),
        logger=lambda _message: None,
    )

    assert attempted == ["leader_left"]
    assert newly_initialized == {"leader_left": recovered_left}
    assert current_failures == {}
    assert robots == {"leader_right": existing_right}
    assert failures == {
        "leader_left": "RuntimeError: no status packet"
    }
