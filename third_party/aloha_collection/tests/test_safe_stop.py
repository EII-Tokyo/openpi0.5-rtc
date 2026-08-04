import subprocess
import sys
import textwrap
import threading
import json
import os
from pathlib import Path

import pytest

from aloha.record_trigger import RecordingPhase
from aloha.safe_stop import (
    SafeStopController,
    publish_safety_state,
    should_defer_s_wakeup,
)
from aloha.safety_state import RecoveryIdentity


ROOT = Path(__file__).resolve().parents[1]
RECORDER = ROOT / "scripts/record_episodes_copy.py"
SERIALIZER = ROOT / "aloha/episode_serialization.py"
SAFE_STOP_SCRIPT = ROOT / "scripts" / "safe_stop_container.sh"


def make_controller(*, retry_input_available=True):
    stop_no_save = threading.Event()
    stop_and_save = threading.Event()
    skip_sleep = threading.Event()
    interrupts = []
    logs = []
    controller = SafeStopController(
        stop_no_save,
        stop_and_save,
        skip_sleep,
        interrupt_main=lambda: interrupts.append("interrupt"),
        logger=logs.append,
        retry_input_available=retry_input_available,
    )
    return controller, stop_no_save, stop_and_save, skip_sleep, interrupts, logs


def test_pose_deviation_policy_defaults_strict_and_s_freezes_relaxed():
    controller, *_ = make_controller()

    assert controller.allow_pose_deviation is False

    controller.request_from_s(wake_main=False)

    assert controller.allow_pose_deviation is True


@pytest.mark.parametrize("request_kind", ["m", "save", "sigint", "sigterm"])
def test_non_s_stop_paths_keep_pose_deviation_policy_strict(request_kind):
    controller, *_ = make_controller()

    if request_kind == "m":
        controller.request_no_save(source="m", wake_main=False)
    elif request_kind == "save":
        assert controller.request_save(skip_sleep=False, source="m")
    elif request_kind == "sigint":
        controller.handle_sigint()
    else:
        controller.handle_sigterm()

    assert controller.allow_pose_deviation is False


def test_first_accepted_stop_request_freezes_pose_deviation_policy():
    strict_controller, *_ = make_controller()
    strict_controller.request_no_save(source="m", wake_main=False)
    strict_controller.request_from_s(wake_main=False)

    relaxed_controller, *_ = make_controller()
    relaxed_controller.request_from_s(wake_main=False)
    relaxed_controller.request_no_save(source="failure", wake_main=False)
    relaxed_controller.force_no_save("cleanup failed")

    assert strict_controller.allow_pose_deviation is False
    assert relaxed_controller.allow_pose_deviation is True


def test_s_retry_during_unsafe_hold_does_not_relax_strict_policy():
    controller, *_ = make_controller()
    controller.enter_unsafe_hold()

    controller.request_from_s(wake_main=False)

    assert controller.allow_pose_deviation is False


def test_named_no_save_request_wakes_main_and_logs_source():
    controller, stop_no_save, stop_and_save, skip_sleep, interrupts, logs = (
        make_controller()
    )
    stop_and_save.set()
    skip_sleep.set()

    controller.request_no_save(source="foot-pedal-failure")

    assert stop_no_save.is_set()
    assert not stop_and_save.is_set()
    assert not skip_sleep.is_set()
    assert interrupts == ["interrupt"]
    assert any("foot-pedal-failure" in message for message in logs)


def test_named_no_save_does_not_turn_unsafe_hold_into_retry():
    controller, stop_no_save, *_rest = make_controller()
    controller.enter_unsafe_hold()

    controller.request_no_save(source="foot-pedal-failure")

    assert not stop_no_save.is_set()
    assert not controller.wait_for_safety_retry(timeout=0.0)


def test_first_sigint_requests_no_save_and_sleep_without_raising():
    controller, stop_no_save, _, skip_sleep, interrupts, logs = make_controller()
    skip_sleep.set()

    controller.handle_sigint()

    assert stop_no_save.is_set()
    assert not skip_sleep.is_set()
    assert interrupts == []
    assert any("安全停止" in message for message in logs)


def test_second_sigint_raises_keyboard_interrupt():
    controller, stop_no_save, _, skip_sleep, interrupts, logs = make_controller()
    controller.handle_sigint()

    with pytest.raises(KeyboardInterrupt):
        controller.handle_sigint()

    assert stop_no_save.is_set()
    assert not skip_sleep.is_set()
    assert any("强制退出阻塞" in message for message in logs)


def test_sigterm_requests_no_save_and_sleep_like_first_sigint():
    controller, stop_no_save, stop_and_save, skip_sleep, interrupts, logs = (
        make_controller()
    )
    stop_and_save.set()
    skip_sleep.set()

    controller.handle_sigterm()

    assert stop_no_save.is_set()
    assert not stop_and_save.is_set()
    assert not skip_sleep.is_set()
    assert interrupts == []
    assert any("SIGTERM" in message for message in logs)


def test_termination_is_blocked_during_unsafe_hold():
    controller, stop_no_save, _, _, _, logs = make_controller()
    controller.enter_unsafe_hold()

    controller.handle_sigint()
    controller.handle_sigterm()

    assert not stop_no_save.is_set()
    assert sum("UNSAFE_HOLD" in message for message in logs) == 2


def test_non_tty_unsafe_hold_never_prompts_for_s_and_names_external_recovery():
    controller, stop_no_save, _, _, _, logs = make_controller(
        retry_input_available=False
    )
    controller.enter_unsafe_hold()

    controller.handle_sigint()
    controller.handle_sigterm()

    assert not stop_no_save.is_set()
    assert all("按 s" not in message for message in logs)
    assert all("Press s" not in message for message in logs)
    assert all("交互终端" in message for message in logs)
    assert all("独立恢复" in message for message in logs)


def test_s_requests_explicit_retry_during_unsafe_hold():
    controller, *_, logs = make_controller()
    controller.enter_unsafe_hold()

    controller.request_from_s()

    assert controller.wait_for_safety_retry(timeout=0.0)
    assert not controller.wait_for_safety_retry(timeout=0.0)
    assert logs == [
        "\n[s] UNSAFE_HOLD 显式重试已请求；"
        "将重试当前独立进程或启动全新 safe-sleep。"
    ]


def test_safety_state_is_published_atomically(tmp_path):
    state_path = tmp_path / "aloha_recorder_safety.json"
    report = type(
        "Report",
        (),
        {
            "results": {
                "leader_left": type(
                    "Result",
                    (),
                    {"status": type("Status", (), {"value": "unresponsive"})()},
                )()
            }
        },
    )()

    publish_safety_state(
        "UNSAFE_HOLD",
        report=report,
        path=state_path,
        recovery=RecoveryIdentity(
            recovery_id="test-recovery",
            owner_pid=os.getpid(),
            source="recorder",
        ),
        context_ok=True,
        monotonic_clock=lambda: 1234.5,
        wall_clock=lambda: "2026-07-30T00:00:00+00:00",
    )

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 2,
        "state": "UNSAFE_HOLD",
        "safe_to_stop": False,
        "recovery_id": "test-recovery",
        "owner_pid": os.getpid(),
        "source": "recorder",
        "context_ok": True,
        "updated_wall_time": "2026-07-30T00:00:00+00:00",
        "updated_monotonic": 1234.5,
        "robots": {
            "leader_left": {
                "status": "unresponsive",
                "phase": "unknown",
                "reason": "",
                "max_error_rad": None,
                "torque_off_verified": False,
            }
        },
    }
    assert list(tmp_path.iterdir()) == [state_path]


def test_host_container_stop_wrapper_is_fail_closed():
    source = SAFE_STOP_SCRIPT.read_text(encoding="utf-8")

    assert "aloha2-collect" in source
    assert "pgrep" in source
    assert "kill -INT" in source
    assert "aloha_recorder_safety.json" in source
    assert "validate_safety_state.py" in source
    assert "expected_recovery_id" in source
    assert "kill -0" in source
    assert "SAFE_TO_STOP" in source
    assert "UNSAFE_HOLD" in source
    assert "docker stop --time 120" in source
    assert "docker kill" not in source
    assert "docker rm" not in source


def test_s_requests_main_thread_interrupt_without_skipping_sleep():
    controller, stop_no_save, _, skip_sleep, interrupts, logs = make_controller()
    skip_sleep.set()

    controller.request_from_s()
    controller.request_from_s()

    assert stop_no_save.is_set()
    assert not skip_sleep.is_set()
    assert interrupts == ["interrupt"]
    assert logs[0] == (
        "\n[s] 已收到：停止采集并丢弃当前未完成 episode；"
        "随后启动独立 safe-sleep。"
    )
    assert logs[1:] == ["\n[s] 安全停止已在进行，忽略重复请求。"]


def test_s_cancels_a_pending_save_request():
    controller, stop_no_save, stop_and_save, skip_sleep, interrupts, _ = make_controller()
    stop_and_save.set()
    skip_sleep.set()

    controller.request_from_s()

    assert stop_no_save.is_set()
    assert not stop_and_save.is_set()
    assert not skip_sleep.is_set()
    assert interrupts == ["interrupt"]


def test_deferred_s_dominates_pending_save_without_interrupting_main():
    controller, stop_no_save, stop_and_save, skip_sleep, interrupts, logs = make_controller()
    stop_and_save.set()
    skip_sleep.set()

    controller.request_from_s(wake_main=False)

    assert stop_no_save.is_set()
    assert not stop_and_save.is_set()
    assert not skip_sleep.is_set()
    assert interrupts == []
    assert logs == [
        "\n[s] 已收到：停止采集并丢弃当前未完成 episode；"
        "随后启动独立 safe-sleep。"
    ]


def test_deferred_s_is_idempotent_and_later_calls_do_not_wake_main():
    controller, stop_no_save, stop_and_save, skip_sleep, interrupts, logs = make_controller()

    controller.request_from_s(wake_main=False)
    controller.request_from_s(wake_main=False)
    controller.request_from_s()

    assert stop_no_save.is_set()
    assert not stop_and_save.is_set()
    assert not skip_sleep.is_set()
    assert interrupts == []
    assert sum("安全停止已在进行" in message for message in logs) == 2


@pytest.mark.parametrize(
    "phase",
    [
        RecordingPhase.RETURNING_TO_RETRY,
        RecordingPhase.RETURNING_TO_SAVE,
    ],
)
def test_s_during_either_return_phase_is_authoritative_without_interrupt(phase):
    controller, stop_no_save, stop_and_save, skip_sleep, interrupts, _ = make_controller()
    stop_and_save.set()
    skip_sleep.set()

    controller.request_from_s(wake_main=not should_defer_s_wakeup(phase))

    assert stop_no_save.is_set()
    assert not stop_and_save.is_set()
    assert not skip_sleep.is_set()
    assert interrupts == []


@pytest.mark.parametrize(
    "phase",
    [
        RecordingPhase.WAITING_FOR_B,
        RecordingPhase.RECORDING,
    ],
)
def test_s_outside_return_phase_still_wakes_main(phase):
    controller, stop_no_save, _, skip_sleep, interrupts, _ = make_controller()

    controller.request_from_s(wake_main=not should_defer_s_wakeup(phase))

    assert stop_no_save.is_set()
    assert not skip_sleep.is_set()
    assert interrupts == ["interrupt"]


def test_save_commands_cannot_override_an_s_request():
    controller, _, stop_and_save, skip_sleep, _, _ = make_controller()
    controller.request_from_s()

    assert not controller.request_save(skip_sleep=True, source="r")
    assert not controller.request_save(skip_sleep=False, source="m")
    assert not stop_and_save.is_set()
    assert not skip_sleep.is_set()


def test_force_no_save_during_cleanup_clears_save_and_skip_atomically():
    controller, stop_no_save, stop_and_save, skip_sleep, interrupts, logs = make_controller()
    stop_and_save.set()
    skip_sleep.set()
    controller.begin_cleanup()

    controller.force_no_save("attempt cleanup failed")

    assert stop_no_save.is_set()
    assert not stop_and_save.is_set()
    assert not skip_sleep.is_set()
    assert interrupts == []
    assert any("attempt cleanup failed" in message for message in logs)


def test_s_during_cleanup_does_not_interrupt_sleep():
    controller, stop_no_save, stop_and_save, skip_sleep, interrupts, logs = make_controller()
    stop_and_save.set()
    controller.begin_cleanup()

    controller.request_from_s()

    assert not stop_no_save.is_set()
    assert stop_and_save.is_set()
    assert not skip_sleep.is_set()
    assert interrupts == []
    assert any("清理阶段" in message for message in logs)


def test_sigint_and_sigterm_cannot_interrupt_sleep_recovery():
    controller, stop_no_save, _, _, _, logs = make_controller()
    controller.handle_sigint()
    controller.begin_cleanup()

    controller.handle_sigint()
    controller.handle_sigterm()

    assert stop_no_save.is_set()
    assert sum("RECOVERY_IN_PROGRESS" in message for message in logs) == 2


def test_cleanup_waits_until_authorized_s_wakeup_is_sent():
    interrupt_entered = threading.Event()
    allow_interrupt_to_finish = threading.Event()
    cleanup_finished = threading.Event()
    stop_no_save = threading.Event()
    stop_and_save = threading.Event()
    skip_sleep = threading.Event()

    def blocking_interrupt():
        interrupt_entered.set()
        allow_interrupt_to_finish.wait(timeout=1.0)

    controller = SafeStopController(
        stop_no_save,
        stop_and_save,
        skip_sleep,
        interrupt_main=blocking_interrupt,
        logger=lambda _: None,
    )
    request_thread = threading.Thread(target=controller.request_from_s)
    cleanup_thread = threading.Thread(
        target=lambda: (controller.begin_cleanup(), cleanup_finished.set())
    )

    request_thread.start()
    assert interrupt_entered.wait(timeout=1.0)
    cleanup_thread.start()

    assert not cleanup_finished.wait(timeout=0.1)

    allow_interrupt_to_finish.set()
    request_thread.join(timeout=1.0)
    cleanup_thread.join(timeout=1.0)
    assert cleanup_finished.is_set()


def test_recorder_routes_s_and_sigint_through_safe_stop_controller():
    source = RECORDER.read_text(encoding="utf-8")

    assert "from aloha.safe_stop import (" in source
    assert "SafeStopController," in source
    s_handler = source.split("def _handle_s_trigger()", 1)[1].split(
        "def _handle_r_trigger()",
        1,
    )[0]
    r_handler = source.split("def _handle_r_trigger()", 1)[1].split(
        "def _handle_ignored_retry_key(",
        1,
    )[0]

    assert "_SAFE_STOP_CONTROLLER.handle_sigint()" in source
    assert "coordinator.request_no_save_from_s()" in s_handler
    assert "SKIP_SLEEP_EVENT.set()" not in s_handler
    assert "STOP_NO_SAVE_EVENT.set()" not in s_handler
    assert "coordinator.request_save(" in r_handler
    assert "except KeyboardInterrupt:" in source
    assert "[shutdown] 安全停止完成" in source
    assert "if STOP_NO_SAVE_EVENT.is_set():" in source
    assert "_SAFE_STOP_CONTROLLER.begin_cleanup()" in source


def test_keyboard_listener_stays_alive_until_program_exit():
    source = RECORDER.read_text(encoding="utf-8")
    handler = source.split("def _handle_keyboard_key(ch: str)", 1)[1].split(
        "def _keyboard_listener():",
        1,
    )[0]
    listener = source.split("def _keyboard_listener():", 1)[1].split(
        "def _return_to_start_position(",
        1,
    )[0]

    assert "PROGRAM_EXIT_EVENT = threading.Event()" in source
    assert "run_keyboard_listener(" in listener
    assert "PROGRAM_EXIT_EVENT" in listener
    assert "_handle_keyboard_key" in listener
    assert "router.handle(ch)" in handler
    assert "break" not in handler


def test_recorder_wires_discard_retry_event_and_keyboard_d():
    source = RECORDER.read_text(encoding="utf-8")

    assert "from aloha.keyboard_commands import RecorderKeyRouter" in source
    assert "DISCARD_AND_RETRY_EVENT = threading.Event()" in source
    assert "DISCARD_AND_RETRY_EVENT.clear()" in source
    assert "discard_and_retry=DISCARD_AND_RETRY_EVENT" in source
    assert "def _handle_d_trigger(source: str)" in source

    d_handler = source.split("def _handle_d_trigger(source: str)", 1)[1].split(
        "def _handle_remote_trigger(",
        1,
    )[0]
    keyboard_handler = source.split("def _handle_keyboard_key(ch: str)", 1)[1].split(
        "def _keyboard_listener():",
        1,
    )[0]

    assert "coordinator.handle_d()" in d_handler
    assert "TriggerResult.DISCARD_STARTED" in d_handler
    assert "TriggerResult.NOT_RECORDING" in d_handler
    assert "'d' 放弃当前 attempt" in keyboard_handler
    assert "router.handle(ch)" in keyboard_handler
    assert "RecorderKeyRouter(" in source


def test_main_stops_and_joins_keyboard_listener_on_program_exit():
    source = RECORDER.read_text(encoding="utf-8")
    main_source = source.split("def main(args:", 1)[1].split(
        'if __name__ == "__main__":',
        1,
    )[0]

    assert "PROGRAM_EXIT_EVENT.clear()" in main_source
    assert "finally:" in main_source
    assert "PROGRAM_EXIT_EVENT.set()" in main_source
    assert "kb_thread.join(" in main_source
    assert "kb_thread.is_alive()" in main_source


def test_attempt_cleanup_is_local_and_process_cleanup_is_finalized_once():
    source = RECORDER.read_text(encoding="utf-8")
    capture = source.split("def capture_one_episode(", 1)[1].split(
        "def get_auto_index(",
        1,
    )[0]
    finalizer = source.split("def finalize_recorder_runtime(", 1)[1].split(
        "def capture_one_episode(",
        1,
    )[0]

    assert "discard_attempt(unfinished_attempt)" in capture
    assert "_SAFE_STOP_CONTROLLER.force_no_save(" in capture
    assert "_SAFE_STOP_CONTROLLER.begin_cleanup()" not in capture
    assert "_run_sleep_pose()" not in capture
    assert "robot_shutdown()" not in capture

    begin = finalizer.index("_SAFE_STOP_CONTROLLER.begin_cleanup()")
    drain = finalizer.index("save_worker.drain(", begin)
    quiesce = finalizer.index("_quiesce_recorder_runtime(runtime)", drain)
    supervise = finalizer.index("report = supervise_recovery(", quiesce)
    assert begin < drain < quiesce < supervise
    assert "recover_robots_to_sleep" not in finalizer
    assert "_restore_post_session_gripper_idle_modes" not in finalizer


def test_recorder_handoff_state_is_owned_by_short_recorder_lease():
    source = RECORDER.read_text(encoding="utf-8")
    finalizer = source.split("def finalize_recorder_runtime(", 1)[1].split(
        "def capture_one_episode(",
        1,
    )[0]

    assert "RecoveryIdentity(" in finalizer
    assert 'source="recorder"' in finalizer
    assert '"EXTERNAL_RECOVERY_REQUIRED"' in finalizer
    assert 'publish_for_lease(lease, "UNSAFE_HOLD")' in finalizer
    assert finalizer.count("lease.release()") == 2


def test_main_wires_proceed_separately_from_confirmed_overwrite_authority():
    source = RECORDER.read_text(encoding="utf-8")
    serializer = SERIALIZER.read_text(encoding="utf-8")
    main_source = source.split("def main(args:", 1)[1].split(
        'if __name__ == "__main__":',
        1,
    )[0]
    capture = source.split("def capture_one_episode(", 1)[1].split(
        "def get_auto_index(",
        1,
    )[0]

    assert "episode_decision = check_episode_index(" in main_source
    assert "if not episode_decision.proceed:" in main_source
    assert "allow_existing=index == initial_episode_idx" in main_source
    assert "and episode_decision.allow_existing" in main_source
    assert "allow_existing=allow_existing" in capture
    assert (
        "allow_existing_destination=payload.allow_existing"
        in serializer
    )
    assert "overwrite=overwrite" not in main_source


def test_script_wires_one_shared_command_lock_and_post_save_diagnostic_commit():
    source = RECORDER.read_text(encoding="utf-8")
    serializer = SERIALIZER.read_text(encoding="utf-8")
    capture = source.split("def capture_one_episode(", 1)[1].split(
        "def get_auto_index(",
        1,
    )[0]

    assert "_COMMAND_LOCK = threading.RLock()" in source
    assert "lock=_COMMAND_LOCK" in source
    assert "RecorderCommandCoordinator(" in source
    assert "daemon=False" in source
    assert "join_motion_thread_safely(" in source
    assert "restore_teleop_modes(" in capture
    retry_restore = capture.index("retry teleop mode restoration failed")
    retry_force_no_save = capture.index(
        "_SAFE_STOP_CONTROLLER.force_no_save(",
        retry_restore - 300,
    )
    assert retry_force_no_save > 0
    hdf5_write = serializer.index("_write_hdf5(")
    diagnostic_commit = serializer.index(
        "payload.artifact.commit_into_existing("
    )
    validation = serializer.index("validate_outputs(")
    assert hdf5_write < diagnostic_commit < validation

    diagnostic_worker = source.split(
        "def _motor6_diagnostics_worker(",
        1,
    )[1].split("def _load_random_start_positions(", 1)[0]
    assert "wait_for_diagnostic_interval(" in diagnostic_worker
    assert "time.sleep(" not in diagnostic_worker
    assert diagnostic_worker.count("if stop_event.is_set():") >= 4
    assert "stop_event=stop_event" in diagnostic_worker

    diagnostic_sample = source.split(
        "def _sample_motor6_diag(",
        1,
    )[1].split("def _motor6_diagnostics_worker(", 1)[0]
    assert "sample_registers_interruptibly(" in diagnostic_sample
    assert "stop_event=stop_event" in diagnostic_sample


def test_discard_return_preparation_uses_bounded_fail_closed_services():
    source = RECORDER.read_text(encoding="utf-8")
    return_to_start = source.split(
        "def _return_to_start_position(",
        1,
    )[1].split("def opening_ceremony(", 1)[0]

    assert "from aloha.interbotix_service import (" in source
    assert "prepare_return_modes(" in return_to_start
    assert "set_operating_modes=_set_operating_modes_bounded" in return_to_start
    assert "torque_on=_torque_on_bounded" in return_to_start
    assert "leader {name} 准备失败" not in return_to_start
    assert "follower {name} 准备失败" not in return_to_start
    assert "robot_set_operating_modes" not in return_to_start


def test_motor_diagnostics_are_opt_in_bounded_and_role_aware():
    source = RECORDER.read_text(encoding="utf-8")
    diagnostic_sample = source.split(
        "def _sample_motor6_diag(",
        1,
    )[1].split("def _motor6_diagnostics_worker(", 1)[0]

    assert "from aloha.motor_diagnostics import (" in source
    assert "read_register_values_with_timeout(" in source
    assert "diagnostic_registers_for_robot(robot.core.robot_name)" in diagnostic_sample
    assert "motor6_diagnostics=False" in source
    assert "default=0.5" in source
    assert "opt-in" in source
    assert "motor6_diagnostics: bool = False" in source
    assert "args.get('motor6_diagnostics', False)" in source
    assert 'args.get("motor6_diagnostics", False)' in source
    assert "args.get('motor6_diagnostics_rate_hz', 0.5)" in source
    assert 'args.get("motor6_diagnostics_rate_hz", 0.5)' in source


def test_recorder_finalizer_uses_only_standalone_sleep_recovery():
    source = RECORDER.read_text(encoding="utf-8")
    accepted_pre_sleep = source.split(
        'print("[pre-sleep] leaders detected:"',
        1,
    )[1].split("# ====== 保存策略 ======", 1)[0]

    assert "robot_set_operating_modes" not in accepted_pre_sleep
    assert "_set_operating_modes_bounded(" in accepted_pre_sleep
    assert "_torque_on_bounded(" in accepted_pre_sleep
    assert "def _run_sleep_pose(" not in source
    assert "recover_robots_to_sleep(" not in source
    assert "supervise_external_recovery" in source
    assert 'Path(__file__).with_name("sleep.py")' in source
    assert "subprocess.run(" not in source


def test_s_interrupts_a_blocked_main_thread_in_a_real_process():
    program = textwrap.dedent(
        """
        import signal
        import threading
        import time

        from aloha.safe_stop import SafeStopController

        stop = threading.Event()
        save = threading.Event()
        skip = threading.Event()
        skip.set()
        controller = SafeStopController(stop, save, skip)
        signal.signal(signal.SIGINT, lambda signum, frame: controller.handle_sigint())
        threading.Timer(0.05, controller.request_from_s).start()
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print(f"stopped={stop.is_set()} skip={skip.is_set()}")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=2.0,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "stopped=True skip=False" in result.stdout


def test_double_sigint_keeps_real_rclpy_context_valid():
    pytest.importorskip("rclpy")
    program = textwrap.dedent(
        """
        import signal
        import threading

        import rclpy
        from rclpy.signals import SignalHandlerOptions

        from aloha.safe_sleep_runtime import initialize_ros_context
        from aloha.safe_stop import SafeStopController

        initialize_ros_context(
            ok=rclpy.ok,
            init=rclpy.init,
            no_signal_handlers=SignalHandlerOptions.NO,
        )
        stop = threading.Event()
        save = threading.Event()
        skip = threading.Event()
        controller = SafeStopController(stop, save, skip)
        signal.signal(
            signal.SIGINT,
            lambda *_: controller.handle_sigint(),
        )
        controller.handle_sigint()
        try:
            controller.handle_sigint()
        except KeyboardInterrupt:
            print(f"context_ok={rclpy.ok()}")
        finally:
            rclpy.shutdown()
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=3.0,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "context_ok=True" in result.stdout


def test_robot_runtime_lifecycle_is_process_scoped():
    source = RECORDER.read_text(encoding="utf-8")
    runtime_factory = source.split(
        "def create_recorder_runtime(",
        1,
    )[1].split("def finalize_recorder_runtime(", 1)[0]
    runtime_finalizer = source.split(
        "def finalize_recorder_runtime(",
        1,
    )[1].split("def capture_one_episode(", 1)[0]
    capture = source.split(
        "def capture_one_episode(",
        1,
    )[1].split("def get_auto_index(", 1)[0]
    main = source.split("def main(", 1)[1].split(
        'if __name__ == "__main__":',
        1,
    )[0]

    assert runtime_factory.count('create_interbotix_global_node("aloha")') == 1
    assert runtime_factory.count("make_real_env(") == 1
    assert runtime_factory.count("robot_startup(") == 1
    assert "_SAFE_STOP_CONTROLLER.begin_cleanup()" in runtime_finalizer
    assert "save_worker.drain(timeout=save_drain_timeout_seconds)" in (
        runtime_finalizer
    )
    assert "SAVE_DRAIN_BEFORE_RECOVERY_TIMEOUT_SECONDS" in source
    assert "SAVE_ABORT_TIMEOUT_SECONDS" in source
    assert "_quiesce_recorder_runtime(runtime)" in runtime_finalizer
    assert "report = supervise_recovery(" in runtime_finalizer
    assert "report is None or not report.safe_to_stop" in runtime_finalizer
    assert "candidate = supervise_recovery(" not in runtime_finalizer
    assert "recover_robots_to_sleep" not in runtime_finalizer

    for process_scoped_call in (
        "create_interbotix_global_node(",
        "make_real_env(",
        "robot_startup(",
        "_SAFE_STOP_CONTROLLER.begin_cleanup()",
        "supervise_recovery(",
        "robot_shutdown()",
    ):
        assert process_scoped_call not in capture

    assert main.count("create_recorder_runtime(") == 1
    assert main.count("finalize_recorder_runtime(") == 1


def test_recorder_owns_ros_signals_and_always_externalizes_recovery():
    source = RECORDER.read_text(encoding="utf-8")
    main = source.split("def main(", 1)[1].split(
        'if __name__ == "__main__":',
        1,
    )[0]
    finalizer = source.split(
        "def finalize_recorder_runtime(",
        1,
    )[1].split("def capture_one_episode(", 1)[0]

    assert "SignalHandlerOptions.NO" in main
    assert "initialize_ros_context(" in main
    assert "RecoveryLease.acquire" in finalizer
    assert "EXTERNAL_RECOVERY_REQUIRED" in finalizer
    assert "supervise_external_recovery" in source
    assert "context_ok=False" in finalizer
    assert "recover_robots_to_sleep" not in finalizer
    assert "SIGUSR1" not in main
    assert "Press s" not in finalizer
    assert 'return "恢复反馈后按 s' not in finalizer


def test_recorder_main_wires_exact_outcome_worker_and_bounded_shutdown():
    source = RECORDER.read_text(encoding="utf-8")
    main = source.split("def main(", 1)[1].split(
        'if __name__ == "__main__":',
        1,
    )[0]

    assert "final_cleanup=lambda runtime, outcome:" in main
    assert "outcome=outcome" in main
    assert "save_worker=save_worker" in main
    assert "robot_name=robot_base" in main
    assert "gravity_compensation_active=(" in main
    assert "sleep_required=" not in main
    assert "save_worker.shutdown(" in main
    assert "timeout=SAVE_ABORT_TIMEOUT_SECONDS" in main
    assert "raise_failure=True" in main
    assert "raise_failure=not session_failed" not in main
    assert "except BaseException as shutdown_error:" in main
    assert "[保存] 后台 worker 收尾失败" in main
    assert "set_retry_input_available(sys.stdin.isatty())" in main


def test_capture_records_terminal_episode_identity_for_drain_confirmation():
    source = RECORDER.read_text(encoding="utf-8")
    capture = source.split(
        "def capture_one_episode(",
        1,
    )[1].split("def get_auto_index(", 1)[0]

    assert "runtime.last_saved_episode_name = dataset_name" in capture
    assert "runtime.terminal_save_source =" in capture


def test_recorder_wires_staging_validation_and_continuous_session():
    source = RECORDER.read_text(encoding="utf-8")
    serializer = SERIALIZER.read_text(encoding="utf-8")
    capture = source.split(
        "def capture_one_episode(",
        1,
    )[1].split("def get_auto_index(", 1)[0]
    main = source.split("def main(", 1)[1].split(
        'if __name__ == "__main__":',
        1,
    )[0]

    assert "StagedEpisode.create(" in capture
    assert "staged.staging_path" in capture
    assert "EpisodeSavePayload(" in capture
    assert "handoff_episode_save(" in capture
    assert "validate_episode_outputs" in serializer
    assert "payload.staged.publish(" in serializer
    assert "SessionOutcome.EXIT_DISCARD_AND_SLEEP" in capture
    assert 'return "OK"' not in capture
    assert 'return "ABORT_SAVE"' not in capture
    assert 'return "ABORT_NO_SAVE"' not in capture

    assert "run_continuous_session(" in main
    assert "find_next_available_episode_index(" in main
    assert "allow_existing=index == initial_episode_idx" in main


def test_recorder_enables_trigger_only_after_episode_preparation():
    source = RECORDER.read_text(encoding="utf-8")
    capture = source.split(
        "def capture_one_episode(",
        1,
    )[1].split("def get_auto_index(", 1)[0]

    preparation_index = capture.index("prepare_episode_start(")
    camera_validation_index = capture.index("if none_cameras:")
    enable_trigger_index = capture.index(
        "_TRIGGER_CONTROLLER.complete_preparation("
    )
    attempt_loop_index = capture.index("accepted_attempt = {}")

    assert preparation_index < camera_validation_index
    assert camera_validation_index < enable_trigger_index < attempt_loop_index


def test_recorder_marks_sample_only_after_action_is_appended():
    source = RECORDER.read_text(encoding="utf-8")
    capture = source.split(
        "def capture_one_episode(",
        1,
    )[1].split("def get_auto_index(", 1)[0]

    append_index = capture.index("attempt.actions.append(action)")
    mark_index = capture.index("_TRIGGER_CONTROLLER.mark_sample_recorded()")

    assert append_index < mark_index


def test_recorder_rejects_empty_actions_before_save_worker_handoff():
    source = RECORDER.read_text(encoding="utf-8")
    capture = source.split(
        "def capture_one_episode(",
        1,
    )[1].split("def get_auto_index(", 1)[0]

    empty_guard_index = capture.index("if not actions:")
    payload_index = capture.index("payload = EpisodeSavePayload(")
    submit_index = capture.index("save_worker.submit(", payload_index)
    empty_guard = capture[empty_guard_index:payload_index]

    assert empty_guard_index < payload_index < submit_index
    assert "SessionOutcome.EXIT_DISCARD_AND_SLEEP" in empty_guard


def test_recorder_spools_images_before_retaining_each_timestep():
    source = RECORDER.read_text(encoding="utf-8")
    capture = source.split(
        "def capture_one_episode(",
        1,
    )[1].split("def get_auto_index(", 1)[0]

    assert "EpisodeImageSpoolWriter(" in capture
    assert capture.count("store_attempt_timestep(attempt, ts)") == 2
    assert "strip_and_spool_timestep(writer, timestep)" in capture
    assert "timestep_transform=(" in capture
    seal_index = capture.index("image_spool = spool_writer.seal(")
    payload_index = capture.index("payload = EpisodeSavePayload(")
    submit_index = capture.index("save_worker.submit(", payload_index)
    clear_index = capture.index("attempt.timesteps.clear()", submit_index)

    assert seal_index < payload_index < submit_index < clear_index
    assert "lambda: save_episode(" not in capture
