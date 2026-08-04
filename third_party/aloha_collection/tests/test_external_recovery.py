from pathlib import Path
import signal
import subprocess
import sys

import pytest

import aloha.external_recovery as external_recovery

from aloha.external_recovery import (
    ExternalRecoveryError,
    ExternalRecoverySession,
    run_external_recovery,
    supervise_external_recovery,
)
from aloha.safe_sleep import (
    RobotSleepResult,
    SafeSleepReport,
    SleepStatus,
)


class FakeProcess:
    def __init__(self, pid=321):
        self.pid = pid
        self.command = None
        self.returncode = None
        self.terminate_calls = 0
        self.wait_calls = []
        self.kill_calls = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        self.returncode = -15

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        return self.returncode

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9


def safe_payload(*, recovery_id="abc", owner_pid=321):
    return {
        "schema_version": 2,
        "state": "SAFE_TO_STOP",
        "safe_to_stop": True,
        "recovery_id": recovery_id,
        "owner_pid": owner_pid,
        "source": "standalone",
        "context_ok": True,
        "robots": {
            "leader_left": {
                "status": "slept_verified",
                "phase": "complete",
                "reason": "verified",
                "max_error_rad": 0.01,
                "torque_off_verified": True,
            }
        },
    }


def process_factory_for(process, calls):
    def factory(command, **kwargs):
        process.command = command
        calls.append(kwargs)
        return process

    return factory


def safe_report():
    return SafeSleepReport(
        results={
            "leader_left": RobotSleepResult(
                robot_name="leader_left",
                status=SleepStatus.SLEPT_VERIFIED,
                max_error_rad=0.01,
                reason="verified",
                phase="complete",
                torque_off_verified=True,
            )
        }
    )


def test_supervisor_returns_first_success_without_waiting_for_restart():
    report = safe_report()
    prepared = []
    attempted = []
    waited = []

    def run_attempt(**kwargs):
        attempted.append(kwargs["recovery_id"])
        return report

    result = supervise_external_recovery(
        session=ExternalRecoverySession(),
        robot_name="aloha_stationary",
        gravity_compensation_active=False,
        sleep_script=Path("/repo/scripts/sleep.py"),
        retry_requested=lambda _timeout: False,
        prepare_attempt=prepared.append,
        wait_for_restart=lambda recovery_id, exc: waited.append(
            (recovery_id, exc)
        ),
        recovery_id_factory=lambda: "attempt-1",
        run_attempt=run_attempt,
    )

    assert result is report
    assert prepared == ["attempt-1"]
    assert attempted == ["attempt-1"]
    assert waited == []


def test_supervisor_preserves_pose_policy_without_retrying_failed_child():
    policies = []

    def run_attempt(**kwargs):
        policies.append(kwargs["allow_pose_deviation"])
        raise ExternalRecoveryError("first child exited")

    with pytest.raises(ExternalRecoveryError, match="first child exited"):
        supervise_external_recovery(
            session=ExternalRecoverySession(),
            robot_name="aloha_stationary",
            gravity_compensation_active=False,
            allow_pose_deviation=True,
            sleep_script=Path("/repo/scripts/sleep.py"),
            retry_requested=lambda _timeout: True,
            prepare_attempt=lambda _recovery_id: None,
            wait_for_restart=lambda *_args: pytest.fail("must not restart"),
            recovery_id_factory=lambda: "attempt-1",
            run_attempt=run_attempt,
            logger=lambda _message: None,
        )

    assert policies == [True]


def test_supervisor_does_not_create_fresh_id_after_child_exit():
    prepared = []
    attempted = []
    waited = []
    id_calls = []

    def prepare_attempt(recovery_id):
        prepared.append(recovery_id)

    def run_attempt(**kwargs):
        attempted.append(kwargs["recovery_id"])
        raise ExternalRecoveryError("child exited")

    def wait_for_restart(recovery_id, exc):
        waited.append((recovery_id, exc))

    def recovery_id_factory():
        if id_calls:
            pytest.fail("supervisor attempted a second recovery ID")
        recovery_id = f"attempt-{len(id_calls) + 1}"
        id_calls.append(recovery_id)
        return recovery_id

    with pytest.raises(ExternalRecoveryError, match="child exited"):
        supervise_external_recovery(
            session=ExternalRecoverySession(),
            robot_name="aloha_stationary",
            gravity_compensation_active=False,
            sleep_script=Path("/repo/scripts/sleep.py"),
            retry_requested=lambda _timeout: True,
            prepare_attempt=prepare_attempt,
            wait_for_restart=wait_for_restart,
            recovery_id_factory=recovery_id_factory,
            run_attempt=run_attempt,
        )

    assert prepared == ["attempt-1"]
    assert attempted == ["attempt-1"]
    assert id_calls == ["attempt-1"]
    assert waited == []


def test_supervisor_propagates_failure_when_logger_breaks():
    attempted = []
    waited = []
    id_calls = []

    def run_attempt(**kwargs):
        attempted.append(kwargs["recovery_id"])
        raise ExternalRecoveryError("child exited")

    def broken_logger(_message):
        raise RuntimeError("logger broke")

    def recovery_id_factory():
        if id_calls:
            pytest.fail("supervisor attempted a second recovery ID")
        id_calls.append("attempt-1")
        return "attempt-1"

    with pytest.raises(ExternalRecoveryError, match="child exited"):
        supervise_external_recovery(
            session=ExternalRecoverySession(),
            robot_name="aloha_stationary",
            gravity_compensation_active=False,
            sleep_script=Path("/repo/scripts/sleep.py"),
            retry_requested=lambda _timeout: True,
            prepare_attempt=lambda _recovery_id: None,
            wait_for_restart=lambda recovery_id, exc: waited.append(
                (recovery_id, exc)
            ),
            recovery_id_factory=recovery_id_factory,
            run_attempt=run_attempt,
            logger=broken_logger,
        )

    assert attempted == ["attempt-1"]
    assert id_calls == ["attempt-1"]
    assert waited == []


def test_supervisor_log_is_single_line_and_bounded():
    recovery_id = "attempt-\n injected-" + "x" * 600
    messages = []

    def run_attempt(**kwargs):
        raise ExternalRecoveryError(
            "child\n exited " + "y" * 600
        )

    with pytest.raises(ExternalRecoveryError, match="child"):
        supervise_external_recovery(
            session=ExternalRecoverySession(),
            robot_name="aloha_stationary",
            gravity_compensation_active=False,
            sleep_script=Path("/repo/scripts/sleep.py"),
            retry_requested=lambda _timeout: True,
            prepare_attempt=lambda _recovery_id: None,
            wait_for_restart=lambda *_args: pytest.fail("must not restart"),
            recovery_id_factory=lambda: recovery_id,
            run_attempt=run_attempt,
            logger=messages.append,
        )

    assert len(messages) == 1
    assert "\n" not in messages[0]
    assert "attempt- injected-" in messages[0]
    assert len(messages[0]) <= 512


def test_supervisor_propagates_prepare_attempt_error_without_running():
    error = RuntimeError("prepare failed")
    attempted = []

    def prepare_attempt(_recovery_id):
        raise error

    with pytest.raises(RuntimeError) as caught:
        supervise_external_recovery(
            session=ExternalRecoverySession(),
            robot_name="aloha_stationary",
            gravity_compensation_active=False,
            sleep_script=Path("/repo/scripts/sleep.py"),
            retry_requested=lambda _timeout: False,
            prepare_attempt=prepare_attempt,
            wait_for_restart=lambda _recovery_id, _exc: None,
            recovery_id_factory=lambda: "attempt-1",
            run_attempt=lambda **kwargs: attempted.append(kwargs),
        )

    assert caught.value is error
    assert attempted == []


def test_supervisor_never_calls_restart_wait_after_child_error():
    prepared = []
    attempted = []
    waited = []

    def run_attempt(**kwargs):
        attempted.append(kwargs["recovery_id"])
        raise ExternalRecoveryError("child exited")

    def wait_for_restart(_recovery_id, _exc):
        waited.append((_recovery_id, _exc))

    with pytest.raises(ExternalRecoveryError, match="child exited"):
        supervise_external_recovery(
            session=ExternalRecoverySession(),
            robot_name="aloha_stationary",
            gravity_compensation_active=False,
            sleep_script=Path("/repo/scripts/sleep.py"),
            retry_requested=lambda _timeout: False,
            prepare_attempt=prepared.append,
            wait_for_restart=wait_for_restart,
            recovery_id_factory=lambda: "attempt-1",
            run_attempt=run_attempt,
            logger=lambda _message: None,
        )

    assert prepared == ["attempt-1"]
    assert attempted == ["attempt-1"]
    assert waited == []


def test_supervisor_propagates_non_recovery_error_without_retrying():
    error = ValueError("unexpected failure")
    prepared = []
    attempted = []
    waited = []

    def run_attempt(**kwargs):
        attempted.append(kwargs["recovery_id"])
        raise error

    with pytest.raises(ValueError) as caught:
        supervise_external_recovery(
            session=ExternalRecoverySession(),
            robot_name="aloha_stationary",
            gravity_compensation_active=False,
            sleep_script=Path("/repo/scripts/sleep.py"),
            retry_requested=lambda _timeout: False,
            prepare_attempt=prepared.append,
            wait_for_restart=lambda recovery_id, exc: waited.append(
                (recovery_id, exc)
            ),
            recovery_id_factory=lambda: "attempt-1",
            run_attempt=run_attempt,
        )

    assert caught.value is error
    assert prepared == ["attempt-1"]
    assert attempted == ["attempt-1"]
    assert waited == []


def test_external_recovery_is_unbuffered_and_accepts_only_child_state(
    monkeypatch,
):
    monkeypatch.setenv("PYTHONPATH", "/installed/aloha")
    process = FakeProcess()
    factory_calls = []

    report = run_external_recovery(
        session=ExternalRecoverySession(),
        robot_name="aloha_stationary",
        recovery_id="abc",
        gravity_compensation_active=False,
        sleep_script=Path("/repo/scripts/sleep.py"),
        process_factory=process_factory_for(
            process,
            factory_calls,
        ),
        read_state=lambda: safe_payload(),
        retry_requested=lambda _timeout: False,
    )

    assert process.command == [
        sys.executable,
        "-u",
        "/repo/scripts/sleep.py",
        "--all",
        "--robot",
        "aloha_stationary",
        "--recovery-id",
        "abc",
    ]
    assert factory_calls[0]["stdin"] is not None
    child_pythonpath = factory_calls[0]["env"]["PYTHONPATH"].split(
        external_recovery.os.pathsep
    )
    assert child_pythonpath[:2] == ["/repo", "/installed/aloha"]
    assert report.safe_to_stop
    assert report.results["leader_left"].torque_off_verified
    assert process.terminate_calls == 0


def test_external_recovery_logs_real_child_pid_after_process_start():
    process = FakeProcess(pid=7654)
    messages = []

    report = run_external_recovery(
        session=ExternalRecoverySession(),
        robot_name="aloha_stationary",
        recovery_id="attempt-42",
        gravity_compensation_active=False,
        sleep_script=Path("/repo/scripts/sleep.py"),
        process_factory=process_factory_for(process, []),
        read_state=lambda: safe_payload(
            recovery_id="attempt-42",
            owner_pid=7654,
        ),
        retry_requested=lambda _timeout: False,
        logger=messages.append,
    )

    assert report.safe_to_stop
    assert messages == [
        "[handoff] 已启动独立 safe-sleep: pid=7654 "
        "recovery_id=attempt-42."
    ]


def test_explicit_gravity_state_is_forwarded_only_when_active():
    process = FakeProcess()

    run_external_recovery(
        session=ExternalRecoverySession(),
        robot_name="aloha_stationary",
        recovery_id="abc",
        gravity_compensation_active=True,
        sleep_script=Path("/repo/scripts/sleep.py"),
        process_factory=process_factory_for(process, []),
        read_state=lambda: safe_payload(),
        retry_requested=lambda _timeout: False,
    )

    assert process.command[-1] == "--gravity-compensation-active"


def test_pose_deviation_policy_is_forwarded_only_when_enabled():
    process = FakeProcess()

    run_external_recovery(
        session=ExternalRecoverySession(),
        robot_name="aloha_stationary",
        recovery_id="abc",
        gravity_compensation_active=False,
        allow_pose_deviation=True,
        sleep_script=Path("/repo/scripts/sleep.py"),
        process_factory=process_factory_for(process, []),
        read_state=lambda: safe_payload(),
        retry_requested=lambda _timeout: False,
    )

    assert process.command[-1] == "--allow-pose-deviation"


@pytest.mark.parametrize(
    "payload,match",
    [
        (safe_payload(recovery_id="other"), "recovery_id"),
        (safe_payload(owner_pid=999), "owner_pid"),
    ],
)
def test_external_recovery_rejects_mismatched_identity(payload, match):
    process = FakeProcess()

    with pytest.raises(ExternalRecoveryError, match=match):
        run_external_recovery(
            session=ExternalRecoverySession(),
            robot_name="aloha_stationary",
            recovery_id="abc",
            gravity_compensation_active=False,
            sleep_script=Path("/repo/scripts/sleep.py"),
            process_factory=process_factory_for(process, []),
            read_state=lambda: payload,
            retry_requested=lambda _timeout: False,
        )


def test_unsafe_hold_is_not_forwarded_as_retry_to_child():
    process = FakeProcess()
    sent = []
    retry_calls = []

    def unsafe_then_exit():
        process.returncode = 2
        return {
            **safe_payload(),
            "state": "UNSAFE_HOLD",
            "safe_to_stop": False,
        }

    def request_retry_once(timeout):
        retry_calls.append(timeout)
        return len(retry_calls) == 1

    with pytest.raises(ExternalRecoveryError, match="returncode=2"):
        run_external_recovery(
            session=ExternalRecoverySession(),
            robot_name="aloha_stationary",
            recovery_id="abc",
            gravity_compensation_active=False,
            sleep_script=Path("/repo/scripts/sleep.py"),
            process_factory=process_factory_for(process, []),
            read_state=unsafe_then_exit,
            retry_requested=request_retry_once,
            signal_process=lambda pid, sig: sent.append((pid, sig)),
        )

    assert sent == []
    assert retry_calls == []


def test_child_exit_before_safe_state_fails_closed():
    process = FakeProcess()
    process.returncode = 2

    with pytest.raises(ExternalRecoveryError, match="exited"):
        run_external_recovery(
            session=ExternalRecoverySession(),
            robot_name="aloha_stationary",
            recovery_id="abc",
            gravity_compensation_active=False,
            sleep_script=Path("/repo/scripts/sleep.py"),
            process_factory=process_factory_for(process, []),
            read_state=lambda: {
                **safe_payload(),
                "state": "RECOVERY_IN_PROGRESS",
                "safe_to_stop": False,
            },
            retry_requested=lambda _timeout: False,
        )


def test_popen_oserror_is_normalized_for_supervisor_retry():
    with pytest.raises(ExternalRecoveryError, match="start.*denied"):
        run_external_recovery(
            session=ExternalRecoverySession(),
            robot_name="aloha_stationary",
            recovery_id="abc",
            gravity_compensation_active=False,
            sleep_script=Path("/repo/scripts/sleep.py"),
            process_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("denied")
            ),
            retry_requested=lambda _timeout: False,
        )


def test_bounded_best_effort_log_is_single_line_and_ignores_logger_exception():
    messages = []
    external_recovery.bounded_best_effort_log(
        messages.append,
        "failure\n" + "x" * 700,
    )
    external_recovery.bounded_best_effort_log(
        lambda _message: (_ for _ in ()).throw(RuntimeError("logger broke")),
        "ignored",
    )

    assert len(messages) == 1
    assert "\n" not in messages[0]
    assert len(messages[0]) <= 512


def test_spawned_child_is_terminated_and_reaped_on_non_safe_exception():
    process = FakeProcess()
    error = RuntimeError("read state failed")

    with pytest.raises(RuntimeError) as raised:
        run_external_recovery(
            session=ExternalRecoverySession(),
            robot_name="aloha_stationary",
            recovery_id="abc",
            gravity_compensation_active=False,
            sleep_script=Path("/repo/scripts/sleep.py"),
            process_factory=process_factory_for(process, []),
            read_state=lambda: (_ for _ in ()).throw(error),
            retry_requested=lambda _timeout: False,
            process_stop_timeout_seconds=0.1,
        )

    assert raised.value is error
    assert process.terminate_calls == 1
    assert process.wait_calls == [0.1]


def test_child_cleanup_escalates_to_kill_and_reports_cleanup_error():
    class BrokenTerminateProcess(FakeProcess):
        def terminate(self):
            self.terminate_calls += 1
            raise OSError("terminate failed")

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            if not self.kill_calls:
                raise TimeoutError("still alive")
            return self.returncode

    process = BrokenTerminateProcess()

    with pytest.raises(
        ExternalRecoveryError,
        match="cleanup.*terminate failed",
    ):
        run_external_recovery(
            session=ExternalRecoverySession(),
            robot_name="aloha_stationary",
            recovery_id="abc",
            gravity_compensation_active=False,
            sleep_script=Path("/repo/scripts/sleep.py"),
            process_factory=process_factory_for(process, []),
            read_state=lambda: (_ for _ in ()).throw(
                RuntimeError("read failed")
            ),
            retry_requested=lambda _timeout: False,
            process_stop_timeout_seconds=0.0,
        )

    assert process.kill_calls == 1
    assert len(process.wait_calls) == 2


def test_supervisor_never_respawns_until_stubborn_child_is_reaped():
    calls = []
    ids = iter(("attempt-1", "attempt-2"))
    id_calls = []

    class StubbornProcess(FakeProcess):
        def __init__(self):
            super().__init__(pid=701)
            self.allowed_to_exit = False

        def poll(self):
            return -9 if self.allowed_to_exit else None

        def terminate(self):
            self.terminate_calls += 1
            if self.terminate_calls == 1:
                raise OSError("terminate callback failed")

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            if not self.allowed_to_exit:
                raise subprocess.TimeoutExpired("sleep.py", timeout)
            return -9

        def kill(self):
            self.kill_calls += 1

    stubborn = StubbornProcess()
    safe_child = FakeProcess(pid=702)
    factory_calls = []

    def process_factory(command, **_kwargs):
        factory_calls.append(command)
        return stubborn if len(factory_calls) == 1 else safe_child

    def recovery_id_factory():
        recovery_id = next(ids)
        id_calls.append(recovery_id)
        return recovery_id

    retry_count = {"value": 0}

    def retry_requested(timeout):
        calls.append(("retry", timeout))
        retry_count["value"] += 1
        if retry_count["value"] == 2:
            stubborn.allowed_to_exit = True
        return True

    def run_attempt(**kwargs):
        recovery_id = kwargs["recovery_id"]
        return run_external_recovery(
            **kwargs,
            process_factory=process_factory,
            read_state=(
                (lambda: (_ for _ in ()).throw(RuntimeError("state failed")))
                if recovery_id == "attempt-1"
                else lambda: safe_payload(
                    recovery_id="attempt-2",
                    owner_pid=safe_child.pid,
                )
            ),
            process_stop_timeout_seconds=0.0,
        )

    with pytest.raises(
        ExternalRecoveryError,
        match="without SAFE_TO_STOP",
    ):
        supervise_external_recovery(
            session=ExternalRecoverySession(),
            robot_name="aloha_stationary",
            gravity_compensation_active=False,
            sleep_script=Path("/repo/scripts/sleep.py"),
            retry_requested=retry_requested,
            prepare_attempt=lambda recovery_id: calls.append(
                ("prepare", recovery_id)
            ),
            wait_for_restart=lambda *_args: pytest.fail(
                "still-live child must not enter ordinary respawn wait"
            ),
            recovery_id_factory=recovery_id_factory,
            run_attempt=run_attempt,
            logger=lambda message: calls.append(("log", message)),
        )

    assert len(factory_calls) == 1
    assert id_calls == ["attempt-1"]
    assert calls.count(("prepare", "attempt-1")) == 1
    assert calls.count(("prepare", "attempt-2")) == 0
    assert retry_count["value"] == 2
    assert stubborn.kill_calls >= 2
    assert all(
        "\n" not in item[1] and len(item[1]) <= 512
        for item in calls
        if item[0] == "log"
    )


class SessionStubbornProcess(FakeProcess):
    def __init__(self, pid):
        super().__init__(pid=pid)
        self.allowed_to_exit = False

    def poll(self):
        return -9 if self.allowed_to_exit else None

    def terminate(self):
        self.terminate_calls += 1

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if not self.allowed_to_exit:
            raise subprocess.TimeoutExpired("sleep.py", timeout)
        return -9

    def kill(self):
        self.kill_calls += 1


def _session_supervisor_kwargs(session, *, retry_requested, run_attempt, logger):
    return dict(
        robot_name="aloha_stationary",
        gravity_compensation_active=False,
        sleep_script=Path("/repo/scripts/sleep.py"),
        retry_requested=retry_requested,
        prepare_attempt=lambda _recovery_id: None,
        wait_for_restart=lambda *_args: pytest.fail(
            "live child must not enter ordinary restart"
        ),
        run_attempt=run_attempt,
        logger=logger,
        session=session,
        still_alive_retry_poll_seconds=0.0,
    )


def test_session_preserves_live_child_when_logger_interrupts_across_calls():
    session = external_recovery.ExternalRecoverySession()
    stubborn = SessionStubbornProcess(pid=801)
    safe_child = FakeProcess(pid=802)
    processes = iter((stubborn, safe_child))
    factory_calls = []
    id_calls = []
    ids = iter(("attempt-1", "attempt-2"))

    def process_factory(command, **_kwargs):
        factory_calls.append(command)
        return next(processes)

    def recovery_id_factory():
        recovery_id = next(ids)
        id_calls.append(recovery_id)
        return recovery_id

    def run_attempt(**kwargs):
        recovery_id = kwargs["recovery_id"]
        return run_external_recovery(
            **kwargs,
            process_factory=process_factory,
            read_state=(
                (lambda: (_ for _ in ()).throw(RuntimeError("state failed")))
                if recovery_id == "attempt-1"
                else lambda: safe_payload(
                    recovery_id="attempt-2",
                    owner_pid=safe_child.pid,
                )
            ),
            process_stop_timeout_seconds=0.0,
        )

    kwargs = _session_supervisor_kwargs(
        session,
        retry_requested=lambda _timeout: True,
        run_attempt=run_attempt,
        logger=lambda _message: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    kwargs["recovery_id_factory"] = recovery_id_factory

    with pytest.raises(KeyboardInterrupt):
        supervise_external_recovery(**kwargs)

    assert session.active.process is stubborn
    assert session.active.recovery_id == "attempt-1"
    assert len(factory_calls) == 1
    assert id_calls == ["attempt-1"]

    stubborn.allowed_to_exit = True
    kwargs["logger"] = lambda _message: None
    with pytest.raises(
        ExternalRecoveryError,
        match="without SAFE_TO_STOP",
    ):
        supervise_external_recovery(**kwargs)

    assert len(factory_calls) == 1
    assert id_calls == ["attempt-1"]


def test_session_preserves_live_child_when_retry_backoff_exits_across_calls():
    session = external_recovery.ExternalRecoverySession()
    stubborn = SessionStubbornProcess(pid=811)
    safe_child = FakeProcess(pid=812)
    processes = iter((stubborn, safe_child))
    factory_calls = []
    id_calls = []
    ids = iter(("attempt-1", "attempt-2"))

    def process_factory(command, **_kwargs):
        factory_calls.append(command)
        return next(processes)

    def recovery_id_factory():
        recovery_id = next(ids)
        id_calls.append(recovery_id)
        return recovery_id

    def run_attempt(**kwargs):
        recovery_id = kwargs["recovery_id"]
        return run_external_recovery(
            **kwargs,
            process_factory=process_factory,
            read_state=(
                (lambda: (_ for _ in ()).throw(RuntimeError("state failed")))
                if recovery_id == "attempt-1"
                else lambda: safe_payload(
                    recovery_id="attempt-2",
                    owner_pid=safe_child.pid,
                )
            ),
            process_stop_timeout_seconds=0.0,
        )

    kwargs = _session_supervisor_kwargs(
        session,
        retry_requested=lambda _timeout: (_ for _ in ()).throw(
            RuntimeError("retry unavailable")
        ),
        run_attempt=run_attempt,
        logger=lambda _message: None,
    )
    kwargs["recovery_id_factory"] = recovery_id_factory
    kwargs["retry_backoff_sleep"] = lambda _timeout: (_ for _ in ()).throw(
        SystemExit("backoff interrupted")
    )

    with pytest.raises(SystemExit, match="backoff interrupted"):
        supervise_external_recovery(**kwargs)

    assert session.active.process is stubborn
    assert session.active.recovery_id == "attempt-1"
    assert len(factory_calls) == 1
    assert id_calls == ["attempt-1"]

    stubborn.allowed_to_exit = True
    kwargs["retry_requested"] = lambda _timeout: True
    kwargs["retry_backoff_sleep"] = lambda _timeout: None
    with pytest.raises(
        ExternalRecoveryError,
        match="without SAFE_TO_STOP",
    ):
        supervise_external_recovery(**kwargs)

    assert len(factory_calls) == 1
    assert id_calls == ["attempt-1"]


def test_safe_session_reentry_returns_cached_report_without_respawn():
    session = external_recovery.ExternalRecoverySession()
    child = FakeProcess(pid=821)
    factory_calls = []
    id_calls = []

    def run_attempt(**kwargs):
        return run_external_recovery(
            **kwargs,
            process_factory=process_factory_for(child, factory_calls),
            read_state=lambda: safe_payload(
                recovery_id=kwargs["recovery_id"],
                owner_pid=child.pid,
            ),
        )

    kwargs = _session_supervisor_kwargs(
        session,
        retry_requested=lambda _timeout: False,
        run_attempt=run_attempt,
        logger=lambda _message: None,
    )
    kwargs["recovery_id_factory"] = lambda: (
        id_calls.append("attempt-1") or "attempt-1"
    )

    first = supervise_external_recovery(**kwargs)
    kwargs["recovery_id_factory"] = lambda: pytest.fail("must not create new ID")
    kwargs["prepare_attempt"] = lambda _recovery_id: pytest.fail(
        "must not prepare"
    )
    kwargs["run_attempt"] = lambda **_kwargs: pytest.fail("must not run")
    second = supervise_external_recovery(**kwargs)

    assert second is first
    assert len(factory_calls) == 1
    assert id_calls == ["attempt-1"]


def test_session_rejects_stale_attempt_token_clear():
    session = external_recovery.ExternalRecoverySession()
    first = session.start_process(
        recovery_id="attempt-1",
        stop_timeout_seconds=0.0,
        process_factory=lambda *_args, **_kwargs: FakeProcess(pid=831),
        command=["sleep.py"],
    )
    session.mark_reaped(first.token)
    second = session.start_process(
        recovery_id="attempt-2",
        stop_timeout_seconds=0.0,
        process_factory=lambda *_args, **_kwargs: FakeProcess(pid=832),
        command=["sleep.py"],
    )

    with pytest.raises(RuntimeError, match="stale recovery attempt"):
        session.mark_reaped(first.token)

    assert session.active is second


def test_session_validates_timeout_before_process_factory():
    session = external_recovery.ExternalRecoverySession()
    factory_calls = []

    class InvalidTimeout:
        def __float__(self):
            raise ValueError("invalid timeout")

    with pytest.raises(ValueError, match="invalid timeout"):
        session.start_process(
            recovery_id="attempt-1",
            stop_timeout_seconds=InvalidTimeout(),
            process_factory=lambda *_args, **_kwargs: (
                factory_calls.append("factory") or FakeProcess(pid=841)
            ),
            command=["sleep.py"],
        )

    assert factory_calls == []
    assert session.active is None


@pytest.mark.parametrize(
    "factory_error",
    (
        KeyboardInterrupt("interrupted after possible spawn"),
        SystemExit("exited after possible spawn"),
    ),
)
def test_session_reserves_attempt_before_factory_base_exception(factory_error):
    session = external_recovery.ExternalRecoverySession()
    factory_calls = []

    def interrupted_factory(*_args, **_kwargs):
        factory_calls.append("attempt-1")
        raise factory_error

    with pytest.raises(type(factory_error), match="possible spawn"):
        session.start_process(
            recovery_id="attempt-1",
            stop_timeout_seconds=0.0,
            process_factory=interrupted_factory,
            command=["sleep.py"],
        )

    assert session.active is not None
    assert session.active.recovery_id == "attempt-1"
    assert session.active.process is None

    with pytest.raises(RuntimeError, match="session owns an attempt"):
        session.start_process(
            recovery_id="attempt-2",
            stop_timeout_seconds=0.0,
            process_factory=lambda *_args, **_kwargs: factory_calls.append(
                "attempt-2"
            ),
            command=["sleep.py"],
        )

    assert factory_calls == ["attempt-1"]


def test_spawning_session_reentry_never_creates_another_attempt():
    session = external_recovery.ExternalRecoverySession()
    id_calls = []
    prepared = []
    factory_calls = []

    def recovery_id_factory():
        recovery_id = f"attempt-{len(id_calls) + 1}"
        id_calls.append(recovery_id)
        return recovery_id

    def run_attempt(**kwargs):
        return run_external_recovery(
            **kwargs,
            process_factory=lambda *_args, **_kwargs: (
                factory_calls.append(kwargs["recovery_id"])
                or (_ for _ in ()).throw(
                    KeyboardInterrupt("factory interrupted")
                )
            ),
            read_state=lambda: pytest.fail("factory must not return"),
        )

    kwargs = _session_supervisor_kwargs(
        session,
        retry_requested=lambda _timeout: True,
        run_attempt=run_attempt,
        logger=lambda _message: None,
    )
    kwargs["recovery_id_factory"] = recovery_id_factory
    kwargs["prepare_attempt"] = prepared.append

    with pytest.raises(KeyboardInterrupt, match="factory interrupted"):
        supervise_external_recovery(**kwargs)

    assert session.active is not None
    assert session.active.process is None

    messages = []

    def interrupted_logger(message):
        messages.append(message)
        raise KeyboardInterrupt("still spawning")

    kwargs["logger"] = interrupted_logger
    for _ in range(2):
        with pytest.raises(KeyboardInterrupt, match="still spawning"):
            supervise_external_recovery(**kwargs)

    assert id_calls == ["attempt-1"]
    assert prepared == ["attempt-1"]
    assert factory_calls == ["attempt-1"]
    assert len(messages) == 2
    assert all("child identity unavailable" in message for message in messages)
    assert all("\n" not in message and len(message) <= 512 for message in messages)
