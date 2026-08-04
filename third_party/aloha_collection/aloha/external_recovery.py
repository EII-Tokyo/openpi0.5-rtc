"""Fresh-process recovery fallback for an invalid parent ROS context."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable

from aloha.safe_sleep import (
    RobotSleepResult,
    SafeSleepReport,
    SleepStatus,
)
from aloha.safety_state import read_safety_state


class ExternalRecoveryError(RuntimeError):
    """Raised when a fresh recovery process cannot prove a safe result."""


@dataclass(frozen=True)
class ProcessStopResult:
    reaped: bool
    failures: tuple[tuple[str, BaseException], ...]


@dataclass(frozen=True)
class ActiveRecoveryAttempt:
    token: object
    recovery_id: str
    process: object | None
    stop_timeout_seconds: float
    phase: str


class ExternalRecoverySession:
    """Persist recovery-child ownership across supervisor invocations."""

    def __init__(self) -> None:
        self._active: ActiveRecoveryAttempt | None = None
        self._safe_report: SafeSleepReport | None = None

    @property
    def active(self) -> ActiveRecoveryAttempt | None:
        return self._active

    @property
    def safe_report(self) -> SafeSleepReport | None:
        return self._safe_report

    def start_process(
        self,
        *,
        recovery_id: str,
        stop_timeout_seconds: float,
        process_factory: Callable[..., object],
        command: list[str],
        attempt_token: object | None = None,
        **process_kwargs,
    ) -> ActiveRecoveryAttempt:
        stop_timeout = max(0.0, float(stop_timeout_seconds))
        if attempt_token is None:
            attempt = self.reserve_attempt(recovery_id)
        else:
            attempt = self._require_active_token(attempt_token)
            if attempt.recovery_id != recovery_id:
                raise RuntimeError("recovery attempt ID mismatch")
        if attempt.phase != "PREPARING":
            raise RuntimeError("recovery attempt is not ready to spawn")
        spawning = ActiveRecoveryAttempt(
            token=attempt.token,
            recovery_id=recovery_id,
            process=None,
            stop_timeout_seconds=stop_timeout,
            phase="SPAWNING",
        )
        self._active = spawning
        process = process_factory(command, **process_kwargs)
        running = ActiveRecoveryAttempt(
            token=attempt.token,
            recovery_id=recovery_id,
            process=process,
            stop_timeout_seconds=stop_timeout,
            phase="RUNNING",
        )
        self._active = running
        return running

    def reserve_attempt(self, recovery_id: str) -> ActiveRecoveryAttempt:
        if self._safe_report is not None or self._active is not None:
            raise RuntimeError(
                "cannot start a recovery child while a session owns an attempt"
            )
        attempt = ActiveRecoveryAttempt(
            token=object(),
            recovery_id=recovery_id,
            process=None,
            stop_timeout_seconds=0.0,
            phase="PREPARING",
        )
        self._active = attempt
        return attempt

    def cancel_preparing(self, token: object) -> None:
        active = self._require_active_token(token)
        if active.phase != "PREPARING" or active.process is not None:
            raise RuntimeError("cannot cancel a recovery attempt after spawn")
        self._active = None

    def _require_active_token(self, token: object) -> ActiveRecoveryAttempt:
        active = self._active
        if active is None or active.token is not token:
            raise RuntimeError("stale recovery attempt token")
        return active

    def mark_reaped(self, token: object) -> None:
        active = self._require_active_token(token)
        if active.process is None:
            raise RuntimeError("cannot mark an unidentified process reaped")
        self._active = None

    def mark_safe(self, token: object, report: SafeSleepReport) -> None:
        self._require_active_token(token)
        self._safe_report = report


class ExternalRecoveryChildStillAlive(ExternalRecoveryError):
    """A recovery child survived bounded cleanup and still owns its ID."""

    def __init__(
        self,
        *,
        process,
        attempt_token: object,
        recovery_id: str,
        stop_timeout_seconds: float,
        original_error: BaseException,
        stop_result: ProcessStopResult,
    ) -> None:
        self.process = process
        self.attempt_token = attempt_token
        self.recovery_id = recovery_id
        self.stop_timeout_seconds = stop_timeout_seconds
        self.original_error = original_error
        self.stop_result = stop_result
        super().__init__(
            "standalone recovery child is still alive after bounded cleanup: "
            f"recovery_id={recovery_id}, pid={getattr(process, 'pid', None)}"
        )


def bounded_best_effort_log(
    logger: Callable[[str], None],
    message: object,
    *,
    limit: int = 512,
) -> None:
    """Emit one bounded line; ordinary logger failures never alter control."""

    try:
        bounded = " ".join(str(message).split())[: max(0, int(limit))]
        logger(bounded)
    except Exception:
        pass


def _stop_process_bounded(process, *, timeout: float) -> ProcessStopResult:
    """Best-effort terminate, escalate, and reap a spawned recovery child."""

    failures: list[tuple[str, BaseException]] = []

    def attempt(stage, operation):
        try:
            return operation()
        except BaseException as exc:
            failures.append((stage, exc))
            return None

    returncode = attempt("poll", process.poll)
    if returncode is not None:
        return ProcessStopResult(True, tuple(failures))

    attempt("terminate", process.terminate)
    reaped = False
    try:
        wait_result = process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, TimeoutError) as exc:
        failures.append(("wait_timeout", exc))
        attempt("kill", process.kill)
        wait_result = attempt(
            "wait_after_kill",
            lambda: process.wait(timeout=timeout),
        )
        reaped = wait_result is not None
    except BaseException as exc:
        failures.append(("wait", exc))
        attempt("kill", process.kill)
        wait_result = attempt(
            "wait_after_kill",
            lambda: process.wait(timeout=timeout),
        )
        reaped = wait_result is not None
    else:
        reaped = wait_result is not None
        if not reaped:
            attempt("kill", process.kill)
            wait_result = attempt(
                "wait_after_kill",
                lambda: process.wait(timeout=timeout),
            )
            reaped = wait_result is not None

    final_returncode = attempt("poll_after_cleanup", process.poll)
    if final_returncode is not None:
        reaped = True
    return ProcessStopResult(reaped, tuple(failures))


def _report_from_safe_payload(payload: dict) -> SafeSleepReport:
    robots = payload.get("robots")
    if not isinstance(robots, dict) or not robots:
        raise ExternalRecoveryError(
            "SAFE_TO_STOP state has no robot results"
        )
    results = {}
    for robot_name, item in robots.items():
        try:
            status = SleepStatus(item["status"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExternalRecoveryError(
                f"{robot_name} has invalid recovery status"
            ) from exc
        result = RobotSleepResult(
            robot_name=robot_name,
            status=status,
            max_error_rad=item.get("max_error_rad"),
            reason=str(item.get("reason", "")),
            phase=str(item.get("phase", "unknown")),
            torque_off_verified=(
                item.get("torque_off_verified") is True
            ),
        )
        results[robot_name] = result
    report = SafeSleepReport(results=results)
    if not report.safe_to_stop:
        raise ExternalRecoveryError(
            "SAFE_TO_STOP payload contains an unverified robot"
        )
    return report


def run_external_recovery(
    *,
    robot_name: str,
    recovery_id: str,
    session: ExternalRecoverySession,
    gravity_compensation_active: bool,
    sleep_script: Path,
    allow_pose_deviation: bool = False,
    process_factory: Callable[..., object] = subprocess.Popen,
    read_state: Callable[[], dict] = read_safety_state,
    retry_requested: Callable[[float], bool],
    signal_process: Callable[[int, int], None] = os.kill,
    poll_seconds: float = 1.0,
    process_stop_timeout_seconds: float = 5.0,
    attempt_token: object | None = None,
    logger: Callable[[str], None] = print,
) -> SafeSleepReport:
    """Wait for one matching standalone child to prove every arm safe."""

    command = [
        sys.executable,
        "-u",
        str(sleep_script),
        "--all",
        "--robot",
        robot_name,
        "--recovery-id",
        recovery_id,
    ]
    if gravity_compensation_active:
        command.append("--gravity-compensation-active")
    if allow_pose_deviation:
        command.append("--allow-pose-deviation")
    child_env = os.environ.copy()
    repository_root = str(sleep_script.resolve().parent.parent)
    inherited_pythonpath = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = (
        repository_root
        if not inherited_pythonpath
        else os.pathsep.join((repository_root, inherited_pythonpath))
    )
    stop_timeout = max(0.0, float(process_stop_timeout_seconds))
    try:
        attempt = session.start_process(
            recovery_id=recovery_id,
            stop_timeout_seconds=stop_timeout,
            process_factory=process_factory,
            command=command,
            attempt_token=attempt_token,
            stdin=subprocess.DEVNULL,
            env=child_env,
        )
    except OSError as exc:
        raise ExternalRecoveryError(
            f"failed to start standalone recovery: {exc}"
        ) from exc
    process = attempt.process
    bounded_best_effort_log(
        logger,
        "[handoff] 已启动独立 safe-sleep: "
        f"pid={process.pid} recovery_id={recovery_id}.",
    )

    try:
        while True:
            payload = None
            try:
                payload = read_state()
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                pass

            if payload is not None and payload.get("schema_version") == 2:
                observed_id = payload.get("recovery_id")
                if observed_id != recovery_id:
                    raise ExternalRecoveryError(
                        "external recovery_id mismatch: "
                        f"expected {recovery_id}, observed {observed_id}"
                    )
                source = payload.get("source")
                state = payload.get("state")
                if source == "standalone":
                    owner_pid = payload.get("owner_pid")
                    if owner_pid != process.pid:
                        raise ExternalRecoveryError(
                            "external recovery owner_pid mismatch: "
                            f"expected {process.pid}, observed {owner_pid}"
                        )
                    if (
                        state == "SAFE_TO_STOP"
                        and payload.get("safe_to_stop") is True
                    ):
                        report = _report_from_safe_payload(payload)
                        session.mark_safe(attempt.token, report)
                        return report
            returncode = process.poll()
            if returncode is not None:
                raise ExternalRecoveryError(
                    "standalone recovery exited before SAFE_TO_STOP: "
                    f"returncode={returncode}"
                )
            if payload is None or payload.get("state") != "UNSAFE_HOLD":
                retry_requested(poll_seconds)
            else:
                time.sleep(poll_seconds)
    except BaseException as exc:
        stop_result = _stop_process_bounded(
            process,
            timeout=stop_timeout,
        )
        if not stop_result.reaped:
            raise ExternalRecoveryChildStillAlive(
                process=process,
                attempt_token=attempt.token,
                recovery_id=recovery_id,
                stop_timeout_seconds=stop_timeout,
                original_error=exc,
                stop_result=stop_result,
            ) from exc
        session.mark_reaped(attempt.token)
        if stop_result.failures:
            details = "; ".join(
                f"{stage}: {type(error).__name__}: {error}"
                for stage, error in stop_result.failures
            )
            raise ExternalRecoveryError(
                f"standalone child cleanup failed: {details}"[:512]
            ) from exc
        raise


def _new_recovery_id() -> str:
    return uuid.uuid4().hex


def _wait_for_active_reap(
    *,
    session: ExternalRecoverySession,
    retry_requested: Callable[[float], bool],
    logger: Callable[[str], None],
    retry_poll_seconds: float,
    retry_backoff_sleep: Callable[[float], None],
) -> None:
    while session.active is not None:
        try:
            retry = retry_requested(retry_poll_seconds)
        except BaseException as retry_error:
            bounded_best_effort_log(
                logger,
                "[external-recovery] retry callback failed while the "
                f"same child remains alive: {retry_error}",
            )
            retry_backoff_sleep(retry_poll_seconds)
            continue
        if not retry:
            continue
        active = session.active
        if active is None:
            return
        if active.process is None:
            bounded_best_effort_log(
                logger,
                "[external-recovery] child identity unavailable; refusing "
                f"cleanup or respawn: recovery_id={active.recovery_id}, "
                f"phase={active.phase}",
            )
            continue
        stop_result = _stop_process_bounded(
            active.process,
            timeout=active.stop_timeout_seconds,
        )
        if stop_result.failures:
            details = "; ".join(
                f"{stage}: {type(error).__name__}: {error}"
                for stage, error in stop_result.failures
            )
            bounded_best_effort_log(
                logger,
                f"[external-recovery] same-child cleanup: {details}",
            )
        if stop_result.reaped:
            session.mark_reaped(active.token)


def supervise_external_recovery(
    *,
    robot_name: str,
    gravity_compensation_active: bool,
    sleep_script: Path,
    retry_requested: Callable[[float], bool],
    prepare_attempt: Callable[[str], None],
    wait_for_restart: Callable[[str, ExternalRecoveryError], None],
    session: ExternalRecoverySession,
    allow_pose_deviation: bool = False,
    recovery_id_factory: Callable[[], str] = _new_recovery_id,
    run_attempt: Callable[..., SafeSleepReport] = run_external_recovery,
    logger: Callable[[str], None] = print,
    still_alive_retry_poll_seconds: float = 1.0,
    retry_backoff_sleep: Callable[[float], None] = time.sleep,
) -> SafeSleepReport:
    """Run one standalone child and require that exact child to prove safe."""

    while True:
        if session.safe_report is not None:
            return session.safe_report
        if session.active is not None:
            _wait_for_active_reap(
                session=session,
                retry_requested=retry_requested,
                logger=logger,
                retry_poll_seconds=still_alive_retry_poll_seconds,
                retry_backoff_sleep=retry_backoff_sleep,
            )
            raise ExternalRecoveryError(
                "previous standalone recovery exited without SAFE_TO_STOP"
            )
        recovery_id = recovery_id_factory()
        reservation = session.reserve_attempt(recovery_id)
        try:
            prepare_attempt(recovery_id)
        except BaseException:
            session.cancel_preparing(reservation.token)
            raise
        try:
            report = run_attempt(
                robot_name=robot_name,
                recovery_id=recovery_id,
                gravity_compensation_active=gravity_compensation_active,
                allow_pose_deviation=allow_pose_deviation,
                sleep_script=sleep_script,
                retry_requested=retry_requested,
                session=session,
                attempt_token=reservation.token,
                logger=logger,
            )
            if session.safe_report is None:
                active = session.active
                if (
                    active is not None
                    and active.token is reservation.token
                    and active.phase == "PREPARING"
                ):
                    session.cancel_preparing(reservation.token)
            return report
        except ExternalRecoveryChildStillAlive as exc:
            active = session.active
            if active is None or active.token is not exc.attempt_token:
                raise RuntimeError(
                    "external recovery lost ownership of a live child"
                ) from exc
            bounded_best_effort_log(
                logger,
                f"[external-recovery] {recovery_id} child remains alive; "
                "refusing to spawn another child",
            )
            _wait_for_active_reap(
                session=session,
                retry_requested=retry_requested,
                logger=logger,
                retry_poll_seconds=still_alive_retry_poll_seconds,
                retry_backoff_sleep=retry_backoff_sleep,
            )
            raise ExternalRecoveryError(
                "standalone recovery exited without SAFE_TO_STOP"
            ) from exc
        except ExternalRecoveryError as exc:
            active = session.active
            if (
                active is not None
                and active.token is reservation.token
                and active.phase == "PREPARING"
            ):
                session.cancel_preparing(reservation.token)
            if session.active is not None:
                raise RuntimeError(
                    "external recovery failed while its child remains active"
                ) from exc
            bounded_best_effort_log(
                logger,
                f"[external-recovery] {recovery_id} failed: {exc}",
            )
            raise
        except BaseException:
            active = session.active
            if (
                active is not None
                and active.token is reservation.token
                and active.phase == "PREPARING"
            ):
                session.cancel_preparing(reservation.token)
            raise
