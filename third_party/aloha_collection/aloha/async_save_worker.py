"""Bounded child-process execution for accepted episode save jobs."""

from __future__ import annotations

import multiprocessing
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SaveJob:
    """One accepted episode whose ownership transfers to the save worker."""

    name: str
    payload: Any


@dataclass(frozen=True)
class _PendingSave:
    name: str
    staged: Any | None
    artifact: Any | None


def _default_save(payload: Any) -> None:
    from aloha.episode_serialization import save_episode

    save_episode(payload)


def _save_process_main(connection: Any, save_fn: Callable[[Any], None]) -> None:
    """Receive small descriptors and execute saves in a clean interpreter."""
    try:
        while True:
            message = connection.recv()
            if message is None:
                return
            job_id, job = message
            try:
                save_fn(job.payload)
            except BaseException as exc:
                connection.send(
                    (
                        job_id,
                        False,
                        type(exc).__name__,
                        str(exc),
                        traceback.format_exc(),
                    )
                )
            else:
                connection.send((job_id, True, "", "", ""))
    except (EOFError, BrokenPipeError, OSError):
        return
    finally:
        connection.close()


class SaveWorker:
    """Run accepted save jobs in a persistent spawned process."""

    def __init__(
        self,
        *,
        capacity: int = 1,
        save_fn: Callable[[Any], None] | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least one")
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        self._connection = parent_connection
        self._capacity = threading.BoundedSemaphore(capacity)
        self._condition = threading.Condition()
        self._send_lock = threading.Lock()
        self._pending: dict[int, _PendingSave] = {}
        self._next_job_id = 1
        self._closed = False
        self._aborted = False
        self._abort_complete = False
        self._abort_error: BaseException | None = None
        self._ownership_cleanup_failures: list[
            tuple[str, str, BaseException]
        ] = []
        self._failure: tuple[str, str] | None = None
        self._process = context.Process(
            target=_save_process_main,
            args=(child_connection, save_fn or _default_save),
            name="aloha-episode-save-process",
            daemon=False,
        )
        self._process.start()
        child_connection.close()
        self._monitor = threading.Thread(
            target=self._monitor_results,
            name="aloha-save-process-monitor",
            daemon=False,
        )
        self._monitor.start()

    @property
    def process_pid(self) -> int | None:
        return self._process.pid

    @property
    def is_alive(self) -> bool:
        return self._process.is_alive()

    @property
    def is_busy(self) -> bool:
        with self._condition:
            return bool(self._pending)

    @property
    def failed(self) -> bool:
        with self._condition:
            return self._failure is not None

    def _failure_error(self) -> RuntimeError | None:
        with self._condition:
            failure = self._failure
        if failure is None:
            return None
        job_name, detail = failure
        return RuntimeError(
            f"background save failed for {job_name}: {detail}"
        )

    def raise_if_failed(self) -> None:
        error = self._failure_error()
        if error is not None:
            raise error

    def submit(self, job: SaveJob, *, timeout: float | None = None) -> None:
        if not isinstance(job, SaveJob):
            raise TypeError("job must be a SaveJob")
        self.raise_if_failed()
        with self._condition:
            if self._closed:
                raise RuntimeError("save worker has been shut down")

        acquired = (
            self._capacity.acquire()
            if timeout is None
            else self._capacity.acquire(timeout=max(0.0, float(timeout)))
        )
        if not acquired:
            raise TimeoutError("save worker capacity is still occupied")

        job_id: int | None = None
        try:
            self.raise_if_failed()
            with self._condition:
                if self._closed:
                    raise RuntimeError("save worker has been shut down")
                job_id = self._next_job_id
                self._next_job_id += 1
                self._pending[job_id] = _PendingSave(
                    name=job.name,
                    staged=getattr(job.payload, "staged", None),
                    artifact=getattr(job.payload, "artifact", None),
                )
            with self._send_lock:
                self._connection.send((job_id, job))
        except BaseException:
            if job_id is not None:
                with self._condition:
                    removed = self._pending.pop(job_id, None)
                    self._condition.notify_all()
                if removed is not None:
                    self._capacity.release()
            else:
                self._capacity.release()
            raise

    def _monitor_results(self) -> None:
        while True:
            try:
                if self._connection.poll(0.05):
                    result = self._connection.recv()
                    self._complete_result(result)
                    continue
            except (EOFError, BrokenPipeError, OSError):
                time.sleep(0.05)

            if not self._process.is_alive():
                self._process.join(timeout=0)
                with self._condition:
                    pending = tuple(self._pending.items())
                    closed = self._closed
                    if (
                        not pending
                        and not closed
                        and self._failure is None
                    ):
                        self._failure = (
                            "<save-process>",
                            "save process exited unexpectedly with "
                            f"exit code {self._process.exitcode}",
                        )
                        self._condition.notify_all()
                if pending:
                    exit_code = self._process.exitcode
                    for job_id, job in pending:
                        self._complete_abnormal_exit(
                            job_id,
                            job,
                            exit_code,
                        )
                if closed or not pending:
                    return

    def _complete_result(self, result: tuple[Any, ...]) -> None:
        job_id, succeeded, error_type, message, remote_traceback = result
        with self._condition:
            job = self._pending.pop(job_id, None)
            if job is None:
                return
            if not succeeded and self._failure is None:
                detail = f"{error_type}: {message}"
                if remote_traceback:
                    detail = f"{detail}\n{remote_traceback}"
                self._failure = (job.name, detail)
            self._condition.notify_all()
        self._capacity.release()

    def _complete_abnormal_exit(
        self,
        job_id: int,
        job: _PendingSave,
        exit_code: int | None,
    ) -> None:
        cleanup_failures = self._discard_job_ownership(job)
        with self._condition:
            removed = self._pending.pop(job_id, None)
            if removed is None:
                return
            detail = f"save process exited with exit code {exit_code}"
            if cleanup_failures:
                cleanup_detail = "; ".join(
                    f"{stage}: {type(error).__name__}: {error}"
                    for stage, error in cleanup_failures
                )
                detail = f"{detail}; ownership cleanup: {cleanup_detail}"
                self._ownership_cleanup_failures.extend(
                    (job.name, stage, error)
                    for stage, error in cleanup_failures
                )
            if self._failure is None:
                self._failure = (
                    job.name,
                    detail,
                )
            elif cleanup_failures:
                failed_job, existing_detail = self._failure
                self._failure = (
                    failed_job,
                    f"{existing_detail}; {job.name}: {detail}",
                )
            self._condition.notify_all()
        try:
            self._capacity.release()
        except BaseException as exc:
            with self._condition:
                self._ownership_cleanup_failures.append(
                    (job.name, "capacity.release", exc)
                )
                failed_job, existing_detail = self._failure or (
                    job.name,
                    detail,
                )
                self._failure = (
                    failed_job,
                    f"{existing_detail}; capacity.release: "
                    f"{type(exc).__name__}: {exc}",
                )
                self._condition.notify_all()

    @staticmethod
    def _discard_job_ownership(
        job: _PendingSave,
    ) -> list[tuple[str, BaseException]]:
        failures = []
        staged = job.staged
        if staged is not None:
            try:
                staged.discard()
            except BaseException as exc:
                failures.append(("staged.discard", exc))
        artifact = job.artifact
        if artifact is not None:
            try:
                artifact.discard()
            except BaseException as exc:
                failures.append(("artifact.discard", exc))
        return failures

    def drain(self, *, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._pending:
                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("timed out draining save worker")
                self._condition.wait(timeout=remaining)
        self.raise_if_failed()

    def abort(self, *, timeout: float | None = None) -> None:
        """Forcibly stop saving and discard all pending parent ownership."""

        deadline = (
            None
            if timeout is None
            else time.monotonic() + max(0.0, float(timeout))
        )
        with self._condition:
            if self._abort_complete:
                abort_error = getattr(self, "_abort_error", None)
                if abort_error is not None:
                    raise abort_error
                return
            self._closed = True
            self._aborted = True

        failures: list[tuple[str, BaseException]] = []

        def attempt(stage, operation, default=None):
            try:
                return operation()
            except BaseException as exc:
                failures.append((stage, exc))
                return default

        process_was_alive = attempt(
            "process.is_alive",
            self._process.is_alive,
            True,
        )
        if process_was_alive:
            attempt("process.terminate", self._process.terminate)

        def remaining() -> float | None:
            if deadline is None:
                return None
            return max(0.0, deadline - time.monotonic())

        attempt(
            "process.join",
            lambda: self._process.join(timeout=remaining()),
        )
        attempt(
            "monitor.join",
            lambda: self._monitor.join(timeout=remaining()),
        )

        process_alive = attempt(
            "process.is_alive.final",
            self._process.is_alive,
            True,
        )
        monitor_alive = attempt(
            "monitor.is_alive",
            self._monitor.is_alive,
            True,
        )
        attempt("connection.close", self._connection.close)
        if process_alive or monitor_alive:
            failures.append(
                (
                    "live_timeout",
                    TimeoutError(
                        f"process_alive={process_alive}, "
                        f"monitor_alive={monitor_alive}"
                    ),
                )
            )
        else:
            with self._condition:
                pending = tuple(self._pending.items())
                self._pending.clear()
                self._condition.notify_all()
            for _job_id, job in pending:
                failures.extend(self._discard_job_ownership(job))
                attempt("capacity.release", self._capacity.release)

            with self._condition:
                failures.extend(
                    (f"{job_name}.{stage}", error)
                    for job_name, stage, error
                    in getattr(self, "_ownership_cleanup_failures", ())
                )

        with self._condition:
            pending_remains = bool(self._pending)
            if not pending_remains and not process_alive and not monitor_alive:
                self._abort_complete = True
        if pending_remains:
            failures.append(
                (
                    "pending",
                    RuntimeError("save worker abort left pending ownership"),
                )
            )
        if failures:
            details = "; ".join(
                f"{stage}: {type(error).__name__}: {error}"
                for stage, error in failures
            )
            error_type = (
                TimeoutError
                if (process_alive or monitor_alive)
                else RuntimeError
            )
            abort_error = error_type(
                f"save worker abort errors: {details}"[:512]
            )
            if self._abort_complete:
                self._abort_error = abort_error
            raise abort_error

    def shutdown(
        self,
        *,
        timeout: float | None = None,
        raise_failure: bool = True,
    ) -> None:
        with self._condition:
            already_closed = self._closed
            if not already_closed:
                self._closed = True
        if already_closed:
            if self._aborted:
                if not self._abort_complete:
                    self.abort(timeout=timeout)
                abort_error = getattr(self, "_abort_error", None)
                if raise_failure and abort_error is not None:
                    raise abort_error
                return
            if raise_failure:
                self.raise_if_failed()
            return

        drain_error: BaseException | None = None
        try:
            self.drain(timeout=timeout)
        except BaseException as exc:
            drain_error = exc
        finally:
            if self._process.is_alive():
                try:
                    with self._send_lock:
                        self._connection.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
            self._process.join(timeout=timeout)
            self._monitor.join(timeout=timeout)
            self._connection.close()

        if self._process.is_alive() or self._monitor.is_alive():
            raise TimeoutError("save worker did not shut down")
        if drain_error is not None and raise_failure:
            raise drain_error
