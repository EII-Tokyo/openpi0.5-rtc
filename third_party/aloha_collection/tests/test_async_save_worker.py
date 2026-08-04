import os
from pathlib import Path
import signal
import threading
import time
from types import SimpleNamespace

import pytest

from aloha.async_save_worker import SaveJob, SaveWorker
from aloha.episode_storage import StagedEpisode


def _run_test_save(payload):
    marker = Path(payload["marker"])
    marker.with_suffix(".started").write_text(str(os.getpid()), encoding="utf-8")
    release = marker.with_suffix(".release")
    deadline = time.monotonic() + float(payload.get("wait_seconds", 0.0))
    while time.monotonic() < deadline and not release.exists():
        time.sleep(0.005)
    if payload.get("crash"):
        os._exit(23)
    if payload.get("error"):
        raise RuntimeError(payload["error"])
    marker.write_text("saved", encoding="utf-8")


def _run_test_save_namespace(payload):
    _run_test_save(vars(payload))


class FileDiscardTracker:
    def __init__(self, marker):
        self.marker = str(marker)

    def discard(self):
        Path(self.marker).write_text("discarded", encoding="utf-8")


class RaisingDiscardTracker:
    def __init__(self, marker, message):
        self.marker = str(marker)
        self.message = message

    def discard(self):
        Path(self.marker).write_text("attempted", encoding="utf-8")
        raise RuntimeError(self.message)


def _wait_for(path, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


def _job(tmp_path, name, **options):
    marker = tmp_path / name
    return SaveJob(
        name,
        {
            "marker": str(marker),
            **options,
        },
    )


def test_submit_runs_save_in_spawned_process(tmp_path):
    worker = SaveWorker(save_fn=_run_test_save)
    try:
        job = _job(tmp_path, "episode_1")
        worker.submit(job)
        worker.drain(timeout=3.0)

        child_pid = int(
            (tmp_path / "episode_1.started").read_text(encoding="utf-8")
        )
        assert child_pid == worker.process_pid
        assert child_pid != os.getpid()
        assert (tmp_path / "episode_1").read_text(encoding="utf-8") == "saved"
    finally:
        worker.shutdown(raise_failure=False)


def test_capacity_includes_running_job_and_prevents_unbounded_backlog(tmp_path):
    worker = SaveWorker(capacity=1, save_fn=_run_test_save)
    try:
        worker.submit(
            _job(tmp_path, "episode_1", wait_seconds=2.0)
        )
        _wait_for(tmp_path / "episode_1.started")

        with pytest.raises(TimeoutError, match="save worker capacity"):
            worker.submit(_job(tmp_path, "episode_2"), timeout=0.02)
    finally:
        (tmp_path / "episode_1.release").touch()
        worker.shutdown(raise_failure=False)


def test_job_failure_is_latched_and_rejects_later_submissions(tmp_path):
    worker = SaveWorker(save_fn=_run_test_save)
    worker.submit(_job(tmp_path, "episode_3", error="encoder failed"))

    with pytest.raises(RuntimeError, match="episode_3.*encoder failed"):
        worker.drain(timeout=3.0)
    with pytest.raises(RuntimeError, match="episode_3.*encoder failed"):
        worker.submit(_job(tmp_path, "episode_4"))
    with pytest.raises(RuntimeError, match="episode_3.*encoder failed"):
        worker.shutdown()


def test_abnormal_child_exit_is_reported_and_drain_does_not_hang(tmp_path):
    worker = SaveWorker(save_fn=_run_test_save)
    worker.submit(_job(tmp_path, "episode_crash", crash=True))

    with pytest.raises(RuntimeError, match="episode_crash.*exit code 23"):
        worker.drain(timeout=3.0)
    worker.shutdown(raise_failure=False)


def test_abnormal_child_exit_discards_parent_owned_staging(tmp_path):
    staged = StagedEpisode.create(tmp_path, 8)
    payload = SimpleNamespace(
        marker=str(tmp_path / "episode_owned"),
        staged=staged,
        artifact=None,
        crash=True,
    )

    worker = SaveWorker(save_fn=_run_test_save_namespace)
    worker.submit(SaveJob("episode_8", payload))

    with pytest.raises(RuntimeError, match="episode_8.*exit code 23"):
        worker.drain(timeout=3.0)
    worker.shutdown(raise_failure=False)

    assert not staged.staging_path.exists()
    assert not staged.claim.claim_path.exists()


def test_idle_child_exit_rejects_the_next_submission(tmp_path):
    worker = SaveWorker(save_fn=_run_test_save)
    os.kill(worker.process_pid, signal.SIGKILL)
    deadline = time.monotonic() + 3.0
    while not worker.failed and time.monotonic() < deadline:
        time.sleep(0.01)

    with pytest.raises(RuntimeError, match="save process.*exit code"):
        worker.submit(_job(tmp_path, "episode_after_crash"))
    worker.shutdown(raise_failure=False)


def test_shutdown_waits_for_accepted_job_and_closes_worker(tmp_path):
    worker = SaveWorker(save_fn=_run_test_save)
    worker.submit(_job(tmp_path, "episode_5", wait_seconds=2.0))
    _wait_for(tmp_path / "episode_5.started")
    (tmp_path / "episode_5.release").touch()

    worker.shutdown(timeout=3.0)

    assert (tmp_path / "episode_5").is_file()
    assert not worker.is_alive
    with pytest.raises(RuntimeError, match="shut down"):
        worker.submit(_job(tmp_path, "episode_6"))


def test_abort_bounds_hanging_save_and_discards_pending_ownership(tmp_path):
    staged = StagedEpisode.create(tmp_path, 9)
    artifact_marker = tmp_path / "artifact.discarded"
    payload = SimpleNamespace(
        marker=str(tmp_path / "episode_hanging"),
        staged=staged,
        artifact=FileDiscardTracker(artifact_marker),
        wait_seconds=60.0,
    )
    worker = SaveWorker(save_fn=_run_test_save_namespace)
    try:
        worker.submit(SaveJob("episode_9", payload))
        _wait_for(tmp_path / "episode_hanging.started")

        started = time.monotonic()
        worker.abort(timeout=3.0)
        elapsed = time.monotonic() - started

        assert elapsed < 3.0
        assert not worker.is_alive
        assert not worker._monitor.is_alive()
        assert not worker.is_busy
        assert not staged.staging_path.exists()
        assert not staged.claim.claim_path.exists()
        assert artifact_marker.read_text(encoding="utf-8") == "discarded"

        worker.abort(timeout=0.1)
        worker.shutdown(timeout=0.1)
    finally:
        (tmp_path / "episode_hanging.release").touch()
        worker.shutdown(timeout=3.0, raise_failure=False)


def test_abort_timeout_closes_connection_and_reports_live_components():
    class StillAlive:
        def is_alive(self):
            return True

        def terminate(self):
            return None

        def join(self, timeout=None):
            return None

    connection = SimpleNamespace(
        closed=False,
        close=lambda: setattr(connection, "closed", True),
    )
    worker = object.__new__(SaveWorker)
    worker._condition = threading.Condition()
    worker._closed = False
    worker._aborted = False
    worker._abort_complete = False
    worker._process = StillAlive()
    worker._monitor = StillAlive()
    worker._connection = connection
    worker._pending = {1: object()}

    with pytest.raises(
        TimeoutError,
        match="process_alive=True.*monitor_alive=True",
    ):
        worker.abort(timeout=0.0)

    assert connection.closed
    with pytest.raises(TimeoutError, match="process_alive=True"):
        worker.shutdown(timeout=0.0, raise_failure=False)


def test_abort_continues_cleanup_after_terminate_error(tmp_path):
    calls = []

    class StopsDuringJoin:
        alive = True

        def is_alive(self):
            calls.append("process_is_alive")
            return self.alive

        def terminate(self):
            calls.append("terminate")
            raise OSError("terminate failed")

        def join(self, timeout=None):
            calls.append("process_join")
            self.alive = False

    class Monitor:
        def join(self, timeout=None):
            calls.append("monitor_join")

        def is_alive(self):
            return False

    def discard_staged():
        calls.append("staged_discard")
        raise RuntimeError("staged discard failed")

    staged = SimpleNamespace(discard=discard_staged)
    artifact = SimpleNamespace(
        discard=lambda: calls.append("artifact_discard")
    )
    connection = SimpleNamespace(close=lambda: calls.append("connection_close"))
    worker = object.__new__(SaveWorker)
    worker._condition = threading.Condition()
    worker._closed = False
    worker._aborted = False
    worker._abort_complete = False
    worker._process = StopsDuringJoin()
    worker._monitor = Monitor()
    worker._connection = connection
    worker._pending = {
        1: SimpleNamespace(
            name="episode_10",
            staged=staged,
            artifact=artifact,
        )
    }
    worker._capacity = SimpleNamespace(
        release=lambda: calls.append("capacity_release")
    )

    with pytest.raises(
        RuntimeError,
        match="terminate.*terminate failed.*staged.discard.*staged discard failed",
    ):
        worker.abort(timeout=0.1)

    assert "process_join" in calls
    assert "monitor_join" in calls
    assert "connection_close" in calls
    assert "staged_discard" in calls
    assert "artifact_discard" in calls
    assert not worker._pending
    with pytest.raises(RuntimeError, match="terminate.*staged.discard"):
        worker.abort(timeout=0.1)


def test_abort_aggregates_terminate_error_with_live_timeout():
    class Unstoppable:
        def is_alive(self):
            return True

        def terminate(self):
            raise OSError("terminate denied")

        def join(self, timeout=None):
            return None

    connection = SimpleNamespace(close=lambda: None)
    worker = object.__new__(SaveWorker)
    worker._condition = threading.Condition()
    worker._closed = False
    worker._aborted = False
    worker._abort_complete = False
    worker._process = Unstoppable()
    worker._monitor = Unstoppable()
    worker._connection = connection
    worker._pending = {1: object()}

    with pytest.raises(
        TimeoutError,
        match="terminate denied.*process_alive=True.*monitor_alive=True",
    ):
        worker.abort(timeout=0.0)


def test_real_abort_surfaces_parent_ownership_discard_failures(tmp_path):
    staged_marker = tmp_path / "staged.discard"
    artifact_marker = tmp_path / "artifact.discard"
    payload = SimpleNamespace(
        marker=str(tmp_path / "episode_discard_failure"),
        staged=RaisingDiscardTracker(staged_marker, "staged refused"),
        artifact=RaisingDiscardTracker(artifact_marker, "artifact refused"),
        wait_seconds=60.0,
    )
    worker = SaveWorker(save_fn=_run_test_save_namespace)
    worker.submit(SaveJob("episode_discard_failure", payload))
    _wait_for(tmp_path / "episode_discard_failure.started")

    with pytest.raises(
        RuntimeError,
        match="staged.discard.*staged refused.*artifact.discard.*artifact refused",
    ):
        worker.abort(timeout=3.0)

    assert staged_marker.read_text(encoding="utf-8") == "attempted"
    assert artifact_marker.read_text(encoding="utf-8") == "attempted"
    with pytest.raises(
        RuntimeError,
        match=(
            "episode_discard_failure.*exit code.*"
            "staged.discard.*artifact.discard"
        ),
    ):
        worker.drain(timeout=0.1)
    with pytest.raises(RuntimeError, match="staged.discard.*artifact.discard"):
        worker.shutdown(timeout=0.1)
