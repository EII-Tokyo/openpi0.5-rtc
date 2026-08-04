import threading

import pytest

from aloha.continuous_recorder import (
    EpisodeFinalizer,
    SessionOutcome,
    finalize_staged_episode,
    run_continuous_session,
)


class DeliberateBaseException(BaseException):
    pass


class FinalizerHarness:
    def __init__(
        self,
        *,
        stop_no_save=False,
        stop_and_save=False,
        skip_sleep=False,
        complete_save_result=True,
    ):
        self.lock = threading.RLock()
        self.stop_no_save = threading.Event()
        self.stop_and_save = threading.Event()
        self.skip_sleep = threading.Event()
        self.calls = []
        self.logs = []
        self.complete_save_result = complete_save_result

        if stop_no_save:
            self.stop_no_save.set()
        if stop_and_save:
            self.stop_and_save.set()
        if skip_sleep:
            self.skip_sleep.set()

        self.finalizer = EpisodeFinalizer(
            lock=self.lock,
            stop_no_save=self.stop_no_save,
            stop_and_save=self.stop_and_save,
            skip_sleep=self.skip_sleep,
            complete_save=self.complete_save,
            logger=self.logs.append,
        )

    def _record_locked_call(self, name):
        assert self.lock._is_owned()
        self.calls.append(name)

    def publish(self):
        self._record_locked_call("publish")

    def discard(self):
        self._record_locked_call("discard")

    def complete_save(self):
        self._record_locked_call("complete_save")
        return self.complete_save_result


def test_session_outcome_values_and_sleep_policy_are_explicit():
    assert {outcome.name: outcome.value for outcome in SessionOutcome} == {
        "CONTINUE_NEXT_EPISODE": "continue_next_episode",
        "EXIT_SAVE_AND_SLEEP": "exit_save_and_sleep",
        "EXIT_DISCARD_AND_SLEEP": "exit_discard_and_sleep",
        "EXIT_SAVE_WITHOUT_SLEEP": "exit_save_without_sleep",
        "EXIT_FAILURE_AND_SLEEP": "exit_failure_and_sleep",
    }
    assert {
        outcome: outcome.requires_sleep
        for outcome in SessionOutcome
    } == {
        SessionOutcome.CONTINUE_NEXT_EPISODE: True,
        SessionOutcome.EXIT_SAVE_AND_SLEEP: True,
        SessionOutcome.EXIT_DISCARD_AND_SLEEP: True,
        SessionOutcome.EXIT_SAVE_WITHOUT_SLEEP: False,
        SessionOutcome.EXIT_FAILURE_AND_SLEEP: True,
    }


def test_normal_save_publishes_then_completes_trigger_under_lock():
    harness = FinalizerHarness()

    result = harness.finalizer.finalize_validated(
        publish=harness.publish,
        discard=harness.discard,
    )

    assert result is SessionOutcome.CONTINUE_NEXT_EPISODE
    assert harness.calls == ["publish", "complete_save"]
    assert harness.logs == []


def test_s_discards_under_lock_and_never_publishes():
    harness = FinalizerHarness(stop_no_save=True)

    result = harness.finalizer.finalize_validated(
        publish=harness.publish,
        discard=harness.discard,
    )

    assert result is SessionOutcome.EXIT_DISCARD_AND_SLEEP
    assert harness.calls == ["discard"]
    assert harness.logs == []


@pytest.mark.parametrize(
    ("skip_sleep", "expected"),
    [
        (False, SessionOutcome.EXIT_SAVE_AND_SLEEP),
        (True, SessionOutcome.EXIT_SAVE_WITHOUT_SLEEP),
    ],
)
def test_m_and_r_publish_under_lock_then_exit(skip_sleep, expected):
    harness = FinalizerHarness(stop_and_save=True, skip_sleep=skip_sleep)

    result = harness.finalizer.finalize_validated(
        publish=harness.publish,
        discard=harness.discard,
    )

    assert result is expected
    assert harness.calls == ["publish"]
    assert harness.logs == []


@pytest.mark.parametrize("stop_and_save", [False, True])
def test_publish_exception_is_logged_and_fails_closed(stop_and_save):
    harness = FinalizerHarness(stop_and_save=stop_and_save)

    def failing_publish():
        harness._record_locked_call("publish")
        raise RuntimeError("publication broke")

    result = harness.finalizer.finalize_validated(
        publish=failing_publish,
        discard=harness.discard,
    )

    assert result is SessionOutcome.EXIT_FAILURE_AND_SLEEP
    assert harness.calls == ["publish", "discard"]
    assert len(harness.logs) == 1
    assert "publication broke" in harness.logs[0]


def test_publish_base_exception_discards_best_effort_and_fails_closed():
    harness = FinalizerHarness()

    def failing_publish():
        harness._record_locked_call("publish")
        raise DeliberateBaseException("fatal publication")

    result = harness.finalizer.finalize_validated(
        publish=failing_publish,
        discard=harness.discard,
    )

    assert result is SessionOutcome.EXIT_FAILURE_AND_SLEEP
    assert harness.calls == ["publish", "discard"]
    assert len(harness.logs) == 1
    assert "fatal publication" in harness.logs[0]


def test_publish_and_recovery_discard_failures_are_logged_separately():
    harness = FinalizerHarness()

    def failing_publish():
        harness._record_locked_call("publish")
        raise DeliberateBaseException("fatal publication")

    def failing_discard():
        harness._record_locked_call("discard")
        raise DeliberateBaseException("fatal discard")

    result = harness.finalizer.finalize_validated(
        publish=failing_publish,
        discard=failing_discard,
    )

    assert result is SessionOutcome.EXIT_FAILURE_AND_SLEEP
    assert harness.calls == ["publish", "discard"]
    assert len(harness.logs) == 2
    assert "fatal publication" in harness.logs[0]
    assert "fatal discard" in harness.logs[1]


def test_direct_s_discard_base_exception_is_logged_and_fails_closed():
    harness = FinalizerHarness(stop_no_save=True)

    def failing_discard():
        harness._record_locked_call("discard")
        raise DeliberateBaseException("fatal direct discard")

    result = harness.finalizer.finalize_validated(
        publish=harness.publish,
        discard=failing_discard,
    )

    assert result is SessionOutcome.EXIT_FAILURE_AND_SLEEP
    assert harness.calls == ["discard"]
    assert len(harness.logs) == 1
    assert "fatal direct discard" in harness.logs[0]


def test_logger_base_exception_does_not_escape_failure_boundary():
    harness = FinalizerHarness()

    def failing_publish():
        harness._record_locked_call("publish")
        raise DeliberateBaseException("fatal publication")

    def failing_logger(_message):
        raise DeliberateBaseException("logger broke")

    harness.finalizer = EpisodeFinalizer(
        lock=harness.lock,
        stop_no_save=harness.stop_no_save,
        stop_and_save=harness.stop_and_save,
        skip_sleep=harness.skip_sleep,
        complete_save=harness.complete_save,
        logger=failing_logger,
    )

    result = harness.finalizer.finalize_validated(
        publish=failing_publish,
        discard=harness.discard,
    )

    assert result is SessionOutcome.EXIT_FAILURE_AND_SLEEP
    assert harness.calls == ["publish", "discard"]


def test_complete_save_false_is_logged_and_fails_closed():
    harness = FinalizerHarness(complete_save_result=False)

    result = harness.finalizer.finalize_validated(
        publish=harness.publish,
        discard=harness.discard,
    )

    assert result is SessionOutcome.EXIT_FAILURE_AND_SLEEP
    assert harness.calls == ["publish", "complete_save"]
    assert len(harness.logs) == 1
    assert "complete_save" in harness.logs[0]


def test_complete_save_base_exception_is_logged_without_discarding_publication():
    harness = FinalizerHarness()

    def failing_complete_save():
        harness._record_locked_call("complete_save")
        raise DeliberateBaseException("trigger completion broke")

    harness.finalizer = EpisodeFinalizer(
        lock=harness.lock,
        stop_no_save=harness.stop_no_save,
        stop_and_save=harness.stop_and_save,
        skip_sleep=harness.skip_sleep,
        complete_save=failing_complete_save,
        logger=harness.logs.append,
    )

    result = harness.finalizer.finalize_validated(
        publish=harness.publish,
        discard=harness.discard,
    )

    assert result is SessionOutcome.EXIT_FAILURE_AND_SLEEP
    assert harness.calls == ["publish", "complete_save"]
    assert len(harness.logs) == 1
    assert "trigger completion broke" in harness.logs[0]


def test_stop_mutation_waits_for_atomic_normal_finalization_then_s_wins():
    lock = threading.RLock()
    stop_no_save = threading.Event()
    stop_and_save = threading.Event()
    skip_sleep = threading.Event()
    publish_started = threading.Event()
    mutation_attempted = threading.Event()
    allow_publish_to_finish = threading.Event()
    mutation_acquired = threading.Event()
    calls = []
    result = []

    def publish():
        calls.append("publish")
        publish_started.set()
        assert allow_publish_to_finish.wait(timeout=1)

    def complete_save():
        calls.append("complete_save")
        assert mutation_attempted.is_set()
        assert not mutation_acquired.is_set()
        return True

    def discard():
        calls.append("discard")

    finalizer = EpisodeFinalizer(
        lock=lock,
        stop_no_save=stop_no_save,
        stop_and_save=stop_and_save,
        skip_sleep=skip_sleep,
        complete_save=complete_save,
    )

    def finalize():
        result.append(
            finalizer.finalize_validated(publish=publish, discard=discard)
        )

    def request_both_stops():
        assert publish_started.wait(timeout=1)
        mutation_attempted.set()
        with lock:
            mutation_acquired.set()
            stop_and_save.set()
            stop_no_save.set()

    finalizer_thread = threading.Thread(target=finalize)
    mutator_thread = threading.Thread(target=request_both_stops)
    finalizer_thread.start()
    mutator_thread.start()
    assert mutation_attempted.wait(timeout=1)
    allow_publish_to_finish.set()
    finalizer_thread.join(timeout=1)
    mutator_thread.join(timeout=1)

    assert not finalizer_thread.is_alive()
    assert not mutator_thread.is_alive()
    assert result == [SessionOutcome.CONTINUE_NEXT_EPISODE]
    assert mutation_acquired.is_set()
    assert calls == ["publish", "complete_save"]

    assert finalizer.finalize_validated(
        publish=publish,
        discard=discard,
    ) is SessionOutcome.EXIT_DISCARD_AND_SLEEP
    assert calls == ["publish", "complete_save", "discard"]


def test_two_episodes_share_one_runtime_and_cleanup_once():
    calls = []
    runtime = object()
    outcomes = iter(
        [
            SessionOutcome.CONTINUE_NEXT_EPISODE,
            SessionOutcome.EXIT_DISCARD_AND_SLEEP,
        ]
    )

    result = run_continuous_session(
        create_runtime=lambda: calls.append("startup") or runtime,
        capture_episode=lambda seen_runtime, index: (
            calls.append(("capture", seen_runtime, index))
            or next(outcomes)
        ),
        next_index=lambda index: index + 1,
        initial_index=6,
        final_cleanup=lambda seen_runtime, outcome: calls.append(
            ("cleanup", seen_runtime, outcome)
        ),
    )

    assert result is SessionOutcome.EXIT_DISCARD_AND_SLEEP
    assert calls == [
        "startup",
        ("capture", runtime, 6),
        ("capture", runtime, 7),
        ("cleanup", runtime, SessionOutcome.EXIT_DISCARD_AND_SLEEP),
    ]


def test_r_cleans_up_once_without_sleep():
    calls = []
    runtime = object()

    result = run_continuous_session(
        create_runtime=lambda: calls.append("startup") or runtime,
        capture_episode=lambda seen_runtime, index: (
            calls.append(("capture", seen_runtime, index))
            or SessionOutcome.EXIT_SAVE_WITHOUT_SLEEP
        ),
        next_index=lambda index: pytest.fail("next_index must not run"),
        initial_index=3,
        final_cleanup=lambda seen_runtime, outcome: calls.append(
            ("cleanup", seen_runtime, outcome)
        ),
    )

    assert result is SessionOutcome.EXIT_SAVE_WITHOUT_SLEEP
    assert calls == [
        "startup",
        ("capture", runtime, 3),
        ("cleanup", runtime, SessionOutcome.EXIT_SAVE_WITHOUT_SLEEP),
    ]


def test_invalid_capture_result_cleans_up_with_failure_outcome_and_raises():
    calls = []
    runtime = object()
    invalid_outcome = {"outcome": "continue"}

    with pytest.raises(
        TypeError,
        match=r"capture_episode returned .*dict.*SessionOutcome",
    ):
        run_continuous_session(
            create_runtime=lambda: runtime,
            capture_episode=lambda seen_runtime, index: invalid_outcome,
            next_index=lambda index: pytest.fail("next_index must not run"),
            initial_index=4,
            final_cleanup=lambda seen_runtime, outcome: calls.append(
                ("cleanup", seen_runtime, outcome)
            ),
        )

    assert calls == [
        ("cleanup", runtime, SessionOutcome.EXIT_FAILURE_AND_SLEEP)
    ]


def test_create_runtime_failure_propagates_without_final_cleanup():
    calls = []

    def create_runtime():
        calls.append("startup")
        raise RuntimeError("startup factory rolled back")

    with pytest.raises(RuntimeError, match="startup factory rolled back"):
        run_continuous_session(
            create_runtime=create_runtime,
            capture_episode=lambda runtime, index: pytest.fail(
                "capture_episode must not run"
            ),
            next_index=lambda index: pytest.fail("next_index must not run"),
            initial_index=0,
            final_cleanup=lambda runtime, outcome: calls.append("cleanup"),
        )

    assert calls == ["startup"]


def test_cleanup_failure_after_terminal_outcome_propagates():
    runtime = object()

    def failing_cleanup(seen_runtime, outcome):
        assert seen_runtime is runtime
        assert outcome is SessionOutcome.EXIT_SAVE_AND_SLEEP
        raise RuntimeError("cleanup broke")

    with pytest.raises(RuntimeError, match="cleanup broke"):
        run_continuous_session(
            create_runtime=lambda: runtime,
            capture_episode=lambda seen_runtime, index: (
                SessionOutcome.EXIT_SAVE_AND_SLEEP
            ),
            next_index=lambda index: pytest.fail("next_index must not run"),
            initial_index=0,
            final_cleanup=failing_cleanup,
        )


def test_loop_exception_cleans_up_with_failure_outcome():
    calls = []
    runtime = object()

    def fail_capture(_runtime, _index):
        raise RuntimeError("capture broke")

    with pytest.raises(RuntimeError, match="capture broke"):
        run_continuous_session(
            create_runtime=lambda: runtime,
            capture_episode=fail_capture,
            next_index=lambda index: index + 1,
            initial_index=0,
            final_cleanup=lambda seen, outcome: calls.append((seen, outcome)),
        )

    assert calls == [(runtime, SessionOutcome.EXIT_FAILURE_AND_SLEEP)]


@pytest.mark.parametrize("failure_source", ["capture", "next_index"])
def test_loop_exception_cleans_up_once_then_propagates(failure_source):
    calls = []
    runtime = object()

    def capture_episode(seen_runtime, index):
        calls.append(("capture", seen_runtime, index))
        if failure_source == "capture":
            raise RuntimeError("capture broke")
        return SessionOutcome.CONTINUE_NEXT_EPISODE

    def next_index(index):
        calls.append(("next_index", index))
        raise RuntimeError("index broke")

    expected_message = (
        "capture broke" if failure_source == "capture" else "index broke"
    )
    with pytest.raises(RuntimeError, match=expected_message):
        run_continuous_session(
            create_runtime=lambda: calls.append("startup") or runtime,
            capture_episode=capture_episode,
            next_index=next_index,
            initial_index=11,
            final_cleanup=lambda seen_runtime, outcome: calls.append(
                ("cleanup", seen_runtime, outcome)
            ),
        )

    expected = ["startup", ("capture", runtime, 11)]
    if failure_source == "next_index":
        expected.append(("next_index", 11))
    expected.append(
        ("cleanup", runtime, SessionOutcome.EXIT_FAILURE_AND_SLEEP)
    )
    assert calls == expected


@pytest.mark.parametrize(
    "failure_type",
    [KeyboardInterrupt, SystemExit],
)
def test_process_base_exception_cleans_up_with_failure_outcome(failure_type):
    calls = []
    runtime = object()

    def capture_episode(seen_runtime, index):
        calls.append(("capture", seen_runtime, index))
        raise failure_type()

    with pytest.raises(failure_type):
        run_continuous_session(
            create_runtime=lambda: runtime,
            capture_episode=capture_episode,
            next_index=lambda index: pytest.fail("next_index must not run"),
            initial_index=8,
            final_cleanup=lambda seen_runtime, outcome: calls.append(
                ("cleanup", seen_runtime, outcome)
            ),
        )

    assert calls == [
        ("capture", runtime, 8),
        ("cleanup", runtime, SessionOutcome.EXIT_FAILURE_AND_SLEEP),
    ]


@pytest.mark.parametrize("failure_source", ["capture", "next_index"])
def test_cleanup_failure_does_not_mask_active_loop_exception(failure_source):
    logs = []
    runtime = object()

    def capture_episode(seen_runtime, index):
        if failure_source == "capture":
            raise RuntimeError("capture broke")
        return SessionOutcome.CONTINUE_NEXT_EPISODE

    def next_index(index):
        raise RuntimeError("index broke")

    def failing_cleanup(seen_runtime, outcome):
        assert seen_runtime is runtime
        assert outcome is SessionOutcome.EXIT_FAILURE_AND_SLEEP
        raise DeliberateBaseException("cleanup base failure")

    expected_message = (
        "capture broke" if failure_source == "capture" else "index broke"
    )
    with pytest.raises(RuntimeError, match=expected_message):
        run_continuous_session(
            create_runtime=lambda: runtime,
            capture_episode=capture_episode,
            next_index=next_index,
            initial_index=2,
            final_cleanup=failing_cleanup,
            logger=logs.append,
        )

    assert len(logs) == 1
    assert "cleanup base failure" in logs[0]


def test_logger_failure_does_not_mask_active_capture_exception():
    runtime = object()

    def failing_capture(seen_runtime, index):
        raise RuntimeError("capture remains primary")

    def failing_cleanup(seen_runtime, outcome):
        raise RuntimeError("cleanup is secondary")

    def failing_logger(_message):
        raise DeliberateBaseException("logger is tertiary")

    with pytest.raises(RuntimeError, match="capture remains primary"):
        run_continuous_session(
            create_runtime=lambda: runtime,
            capture_episode=failing_capture,
            next_index=lambda index: index + 1,
            initial_index=0,
            final_cleanup=failing_cleanup,
            logger=failing_logger,
        )


class FakeStagedEpisode:
    def __init__(self):
        self.calls = []

    def publish(self, *, allow_existing_destination=False):
        self.calls.append(("publish", allow_existing_destination))

    def discard(self):
        self.calls.append(("discard",))


def test_staged_episode_validates_publishes_and_continues():
    harness = FinalizerHarness()
    staged = FakeStagedEpisode()
    calls = []

    outcome = finalize_staged_episode(
        staged=staged,
        validate=lambda: calls.append("validate"),
        finalizer=harness.finalizer,
        allow_existing_destination=True,
    )

    assert outcome is SessionOutcome.CONTINUE_NEXT_EPISODE
    assert calls == ["validate"]
    assert staged.calls == [("publish", True)]
    assert harness.calls == ["complete_save"]


def test_staged_episode_validation_failure_discards_and_never_advances():
    harness = FinalizerHarness()
    staged = FakeStagedEpisode()
    logs = []

    def fail_validation():
        raise RuntimeError("bad staged outputs")

    outcome = finalize_staged_episode(
        staged=staged,
        validate=fail_validation,
        finalizer=harness.finalizer,
        logger=logs.append,
    )

    assert outcome is SessionOutcome.EXIT_FAILURE_AND_SLEEP
    assert staged.calls == [("discard",)]
    assert harness.calls == []
    assert any("bad staged outputs" in message for message in logs)


@pytest.mark.parametrize(
    ("stop_no_save", "stop_and_save", "skip_sleep", "expected", "calls"),
    [
        (
            True,
            False,
            False,
            SessionOutcome.EXIT_DISCARD_AND_SLEEP,
            [("discard",)],
        ),
        (
            False,
            True,
            False,
            SessionOutcome.EXIT_SAVE_AND_SLEEP,
            [("publish", False)],
        ),
        (
            False,
            True,
            True,
            SessionOutcome.EXIT_SAVE_WITHOUT_SLEEP,
            [("publish", False)],
        ),
    ],
)
def test_staged_episode_preserves_s_m_r_finalization(
    stop_no_save,
    stop_and_save,
    skip_sleep,
    expected,
    calls,
):
    harness = FinalizerHarness(
        stop_no_save=stop_no_save,
        stop_and_save=stop_and_save,
        skip_sleep=skip_sleep,
    )
    staged = FakeStagedEpisode()

    outcome = finalize_staged_episode(
        staged=staged,
        validate=lambda: None,
        finalizer=harness.finalizer,
    )

    assert outcome is expected
    assert staged.calls == calls
    assert harness.calls == []


def test_max_timestep_episode_publishes_and_exits_without_trigger_completion():
    harness = FinalizerHarness()
    staged = FakeStagedEpisode()

    outcome = finalize_staged_episode(
        staged=staged,
        validate=lambda: None,
        finalizer=harness.finalizer,
        continue_after_save=False,
    )

    assert outcome is SessionOutcome.EXIT_SAVE_AND_SLEEP
    assert staged.calls == [("publish", False)]
    assert harness.calls == []
