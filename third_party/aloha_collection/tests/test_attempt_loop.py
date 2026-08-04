import threading
import time

import pytest
import aloha.episode_attempt as episode_attempt

from aloha.episode_attempt import (
    AttemptDecision,
    AttemptOutcome,
    EpisodeAttempt,
    EpisodeAttemptRunner,
    cleanup_attempt_artifact,
    prepare_return_modes,
    restore_teleop_modes,
    run_motion_worker_safely,
    sample_registers_interruptibly,
    stop_diagnostic_worker,
    wait_for_diagnostic_interval,
)


def test_return_mode_timeout_fails_before_torque_followers_or_motion_setup():
    calls = []
    leader = object()
    follower = object()

    def set_operating_modes(*args):
        calls.append(("mode", args))
        raise TimeoutError("leader_left arm operating mode timed out")

    with pytest.raises(TimeoutError, match="leader_left arm operating mode"):
        prepare_return_modes(
            {"leader_left": leader},
            {"follower_left": follower},
            continuous_roll_joints=False,
            set_operating_modes=set_operating_modes,
            set_follower_arm_mode=lambda *args, **kwargs: calls.append(
                ("follower", args, kwargs)
            ),
            configure_follower_gripper=lambda *args: calls.append(
                ("gripper", args)
            ),
            torque_on=lambda *args: calls.append(("torque", args)),
        )

    assert calls == [("mode", (leader, "group", "arm", "position"))]


def test_prepare_return_modes_configures_each_named_follower():
    calls = []
    followers = {
        "follower_left": object(),
        "follower_right": object(),
    }

    prepare_return_modes(
        {},
        followers,
        continuous_roll_joints=False,
        set_operating_modes=lambda *args: calls.append(("mode", args)),
        set_follower_arm_mode=lambda bot, **kwargs: calls.append(
            ("follower_arm", bot, kwargs)
        ),
        configure_follower_gripper=lambda name, bot: calls.append(
            ("follower_gripper", name, bot)
        ),
        torque_on=lambda bot: calls.append(("torque_on", bot)),
    )

    for name, follower in followers.items():
        assert ("follower_gripper", name, follower) in calls
        assert ("torque_on", follower) in calls


def make_runner(
    decisions,
    calls,
    *,
    return_results=None,
    should_exit=None,
    auto_start_after_retry=False,
):
    decision_iter = iter(decisions)
    return_iter = iter(return_results or [True] * len(decisions))
    exit_checks = iter(should_exit or [])

    def wait_for_start(attempt):
        calls.append("wait_for_b")

    def collect(attempt):
        calls.append("collect")
        attempt.timesteps.append(f"ts-{len(calls)}")
        attempt.actions.append(f"action-{len(calls)}")
        attempt.dt_history.append(f"dt-{len(calls)}")
        return next(decision_iter)

    def return_to_start(attempt, decision):
        suffix = "discard" if decision is AttemptDecision.DISCARD else "save"
        calls.append(f"return_to_start_for_{suffix}")
        return next(return_iter)

    def discard_attempt(attempt):
        calls.append("discard_artifact")

    def commit_attempt(attempt):
        calls.append("commit_artifact")

    def complete_retry(*, auto_start):
        assert auto_start is auto_start_after_retry
        calls.append("complete_retry")

    def reset_timestep():
        calls.append("reset_attempt")

    def is_exit_requested():
        return next(exit_checks, False)

    return EpisodeAttemptRunner(
        dataset_name="episode_7",
        start_pose=("same", "pose"),
        wait_for_start=wait_for_start,
        collect=collect,
        return_to_start=return_to_start,
        discard_attempt=discard_attempt,
        commit_attempt=commit_attempt,
        complete_retry=complete_retry,
        reset_timestep=reset_timestep,
        is_exit_requested=is_exit_requested,
        auto_start_after_retry=auto_start_after_retry,
    )


def test_discard_retries_then_commits_only_the_accepted_attempt():
    calls = []
    runner = make_runner(
        [AttemptDecision.DISCARD, AttemptDecision.SAVE],
        calls,
    )

    outcome = runner.run()

    assert calls == [
        "wait_for_b",
        "collect",
        "return_to_start_for_discard",
        "discard_artifact",
        "complete_retry",
        "reset_attempt",
        "wait_for_b",
        "collect",
        "return_to_start_for_save",
        "commit_artifact",
    ]
    assert outcome is AttemptOutcome.SAVE


def test_each_attempt_has_fresh_buffers_and_same_session_identity():
    seen = []

    def collect(attempt):
        seen.append(attempt)
        assert attempt.dataset_name == "episode_7"
        assert attempt.start_pose == ("same", "pose")
        attempt.timesteps.append(len(seen))
        attempt.actions.append(len(seen))
        attempt.dt_history.append(len(seen))
        return (
            AttemptDecision.DISCARD
            if len(seen) == 1
            else AttemptDecision.SAVE
        )

    runner = EpisodeAttemptRunner(
        dataset_name="episode_7",
        start_pose=("same", "pose"),
        wait_for_start=lambda _attempt: None,
        collect=collect,
        return_to_start=lambda _attempt, _decision: True,
        discard_attempt=lambda _attempt: None,
        commit_attempt=lambda _attempt: None,
        complete_retry=lambda **_kwargs: None,
        reset_timestep=lambda: None,
        is_exit_requested=lambda: False,
    )

    assert runner.run() is AttemptOutcome.SAVE
    assert seen[0] is not seen[1]
    assert seen[0].timesteps == [1]
    assert seen[1].timesteps == [2]
    assert seen[0].actions == [1]
    assert seen[1].actions == [2]
    assert seen[0].dt_history == [1]
    assert seen[1].dt_history == [2]


def test_cleanup_does_not_unlink_artifact_while_diagnostics_are_alive():
    calls = []
    logs = []
    forced = []

    class Artifact:
        diagnostic_path = "/tmp/.episode_7.attempt-alive.jsonl"

        def discard(self):
            calls.append("unlink")

    attempt = EpisodeAttempt("episode_7", "pose")
    attempt.resources["artifact"] = Artifact()

    cleaned = cleanup_attempt_artifact(
        attempt,
        stop_diagnostics=lambda _attempt: (
            calls.append("join") or (_ for _ in ()).throw(
                RuntimeError("diagnostic thread still alive after timeout")
            )
        ),
        force_no_save=forced.append,
        logger=logs.append,
    )

    assert cleaned is False
    assert calls == ["join"]
    assert forced == ["attempt cleanup failed"]
    assert any(
        "/tmp/.episode_7.attempt-alive.jsonl" in message
        and "still alive" in message
        for message in logs
    )


def test_stop_diagnostic_worker_sets_event_joins_and_detects_timeout():
    calls = []

    class StopEvent:
        def set(self):
            calls.append("set")

    class NeverStops:
        def join(self, timeout):
            calls.append(("join", timeout))

        def is_alive(self):
            return True

    attempt = EpisodeAttempt("episode_7", "pose")
    attempt.resources["diagnostic_stop_event"] = StopEvent()
    attempt.resources["diagnostic_thread"] = NeverStops()

    with pytest.raises(RuntimeError, match="still alive after 0.25 seconds"):
        stop_diagnostic_worker(attempt, timeout=0.25)

    assert calls == ["set", ("join", 0.25)]
    assert "diagnostics_stopped" not in attempt.resources


def test_request_diagnostic_stop_sets_event_without_joining():
    calls = []

    class StopEvent:
        def set(self):
            calls.append("set")

    class Thread:
        def join(self, _timeout):
            calls.append("join")

    attempt = EpisodeAttempt("episode_7", "pose")
    attempt.resources["diagnostic_stop_event"] = StopEvent()
    attempt.resources["diagnostic_thread"] = Thread()

    episode_attempt.request_diagnostic_stop(attempt)
    episode_attempt.request_diagnostic_stop(attempt)

    assert calls == ["set"]
    assert attempt.resources["diagnostic_stop_requested"] is True
    assert "diagnostics_stopped" not in attempt.resources


def test_request_diagnostic_stop_without_worker_is_idempotent():
    attempt = EpisodeAttempt("episode_7", "pose")

    episode_attempt.request_diagnostic_stop(attempt)
    episode_attempt.request_diagnostic_stop(attempt)

    assert attempt.resources["diagnostic_stop_requested"] is True


def test_motion_worker_timeout_waits_for_completion_before_raising():
    release = threading.Event()
    worker_started = threading.Event()
    returned = threading.Event()
    logs = []

    def worker():
        worker_started.set()
        release.wait()

    def invoke():
        with pytest.raises(TimeoutError, match="nominal timeout"):
            run_motion_worker_safely(worker, nominal_timeout=0.01, logger=logs.append)
        returned.set()

    caller = threading.Thread(target=invoke)
    caller.start()
    assert worker_started.wait(timeout=1.0)
    time.sleep(0.05)
    assert not returned.is_set()
    assert caller.is_alive()

    release.set()
    caller.join(timeout=1.0)

    assert returned.is_set()
    assert any("waiting for safe completion" in message for message in logs)


def test_diagnostic_interval_wait_stops_promptly_at_point_one_hz():
    stop = threading.Event()
    finished = threading.Event()

    worker = threading.Thread(
        target=lambda: (
            wait_for_diagnostic_interval(stop, 10.0),
            finished.set(),
        )
    )
    worker.start()
    time.sleep(0.02)
    started = time.monotonic()
    stop.set()
    worker.join(timeout=0.5)

    assert finished.is_set()
    assert time.monotonic() - started < 0.5


def test_stop_mid_register_sample_exits_worker_within_join_bound():
    stop = threading.Event()
    first_register_started = threading.Event()
    calls = []

    def slow_read(register_name):
        calls.append(register_name)
        first_register_started.set()
        time.sleep(0.05)
        return register_name

    def diagnostic_worker():
        sample_registers_interruptibly(
            ["r1", "r2", "r3"],
            read_register=slow_read,
            stop_event=stop,
        )

    worker = threading.Thread(target=diagnostic_worker)
    worker.start()
    assert first_register_started.wait(timeout=1.0)
    attempt = EpisodeAttempt("episode_7", "pose")
    attempt.resources["diagnostic_stop_event"] = stop
    attempt.resources["diagnostic_thread"] = worker

    stop_diagnostic_worker(attempt, timeout=0.25)

    assert calls == ["r1"]
    assert not worker.is_alive()
    assert attempt.resources["diagnostics_stopped"] is True


class FakeCore:
    def __init__(self, calls):
        self.calls = calls

    def robot_set_operating_modes(self, *args):
        self.calls.append(("mode", args))

    def robot_torque_enable(self, *args):
        self.calls.append(("torque_enable", args))


class FakeRobot:
    def __init__(self, calls):
        self.core = FakeCore(calls)


@pytest.mark.parametrize("gravity_compensation", [False, True])
def test_restore_teleop_modes_restores_followers_and_leaders(gravity_compensation):
    calls = []
    robots = {
        "leader_left": FakeRobot(calls),
        "follower_left": FakeRobot(calls),
        "follower_right": FakeRobot(calls),
    }

    restore_teleop_modes(
        robots,
        gravity_compensation=gravity_compensation,
        continuous_roll_joints=True,
        set_follower_arm_mode=lambda bot, **kwargs: calls.append(
            ("follower_arm", bot, kwargs)
        ),
        set_operating_modes=lambda bot, *args: calls.append(
            ("mode", bot, args)
        ),
        configure_follower_gripper=lambda name, bot: calls.append(
            ("follower_gripper", name, bot)
        ),
        torque_enable=lambda bot, *args: calls.append(
            ("torque_enable", bot, args)
        ),
        torque_on=lambda bot: calls.append(("torque_on", bot)),
        torque_off=lambda bot: calls.append(("torque_off", bot)),
        enable_gravity_compensation=lambda bot: calls.append(
            ("gravity", bot)
        ),
    )

    follower = robots["follower_left"]
    follower_right = robots["follower_right"]
    leader = robots["leader_left"]
    assert ("follower_arm", follower, {"continuous_roll_joints": True}) in calls
    assert (
        "follower_gripper",
        "follower_left",
        follower,
    ) in calls
    assert (
        "follower_gripper",
        "follower_right",
        follower_right,
    ) in calls
    assert ("torque_on", follower) in calls
    assert ("torque_on", follower_right) in calls
    assert ("torque_enable", leader, ("single", "gripper", False)) in calls
    if gravity_compensation:
        assert ("gravity", leader) in calls
        assert ("torque_off", leader) not in calls
    else:
        assert ("torque_off", leader) in calls
        assert ("gravity", leader) not in calls


def test_cleanup_discard_error_marks_no_save_and_caller_can_continue_shutdown():
    logs = []
    forced = []
    shutdown = []

    class Artifact:
        diagnostic_path = "/tmp/.episode_7.attempt-unlink.jsonl"

        def discard(self):
            raise OSError("injected unlink error")

    attempt = EpisodeAttempt("episode_7", "pose")
    attempt.resources["artifact"] = Artifact()

    try:
        assert cleanup_attempt_artifact(
            attempt,
            stop_diagnostics=lambda _attempt: None,
            force_no_save=forced.append,
            logger=logs.append,
        ) is False
    finally:
        shutdown.append("sleep_and_robot_shutdown")

    assert forced == ["attempt cleanup failed"]
    assert shutdown == ["sleep_and_robot_shutdown"]
    assert any(
        "/tmp/.episode_7.attempt-unlink.jsonl" in message
        and "injected unlink error" in message
        for message in logs
    )


def test_retry_return_failure_discards_attempt_and_exits_without_save():
    calls = []
    runner = make_runner(
        [AttemptDecision.DISCARD],
        calls,
        return_results=[False],
    )

    assert runner.run() is AttemptOutcome.EXIT_NO_SAVE
    assert calls == [
        "wait_for_b",
        "collect",
        "return_to_start_for_discard",
        "discard_artifact",
    ]
    assert "commit_artifact" not in calls


def test_s_during_retry_return_discards_and_exits_without_resetting():
    calls = []
    stop_requested = {"value": False}

    def return_to_start(_attempt, _decision):
        calls.append("return_to_start_for_discard")
        stop_requested["value"] = True
        return True

    runner = EpisodeAttemptRunner(
        dataset_name="episode_7",
        start_pose="fixed-pose",
        wait_for_start=lambda _attempt: calls.append("wait_for_b"),
        collect=lambda _attempt: (
            calls.append("collect") or AttemptDecision.DISCARD
        ),
        return_to_start=return_to_start,
        discard_attempt=lambda _attempt: calls.append("discard_artifact"),
        commit_attempt=lambda _attempt: calls.append("commit_artifact"),
        complete_retry=lambda **_kwargs: True,
        reset_timestep=lambda: calls.append("reset_attempt"),
        is_exit_requested=lambda: stop_requested["value"],
    )

    assert runner.run() is AttemptOutcome.EXIT_NO_SAVE
    assert calls == [
        "wait_for_b",
        "collect",
        "return_to_start_for_discard",
        "discard_artifact",
    ]


def test_s_after_collect_wins_before_retry_return_starts():
    calls = []
    stop_requested = {"value": False}

    def collect(_attempt):
        calls.append("collect")
        stop_requested["value"] = True
        return AttemptDecision.DISCARD

    runner = EpisodeAttemptRunner(
        dataset_name="episode_7",
        start_pose="fixed-pose",
        wait_for_start=lambda _attempt: calls.append("wait_for_b"),
        collect=collect,
        return_to_start=lambda _attempt, _decision: (
            calls.append("return_to_start_for_discard") or True
        ),
        discard_attempt=lambda _attempt: calls.append("discard_artifact"),
        commit_attempt=lambda _attempt: calls.append("commit_artifact"),
        complete_retry=lambda **_kwargs: True,
        reset_timestep=lambda: calls.append("reset_attempt"),
        is_exit_requested=lambda: stop_requested["value"],
    )

    assert runner.run() is AttemptOutcome.EXIT_NO_SAVE
    assert calls == ["wait_for_b", "collect", "discard_artifact"]


def test_stop_winning_complete_retry_race_exits_without_resetting():
    calls = []
    runner = EpisodeAttemptRunner(
        dataset_name="episode_7",
        start_pose="fixed-pose",
        wait_for_start=lambda _attempt: calls.append("wait_for_b"),
        collect=lambda _attempt: (
            calls.append("collect") or AttemptDecision.DISCARD
        ),
        return_to_start=lambda _attempt, _decision: (
            calls.append("return_to_start_for_discard") or True
        ),
        discard_attempt=lambda _attempt: calls.append("discard_artifact"),
        commit_attempt=lambda _attempt: calls.append("commit_artifact"),
        complete_retry=lambda **_kwargs: False,
        reset_timestep=lambda: calls.append("reset_attempt"),
        is_exit_requested=lambda: False,
    )

    assert runner.run() is AttemptOutcome.EXIT_NO_SAVE
    assert calls == [
        "wait_for_b",
        "collect",
        "return_to_start_for_discard",
        "discard_artifact",
    ]


def test_exit_during_collection_discards_without_return_or_save():
    calls = []
    runner = make_runner(
        [AttemptDecision.EXIT_NO_SAVE],
        calls,
    )

    assert runner.run() is AttemptOutcome.EXIT_NO_SAVE
    assert calls == ["wait_for_b", "collect", "discard_artifact"]


def test_discard_never_commits_a_discarded_attempt():
    saved_attempts = []
    attempts = []

    def collect(attempt):
        attempts.append(attempt)
        return (
            AttemptDecision.DISCARD
            if len(attempts) == 1
            else AttemptDecision.SAVE
        )

    runner = EpisodeAttemptRunner(
        dataset_name="episode_7",
        start_pose="fixed-pose",
        wait_for_start=lambda _attempt: None,
        collect=collect,
        return_to_start=lambda _attempt, _decision: True,
        discard_attempt=lambda _attempt: None,
        commit_attempt=saved_attempts.append,
        complete_retry=lambda **_kwargs: None,
        reset_timestep=lambda: None,
        is_exit_requested=lambda: False,
    )

    assert runner.run() is AttemptOutcome.SAVE
    assert saved_attempts == [attempts[1]]
    assert all(saved is not attempts[0] for saved in saved_attempts)


def test_gripper_retry_auto_starts_next_attempt_without_waiting():
    calls = []
    runner = make_runner(
        [AttemptDecision.DISCARD, AttemptDecision.SAVE],
        calls,
        auto_start_after_retry=True,
    )

    assert runner.run() is AttemptOutcome.SAVE
    assert calls.count("wait_for_b") == 1
