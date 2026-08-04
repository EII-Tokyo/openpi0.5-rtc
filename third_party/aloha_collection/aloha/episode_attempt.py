"""Lifecycle management for diagnostics produced by one recording attempt."""

from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class AttemptDecision(Enum):
    """The collection result for one in-memory attempt."""

    DISCARD = "discard"
    SAVE = "save"
    EXIT_NO_SAVE = "exit_no_save"


class AttemptOutcome(Enum):
    """The process-level result of the retry loop."""

    SAVE = "save"
    EXIT_NO_SAVE = "exit_no_save"


def guarded_teleop_step(
    *,
    health_check: Callable[[], None],
    read_action: Callable[[], Any],
    command: Callable[[Any], Any],
    clock: Callable[[], float],
) -> tuple[Any, Any, float]:
    """Check source freshness before reading and commanding one action."""
    health_check()
    action = read_action()
    action_read_time = clock()
    return action, command(action), action_read_time


@dataclass(frozen=True)
class EpisodeIndexDecision:
    """Separate permission to proceed from confirmed replacement authority."""

    proceed: bool
    allow_existing: bool


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def find_next_available_episode_index(
    dataset_dir: str | os.PathLike[str],
    *,
    start_index: int,
    data_suffix: str = "hdf5",
    logger: Callable[[str], None] = print,
) -> int:
    """Find the first index free of directory- and legacy-file-form entries.

    This is a point-in-time snapshot and does not reserve the returned index.
    """
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    dataset_path = Path(dataset_dir)
    episode_idx = start_index
    while True:
        episode_dir = dataset_path / f"episode_{episode_idx}"
        legacy_file = dataset_path / f"episode_{episode_idx}.{data_suffix}"
        claim_entry = dataset_path / f".episode_{episode_idx}.claim"
        if (
            not _path_entry_exists(episode_dir)
            and not _path_entry_exists(legacy_file)
            and not _path_entry_exists(claim_entry)
        ):
            return episode_idx
        logger(f"[episode-index] episode_{episode_idx} already exists; skipping.")
        episode_idx += 1


def check_episode_index(
    dataset_dir: str | os.PathLike[str],
    episode_idx: int,
    data_suffix: str = "hdf5",
    *,
    input_fn: Callable[[str], str] = input,
    logger: Callable[[str], None] = print,
) -> EpisodeIndexDecision:
    """Prompt only for an existing path and return explicit overwrite authority."""
    dataset_path = Path(dataset_dir)
    episode_dir = dataset_path / f"episode_{episode_idx}"
    legacy_episode_file = dataset_path / f"episode_{episode_idx}.{data_suffix}"
    existing_path = (
        episode_dir
        if episode_dir.is_dir()
        else legacy_episode_file
        if legacy_episode_file.is_file()
        else None
    )
    if existing_path is None:
        return EpisodeIndexDecision(proceed=True, allow_existing=False)

    user_input = input_fn(
        f"Episode path '{existing_path}' already exists. "
        "Do you want to overwrite it? (y/n): "
    ).strip().lower()
    if user_input == "y":
        logger(f"Overwriting episode {episode_idx}.")
        return EpisodeIndexDecision(proceed=True, allow_existing=True)

    logger("Not overwriting the file. Operation aborted.")
    return EpisodeIndexDecision(proceed=False, allow_existing=False)


@dataclass
class EpisodeAttempt:
    """Fresh mutable buffers and immutable identity for one attempt."""

    dataset_name: str
    start_pose: Any
    timesteps: list[Any] = field(default_factory=list)
    actions: list[Any] = field(default_factory=list)
    dt_history: list[Any] = field(default_factory=list)
    resources: dict[str, Any] = field(default_factory=dict)


def wait_for_diagnostic_interval(
    stop_event: threading.Event,
    timeout: float,
) -> bool:
    """Wait interruptibly; return true when a stop was requested."""
    return stop_event.wait(timeout=max(0.0, timeout))


def sample_registers_interruptibly(
    requests: list[Any],
    *,
    read_register: Callable[[Any], Any],
    stop_event: threading.Event | None,
) -> tuple[list[tuple[Any, Any]], bool]:
    """Check stop between every bounded register call and preserve partial data."""
    results: list[tuple[Any, Any]] = []
    for request in requests:
        if stop_event is not None and stop_event.is_set():
            return results, True
        results.append((request, read_register(request)))
    interrupted = stop_event is not None and stop_event.is_set()
    return results, interrupted


def join_motion_thread_safely(
    thread: threading.Thread,
    *,
    nominal_timeout: float,
    logger: Callable[[str], None],
) -> None:
    """Never return or raise while a robot motion worker is still alive."""
    pending_error: BaseException | None = None
    timed_out = False
    try:
        thread.join(timeout=nominal_timeout)
        timed_out = thread.is_alive()
    except BaseException as exc:
        pending_error = exc
    finally:
        if thread.is_alive():
            logger(
                "[motion] nominal timeout/interruption; "
                "waiting for safe completion before cleanup"
            )
        while thread.is_alive():
            try:
                thread.join()
            except BaseException as exc:
                if pending_error is None:
                    pending_error = exc

    if pending_error is not None:
        raise pending_error
    if timed_out:
        raise TimeoutError(
            "motion worker exceeded nominal timeout but completed safely"
        )


def run_motion_worker_safely(
    worker: Callable[[], None],
    *,
    nominal_timeout: float,
    logger: Callable[[str], None],
) -> None:
    """Start a non-daemon motion worker and guarantee its completion."""
    thread = threading.Thread(target=worker, daemon=False)
    thread.start()
    join_motion_thread_safely(
        thread,
        nominal_timeout=nominal_timeout,
        logger=logger,
    )


def prepare_return_modes(
    leader_bots: dict[str, Any],
    follower_bots: dict[str, Any],
    *,
    continuous_roll_joints: bool,
    set_operating_modes: Callable[..., None],
    set_follower_arm_mode: Callable[..., None],
    configure_follower_gripper: Callable[[str, Any], None],
    torque_on: Callable[[Any], None],
) -> None:
    """Prepare every robot for return motion, failing before motion on error."""
    for leader in leader_bots.values():
        set_operating_modes(leader, "group", "arm", "position")
        set_operating_modes(leader, "single", "gripper", "position")
        torque_on(leader)
    for follower_name, follower in follower_bots.items():
        set_follower_arm_mode(
            follower,
            continuous_roll_joints=continuous_roll_joints,
        )
        configure_follower_gripper(follower_name, follower)
        torque_on(follower)


def restore_teleop_modes(
    robots: dict[str, Any],
    *,
    gravity_compensation: bool,
    continuous_roll_joints: bool,
    set_follower_arm_mode: Callable[..., None],
    set_operating_modes: Callable[..., None],
    configure_follower_gripper: Callable[[str, Any], None],
    torque_enable: Callable[..., None],
    torque_on: Callable[[Any], None],
    torque_off: Callable[[Any], None],
    enable_gravity_compensation: Callable[[Any], None],
) -> None:
    """Restore leader/follower modes required for a fresh teleop attempt."""
    follower_bots = {
        name: bot for name, bot in robots.items() if "follower" in name
    }
    leader_bots = {
        name: bot for name, bot in robots.items() if "leader" in name
    }
    for follower_name, follower in follower_bots.items():
        set_follower_arm_mode(
            follower,
            continuous_roll_joints=continuous_roll_joints,
        )
        configure_follower_gripper(follower_name, follower)
        torque_on(follower)

    for leader in leader_bots.values():
        torque_enable(leader, "single", "gripper", False)
        if gravity_compensation:
            enable_gravity_compensation(leader)
        else:
            torque_off(leader)


def request_diagnostic_stop(attempt: EpisodeAttempt) -> None:
    """Request diagnostic shutdown without waiting for the worker."""
    if attempt.resources.get("diagnostic_stop_requested"):
        return
    stop_event = attempt.resources.get("diagnostic_stop_event")
    if stop_event is not None:
        stop_event.set()
    attempt.resources["diagnostic_stop_requested"] = True


def stop_diagnostic_worker(
    attempt: EpisodeAttempt,
    *,
    timeout: float = 2.0,
) -> None:
    """Stop and join one attempt's diagnostic worker, or fail before unlink."""
    thread = attempt.resources.get("diagnostic_thread")
    request_diagnostic_stop(attempt)
    if thread is not None:
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise RuntimeError(
                f"diagnostic thread still alive after {timeout} seconds"
            )
    attempt.resources["diagnostics_stopped"] = True


def cleanup_attempt_artifact(
    attempt: EpisodeAttempt,
    *,
    stop_diagnostics: Callable[[EpisodeAttempt], None],
    force_no_save: Callable[[str], None],
    logger: Callable[[str], None],
) -> bool:
    """Stop diagnostics and discard its artifact without racing the writer."""
    artifact = attempt.resources.get("artifact")
    artifact_path = (
        str(artifact.diagnostic_path)
        if artifact is not None
        else "<no diagnostic artifact>"
    )
    try:
        stop_diagnostics(attempt)
    except BaseException as exc:
        logger(
            f"[attempt-cleanup] {artifact_path}: "
            f"failed to stop diagnostics safely: {exc}"
        )
        force_no_save("attempt cleanup failed")
        return False

    if artifact is None:
        return True
    try:
        artifact.discard()
    except BaseException as exc:
        logger(f"[attempt-cleanup] {artifact_path}: discard failed: {exc}")
        force_no_save("attempt cleanup failed")
        return False
    return True


class EpisodeAttemptRunner:
    """Orchestrate discard/retry without owning robot-specific operations."""

    def __init__(
        self,
        *,
        dataset_name: str,
        start_pose: Any,
        wait_for_start: Callable[[EpisodeAttempt], None],
        collect: Callable[[EpisodeAttempt], AttemptDecision],
        return_to_start: Callable[[EpisodeAttempt, AttemptDecision], bool],
        discard_attempt: Callable[[EpisodeAttempt], None],
        commit_attempt: Callable[[EpisodeAttempt], None],
        complete_retry: Callable[..., None],
        reset_timestep: Callable[[], None],
        is_exit_requested: Callable[[], bool],
        auto_start_after_retry: bool = False,
    ) -> None:
        self._dataset_name = dataset_name
        self._start_pose = start_pose
        self._wait_for_start = wait_for_start
        self._collect = collect
        self._return_to_start = return_to_start
        self._discard_attempt = discard_attempt
        self._commit_attempt = commit_attempt
        self._complete_retry = complete_retry
        self._reset_timestep = reset_timestep
        self._is_exit_requested = is_exit_requested
        self._auto_start_after_retry = auto_start_after_retry

    def run(self) -> AttemptOutcome:
        """Run fresh attempts until one is accepted or the session must exit."""
        wait_for_start = True
        while True:
            attempt = EpisodeAttempt(self._dataset_name, self._start_pose)
            if wait_for_start:
                self._wait_for_start(attempt)
            if self._is_exit_requested():
                self._discard_attempt(attempt)
                return AttemptOutcome.EXIT_NO_SAVE

            decision = self._collect(attempt)
            if (
                decision is AttemptDecision.EXIT_NO_SAVE
                or self._is_exit_requested()
            ):
                self._discard_attempt(attempt)
                return AttemptOutcome.EXIT_NO_SAVE

            returned = self._return_to_start(attempt, decision)
            if decision is AttemptDecision.DISCARD:
                discarded = self._discard_attempt(attempt)
                if discarded is False:
                    return AttemptOutcome.EXIT_NO_SAVE
                if not returned or self._is_exit_requested():
                    return AttemptOutcome.EXIT_NO_SAVE
                retry_completed = self._complete_retry(
                    auto_start=self._auto_start_after_retry
                )
                if retry_completed is False:
                    return AttemptOutcome.EXIT_NO_SAVE
                self._reset_timestep()
                wait_for_start = not self._auto_start_after_retry
                continue

            if not returned or self._is_exit_requested():
                self._discard_attempt(attempt)
                return AttemptOutcome.EXIT_NO_SAVE
            self._commit_attempt(attempt)
            return AttemptOutcome.SAVE


class AttemptArtifact:
    """Own a temporary diagnostic file until it is discarded or committed."""

    __slots__ = ("__diagnostic_path", "__state")
    __construction_token = object()

    def __init__(
        self,
        diagnostic_path: Path,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not self.__construction_token:
            raise TypeError("use AttemptArtifact.create() to construct artifacts")
        self.__diagnostic_path = diagnostic_path
        self.__state = "active"

    @property
    def diagnostic_path(self) -> Path:
        """The immutable path owned by this attempt."""
        return self.__diagnostic_path

    @classmethod
    def create(
        cls,
        dataset_dir: str | os.PathLike[str],
        episode_name: str,
    ) -> "AttemptArtifact":
        dataset_path = Path(dataset_dir)
        dataset_path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=dataset_path,
            prefix=f".{episode_name}.attempt-",
            suffix=".jsonl",
        ) as temporary:
            diagnostic_path = Path(temporary.name)
        return cls(diagnostic_path, _token=cls.__construction_token)

    def discard(self) -> None:
        """Remove only this attempt's temporary file."""
        if self.__state != "active":
            return
        self.diagnostic_path.unlink(missing_ok=True)
        self.__state = "discarded"

    def commit(
        self,
        final_episode_dir: str | os.PathLike[str],
        *,
        allow_existing: bool = False,
    ) -> Path:
        """Atomically move this attempt's diagnostic into its final directory."""
        if self.__state == "committed":
            raise RuntimeError("attempt artifact is already committed")
        if self.__state == "discarded":
            raise RuntimeError("attempt artifact was discarded")

        episode_path = Path(final_episode_dir)
        created_episode_dir = False
        if episode_path.exists():
            if not allow_existing or not episode_path.is_dir():
                raise FileExistsError(f"episode path already exists: {episode_path}")
        else:
            episode_path.mkdir(parents=False, exist_ok=False)
            created_episode_dir = True
        committed_path = episode_path / "motor6_diagnostics.jsonl"
        try:
            os.replace(self.diagnostic_path, committed_path)
        except BaseException:
            if created_episode_dir:
                try:
                    episode_path.rmdir()
                except OSError:
                    pass
            raise
        self.__state = "committed"
        return committed_path

    def commit_into_existing(
        self,
        final_episode_dir: str | os.PathLike[str],
        *,
        allow_existing_destination: bool = False,
    ) -> Path:
        """Commit after data save succeeds, without creating the episode dir."""
        if self.__state == "committed":
            raise RuntimeError("attempt artifact is already committed")
        if self.__state == "discarded":
            raise RuntimeError("attempt artifact was discarded")

        episode_path = Path(final_episode_dir)
        if not episode_path.is_dir():
            raise FileNotFoundError(
                f"final episode directory does not exist: {episode_path}"
            )
        committed_path = episode_path / "motor6_diagnostics.jsonl"
        if committed_path.exists() and not allow_existing_destination:
            raise FileExistsError(
                f"diagnostic destination already exists: {committed_path}"
            )
        os.replace(self.diagnostic_path, committed_path)
        self.__state = "committed"
        return committed_path
