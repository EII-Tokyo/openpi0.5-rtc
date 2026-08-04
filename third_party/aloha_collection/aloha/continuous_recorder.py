from enum import Enum
from typing import (
    Any,
    Callable,
    ContextManager,
    TypeVar,
)


RuntimeT = TypeVar("RuntimeT")
IndexT = TypeVar("IndexT")
Logger = Callable[[str], Any]


def _best_effort_log(logger: Logger, message: str) -> None:
    try:
        logger(message)
    except BaseException:
        pass


class SessionOutcome(Enum):
    """Terminal or continuation decision for a continuous recording session.

    ``CONTINUE_NEXT_EPISODE`` intentionally requires sleep so a later failure
    during successor allocation still takes the fail-safe cleanup path.
    """

    CONTINUE_NEXT_EPISODE = "continue_next_episode"
    EXIT_SAVE_AND_SLEEP = "exit_save_and_sleep"
    EXIT_DISCARD_AND_SLEEP = "exit_discard_and_sleep"
    EXIT_SAVE_WITHOUT_SLEEP = "exit_save_without_sleep"
    EXIT_FAILURE_AND_SLEEP = "exit_failure_and_sleep"

    @property
    def requires_sleep(self) -> bool:
        return self is not SessionOutcome.EXIT_SAVE_WITHOUT_SLEEP


class EpisodeFinalizer:
    """Atomically arbitrate stop flags and validated-episode callbacks."""

    def __init__(
        self,
        lock: ContextManager[Any],
        stop_no_save: Any,
        stop_and_save: Any,
        skip_sleep: Any,
        complete_save: Callable[[], bool],
        logger: Logger = print,
    ) -> None:
        self._lock = lock
        self._stop_no_save = stop_no_save
        self._stop_and_save = stop_and_save
        self._skip_sleep = skip_sleep
        self._complete_save = complete_save
        self._logger = logger

    def _log(self, message: str) -> None:
        _best_effort_log(self._logger, message)

    def _discard_best_effort(self, discard: Callable[[], Any]) -> bool:
        try:
            discard()
        except BaseException as exc:
            self._log(f"Episode discard failed: {exc!r}")
            return False
        return True

    def _publish_or_discard(
        self,
        publish: Callable[[], Any],
        discard: Callable[[], Any],
    ) -> bool:
        try:
            publish()
        except BaseException as exc:
            self._log(f"Episode publication failed: {exc!r}")
            self._discard_best_effort(discard)
            return False
        return True

    def finalize_validated(
        self,
        publish: Callable[[], Any],
        discard: Callable[[], Any],
        *,
        continue_after_save: bool = True,
    ) -> SessionOutcome:
        """Finalize one validated episode while holding the shared lock."""

        with self._lock:
            if self._stop_no_save.is_set():
                return (
                    SessionOutcome.EXIT_DISCARD_AND_SLEEP
                    if self._discard_best_effort(discard)
                    else SessionOutcome.EXIT_FAILURE_AND_SLEEP
                )

            if self._stop_and_save.is_set():
                if not self._publish_or_discard(publish, discard):
                    return SessionOutcome.EXIT_FAILURE_AND_SLEEP
                return (
                    SessionOutcome.EXIT_SAVE_WITHOUT_SLEEP
                    if self._skip_sleep.is_set()
                    else SessionOutcome.EXIT_SAVE_AND_SLEEP
                )

            if not self._publish_or_discard(publish, discard):
                return SessionOutcome.EXIT_FAILURE_AND_SLEEP

            if not continue_after_save:
                return SessionOutcome.EXIT_SAVE_AND_SLEEP

            try:
                completed = self._complete_save()
            except BaseException as exc:
                self._log(f"Episode trigger completion failed: {exc!r}")
                return SessionOutcome.EXIT_FAILURE_AND_SLEEP

            if not completed:
                self._log("Episode finalization failed: complete_save returned False")
                return SessionOutcome.EXIT_FAILURE_AND_SLEEP

            return SessionOutcome.CONTINUE_NEXT_EPISODE

    def finalize_invalid(
        self,
        discard: Callable[[], Any],
    ) -> SessionOutcome:
        """Discard invalid staged data under the shared command lock."""
        with self._lock:
            self._discard_best_effort(discard)
            return SessionOutcome.EXIT_FAILURE_AND_SLEEP


def finalize_staged_episode(
    *,
    staged: Any,
    validate: Callable[[], Any],
    finalizer: EpisodeFinalizer,
    allow_existing_destination: bool = False,
    continue_after_save: bool = True,
    logger: Logger = print,
) -> SessionOutcome:
    """Validate one owned staging directory and finalize it atomically."""
    try:
        validate()
    except BaseException as exc:
        _best_effort_log(logger, f"Episode validation failed: {exc!r}")
        return finalizer.finalize_invalid(staged.discard)

    return finalizer.finalize_validated(
        publish=lambda: staged.publish(
            allow_existing_destination=allow_existing_destination,
        ),
        discard=staged.discard,
        continue_after_save=continue_after_save,
    )


def run_continuous_session(
    *,
    create_runtime: Callable[[], RuntimeT],
    capture_episode: Callable[[RuntimeT, IndexT], SessionOutcome],
    next_index: Callable[[IndexT], IndexT],
    initial_index: IndexT,
    final_cleanup: Callable[[RuntimeT, SessionOutcome], Any],
    logger: Logger = print,
) -> SessionOutcome:
    """Run episode captures inside one runtime and clean it up exactly once.

    ``create_runtime`` runs before cleanup ownership begins. If it raises, the
    factory must roll back any partial creation itself and ``final_cleanup`` is
    not called.
    """

    runtime = create_runtime()
    index = initial_index
    cleanup_outcome = SessionOutcome.EXIT_FAILURE_AND_SLEEP
    active_session_failure = False
    try:
        while True:
            outcome = capture_episode(runtime, index)
            if not isinstance(outcome, SessionOutcome):
                raise TypeError(
                    "capture_episode returned invalid outcome "
                    f"{outcome!r} of type {type(outcome).__name__}; "
                    "expected SessionOutcome"
                )
            if outcome is not SessionOutcome.CONTINUE_NEXT_EPISODE:
                cleanup_outcome = outcome
                return outcome
            index = next_index(index)
    except BaseException:
        active_session_failure = True
        raise
    finally:
        try:
            final_cleanup(runtime, cleanup_outcome)
        except BaseException as exc:
            if not active_session_failure:
                raise
            _best_effort_log(
                logger,
                f"Final cleanup failed after session error: {exc!r}",
            )
