"""Thread-safe recording state transitions shared by local and remote input."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum


class TriggerResult(Enum):
    STARTED = "started"
    STOPPED = "stopped"
    DISCARD_STARTED = "discard_started"
    NOT_RECORDING = "not_recording"
    NOT_READY = "not_ready"
    NO_SAMPLES = "no_samples"
    WRONG_START_MODE = "wrong_start_mode"
    IGNORED = "ignored"


class RecordingPhase(Enum):
    PREPARING = "preparing"
    WAITING_FOR_B = "waiting_for_b"
    RECORDING = "recording"
    RETURNING_TO_RETRY = "returning_to_retry"
    RETURNING_TO_SAVE = "returning_to_save"


@dataclass(frozen=True)
class RecordingEvents:
    recording_started: threading.Event
    return_to_start: threading.Event
    stop_and_save: threading.Event
    stop_no_save: threading.Event
    skip_sleep: threading.Event
    discard_and_retry: threading.Event = field(default_factory=threading.Event)

    @classmethod
    def create(cls) -> "RecordingEvents":
        return cls(
            recording_started=threading.Event(),
            return_to_start=threading.Event(),
            discard_and_retry=threading.Event(),
            stop_and_save=threading.Event(),
            stop_no_save=threading.Event(),
            skip_sleep=threading.Event(),
        )


class RecordingTriggerController:
    """Apply atomic `b`/`d` transitions to the recorder's retry state machine."""

    def __init__(
        self,
        events: RecordingEvents,
        *,
        start_trigger: str,
        lock: threading.RLock | None = None,
    ):
        if start_trigger not in {"b", "gripper"}:
            raise ValueError(f"unsupported start trigger: {start_trigger}")
        self._events = events
        self._start_trigger = start_trigger
        self._lock = lock or threading.RLock()
        self._phase = RecordingPhase.PREPARING
        self._sample_recorded = False

    @property
    def phase(self) -> RecordingPhase:
        with self._lock:
            return self._phase

    def _sync_external_recording_start(self) -> None:
        if (
            self._phase is RecordingPhase.WAITING_FOR_B
            and self._events.recording_started.is_set()
        ):
            self._phase = RecordingPhase.RECORDING
            self._sample_recorded = False

    def complete_preparation(self, *, auto_start: bool = False) -> bool:
        """Enable recording input only after robot and camera preparation."""
        with self._lock:
            if (
                self._phase is not RecordingPhase.PREPARING
                or self._events.stop_and_save.is_set()
                or self._events.stop_no_save.is_set()
            ):
                return False
            self._events.return_to_start.clear()
            self._events.recording_started.clear()
            self._sample_recorded = False
            if auto_start:
                self._events.recording_started.set()
                self._phase = RecordingPhase.RECORDING
            else:
                self._phase = RecordingPhase.WAITING_FOR_B
            return True

    def mark_sample_recorded(self) -> bool:
        """Latch that at least one action belongs to the active attempt."""
        with self._lock:
            self._sync_external_recording_start()
            if self._phase is not RecordingPhase.RECORDING:
                return False
            self._sample_recorded = True
            return True

    def handle_b(self) -> TriggerResult:
        with self._lock:
            if (
                self._events.stop_and_save.is_set()
                or self._events.stop_no_save.is_set()
            ):
                return TriggerResult.IGNORED

            self._sync_external_recording_start()

            if self._phase is RecordingPhase.PREPARING:
                return TriggerResult.NOT_READY

            if self._phase is RecordingPhase.WAITING_FOR_B:
                if self._start_trigger != "b":
                    return TriggerResult.WRONG_START_MODE
                self._events.recording_started.set()
                self._phase = RecordingPhase.RECORDING
                self._sample_recorded = False
                return TriggerResult.STARTED

            if self._phase is not RecordingPhase.RECORDING:
                return TriggerResult.IGNORED

            if not self._sample_recorded:
                return TriggerResult.NO_SAMPLES

            self._events.return_to_start.set()
            self._phase = RecordingPhase.RETURNING_TO_SAVE
            return TriggerResult.STOPPED

    def handle_d(self) -> TriggerResult:
        with self._lock:
            if (
                self._events.stop_and_save.is_set()
                or self._events.stop_no_save.is_set()
            ):
                return TriggerResult.IGNORED

            self._sync_external_recording_start()

            if self._phase is RecordingPhase.PREPARING:
                return TriggerResult.NOT_READY
            if self._phase is RecordingPhase.WAITING_FOR_B:
                return TriggerResult.NOT_RECORDING
            if self._phase is not RecordingPhase.RECORDING:
                return TriggerResult.IGNORED

            self._events.return_to_start.set()
            self._events.discard_and_retry.set()
            self._phase = RecordingPhase.RETURNING_TO_RETRY
            return TriggerResult.DISCARD_STARTED

    def complete_save(self) -> bool:
        """Complete the synchronous save path after publication."""
        return self.complete_save_handoff()

    def complete_save_handoff(self) -> bool:
        """Release recording controls after ownership moves to a save worker."""
        with self._lock:
            if (
                self._phase is not RecordingPhase.RETURNING_TO_SAVE
                or self._events.stop_and_save.is_set()
                or self._events.stop_no_save.is_set()
            ):
                return False
            self._events.return_to_start.clear()
            self._events.recording_started.clear()
            self._sample_recorded = False
            self._phase = RecordingPhase.PREPARING
            return True

    def complete_retry(self, *, auto_start: bool = False) -> bool:
        with self._lock:
            if (
                self._phase is not RecordingPhase.RETURNING_TO_RETRY
                or self._events.stop_and_save.is_set()
                or self._events.stop_no_save.is_set()
            ):
                return False
            self._events.discard_and_retry.clear()
            self._events.return_to_start.clear()
            self._sample_recorded = False
            if auto_start:
                self._events.recording_started.set()
                self._phase = RecordingPhase.RECORDING
            else:
                self._events.recording_started.clear()
                self._phase = RecordingPhase.WAITING_FOR_B
            return True
