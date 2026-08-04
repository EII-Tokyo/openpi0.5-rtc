"""Atomic coordination across recorder trigger and safe-stop commands."""

from __future__ import annotations

import threading

from aloha.record_trigger import (
    RecordingPhase,
    RecordingTriggerController,
    TriggerResult,
)
from aloha.safe_stop import SafeStopController, should_defer_s_wakeup


class RecorderCommandCoordinator:
    """Use one re-entrant lock for phase decisions and stop event mutation."""

    def __init__(
        self,
        trigger: RecordingTriggerController,
        safe_stop: SafeStopController,
        *,
        lock: threading.RLock,
    ) -> None:
        self._trigger = trigger
        self._safe_stop = safe_stop
        self._lock = lock

    def handle_b(self) -> TriggerResult:
        with self._lock:
            return self._trigger.handle_b()

    def handle_d(self) -> TriggerResult:
        with self._lock:
            return self._trigger.handle_d()

    def request_save(self, *, skip_sleep: bool, source: str) -> bool:
        with self._lock:
            if self._trigger.phase is RecordingPhase.RETURNING_TO_RETRY:
                return False
            return self._safe_stop.request_save(
                skip_sleep=skip_sleep,
                source=source,
            )

    def request_no_save_from_s(self) -> None:
        with self._lock:
            defer_wakeup = should_defer_s_wakeup(self._trigger.phase)
            self._safe_stop.request_from_s(wake_main=not defer_wakeup)

    def request_no_save(self, *, source: str) -> None:
        with self._lock:
            defer_wakeup = should_defer_s_wakeup(self._trigger.phase)
            self._safe_stop.request_no_save(
                source=source,
                wake_main=not defer_wakeup,
            )
