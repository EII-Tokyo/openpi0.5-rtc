"""Dependency-injected terminal command routing for the episode recorder."""

from __future__ import annotations

from collections.abc import Callable

from aloha.record_trigger import RecordingPhase


class RecorderKeyRouter:
    """Route recorder keys while protecting a discard-and-return transition."""

    def __init__(
        self,
        *,
        get_phase: Callable[[], RecordingPhase],
        on_b: Callable[[], None],
        on_d: Callable[[], None],
        on_m: Callable[[], None],
        on_s: Callable[[], None],
        on_r: Callable[[], None],
        on_ignored: Callable[[str], None],
    ):
        self._get_phase = get_phase
        self._callbacks = {
            "b": on_b,
            "d": on_d,
            "m": on_m,
            "s": on_s,
            "r": on_r,
        }
        self._on_ignored = on_ignored

    def handle(self, key: str) -> None:
        callback = self._callbacks.get(key)
        if callback is None:
            return
        if (
            key != "s"
            and self._get_phase() is RecordingPhase.RETURNING_TO_RETRY
        ):
            self._on_ignored(key)
            return
        callback()
