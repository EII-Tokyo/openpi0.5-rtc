from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .adapters import TutorAdapter, adapter_for_app
from .models import Lesson, LessonState, StepResult


class LessonEngine:
    """Small-step lesson state machine with bounded retries."""

    def __init__(self, lesson: Lesson, adapters: dict[str, TutorAdapter] | None = None, fast_mode: bool = False):
        self.lesson = lesson
        self.adapters = adapters or {}
        self.fast_mode = fast_mode
        self.state = LessonState.IDLE
        self.step_index = 0
        self.paused = False
        self.history: list[StepResult] = []
        self.last_checkpoint: Path | None = None

    def _adapter(self, app: str) -> TutorAdapter:
        if app not in self.adapters:
            self.adapters[app] = adapter_for_app(app)
        return self.adapters[app]

    def status(self) -> dict[str, Any]:
        def clean(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            if isinstance(value, list):
                return [clean(item) for item in value]
            return value

        return {
            "lesson": self.lesson.id,
            "mode": self.lesson.mode.value,
            "state": self.state.value,
            "step_index": self.step_index,
            "total_steps": len(self.lesson.steps),
            "paused": self.paused,
            "last_checkpoint": str(self.last_checkpoint) if self.last_checkpoint else None,
            "history": [clean(asdict(item)) for item in self.history[-5:]],
        }

    def preflight(self) -> dict[str, Any]:
        self.state = LessonState.PREFLIGHT
        apps = sorted({step.app for step in self.lesson.steps})
        probes = {app: self._adapter(app).probe() for app in apps}
        self.state = LessonState.OBSERVING
        return {"ok": True, "probes": probes}

    def next_step(self) -> StepResult:
        if self.paused:
            self.state = LessonState.PAUSED
            result = StepResult("", self.state, False, "lesson is paused")
            self.history.append(result)
            return result
        if self.step_index >= len(self.lesson.steps):
            self.state = LessonState.COMPLETED
            result = StepResult("", self.state, True, "lesson completed")
            self.history.append(result)
            return result

        step = self.lesson.steps[self.step_index]
        adapter = self._adapter(step.app)
        before: Path | None = None
        after: Path | None = None
        checkpoint: Path | None = None
        last_message = ""
        for attempt in range(step.retry_limit + 1):
            self.state = LessonState.OBSERVING
            adapter.observe(step)
            self.state = LessonState.POINTING
            located = adapter.locate(step)
            pointed = adapter.point(step)
            self.state = LessonState.WAITING_BEFORE_ACTION
            wait = 0.01 if self.fast_mode else max(0.0, step.pause_duration_seconds)
            time.sleep(wait)
            self.state = LessonState.ACTING
            acted = adapter.act(step)
            time.sleep(wait)
            self.state = LessonState.VERIFYING
            verified = adapter.verify(step)
            ok = bool(located.get("ok") and pointed.get("ok") and acted.get("ok") and verified.get("ok"))
            last_message = "; ".join(
                str(x.get("message") or x.get("reason") or x.get("target") or x)
                for x in [located, pointed, acted, verified]
            )
            if ok:
                if step.checkpoint:
                    checkpoint = adapter.checkpoint(self.lesson.id, step, "after")
                    self.last_checkpoint = checkpoint
                self.state = LessonState.CHECKPOINTED
                self.step_index += 1
                result = StepResult(step.id, self.state, True, last_message, checkpoint, before, after)
                self.history.append(result)
                if self.step_index >= len(self.lesson.steps):
                    self.state = LessonState.COMPLETED
                return result
            if attempt >= step.retry_limit:
                self.state = LessonState.FAILED
                result = StepResult(step.id, self.state, False, f"step failed after retry: {last_message}", checkpoint, before, after)
                self.history.append(result)
                return result
            self.state = LessonState.RECOVERING
        raise AssertionError("unreachable")

    def pause(self) -> dict[str, Any]:
        self.paused = True
        self.state = LessonState.PAUSED
        return self.status()

    def resume(self) -> dict[str, Any]:
        self.paused = False
        if self.state == LessonState.PAUSED:
            self.state = LessonState.OBSERVING
        return self.status()

    def abort(self) -> dict[str, Any]:
        self.state = LessonState.ABORTED
        self.paused = True
        return self.status()

    def step_back(self) -> dict[str, Any]:
        self.step_index = max(0, self.step_index - 1)
        self.state = LessonState.RECOVERING
        return self.status()

    def repeat_step(self) -> StepResult:
        self.step_index = max(0, self.step_index)
        return self.next_step()
