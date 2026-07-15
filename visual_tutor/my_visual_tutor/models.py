from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class LessonState(str, Enum):
    IDLE = "IDLE"
    PREFLIGHT = "PREFLIGHT"
    OBSERVING = "OBSERVING"
    POINTING = "POINTING"
    WAITING_BEFORE_ACTION = "WAITING_BEFORE_ACTION"
    ACTING = "ACTING"
    VERIFYING = "VERIFYING"
    CHECKPOINTED = "CHECKPOINTED"
    PAUSED = "PAUSED"
    RECOVERING = "RECOVERING"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"
    FAILED = "FAILED"


class LessonMode(str, Enum):
    DEMONSTRATE = "demonstrate"
    GUIDED_PRACTICE = "guided-practice"
    BUILD = "build"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class LessonStep:
    id: str
    app: str
    description: str
    action_kind: str
    semantic_target: str
    visual_fallback: str | None = None
    relative_coordinate_fallback: dict[str, float] | None = None
    expected_state: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 10.0
    retry_limit: int = 1
    checkpoint: bool = True
    undo_strategy: str = "adapter_undo"
    pause_duration_seconds: float = 0.8
    safety_class: str = "simulation_only"


@dataclass(frozen=True)
class Lesson:
    id: str
    version: str
    mode: LessonMode
    goal: str
    steps: list[LessonStep]
    max_steps: int = 20


@dataclass
class StepResult:
    step_id: str
    state: LessonState
    ok: bool
    message: str
    checkpoint_path: Path | None = None
    before_snapshot: Path | None = None
    after_snapshot: Path | None = None
