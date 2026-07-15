from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .models import Lesson, LessonMode, LessonStep


SCHEMA_VERSION = "visual-tutor-lesson/v1"


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def load_lesson(path: Path) -> Lesson:
    data = _read_mapping(path)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported lesson schema: {data.get('schema_version')}")
    steps = [
        LessonStep(
            id=str(item["id"]),
            app=str(item["app"]),
            description=str(item["description"]),
            action_kind=str(item["action_kind"]),
            semantic_target=str(item["semantic_target"]),
            visual_fallback=item.get("visual_fallback"),
            relative_coordinate_fallback=item.get("relative_coordinate_fallback"),
            expected_state=item.get("expected_state") or {},
            timeout_seconds=float(item.get("timeout_seconds", 10.0)),
            retry_limit=int(item.get("retry_limit", 1)),
            checkpoint=bool(item.get("checkpoint", True)),
            undo_strategy=str(item.get("undo_strategy", "adapter_undo")),
            pause_duration_seconds=float(item.get("pause_duration_seconds", 0.8)),
            safety_class=str(item.get("safety_class", "simulation_only")),
        )
        for item in data.get("steps", [])
    ]
    if not steps:
        raise ValueError("Lesson must contain at least one step")
    if len(steps) > int(data.get("max_steps", 20)):
        raise ValueError("Lesson exceeds max_steps")
    return Lesson(
        id=str(data["id"]),
        version=str(data["version"]),
        mode=LessonMode(data.get("mode", LessonMode.DEMONSTRATE.value)),
        goal=str(data["goal"]),
        steps=steps,
        max_steps=int(data.get("max_steps", 20)),
    )
