from __future__ import annotations

from pathlib import Path

from my_visual_tutor.engine import LessonEngine
from my_visual_tutor.lesson_io import load_lesson
from my_visual_tutor.models import LessonState


ROOT = Path(__file__).resolve().parents[2]


def test_isaac_dry_run_lesson_completes() -> None:
    lesson = load_lesson(ROOT / "visual_tutor/lessons/isaac_cube_dry_run.yaml")
    engine = LessonEngine(lesson, fast_mode=True)
    preflight = engine.preflight()
    assert preflight["ok"]
    result = engine.next_step()
    assert result.ok
    assert engine.state == LessonState.COMPLETED
    assert engine.last_checkpoint is not None
    assert engine.last_checkpoint.exists()


def test_pause_blocks_next_step() -> None:
    lesson = load_lesson(ROOT / "visual_tutor/lessons/isaac_cube_dry_run.yaml")
    engine = LessonEngine(lesson, fast_mode=True)
    engine.preflight()
    engine.pause()
    result = engine.next_step()
    assert not result.ok
    assert result.state == LessonState.PAUSED


def test_freecad_probe_lesson_fails_safely_when_missing() -> None:
    lesson = load_lesson(ROOT / "visual_tutor/lessons/freecad_minimal_probe.yaml")
    engine = LessonEngine(lesson, fast_mode=True)
    preflight = engine.preflight()
    assert preflight["ok"]
    probe = preflight["probes"]["FreeCAD"]
    assert "available" in probe
    result = engine.next_step()
    if not probe["available"]:
        assert not result.ok
        assert engine.state == LessonState.FAILED
