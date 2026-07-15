#!/usr/bin/env python3
"""Minimal stdio MCP server exposing high-level Visual Tutor tools only."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "visual_tutor") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "visual_tutor"))

from my_visual_tutor.adapters import adapter_for_app  # noqa: E402
from my_visual_tutor.engine import LessonEngine  # noqa: E402
from my_visual_tutor.lesson_io import load_lesson  # noqa: E402
from my_visual_tutor.models import LessonState  # noqa: E402


ENGINE: LessonEngine | None = None


TOOL_NAMES = [
    "probe_app",
    "start_lesson",
    "get_lesson_status",
    "next_step",
    "repeat_step",
    "step_back",
    "pause_lesson",
    "resume_lesson",
    "abort_lesson",
    "capture_current_state",
    "save_checkpoint",
    "restore_checkpoint",
    "finish_lesson",
]


def tool_schema(name: str) -> dict[str, Any]:
    schemas: dict[str, dict[str, Any]] = {
        "probe_app": {
            "type": "object",
            "properties": {"app": {"type": "string", "enum": ["FreeCAD", "Isaac Sim", "freecad", "isaac"]}},
            "required": ["app"],
        },
        "start_lesson": {
            "type": "object",
            "properties": {"lesson_path": {"type": "string"}},
            "required": ["lesson_path"],
        },
        "restore_checkpoint": {
            "type": "object",
            "properties": {"checkpoint_path": {"type": "string"}},
            "required": ["checkpoint_path"],
        },
    }
    return schemas.get(name, {"type": "object", "properties": {}})


def text_result(payload: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(payload, indent=2, ensure_ascii=False, default=str)}]}


def require_engine() -> LessonEngine:
    if ENGINE is None:
        raise RuntimeError("No active lesson. Call start_lesson first.")
    return ENGINE


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    global ENGINE
    if name == "probe_app":
        return text_result(adapter_for_app(arguments["app"]).probe())
    if name == "start_lesson":
        lesson_path = Path(arguments["lesson_path"]).expanduser()
        if not lesson_path.is_absolute():
            lesson_path = REPO_ROOT / lesson_path
        lesson = load_lesson(lesson_path)
        ENGINE = LessonEngine(lesson)
        return text_result({"started": True, "preflight": ENGINE.preflight(), "status": ENGINE.status()})
    if name == "get_lesson_status":
        return text_result(require_engine().status())
    if name == "next_step":
        result = require_engine().next_step()
        return text_result({"result": result.__dict__, "status": require_engine().status()})
    if name == "repeat_step":
        result = require_engine().repeat_step()
        return text_result({"result": result.__dict__, "status": require_engine().status()})
    if name == "step_back":
        return text_result(require_engine().step_back())
    if name == "pause_lesson":
        return text_result(require_engine().pause())
    if name == "resume_lesson":
        return text_result(require_engine().resume())
    if name == "abort_lesson":
        return text_result(require_engine().abort())
    if name == "capture_current_state":
        return text_result({"status": require_engine().status(), "note": "no screenshot capture in minimal headless test"})
    if name == "save_checkpoint":
        engine = require_engine()
        idx = min(engine.step_index, len(engine.lesson.steps) - 1)
        step = engine.lesson.steps[idx]
        checkpoint = engine._adapter(step.app).checkpoint(engine.lesson.id, step, "manual")
        return text_result({"checkpoint": str(checkpoint), "status": engine.status()})
    if name == "restore_checkpoint":
        engine = require_engine()
        idx = min(engine.step_index, len(engine.lesson.steps) - 1)
        step = engine.lesson.steps[idx]
        restored = engine._adapter(step.app).restore(Path(arguments["checkpoint_path"]))
        engine.state = LessonState.RECOVERING
        return text_result({"restore": restored, "status": engine.status()})
    if name == "finish_lesson":
        engine = require_engine()
        engine.state = LessonState.COMPLETED
        return text_result(engine.status())
    raise ValueError(f"Unknown tool: {name}")


def handle(request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    req_id = request.get("id")
    try:
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "my-gui-teacher", "version": "0.1.0"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": name,
                        "description": f"Visual Tutor high-level operation: {name}",
                        "inputSchema": tool_schema(name),
                    }
                    for name in TOOL_NAMES
                ]
            }
        elif method == "tools/call":
            params = request.get("params", {})
            result = call_tool(params["name"], params.get("arguments") or {})
        elif method == "notifications/initialized":
            return None
        else:
            raise ValueError(f"Unsupported method: {method}")
        return {"jsonrpc": "2.0", "id": req_id, "result": result}
    except Exception as exc:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": str(exc)}}


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        response = handle(json.loads(line))
        if response is not None:
            print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
