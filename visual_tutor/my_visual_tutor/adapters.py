from __future__ import annotations

import json
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .models import LessonStep


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = REPO_ROOT / "visual_tutor/checkpoints"


class TutorAdapter(ABC):
    name: str

    @abstractmethod
    def probe(self) -> dict[str, Any]:
        raise NotImplementedError

    def launch(self) -> dict[str, Any]:
        return {"ok": False, "reason": "launch is not implemented for this adapter"}

    def observe(self, step: LessonStep | None = None) -> dict[str, Any]:
        return {"ok": True, "step": step.id if step else None, "timestamp": time.time()}

    def locate(self, step: LessonStep) -> dict[str, Any]:
        return {"ok": True, "target": step.semantic_target, "method": "semantic_target"}

    def point(self, step: LessonStep) -> dict[str, Any]:
        return {"ok": True, "target": step.semantic_target, "visible": False, "message": "dry-run pointer only"}

    def act(self, step: LessonStep) -> dict[str, Any]:
        return {"ok": True, "action_kind": step.action_kind, "message": "dry-run action accepted"}

    def verify(self, step: LessonStep) -> dict[str, Any]:
        return {"ok": True, "expected_state": step.expected_state, "message": "minimal adapter verification"}

    def checkpoint(self, lesson_id: str, step: LessonStep, label: str) -> Path:
        directory = CHECKPOINT_ROOT / self.name / lesson_id / step.id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{label}.json"
        payload = {
            "adapter": self.name,
            "lesson_id": lesson_id,
            "step_id": step.id,
            "timestamp": time.time(),
            "probe": self.probe(),
            "note": "This is a lightweight checkpoint. App-specific screenshots/documents are added by native adapters.",
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def undo(self, step: LessonStep) -> dict[str, Any]:
        return {"ok": False, "reason": "undo not implemented"}

    def restore(self, checkpoint_path: Path) -> dict[str, Any]:
        return {"ok": checkpoint_path.exists(), "checkpoint": str(checkpoint_path)}

    def shutdown(self) -> dict[str, Any]:
        return {"ok": True, "message": "no process owned by adapter"}


class FreeCADAdapter(TutorAdapter):
    name = "freecad"

    def probe(self) -> dict[str, Any]:
        return {
            "app": "FreeCAD",
            "available": shutil.which("FreeCAD") is not None or shutil.which("freecad") is not None,
            "freecad": shutil.which("FreeCAD") or shutil.which("freecad"),
            "freecadcmd": shutil.which("FreeCADCmd") or shutil.which("freecadcmd"),
            "xdotool": shutil.which("xdotool"),
            "wmctrl": shutil.which("wmctrl"),
            "scrot": shutil.which("scrot"),
            "dogtail": shutil.which("dogtail-detect") or shutil.which("dogtail-run-headless"),
            "control_route": "Dogtail/AT-SPI -> visual -> window-relative coordinates",
            "status": "probe-only until FreeCAD is installed" if shutil.which("FreeCAD") is None and shutil.which("freecad") is None else "ready_for_visible_preflight",
        }

    def launch(self) -> dict[str, Any]:
        probe = self.probe()
        if not probe["available"]:
            return {"ok": False, "reason": "FreeCAD is not installed; no GUI launch attempted"}
        return {"ok": False, "reason": "visible launch intentionally disabled in v1 tests; use explicit lesson run after calibration"}

    def verify(self, step: LessonStep) -> dict[str, Any]:
        probe = self.probe()
        if not probe["available"]:
            return {"ok": False, "reason": "FreeCAD is missing, cannot verify FreeCAD document state"}
        return {"ok": True, "message": "FreeCAD is available; detailed document verification requires native FreeCAD Python session"}


class IsaacSimAdapter(TutorAdapter):
    name = "isaac"

    def probe(self) -> dict[str, Any]:
        venv = REPO_ROOT / ".venv_issac/bin/python"
        launcher = REPO_ROOT / "examples/aloha_isaac/scripts/open_workcell_gui.py"
        extension = REPO_ROOT / "visual_tutor/isaac_extensions/my.isaac.visual_tutor"
        return {
            "app": "Isaac Sim",
            "simulation_only": True,
            "real_robot_control_disabled": True,
            "ros_publish_disabled": True,
            "timeline_starts_paused": True,
            "isaac_python": str(venv) if venv.exists() else None,
            "launcher": str(launcher) if launcher.exists() else None,
            "extension_path": str(extension),
            "extension_exists": extension.exists(),
            "control_route": "omni.kit.ui_test/widget query -> internal UI action -> screenshot -> viewport-relative fallback",
        }

    def act(self, step: LessonStep) -> dict[str, Any]:
        if step.safety_class != "simulation_only":
            return {"ok": False, "reason": f"unsupported safety class: {step.safety_class}"}
        return {"ok": True, "action_kind": step.action_kind, "message": "Isaac simulation-only dry-run action accepted"}

    def verify(self, step: LessonStep) -> dict[str, Any]:
        expected = step.expected_state or {}
        if expected.get("timeline") == "paused":
            return {"ok": True, "message": "timeline paused requirement recorded; live Isaac verification requires enabled extension"}
        return {"ok": True, "message": "minimal Isaac adapter verification"}


def adapter_for_app(app: str) -> TutorAdapter:
    key = app.lower().replace("_", "-").replace(" ", "-")
    if key in {"freecad", "free-cad"}:
        return FreeCADAdapter()
    if key in {"isaac", "isaac-sim", "isaacsim"}:
        return IsaacSimAdapter()
    raise ValueError(f"Unsupported app: {app}")
