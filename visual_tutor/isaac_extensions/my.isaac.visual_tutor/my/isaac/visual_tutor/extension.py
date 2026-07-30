from __future__ import annotations

import json
from pathlib import Path

import carb.eventdispatcher
import omni.ext
import omni.kit.app
import omni.timeline
import omni.ui as ui
import omni.usd

from .grasp_editor_bridge import ApprovedGraspEditorBridge

_extension_instance: VisualTutorExtension | None = None


def get_extension_instance() -> VisualTutorExtension | None:
    """Return the live Kit-created extension instance, if it is enabled."""
    return _extension_instance


def get_live_bridge() -> ApprovedGraspEditorBridge | None:
    """Return the one bridge owned by the live extension instance."""
    extension = get_extension_instance()
    return extension.live_bridge if extension is not None else None


class VisualTutorExtension(omni.ext.IExt):
    """Simulation-only Isaac Visual Tutor panel.

    This extension intentionally starts in a passive mode. It does not press Play,
    publish ROS messages, or control a robot. It exposes a small status panel and
    writes JSON snapshots for the project-local MCP server to inspect later.
    """

    def on_startup(self, ext_id: str) -> None:
        global _extension_instance  # noqa: PLW0603 - Kit module accessor.

        self._ext_id = ext_id
        self._live_bridge = ApprovedGraspEditorBridge()
        self._update_subscription = (
            carb.eventdispatcher.get_eventdispatcher().observe_event(
                event_name=omni.kit.app.GLOBAL_EVENT_UPDATE,
                on_event=self._on_app_update,
                observer_name=(
                    "my.isaac.visual_tutor."
                    "VisualTutorExtension._on_app_update"
                ),
            )
        )
        self._window = ui.Window("My Isaac Visual Tutor", width=360, height=220)
        self._status = "idle"
        _extension_instance = self
        self._build_ui()

    def on_shutdown(self) -> None:
        global _extension_instance  # noqa: PLW0603 - Kit module accessor.

        self._update_subscription = None
        self._live_bridge = None
        self._window = None
        if _extension_instance is self:
            _extension_instance = None

    @property
    def live_bridge(self) -> ApprovedGraspEditorBridge | None:
        return self._live_bridge

    def _on_app_update(self, _event: object) -> None:
        if self._live_bridge is not None:
            self._live_bridge.note_app_update()

    def _build_ui(self) -> None:
        with self._window.frame, ui.VStack(spacing=8):
            ui.Label("My Isaac Visual Tutor", height=24)
            ui.Label("simulation_only = true", height=22)
            ui.Label("timeline_starts_paused = true", height=22)
            ui.Button("Capture State", clicked_fn=self._capture_state)
            ui.Button("Pause Timeline", clicked_fn=self._pause_timeline)
            ui.Button("Clear Status", clicked_fn=self._clear_status)

    def _snapshot_path(self) -> Path:
        root = Path("/home/eii/project/openpi0.5-rtc-reward-learning/visual_tutor/checkpoints/isaac_extension")
        root.mkdir(parents=True, exist_ok=True)
        return root / "latest_state.json"

    def _pause_timeline(self) -> None:
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        self._status = "timeline_paused"
        self._capture_state()

    def _clear_status(self) -> None:
        self._status = "idle"
        self._capture_state()

    def _capture_state(self) -> None:
        if self._live_bridge is None:
            return
        payload = self._live_bridge.capture_state()
        payload["panel_status"] = self._status
        payload["safety"] = {
            "simulation_only": True,
            "real_robot_control_disabled": True,
            "ros_publish_disabled": True,
        }
        self._snapshot_path().write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
