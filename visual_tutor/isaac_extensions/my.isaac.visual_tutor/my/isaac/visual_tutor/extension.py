from __future__ import annotations

import json
from pathlib import Path

import omni.ext
import omni.timeline
import omni.ui as ui
import omni.usd


class VisualTutorExtension(omni.ext.IExt):
    """Simulation-only Isaac Visual Tutor panel.

    This extension intentionally starts in a passive mode. It does not press Play,
    publish ROS messages, or control a robot. It exposes a small status panel and
    writes JSON snapshots for the project-local MCP server to inspect later.
    """

    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        self._window = ui.Window("My Isaac Visual Tutor", width=360, height=220)
        self._status = "idle"
        self._build_ui()

    def on_shutdown(self) -> None:
        self._window = None

    def _build_ui(self) -> None:
        with self._window.frame:
            with ui.VStack(spacing=8):
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
        context = omni.usd.get_context()
        stage = context.get_stage()
        selection = list(context.get_selection().get_selected_prim_paths()) if context else []
        payload = {
            "extension": "my.isaac.visual_tutor",
            "status": self._status,
            "stage_identifier": stage.GetRootLayer().identifier if stage else None,
            "selection": selection,
            "timeline_playing": omni.timeline.get_timeline_interface().is_playing(),
            "safety": {
                "simulation_only": True,
                "real_robot_control_disabled": True,
                "ros_publish_disabled": True,
            },
        }
        self._snapshot_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
