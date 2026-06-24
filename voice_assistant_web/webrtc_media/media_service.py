from __future__ import annotations

from dataclasses import dataclass
import subprocess
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
from pydantic import Field


GST_REQUIRED_PLUGINS = ("webrtcbin", "videotestsrc", "videoconvert", "fakesink")

app = FastAPI(title="EII Camera WebRTC Media Service")


@dataclass(frozen=True)
class CommandResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int | None


class VideoTestSrcSmokeRequest(BaseModel):
    num_buffers: int = Field(default=30, ge=1, le=300)


def _run_command(command: list[str], timeout: float) -> CommandResult:
    try:
        completed = subprocess.run(command, capture_output=True, check=False, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        return CommandResult(ok=False, stdout="", stderr=str(exc), returncode=None)
    except subprocess.TimeoutExpired as exc:
        return CommandResult(ok=False, stdout=exc.stdout or "", stderr=exc.stderr or "command timed out", returncode=None)
    return CommandResult(
        ok=completed.returncode == 0,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


def _import_gst_webrtc_bindings() -> None:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstWebRTC", "1.0")
    from gi.repository import Gst  # noqa: F401
    from gi.repository import GstWebRTC  # noqa: F401


def probe_python_gstreamer_bindings() -> dict[str, Any]:
    try:
        _import_gst_webrtc_bindings()
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
        }
    return {
        "available": True,
        "error": None,
    }


def probe_gstreamer() -> dict[str, Any]:
    plugins: dict[str, dict[str, Any]] = {}
    for plugin in GST_REQUIRED_PLUGINS:
        result = _run_command(["gst-inspect-1.0", plugin], timeout=5)
        plugins[plugin] = {
            "available": result.ok,
            "error": None if result.ok else (result.stderr or result.stdout),
            "returncode": result.returncode,
        }
    python_bindings = probe_python_gstreamer_bindings()
    return {
        "available": all(plugin["available"] for plugin in plugins.values()) and python_bindings["available"],
        "plugins": plugins,
        "python_bindings": python_bindings,
    }


def run_videotestsrc_smoke(num_buffers: int = 30) -> dict[str, Any]:
    command = [
        "gst-launch-1.0",
        "-q",
        "videotestsrc",
        f"num-buffers={num_buffers}",
        "!",
        "videoconvert",
        "!",
        "fakesink",
    ]
    result = _run_command(command, timeout=10)
    return {
        "ok": result.ok,
        "command": command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/media/gstreamer")
def gstreamer_status() -> dict[str, Any]:
    return probe_gstreamer()


@app.post("/api/media/smoke/videotestsrc")
def videotestsrc_smoke(request: VideoTestSrcSmokeRequest) -> dict[str, Any]:
    return run_videotestsrc_smoke(num_buffers=request.num_buffers)
