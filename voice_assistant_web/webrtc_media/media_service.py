from __future__ import annotations

from dataclasses import dataclass
import asyncio
import ipaddress
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
import uuid
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional

import cv2
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import numpy as np
from pydantic import BaseModel
from pydantic import Field


LOGGER = logging.getLogger(__name__)
GST_REQUIRED_PLUGINS = ("webrtcbin", "nicesrc", "nicesink", "videotestsrc", "videoconvert", "fakesink")
ROS_CAMERA_TOPICS: Dict[str, str] = {
    "cam_high": "/cam_high",
    "cam_low": "/cam_low",
    "cam_left_wrist": "/cam_left_wrist",
    "cam_right_wrist": "/cam_right_wrist",
}

app = FastAPI(title="EII Camera WebRTC Media Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
AIORTC_SESSIONS: Dict[str, Dict[str, Any]] = {}


@dataclass(frozen=True)
class CommandResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: Optional[int]


class VideoTestSrcSmokeRequest(BaseModel):
    num_buffers: int = Field(default=30, ge=1, le=300)


class RosCameraSmokeRequest(BaseModel):
    camera_name: str = Field(default="cam_high")
    timeout_seconds: float = Field(default=3.0, ge=0.2, le=15.0)
    jpeg_quality: int = Field(default=90, ge=10, le=100)


class AiortcOfferRequest(BaseModel):
    sdp: str
    type: str = Field(default="offer")
    fps: Optional[float] = Field(default=None)


class AiortcOfferResponse(BaseModel):
    sdp: str
    type: str = Field(default="answer")
    session_id: str


def clamp_aiortc_fps(fps: float | None) -> float:
    if fps is None or fps <= 0:
        return 15.0
    return max(1.0, min(float(fps), 30.0))


def _run_command(command: List[str], timeout: float) -> CommandResult:
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


def probe_python_gstreamer_bindings() -> Dict[str, Any]:
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


def probe_gstreamer() -> Dict[str, Any]:
    plugins: Dict[str, Dict[str, Any]] = {}
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


def probe_webrtc_runtime() -> Dict[str, Any]:
    try:
        return _probe_webrtc_runtime()
    except Exception as exc:
        return {
            "available": False,
            "ready": False,
            "sink_request_pad": False,
            "error": str(exc),
        }


def _probe_webrtc_runtime() -> Dict[str, Any]:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    pipeline = Gst.Pipeline.new("eii-webrtc-runtime-probe")
    webrtc = Gst.ElementFactory.make("webrtcbin", "webrtc")
    if webrtc is None:
        return {
            "available": False,
            "ready": False,
            "sink_request_pad": False,
            "error": "Could not create webrtcbin",
        }
    pipeline.add(webrtc)
    requested_pad = None
    try:
        state_result = pipeline.set_state(Gst.State.READY)
        ready_result, state, pending = pipeline.get_state(2 * Gst.SECOND)
        ready = state_result == Gst.StateChangeReturn.SUCCESS and ready_result == Gst.StateChangeReturn.SUCCESS
        if not ready:
            bus = pipeline.get_bus()
            message = bus.timed_pop_filtered(0, Gst.MessageType.ERROR)
            error = None
            if message is not None:
                parsed_error, debug = message.parse_error()
                error = f"{parsed_error}: {debug}"
            return {
                "available": False,
                "ready": False,
                "sink_request_pad": False,
                "state_result": str(state_result),
                "state": str(state),
                "pending": str(pending),
                "error": error or "webrtcbin did not reach READY",
            }
        requested_pad = webrtc.get_request_pad("sink_%u")
        return {
            "available": requested_pad is not None,
            "ready": True,
            "sink_request_pad": requested_pad is not None,
            "state_result": str(state_result),
            "state": str(state),
            "pending": str(pending),
            "error": None if requested_pad is not None else "Could not request webrtcbin sink pad",
        }
    finally:
        if requested_pad is not None:
            webrtc.release_request_pad(requested_pad)
        pipeline.set_state(Gst.State.NULL)


def parse_webrtc_offer_message(message: Dict[str, Any]) -> Dict[str, Any]:
    if message.get("type") != "offer" or not isinstance(message.get("sdp"), str):
        return {
            "ok": False,
            "error": "Expected offer message with string sdp",
        }
    return {
        "ok": True,
        "sdp": message["sdp"],
    }


def parse_webrtc_answer_message(message: Dict[str, Any]) -> Dict[str, Any]:
    if message.get("type") != "answer" or not isinstance(message.get("sdp"), str):
        return {
            "ok": False,
            "error": "Expected answer message with string sdp",
        }
    return {
        "ok": True,
        "sdp": message["sdp"],
    }


def parse_webrtc_ice_message(message: Dict[str, Any]) -> Dict[str, Any]:
    if message.get("type") != "ice":
        return {
            "ok": False,
            "error": "Expected ice message",
        }
    candidate = message.get("candidate")
    if candidate is None:
        candidate = ""
    if not isinstance(candidate, str):
        return {
            "ok": False,
            "error": "Expected ICE candidate string or null",
        }
    return {
        "ok": True,
        "candidate": candidate,
        "sdp_mline_index": int(message.get("sdpMLineIndex", 0) or 0),
    }


def extract_sdp_ice_candidates(sdp: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    current_mline_index = -1
    for raw_line in sdp.replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if line.startswith("m="):
            current_mline_index += 1
            continue
        if line.startswith("a=candidate:") and current_mline_index >= 0:
            candidates.append(
                {
                    "sdp_mline_index": current_mline_index,
                    "candidate": line[2:],
                }
            )
    return candidates


def _candidate_ip(candidate: str) -> Optional[ipaddress._BaseAddress]:
    parts = candidate.split()
    if len(parts) < 5:
        return None
    try:
        return ipaddress.ip_address(parts[4])
    except ValueError:
        return None


def should_publish_webrtc_candidate(candidate: str) -> bool:
    address = _candidate_ip(candidate)
    if address is None:
        return False
    if address.version != 4:
        return False
    if address in ipaddress.ip_network("172.16.0.0/12"):
        return False
    return address.is_private or address.is_loopback


def filter_webrtc_sdp_candidates(sdp: str) -> str:
    filtered_lines: List[str] = []
    for raw_line in sdp.replace("\r\n", "\n").split("\n"):
        if raw_line.startswith("a=candidate:") and not should_publish_webrtc_candidate(raw_line[2:]):
            continue
        filtered_lines.append(raw_line)
    return "\r\n".join(filtered_lines)


def build_webrtc_test_page(title: str, websocket_path: str, description: str) -> str:
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; background: #111827; color: #f9fafb; }
    main { max-width: 980px; margin: 0 auto; padding: 24px; }
    video { width: 100%; aspect-ratio: 16 / 9; background: #020617; border: 1px solid #374151; }
    button { padding: 10px 14px; margin-right: 8px; font-weight: 700; }
    pre { white-space: pre-wrap; background: #020617; padding: 12px; min-height: 140px; }
  </style>
</head>
<body>
  <main>
    <h1>__TITLE__</h1>
    <p>__DESCRIPTION__</p>
    <video id="video" autoplay playsinline muted></video>
    <p>
      <button id="start">Start</button>
      <button id="stop">Stop</button>
    </p>
    <pre id="log"></pre>
  </main>
  <script>
    const video = document.getElementById('video');
    const logEl = document.getElementById('log');
    let pc = null;
    let ws = null;
    let statsTimer = null;
    let pendingRemoteIce = [];

    function log(message) {
      logEl.textContent += `${new Date().toISOString()} ${message}\\n`;
      logEl.scrollTop = logEl.scrollHeight;
    }

    async function start() {
      if (pc) return;
      const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      ws = new WebSocket(`${wsProtocol}//${location.host}__WEBSOCKET_PATH__`);
      pc = new RTCPeerConnection({ iceServers: [] });
      window.__eiiPc = pc;
      pc.addTransceiver('video', { direction: 'recvonly' });
      pc.ontrack = (event) => {
        video.srcObject = event.streams[0];
        log('received remote track');
      };
      pc.onicecandidate = (event) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          if (event.candidate) {
            ws.send(JSON.stringify({
              type: 'ice',
              candidate: event.candidate.candidate,
              sdpMLineIndex: event.candidate.sdpMLineIndex,
            }));
          } else {
            ws.send(JSON.stringify({ type: 'ice', candidate: null, sdpMLineIndex: 0 }));
          }
        }
      };
      pc.onconnectionstatechange = () => log(`pc connectionState=${pc.connectionState}`);
      pc.oniceconnectionstatechange = () => log(`pc iceConnectionState=${pc.iceConnectionState}`);
      statsTimer = window.setInterval(async () => {
        if (!pc) return;
        const stats = await pc.getStats();
        for (const report of stats.values()) {
          if (report.type === 'inbound-rtp' && report.kind === 'video') {
            log(`stats bytes=${report.bytesReceived || 0} packets=${report.packetsReceived || 0} frames=${report.framesDecoded || 0}`);
          }
          if (report.type === 'transport') {
            log(`stats ice=${report.iceState || 'n/a'} dtls=${report.dtlsState || 'n/a'}`);
          }
        }
      }, 1000);
      ws.onmessage = async (event) => {
        const message = JSON.parse(event.data);
        log(`ws recv ${message.type}`);
        if (message.type === 'offer') {
          await pc.setRemoteDescription({ type: 'offer', sdp: message.sdp });
          const answer = await pc.createAnswer();
          await pc.setLocalDescription(answer);
          ws.send(JSON.stringify({ type: 'answer', sdp: answer.sdp }));
          log('sent answer');
          for (const candidate of pendingRemoteIce) {
            await pc.addIceCandidate(candidate);
          }
          pendingRemoteIce = [];
        } else if (message.type === 'ice') {
          const candidate = { candidate: message.candidate, sdpMLineIndex: message.sdpMLineIndex };
          if (pc.remoteDescription) {
            await pc.addIceCandidate(candidate);
          } else {
            pendingRemoteIce.push(candidate);
          }
        } else if (message.type === 'error') {
          log(`error ${message.error}`);
        } else if (message.type === 'event') {
          log(`server event ${message.name}: ${message.value}`);
        }
      };
      await new Promise((resolve) => { ws.onopen = resolve; });
      ws.send(JSON.stringify({ type: 'start' }));
      log('sent start');
    }

    function stop() {
      if (ws) ws.close();
      if (pc) pc.close();
      if (statsTimer) window.clearInterval(statsTimer);
      ws = null;
      pc = null;
      window.__eiiPc = null;
      statsTimer = null;
      pendingRemoteIce = [];
      video.srcObject = null;
      log('stopped');
    }

    document.getElementById('start').addEventListener('click', () => start().catch((error) => log(error.stack || error)));
    document.getElementById('stop').addEventListener('click', stop);
  </script>
</body>
</html>
"""
    return (
        html.replace("__TITLE__", title)
        .replace("__DESCRIPTION__", description)
        .replace("__WEBSOCKET_PATH__", websocket_path)
    )


def build_videotestsrc_webrtc_test_page() -> str:
    return build_webrtc_test_page(
        title="WebRTC Videotestsrc",
        websocket_path="/ws/media/webrtc/videotestsrc",
        description="This page tests only the media sidecar. It does not control the robot.",
    )


def build_ros_camera_webrtc_test_page(camera_name: str) -> str:
    validation = validate_ros_camera_name(camera_name)
    if not validation["ok"]:
        raise ValueError(validation["error"])
    return build_webrtc_test_page(
        title=f"WebRTC ROS Camera {camera_name}",
        websocket_path=f"/ws/media/webrtc/ros-camera/{camera_name}",
        description="This page tests one real ROS camera stream through the media sidecar.",
    )


def build_aiortc_ros_camera_test_page(camera_name: str) -> str:
    validation = validate_ros_camera_name(camera_name)
    if not validation["ok"]:
        raise ValueError(validation["error"])
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>aiortc ROS Camera __CAMERA__</title>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; background: #111827; color: #f9fafb; }
    main { max-width: 980px; margin: 0 auto; padding: 24px; }
    video { width: 100%; aspect-ratio: 16 / 9; background: #020617; border: 1px solid #374151; }
    button { padding: 10px 14px; margin-right: 8px; font-weight: 700; }
    pre { white-space: pre-wrap; background: #020617; padding: 12px; min-height: 140px; }
  </style>
</head>
<body>
  <main>
    <h1>aiortc ROS Camera __CAMERA__</h1>
    <p>This page tests one real ROS camera stream through aiortc.</p>
    <video id="video" autoplay playsinline muted></video>
    <p>
      <button id="start">Start</button>
      <button id="stop">Stop</button>
    </p>
    <pre id="log"></pre>
  </main>
  <script>
    const video = document.getElementById('video');
    const logEl = document.getElementById('log');
    let pc = null;
    let statsTimer = null;

    function log(message) {
      logEl.textContent += `${new Date().toISOString()} ${message}\\n`;
      logEl.scrollTop = logEl.scrollHeight;
    }

    function waitForIceGatheringComplete(peerConnection) {
      if (peerConnection.iceGatheringState === 'complete') return Promise.resolve();
      return new Promise((resolve) => {
        function checkState() {
          if (peerConnection.iceGatheringState === 'complete') {
            peerConnection.removeEventListener('icegatheringstatechange', checkState);
            resolve();
          }
        }
        peerConnection.addEventListener('icegatheringstatechange', checkState);
      });
    }

    async function start() {
      if (pc) return;
      pc = new RTCPeerConnection({ iceServers: [] });
      window.__eiiPc = pc;
      pc.addTransceiver('video', { direction: 'recvonly' });
      pc.ontrack = (event) => {
        video.srcObject = event.streams[0];
        log('received remote track');
      };
      pc.onconnectionstatechange = () => log(`pc connectionState=${pc.connectionState}`);
      pc.oniceconnectionstatechange = () => log(`pc iceConnectionState=${pc.iceConnectionState}`);
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await waitForIceGatheringComplete(pc);
      log('local offer ready');
      const response = await fetch('/api/media/aiortc/ros-camera/__CAMERA__/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
      });
      if (!response.ok) throw new Error(`offer failed ${response.status}: ${await response.text()}`);
      const answer = await response.json();
      await pc.setRemoteDescription(answer);
      log('remote answer set');
      statsTimer = window.setInterval(async () => {
        if (!pc) return;
        const stats = await pc.getStats();
        for (const report of stats.values()) {
          if (report.type === 'inbound-rtp' && report.kind === 'video') {
            log(`stats bytes=${report.bytesReceived || 0} packets=${report.packetsReceived || 0} frames=${report.framesDecoded || 0}`);
          }
          if (report.type === 'transport') {
            log(`stats ice=${report.iceState || 'n/a'} dtls=${report.dtlsState || 'n/a'}`);
          }
        }
      }, 1000);
    }

    function stop() {
      if (statsTimer) window.clearInterval(statsTimer);
      if (pc) pc.close();
      statsTimer = null;
      pc = null;
      window.__eiiPc = null;
      video.srcObject = null;
      log('stopped');
    }

    document.getElementById('start').addEventListener('click', () => start().catch((error) => log(error.stack || error)));
    document.getElementById('stop').addEventListener('click', stop);
  </script>
</body>
</html>
"""
    return html.replace("__CAMERA__", camera_name)


def configure_webrtc_element(webrtc: Any, bundle_policy: Any) -> None:
    # Browsers offer BUNDLE by default. Leaving webrtcbin at NONE creates
    # mismatched ICE/DTLS negotiation where ICE can connect but media stalls.
    webrtc.set_property("bundle-policy", bundle_policy.MAX_BUNDLE)


class BaseWebRTCSession:
    def __init__(self, send_json_threadsafe: Callable[[Dict[str, Any]], None]) -> None:
        self._send_json_threadsafe = send_json_threadsafe
        self._pipeline = None
        self._webrtc = None
        self._requested_pad = None
        self._bus = None
        self._main_loop = None
        self._main_loop_thread: Optional[threading.Thread] = None

    @property
    def session_label(self) -> str:
        return "webrtc"

    def start_with_offer(self, offer_sdp: str) -> str:
        self._emit_event("session", "start")
        self._start_main_loop()
        self._build_pipeline()
        self._prepare_pipeline()
        self._set_remote_description(offer_sdp, self._webrtc_module().WebRTCSDPType.OFFER)
        answer_sdp = self._create_answer()
        self._start_pipeline()
        self._emit_event("session", "answer-ready")
        return answer_sdp

    def start_with_server_offer(self) -> str:
        self._emit_event("session", "start-server-offer")
        self._start_main_loop()
        self._build_pipeline()
        self._prepare_pipeline()
        offer_sdp = self._create_offer()
        self._start_pipeline()
        self._emit_event("session", "offer-ready")
        return offer_sdp

    def accept_answer(self, answer_sdp: str) -> None:
        self._set_remote_description(answer_sdp, self._webrtc_module().WebRTCSDPType.ANSWER)

    def add_ice_candidate(self, candidate: str, sdp_mline_index: int) -> None:
        if candidate == "":
            self._emit_event("browser-ice.end", f"mline={int(sdp_mline_index)}")
            return
        if self._webrtc is not None:
            self._webrtc.emit("add-ice-candidate", int(sdp_mline_index), candidate)
            self._emit_event("browser-ice", f"mline={int(sdp_mline_index)}")

    def close(self) -> None:
        if self._pipeline is None:
            return
        try:
            if self._requested_pad is not None and self._webrtc is not None:
                self._webrtc.release_request_pad(self._requested_pad)
        finally:
            self._pipeline.set_state(self._gst().State.NULL)
            if self._main_loop is not None:
                self._main_loop.quit()
            if self._main_loop_thread is not None and self._main_loop_thread.is_alive():
                self._main_loop_thread.join(timeout=1.0)
            self._pipeline = None
            self._webrtc = None
            self._requested_pad = None
            self._bus = None
            self._main_loop = None
            self._main_loop_thread = None

    def _emit_event(self, name: str, value: Any) -> None:
        text = str(value)
        LOGGER.info("%s %s=%s", self.session_label, name, text)
        self._send_json_threadsafe(
            {
                "type": "event",
                "name": name,
                "value": text,
            }
        )

    def _gst(self) -> Any:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        return Gst

    def _webrtc_module(self) -> Any:
        import gi

        gi.require_version("GstWebRTC", "1.0")
        from gi.repository import GstWebRTC

        return GstWebRTC

    def _sdp_module(self) -> Any:
        import gi

        gi.require_version("GstSdp", "1.0")
        from gi.repository import GstSdp

        return GstSdp

    def _start_main_loop(self) -> None:
        if self._main_loop is not None:
            return
        import gi

        gi.require_version("GLib", "2.0")
        from gi.repository import GLib

        self._main_loop = GLib.MainLoop()
        self._main_loop_thread = threading.Thread(target=self._main_loop.run, daemon=True)
        self._main_loop_thread.start()
        self._emit_event("glib-loop", "started")

    def _build_pipeline(self) -> None:
        raise NotImplementedError

    def _attach_webrtc_diagnostics(self, pipeline: Any, webrtc: Any) -> None:
        webrtc.connect("on-ice-candidate", self._on_ice_candidate)
        self._connect_webrtc_notify_signals(webrtc)
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)
        self._pipeline = pipeline
        self._webrtc = webrtc
        self._bus = bus
        self._emit_event("pipeline", "built")

    def _connect_webrtc_notify_signals(self, webrtc: Any) -> None:
        supported = {property_.name for property_ in webrtc.list_properties()}
        for property_name in (
            "signaling-state",
            "ice-gathering-state",
            "ice-connection-state",
            "connection-state",
        ):
            if property_name in supported:
                webrtc.connect(f"notify::{property_name}", self._on_webrtc_property_changed, property_name)

    def _on_webrtc_property_changed(self, webrtc: Any, _param: Any, property_name: str) -> None:
        try:
            value = webrtc.get_property(property_name)
        except Exception as exc:
            value = f"read-failed: {exc}"
        self._emit_event(f"webrtc.{property_name}", value)

    def _on_bus_message(self, _bus: Any, message: Any) -> None:
        Gst = self._gst()
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            self._emit_event("gst.error", f"{error}: {debug}")
            return
        if message.type == Gst.MessageType.WARNING:
            warning, debug = message.parse_warning()
            self._emit_event("gst.warning", f"{warning}: {debug}")
            return
        if message.type == Gst.MessageType.STATE_CHANGED and message.src == self._pipeline:
            old, new, pending = message.parse_state_changed()
            self._emit_event("pipeline.state", f"{old.value_nick}->{new.value_nick}, pending={pending.value_nick}")

    def _set_remote_description(self, sdp: str, sdp_type: Any) -> None:
        Gst = self._gst()
        GstWebRTC = self._webrtc_module()
        GstSdp = self._sdp_module()
        result, message = GstSdp.SDPMessage.new()
        if result != GstSdp.SDPResult.OK:
            raise RuntimeError(f"Could not allocate SDP message: {result}")
        parse_result = GstSdp.sdp_message_parse_buffer(sdp.encode("utf-8"), message)
        if parse_result != GstSdp.SDPResult.OK:
            raise RuntimeError(f"Could not parse browser SDP: {parse_result}")
        offer = GstWebRTC.WebRTCSessionDescription.new(sdp_type, message)
        promise = Gst.Promise.new()
        self._webrtc.emit("set-remote-description", offer, promise)
        promise.wait()
        self._emit_event("remote-description", "set")
        sdp_candidates = extract_sdp_ice_candidates(sdp)
        for candidate in sdp_candidates:
            self.add_ice_candidate(candidate["candidate"], candidate["sdp_mline_index"])
        self._emit_event("remote-description.candidates", len(sdp_candidates))

    def _prepare_pipeline(self) -> None:
        Gst = self._gst()
        result = self._pipeline.set_state(Gst.State.READY)
        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Could not prepare WebRTC pipeline")
        ready_result, _state, _pending = self._pipeline.get_state(2 * Gst.SECOND)
        if ready_result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("WebRTC pipeline did not reach READY")
        self._emit_event("pipeline.prepare", str(ready_result))

    def _create_answer(self) -> str:
        Gst = self._gst()
        promise = Gst.Promise.new()
        self._webrtc.emit("create-answer", None, promise)
        promise.wait()
        reply = promise.get_reply()
        answer = reply.get_value("answer") if reply is not None else None
        if answer is None:
            raise RuntimeError("webrtcbin did not create an answer")
        local_promise = Gst.Promise.new()
        self._webrtc.emit("set-local-description", answer, local_promise)
        local_promise.wait()
        self._emit_event("local-description", "set")
        return filter_webrtc_sdp_candidates(answer.sdp.as_text())

    def _create_offer(self) -> str:
        Gst = self._gst()
        promise = Gst.Promise.new()
        self._webrtc.emit("create-offer", None, promise)
        promise.wait()
        reply = promise.get_reply()
        offer = reply.get_value("offer") if reply is not None else None
        if offer is None:
            raise RuntimeError("webrtcbin did not create an offer")
        local_promise = Gst.Promise.new()
        self._webrtc.emit("set-local-description", offer, local_promise)
        local_promise.wait()
        self._emit_event("local-description", "set-offer")
        return filter_webrtc_sdp_candidates(offer.sdp.as_text())

    def _start_pipeline(self) -> None:
        Gst = self._gst()
        result = self._pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Could not start WebRTC pipeline")
        self._emit_event("pipeline.start", str(result))

    def _on_ice_candidate(self, _webrtc: Any, mline_index: int, candidate: str) -> None:
        if not should_publish_webrtc_candidate(candidate):
            self._emit_event("server-ice.filtered", candidate)
            return
        self._send_json_threadsafe(
            {
                "type": "ice",
                "sdpMLineIndex": int(mline_index),
                "candidate": candidate,
            }
        )


class VideoTestSrcWebRTCSession(BaseWebRTCSession):
    @property
    def session_label(self) -> str:
        return "videotestsrc webrtc"

    def _build_pipeline(self) -> None:
        Gst = self._gst()
        Gst.init(None)
        pipeline = Gst.Pipeline.new("eii-videotestsrc-webrtc")
        webrtc = Gst.ElementFactory.make("webrtcbin", "webrtc")
        elements = [
            webrtc,
            Gst.ElementFactory.make("videotestsrc", "src"),
            Gst.ElementFactory.make("videoconvert", "convert"),
            Gst.ElementFactory.make("queue", "queue"),
            Gst.ElementFactory.make("vp8enc", "encoder"),
            Gst.ElementFactory.make("rtpvp8pay", "payloader"),
            Gst.ElementFactory.make("capsfilter", "caps"),
        ]
        if any(element is None for element in elements):
            raise RuntimeError("Could not create all GStreamer WebRTC elements")
        configure_webrtc_element(webrtc, self._webrtc_module().WebRTCBundlePolicy)
        _, src, convert, queue, encoder, payloader, capsfilter = elements
        src.set_property("is-live", True)
        encoder.set_property("deadline", 1)
        encoder.set_property("keyframe-max-dist", 30)
        payloader.set_property("pt", 96)
        capsfilter.set_property(
            "caps",
            Gst.Caps.from_string("application/x-rtp,media=video,encoding-name=VP8,payload=96,clock-rate=90000"),
        )
        for element in elements:
            pipeline.add(element)
        linked = (
            src.link(convert)
            and convert.link(queue)
            and queue.link(encoder)
            and encoder.link(payloader)
            and payloader.link(capsfilter)
        )
        if not linked:
            raise RuntimeError("Could not link videotestsrc WebRTC pipeline")
        requested_pad = webrtc.get_request_pad("sink_%u")
        if requested_pad is None:
            raise RuntimeError("Could not request webrtcbin sink pad")
        if capsfilter.get_static_pad("src").link(requested_pad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Could not link RTP payloader to webrtcbin")
        self._requested_pad = requested_pad
        self._attach_webrtc_diagnostics(pipeline, webrtc)


class RosCameraWebRTCSession(BaseWebRTCSession):
    def __init__(self, camera_name: str, send_json_threadsafe: Callable[[Dict[str, Any]], None]) -> None:
        validation = validate_ros_camera_name(camera_name)
        if not validation["ok"]:
            raise ValueError(validation["error"])
        super().__init__(send_json_threadsafe)
        self.camera_name = camera_name
        self.topic = validation["topic"]
        self._subscriber = None
        self._appsrc = None
        self._frame_count = 0
        self._last_frame_time = 0.0

    @property
    def session_label(self) -> str:
        return f"ros-camera {self.camera_name} webrtc"

    def close(self) -> None:
        if self._subscriber is not None:
            self._subscriber.unregister()
            self._subscriber = None
        super().close()

    def _build_pipeline(self) -> None:
        Gst = self._gst()
        Gst.init(None)
        pipeline = Gst.Pipeline.new(f"eii-{self.camera_name}-webrtc")
        webrtc = Gst.ElementFactory.make("webrtcbin", "webrtc")
        elements = [
            webrtc,
            Gst.ElementFactory.make("appsrc", "src"),
            Gst.ElementFactory.make("videoconvert", "convert"),
            Gst.ElementFactory.make("queue", "queue"),
            Gst.ElementFactory.make("vp8enc", "encoder"),
            Gst.ElementFactory.make("rtpvp8pay", "payloader"),
            Gst.ElementFactory.make("capsfilter", "caps"),
        ]
        if any(element is None for element in elements):
            raise RuntimeError("Could not create all ROS camera WebRTC elements")
        configure_webrtc_element(webrtc, self._webrtc_module().WebRTCBundlePolicy)
        _, appsrc, convert, queue, encoder, payloader, capsfilter = elements
        appsrc.set_property("is-live", True)
        appsrc.set_property("format", Gst.Format.TIME)
        appsrc.set_property("do-timestamp", True)
        appsrc.set_property("caps", Gst.Caps.from_string("video/x-raw,format=BGR,width=640,height=480,framerate=30/1"))
        encoder.set_property("deadline", 1)
        encoder.set_property("keyframe-max-dist", 30)
        payloader.set_property("pt", 96)
        capsfilter.set_property(
            "caps",
            Gst.Caps.from_string("application/x-rtp,media=video,encoding-name=VP8,payload=96,clock-rate=90000"),
        )
        for element in elements:
            pipeline.add(element)
        linked = (
            appsrc.link(convert)
            and convert.link(queue)
            and queue.link(encoder)
            and encoder.link(payloader)
            and payloader.link(capsfilter)
        )
        if not linked:
            raise RuntimeError("Could not link ROS camera WebRTC pipeline")
        requested_pad = webrtc.get_request_pad("sink_%u")
        if requested_pad is None:
            raise RuntimeError("Could not request webrtcbin sink pad")
        if capsfilter.get_static_pad("src").link(requested_pad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Could not link RTP payloader to webrtcbin")
        self._appsrc = appsrc
        self._requested_pad = requested_pad
        self._attach_webrtc_diagnostics(pipeline, webrtc)
        self._subscribe_ros_camera()

    def _subscribe_ros_camera(self) -> None:
        rospy = _ensure_rospy_node()
        from aloha.msg import RGBGrayscaleImage

        self._subscriber = rospy.Subscriber(self.topic, RGBGrayscaleImage, self._on_ros_image, queue_size=1)
        self._emit_event("ros.subscribe", self.topic)

    def _on_ros_image(self, message: Any) -> None:
        if self._appsrc is None or not getattr(message, "images", None):
            return
        try:
            frame = image_msg_to_bgr(message.images[0])
            if frame.shape[:2] != (480, 640):
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            contiguous = np.ascontiguousarray(frame)
            Gst = self._gst()
            buffer = Gst.Buffer.new_allocate(None, int(contiguous.nbytes), None)
            buffer.fill(0, contiguous.tobytes())
            result = self._appsrc.emit("push-buffer", buffer)
            self._frame_count += 1
            now = time.time()
            if now - self._last_frame_time >= 2.0:
                self._last_frame_time = now
                self._emit_event("ros.frames", f"{self._frame_count}, push={result}")
        except Exception as exc:
            self._emit_event("ros.frame-error", exc)


class RosCameraFrameSource:
    def __init__(self, camera_name: str) -> None:
        validation = validate_ros_camera_name(camera_name)
        if not validation["ok"]:
            raise ValueError(validation["error"])
        self.camera_name = camera_name
        self.topic = validation["topic"]
        self._condition = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._frame_index = 0
        self._closed = False
        self._subscriber = None

    def start(self) -> None:
        rospy = _ensure_rospy_node()
        from aloha.msg import RGBGrayscaleImage

        self._subscriber = rospy.Subscriber(self.topic, RGBGrayscaleImage, self._on_ros_image, queue_size=1)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._subscriber is not None:
            self._subscriber.unregister()
            self._subscriber = None

    def wait_for_next_frame(self, last_seen_index: int, timeout_seconds: float = 1.0) -> Optional[Dict[str, Any]]:
        deadline = time.time() + timeout_seconds
        with self._condition:
            while not self._closed and self._frame_index <= last_seen_index:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)
            if self._closed or self._frame is None:
                return None
            return {
                "frame": self._frame.copy(),
                "frame_index": self._frame_index,
            }

    def _on_ros_image(self, message: Any) -> None:
        if not getattr(message, "images", None):
            return
        try:
            frame = image_msg_to_bgr(message.images[0])
            if frame.shape[:2] != (480, 640):
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            with self._condition:
                self._frame = np.ascontiguousarray(frame)
                self._frame_index += 1
                self._condition.notify_all()
        except Exception as exc:
            LOGGER.warning("aiortc ros camera frame error: %s", exc)


def build_aiortc_video_track_class() -> Any:
    import av
    from aiortc import VideoStreamTrack

    class RosCameraVideoTrack(VideoStreamTrack):
        def __init__(self, source: RosCameraFrameSource, fps: float) -> None:
            super().__init__()
            self._source = source
            self._fps = clamp_aiortc_fps(fps)
            self._last_seen_index = 0
            self._next_frame_time = time.monotonic()

        async def recv(self) -> Any:
            now = time.monotonic()
            if self._next_frame_time > now:
                await asyncio.sleep(self._next_frame_time - now)
            self._next_frame_time = max(self._next_frame_time + 1.0 / self._fps, time.monotonic())
            loop = asyncio.get_running_loop()
            frame_info = await loop.run_in_executor(
                None,
                self._source.wait_for_next_frame,
                self._last_seen_index,
                1.0,
            )
            if frame_info is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                self._last_seen_index = int(frame_info["frame_index"])
                frame = frame_info["frame"]
            pts, time_base = await self.next_timestamp()
            video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            video_frame.pts = pts
            video_frame.time_base = time_base or Fraction(1, 90000)
            return video_frame

    return RosCameraVideoTrack


def build_nvenc_h264_packet_track_class() -> Any:
    import av
    from aiortc import VideoStreamTrack

    class NvencH264PacketTrack(VideoStreamTrack):
        def __init__(self, source: RosCameraFrameSource, fps: float, bitrate: int = 2_000_000) -> None:
            super().__init__()
            self._source = source
            self._fps = clamp_aiortc_fps(fps)
            self._bitrate = bitrate
            self._packet_queue: "queue.Queue[Any]" = queue.Queue(maxsize=120)
            self._stop_event = threading.Event()
            self._pts = 0
            self._pts_step = max(1, int(90000 / self._fps))
            self._process = self._start_ffmpeg()
            self._writer_thread = threading.Thread(target=self._write_frames, daemon=True)
            self._reader_thread = threading.Thread(target=self._read_packets, daemon=True)
            self._writer_thread.start()
            self._reader_thread.start()

        def close(self) -> None:
            self._stop_event.set()
            try:
                if self._process.stdin:
                    self._process.stdin.close()
            except Exception:
                pass
            try:
                self._process.terminate()
                self._process.wait(timeout=1.0)
            except Exception:
                self._process.kill()

        async def recv(self) -> Any:
            loop = asyncio.get_running_loop()
            packet = await loop.run_in_executor(None, self._packet_queue.get)
            packet.pts = self._pts
            packet.dts = self._pts
            packet.time_base = Fraction(1, 90000)
            self._pts += self._pts_step
            return packet

        def _start_ffmpeg(self) -> subprocess.Popen:
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                "640x480",
                "-r",
                str(self._fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "h264_nvenc",
                "-preset",
                "llhp",
                "-bf",
                "0",
                "-g",
                str(max(1, int(self._fps))),
                "-b:v",
                str(self._bitrate),
                "-maxrate",
                str(self._bitrate),
                "-bufsize",
                str(self._bitrate * 2),
                "-f",
                "h264",
                "pipe:1",
            ]
            LOGGER.info("starting NVENC ffmpeg: %s", " ".join(command))
            return subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

        def _write_frames(self) -> None:
            last_seen_index = 0
            next_frame_time = time.monotonic()
            while not self._stop_event.is_set():
                now = time.monotonic()
                if next_frame_time > now:
                    time.sleep(next_frame_time - now)
                next_frame_time = max(next_frame_time + 1.0 / self._fps, time.monotonic())
                frame_info = self._source.wait_for_next_frame(last_seen_index, 1.0)
                if frame_info is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    last_seen_index = int(frame_info["frame_index"])
                    frame = frame_info["frame"]
                try:
                    if self._process.stdin is None:
                        return
                    self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
                    self._process.stdin.flush()
                except Exception as exc:
                    LOGGER.warning("NVENC ffmpeg stdin write failed: %s", exc)
                    self._stop_event.set()
                    return

        def _read_packets(self) -> None:
            try:
                if self._process.stdout is None:
                    return
                container = av.open(self._process.stdout, format="h264", mode="r")
                for packet in container.demux(video=0):
                    if self._stop_event.is_set():
                        return
                    if not packet.size:
                        continue
                    try:
                        self._packet_queue.put(packet, timeout=1.0)
                    except queue.Full:
                        LOGGER.warning("NVENC packet queue is full; dropping packet")
            except Exception as exc:
                if not self._stop_event.is_set():
                    stderr = b""
                    try:
                        if self._process.stderr is not None:
                            stderr = self._process.stderr.read()[-500:]
                    except Exception:
                        pass
                    LOGGER.warning("NVENC ffmpeg packet reader failed: %s stderr=%s", exc, stderr.decode("utf-8", "ignore"))

    return NvencH264PacketTrack


async def close_aiortc_session(session_id: str) -> bool:
    session = AIORTC_SESSIONS.pop(session_id, None)
    if session is None:
        return False
    source = session["source"]
    pc = session["pc"]
    track = session.get("track")
    if track is not None and hasattr(track, "close"):
        track.close()
    source.close()
    await pc.close()
    LOGGER.info("aiortc session %s closed", session_id)
    return True


async def close_aiortc_session_if_unconnected(session_id: str, timeout_seconds: float = 15.0) -> None:
    await asyncio.sleep(timeout_seconds)
    session = AIORTC_SESSIONS.get(session_id)
    if session is None:
        return
    if session.get("connected_at") is None:
        LOGGER.info("aiortc session %s did not connect within %.1fs; closing", session_id, timeout_seconds)
        await close_aiortc_session(session_id)


async def create_aiortc_ros_camera_answer(camera_name: str, offer: AiortcOfferRequest) -> AiortcOfferResponse:
    from aiortc import RTCConfiguration
    from aiortc import RTCPeerConnection
    from aiortc import RTCSessionDescription

    source = RosCameraFrameSource(camera_name)
    source.start()
    fps = clamp_aiortc_fps(offer.fps)
    encoder_mode = os.getenv("EII_WEBRTC_ENCODER", "raw").strip().lower()
    if encoder_mode == "nvenc":
        track_class = build_nvenc_h264_packet_track_class()
    else:
        track_class = build_aiortc_video_track_class()
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    session_id = uuid.uuid4().hex
    track = track_class(source, fps)
    AIORTC_SESSIONS[session_id] = {
        "camera_name": camera_name,
        "created_at": time.time(),
        "connected_at": None,
        "fps": fps,
        "encoder": encoder_mode,
        "pc": pc,
        "source": source,
        "track": track,
    }
    pc.addTrack(track)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        LOGGER.info("aiortc %s session=%s connectionState=%s", camera_name, session_id, pc.connectionState)
        if pc.connectionState == "connected" and session_id in AIORTC_SESSIONS:
            AIORTC_SESSIONS[session_id]["connected_at"] = time.time()
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            await close_aiortc_session(session_id)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    asyncio.create_task(close_aiortc_session_if_unconnected(session_id))
    return AiortcOfferResponse(sdp=pc.localDescription.sdp, type=pc.localDescription.type, session_id=session_id)


def run_videotestsrc_smoke(num_buffers: int = 30) -> Dict[str, Any]:
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


def get_ros_camera_config() -> Dict[str, Any]:
    return {
        "available": True,
        "cameras": {name: {"topic": topic} for name, topic in ROS_CAMERA_TOPICS.items()},
    }


def validate_ros_camera_name(camera_name: str) -> Dict[str, Any]:
    topic = ROS_CAMERA_TOPICS.get(camera_name)
    if topic is None:
        return {
            "ok": False,
            "error": f"Unknown camera {camera_name!r}",
            "known_cameras": sorted(ROS_CAMERA_TOPICS),
        }
    return {
        "ok": True,
        "error": None,
        "topic": topic,
    }


def image_msg_to_bgr(image_msg: Any) -> np.ndarray:
    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
    }
    channels = channels_by_encoding.get(getattr(image_msg, "encoding", None))
    if channels is None:
        raise ValueError(f"Unsupported image encoding: {getattr(image_msg, 'encoding', None)!r}")

    width = int(getattr(image_msg, "width", 0) or 0)
    height = int(getattr(image_msg, "height", 0) or 0)
    expected_size = width * height * channels
    frame = np.frombuffer(getattr(image_msg, "data", b""), dtype=np.uint8)
    if frame.size < expected_size:
        raise ValueError(f"Camera frame was truncated: got {frame.size}, expected {expected_size}")
    frame = frame[:expected_size].reshape((height, width, channels))

    encoding = image_msg.encoding
    if encoding == "rgb8":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if encoding == "bgr8":
        # Current robot camera diagnostics show RGB semantics are expected downstream.
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if encoding == "mono8":
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def build_jpeg_fakesink_command(jpeg_path: str) -> List[str]:
    return [
        "gst-launch-1.0",
        "-q",
        "filesrc",
        f"location={jpeg_path}",
        "!",
        "jpegdec",
        "!",
        "videoconvert",
        "!",
        "fakesink",
    ]


def _ensure_rospy_node() -> Any:
    import rospy

    if not rospy.core.is_initialized():
        rospy.init_node("eii_webrtc_media_probe", anonymous=True, disable_signals=True)
    return rospy


def capture_ros_camera_frame(camera_name: str, timeout_seconds: float) -> Dict[str, Any]:
    validation = validate_ros_camera_name(camera_name)
    if not validation["ok"]:
        raise ValueError(validation["error"])

    rospy = _ensure_rospy_node()
    from aloha.msg import RGBGrayscaleImage

    topic = validation["topic"]
    event = threading.Event()
    holder: Dict[str, Any] = {}

    def callback(message: Any) -> None:
        if event.is_set():
            return
        holder["message"] = message
        holder["received_at"] = time.time()
        event.set()

    subscriber = rospy.Subscriber(topic, RGBGrayscaleImage, callback, queue_size=1)
    started_at = time.time()
    try:
        if not event.wait(timeout=timeout_seconds):
            raise TimeoutError(f"Timed out waiting for {topic}")
    finally:
        subscriber.unregister()

    message = holder["message"]
    if not getattr(message, "images", None):
        raise ValueError(f"{topic} message contains no images")
    image_msg = message.images[0]
    frame = image_msg_to_bgr(image_msg)
    return {
        "camera_name": camera_name,
        "topic": topic,
        "frame": frame,
        "encoding": getattr(image_msg, "encoding", None),
        "width": int(getattr(image_msg, "width", 0) or 0),
        "height": int(getattr(image_msg, "height", 0) or 0),
        "wait_seconds": max(0.0, holder["received_at"] - started_at),
    }


def run_ros_camera_smoke(camera_name: str, timeout_seconds: float = 3.0, jpeg_quality: int = 90) -> Dict[str, Any]:
    captured = capture_ros_camera_frame(camera_name, timeout_seconds)
    encode_args = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    ok, jpeg = cv2.imencode(".jpg", captured["frame"], encode_args)
    if not ok:
        raise RuntimeError("cv2.imencode returned false")

    fd, jpeg_path = tempfile.mkstemp(prefix=f"eii-{camera_name}-", suffix=".jpg")
    os.close(fd)
    try:
        with open(jpeg_path, "wb") as file:
            file.write(jpeg.tobytes())
        command = build_jpeg_fakesink_command(jpeg_path)
        result = _run_command(command, timeout=10)
    finally:
        try:
            os.unlink(jpeg_path)
        except FileNotFoundError:
            pass

    return {
        "ok": result.ok,
        "camera_name": camera_name,
        "topic": captured["topic"],
        "encoding": captured["encoding"],
        "width": captured["width"],
        "height": captured["height"],
        "wait_seconds": captured["wait_seconds"],
        "jpeg_bytes": len(jpeg.tobytes()),
        "command": command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/media/gstreamer")
def gstreamer_status() -> Dict[str, Any]:
    return probe_gstreamer()


@app.get("/api/media/webrtc/runtime")
def webrtc_runtime_status() -> Dict[str, Any]:
    return probe_webrtc_runtime()


@app.get("/webrtc/videotestsrc", response_class=HTMLResponse)
def videotestsrc_webrtc_page() -> HTMLResponse:
    return HTMLResponse(build_videotestsrc_webrtc_test_page())


@app.get("/webrtc/ros-camera/{camera_name}", response_class=HTMLResponse)
def ros_camera_webrtc_page(camera_name: str) -> HTMLResponse:
    try:
        return HTMLResponse(build_ros_camera_webrtc_test_page(camera_name))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/webrtc/aiortc/ros-camera/{camera_name}", response_class=HTMLResponse)
def aiortc_ros_camera_page(camera_name: str) -> HTMLResponse:
    try:
        return HTMLResponse(build_aiortc_ros_camera_test_page(camera_name))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/media/smoke/videotestsrc")
def videotestsrc_smoke(request: VideoTestSrcSmokeRequest) -> Dict[str, Any]:
    return run_videotestsrc_smoke(num_buffers=request.num_buffers)


@app.get("/api/media/ros/cameras")
def ros_camera_config() -> Dict[str, Any]:
    return get_ros_camera_config()


@app.post("/api/media/smoke/ros-camera")
def ros_camera_smoke(request: RosCameraSmokeRequest) -> Dict[str, Any]:
    try:
        return run_ros_camera_smoke(
            camera_name=request.camera_name,
            timeout_seconds=request.timeout_seconds,
            jpeg_quality=request.jpeg_quality,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/media/aiortc/ros-camera/{camera_name}/offer", response_model=AiortcOfferResponse)
async def aiortc_ros_camera_offer(camera_name: str, request: AiortcOfferRequest) -> AiortcOfferResponse:
    try:
        return await create_aiortc_ros_camera_answer(camera_name, request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"aiortc dependency is not available: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/media/aiortc/sessions")
def list_aiortc_sessions() -> Dict[str, Any]:
    now = time.time()
    return {
        "count": len(AIORTC_SESSIONS),
        "sessions": [
            {
                "session_id": session_id,
                "camera_name": session["camera_name"],
                "age_seconds": now - float(session["created_at"]),
                "connected": session.get("connected_at") is not None,
                "connection_state": session["pc"].connectionState,
                "fps": session["fps"],
                "encoder": session["encoder"],
            }
            for session_id, session in AIORTC_SESSIONS.items()
        ],
    }


@app.delete("/api/media/aiortc/sessions/{session_id}")
async def delete_aiortc_session(session_id: str) -> Dict[str, Any]:
    closed = await close_aiortc_session(session_id)
    if not closed:
        raise HTTPException(status_code=404, detail="aiortc session not found")
    return {"status": "closed", "session_id": session_id}


async def _run_webrtc_socket(websocket: WebSocket, session: BaseWebRTCSession) -> None:
    await websocket.accept()
    loop = asyncio.get_running_loop()

    try:
        while True:
            message = await websocket.receive_json()
            if message.get("type") == "start":
                try:
                    offer_sdp = await loop.run_in_executor(None, session.start_with_server_offer)
                except Exception as exc:
                    await websocket.send_json({"type": "error", "error": str(exc)})
                    continue
                await websocket.send_json({"type": "offer", "sdp": offer_sdp})
                continue
            if message.get("type") == "answer":
                parsed_answer = parse_webrtc_answer_message(message)
                if not parsed_answer["ok"]:
                    await websocket.send_json({"type": "error", "error": parsed_answer["error"]})
                    continue
                try:
                    await loop.run_in_executor(None, session.accept_answer, parsed_answer["sdp"])
                except Exception as exc:
                    await websocket.send_json({"type": "error", "error": str(exc)})
                continue
            if message.get("type") == "ice":
                parsed_ice = parse_webrtc_ice_message(message)
                if parsed_ice["ok"]:
                    session.add_ice_candidate(parsed_ice["candidate"], parsed_ice["sdp_mline_index"])
                else:
                    await websocket.send_json({"type": "error", "error": parsed_ice["error"]})
                continue
            parsed = parse_webrtc_offer_message(message)
            if not parsed["ok"]:
                await websocket.send_json({"type": "error", "error": parsed["error"]})
                continue
            try:
                answer_sdp = await loop.run_in_executor(None, session.start_with_offer, parsed["sdp"])
            except Exception as exc:
                await websocket.send_json({"type": "error", "error": str(exc)})
                continue
            await websocket.send_json({"type": "answer", "sdp": answer_sdp})
    except WebSocketDisconnect:
        return
    finally:
        session.close()


@app.websocket("/ws/media/webrtc/videotestsrc")
async def videotestsrc_webrtc_socket(websocket: WebSocket) -> None:
    loop = asyncio.get_running_loop()

    def send_json_threadsafe(message: Dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(websocket.send_json(message), loop)

    await _run_webrtc_socket(websocket, VideoTestSrcWebRTCSession(send_json_threadsafe))


@app.websocket("/ws/media/webrtc/ros-camera/{camera_name}")
async def ros_camera_webrtc_socket(websocket: WebSocket, camera_name: str) -> None:
    loop = asyncio.get_running_loop()

    def send_json_threadsafe(message: Dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(websocket.send_json(message), loop)

    try:
        session = RosCameraWebRTCSession(camera_name, send_json_threadsafe)
    except ValueError:
        await websocket.close(code=1008)
        return
    await _run_webrtc_socket(websocket, session)
