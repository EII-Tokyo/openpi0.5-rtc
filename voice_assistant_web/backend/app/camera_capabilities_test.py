import pytest
from fastapi import HTTPException

from voice_assistant_web.backend.app import main
from voice_assistant_web.backend.app.schemas import CameraWebRTCSessionRequest


class _FakeCameraBridge:
    camera_names = ("cam_high", "cam_low")

    def __init__(self):
        self.calls = 0

    def snapshot_jpeg_b64_all(self):
        self.calls += 1
        return {"cam_high": "encoded"}

    def get_diagnostics(self):
        return {
            "bridge_running": True,
            "bridge_error": None,
            "jpeg_quality": 70,
            "cameras": {
                "cam_high": {
                    "has_frame": True,
                    "frame_age_seconds": 0.1,
                    "source_fps_recent": 9.5,
                    "encoded_fps_recent": 9.5,
                    "raw_frames_total": 12,
                    "encoded_frames_total": 12,
                    "dropped_frames_total": 0,
                    "error_count": 0,
                    "last_error": None,
                    "last_encoding": "rgb8",
                    "last_width": 640,
                    "last_height": 480,
                    "latest_jpeg_bytes": 12345,
                    "encode_ms_mean_recent": 3.2,
                    "encode_ms_max_recent": 4.1,
                }
            },
        }


def test_camera_capabilities_default_to_mjpeg(monkeypatch):
    monkeypatch.setattr(main.settings, "camera_transport", "mjpeg")
    monkeypatch.setattr(main.settings, "camera_webrtc_enabled", False)
    monkeypatch.setattr(main.settings, "realtime_include_camera_frames", False)
    monkeypatch.setattr(main.settings, "camera_webrtc_media_url", "http://127.0.0.1:8013")
    monkeypatch.setattr(main.camera_bridge, "camera_names", ("cam_high", "cam_low"))

    response = main.camera_capabilities()

    assert response.preferred_transport == "mjpeg"
    assert response.transports == ["mjpeg", "jpeg_ws"]
    assert response.cameras == ["cam_high", "cam_low"]
    assert response.include_realtime_frames is False
    assert response.webrtc["enabled"] is False
    assert response.webrtc["media_service_url"] == "http://127.0.0.1:8013"
    assert response.webrtc["media_service_attached"] is False


def test_realtime_camera_frames_disabled_by_default(monkeypatch):
    bridge = _FakeCameraBridge()
    monkeypatch.setattr(main, "camera_bridge", bridge)
    monkeypatch.setattr(main.settings, "realtime_include_camera_frames", False)

    assert main._realtime_camera_frames() == {}
    assert bridge.calls == 0


def test_realtime_camera_frames_can_use_legacy_jpeg_ws(monkeypatch):
    bridge = _FakeCameraBridge()
    monkeypatch.setattr(main, "camera_bridge", bridge)
    monkeypatch.setattr(main.settings, "realtime_include_camera_frames", True)

    assert main._realtime_camera_frames() == {"cam_high": "encoded"}
    assert bridge.calls == 1


def test_camera_diagnostics_returns_bridge_metrics(monkeypatch):
    bridge = _FakeCameraBridge()
    monkeypatch.setattr(main, "camera_bridge", bridge)

    response = main.camera_diagnostics()

    assert response.bridge_running is True
    assert response.cameras["cam_high"].has_frame is True
    assert response.cameras["cam_high"].source_fps_recent == 9.5
    assert response.cameras["cam_high"].encode_ms_mean_recent == 3.2


def test_camera_stream_interval_uses_default_when_fps_missing(monkeypatch):
    monkeypatch.setattr(main.settings, "camera_mjpeg_default_fps", 20.0)
    monkeypatch.setattr(main.settings, "camera_mjpeg_max_fps", 30.0)

    assert main._camera_stream_interval(None) == 0.05


def test_camera_stream_interval_clamps_requested_fps(monkeypatch):
    monkeypatch.setattr(main.settings, "camera_mjpeg_default_fps", 20.0)
    monkeypatch.setattr(main.settings, "camera_mjpeg_max_fps", 30.0)

    assert main._camera_stream_interval(120.0) == 1.0 / 30.0
    assert main._camera_stream_interval(0.0) == 0.05
    assert main._camera_stream_interval(-5.0) == 0.05


def test_webrtc_session_creation_is_disabled_by_default(monkeypatch):
    monkeypatch.setattr(main.settings, "camera_webrtc_enabled", False)

    with pytest.raises(HTTPException) as exc_info:
        main.create_webrtc_camera_session(CameraWebRTCSessionRequest(cameras=["cam_high"]))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "WebRTC camera transport is disabled"


def test_webrtc_session_rejects_unknown_camera(monkeypatch):
    monkeypatch.setattr(main.settings, "camera_webrtc_enabled", True)
    monkeypatch.setattr(main.camera_bridge, "camera_names", ("cam_high", "cam_low"))
    main.webrtc_sessions.clear()

    with pytest.raises(HTTPException) as exc_info:
        main.create_webrtc_camera_session(CameraWebRTCSessionRequest(cameras=["cam_missing"]))

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Unknown camera cam_missing"


def test_webrtc_session_lifecycle(monkeypatch):
    monkeypatch.setattr(main.settings, "camera_webrtc_enabled", True)
    monkeypatch.setattr(main.settings, "camera_webrtc_session_ttl_seconds", 30.0)
    monkeypatch.setattr(main.camera_bridge, "camera_names", ("cam_high", "cam_low"))
    main.webrtc_sessions.clear()

    created = main.create_webrtc_camera_session(CameraWebRTCSessionRequest(cameras=["cam_high", "cam_low"]))

    assert created.session_id
    assert created.status == "signaling"
    assert created.cameras == ["cam_high", "cam_low"]
    assert created.signaling_url.endswith(f"/ws/cameras/webrtc/{created.session_id}")
    assert created.fallback_transport == "mjpeg"

    deleted = main.delete_webrtc_camera_session(created.session_id)

    assert deleted.status == "closed"
