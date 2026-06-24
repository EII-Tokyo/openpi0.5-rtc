from voice_assistant_web.backend.app import main


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
    monkeypatch.setattr(main.camera_bridge, "camera_names", ("cam_high", "cam_low"))

    response = main.camera_capabilities()

    assert response.preferred_transport == "mjpeg"
    assert response.transports == ["mjpeg", "jpeg_ws"]
    assert response.cameras == ["cam_high", "cam_low"]
    assert response.include_realtime_frames is False
    assert response.webrtc["enabled"] is False


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
