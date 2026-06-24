from voice_assistant_web.backend.app import main


class _FakeCameraBridge:
    camera_names = ("cam_high", "cam_low")

    def __init__(self):
        self.calls = 0

    def snapshot_jpeg_b64_all(self):
        self.calls += 1
        return {"cam_high": "encoded"}


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
