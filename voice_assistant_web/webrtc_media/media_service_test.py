from voice_assistant_web.webrtc_media import media_service


class FakeImageMessage:
    def __init__(self, encoding="rgb8", width=2, height=1, data=None):
        self.encoding = encoding
        self.width = width
        self.height = height
        self.data = data if data is not None else bytes([255, 0, 0, 0, 255, 0])


def test_probe_gstreamer_reports_missing_plugins(monkeypatch):
    def fake_run(command, timeout):
        assert command[0] == "gst-inspect-1.0"
        if command[1] == "webrtcbin":
            return media_service.CommandResult(ok=False, stdout="", stderr="missing", returncode=1)
        return media_service.CommandResult(ok=True, stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(media_service, "_run_command", fake_run)

    status = media_service.probe_gstreamer()

    assert status["available"] is False
    assert status["plugins"]["webrtcbin"]["available"] is False
    assert status["plugins"]["webrtcbin"]["error"] == "missing"


def test_probe_gstreamer_reports_available_plugins(monkeypatch):
    def fake_run(command, timeout):
        return media_service.CommandResult(ok=True, stdout=f"{command[-1]} ok", stderr="", returncode=0)

    monkeypatch.setattr(media_service, "_run_command", fake_run)
    monkeypatch.setattr(media_service, "_import_gst_webrtc_bindings", lambda: None)

    status = media_service.probe_gstreamer()

    assert status["available"] is True
    assert status["plugins"]["webrtcbin"]["available"] is True
    assert status["plugins"]["videotestsrc"]["available"] is True


def test_probe_gstreamer_reports_missing_python_bindings(monkeypatch):
    def fake_import():
        raise ModuleNotFoundError("No module named 'gi'")

    monkeypatch.setattr(media_service, "_import_gst_webrtc_bindings", fake_import)
    monkeypatch.setattr(
        media_service,
        "_run_command",
        lambda command, timeout: media_service.CommandResult(ok=True, stdout="ok", stderr="", returncode=0),
    )

    status = media_service.probe_gstreamer()

    assert status["available"] is False
    assert status["python_bindings"]["available"] is False
    assert "No module named 'gi'" in status["python_bindings"]["error"]


def test_run_videotestsrc_smoke_uses_finite_pipeline(monkeypatch):
    captured = {}

    def fake_run(command, timeout):
        captured["command"] = command
        captured["timeout"] = timeout
        return media_service.CommandResult(ok=True, stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(media_service, "_run_command", fake_run)

    result = media_service.run_videotestsrc_smoke(num_buffers=7)

    assert result["ok"] is True
    assert captured["timeout"] == 10
    assert captured["command"] == [
        "gst-launch-1.0",
        "-q",
        "videotestsrc",
        "num-buffers=7",
        "!",
        "videoconvert",
        "!",
        "fakesink",
    ]


def test_ros_camera_config_exposes_expected_topics():
    config = media_service.get_ros_camera_config()

    assert config["available"] is True
    assert config["cameras"]["cam_high"]["topic"] == "/cam_high"
    assert config["cameras"]["cam_right_wrist"]["topic"] == "/cam_right_wrist"


def test_validate_ros_camera_name_rejects_unknown_camera():
    result = media_service.validate_ros_camera_name("cam_missing")

    assert result["ok"] is False
    assert "Unknown camera" in result["error"]


def test_image_msg_to_bgr_converts_rgb_payload_to_bgr():
    image = FakeImageMessage(encoding="rgb8", width=2, height=1, data=bytes([255, 0, 0, 0, 255, 0]))

    frame = media_service.image_msg_to_bgr(image)

    assert frame.shape == (1, 2, 3)
    assert frame[0, 0].tolist() == [0, 0, 255]
    assert frame[0, 1].tolist() == [0, 255, 0]


def test_build_real_camera_jpeg_fakesink_command():
    command = media_service.build_jpeg_fakesink_command("/tmp/frame.jpg")

    assert command == [
        "gst-launch-1.0",
        "-q",
        "filesrc",
        "location=/tmp/frame.jpg",
        "!",
        "jpegdec",
        "!",
        "videoconvert",
        "!",
        "fakesink",
    ]
