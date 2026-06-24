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
    assert status["plugins"]["nicesrc"]["available"] is True
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


def test_probe_webrtc_runtime_reports_structured_error(monkeypatch):
    def fake_probe():
        raise RuntimeError("libnice elements are not available")

    monkeypatch.setattr(media_service, "_probe_webrtc_runtime", fake_probe)

    status = media_service.probe_webrtc_runtime()

    assert status["available"] is False
    assert status["ready"] is False
    assert status["sink_request_pad"] is False
    assert status["error"] == "libnice elements are not available"


def test_videotestsrc_webrtc_page_contains_signaling_endpoint():
    html = media_service.build_videotestsrc_webrtc_test_page()

    assert "RTCPeerConnection" in html
    assert "/ws/media/webrtc/videotestsrc" in html
    assert "type: 'start'" in html
    assert "type: 'answer'" in html
    assert "<video" in html
    assert "getStats" in html
    assert "server event" in html


def test_ros_camera_webrtc_page_contains_camera_signaling_endpoint():
    html = media_service.build_ros_camera_webrtc_test_page("cam_high")

    assert "WebRTC ROS Camera cam_high" in html
    assert "/ws/media/webrtc/ros-camera/cam_high" in html
    assert "RTCPeerConnection" in html


def test_aiortc_ros_camera_page_posts_offer_endpoint():
    html = media_service.build_aiortc_ros_camera_test_page("cam_high")

    assert "aiortc ROS Camera cam_high" in html
    assert "/api/media/aiortc/ros-camera/cam_high/offer" in html
    assert "waitForIceGatheringComplete" in html


def test_clamp_aiortc_fps_bounds_requested_frame_rate():
    assert media_service.clamp_aiortc_fps(None) == 15.0
    assert media_service.clamp_aiortc_fps(0.0) == 15.0
    assert media_service.clamp_aiortc_fps(5.0) == 5.0
    assert media_service.clamp_aiortc_fps(120.0) == 30.0


def test_parse_webrtc_offer_message_requires_offer_sdp():
    message = {"type": "offer", "sdp": "v=0\r\n"}

    parsed = media_service.parse_webrtc_offer_message(message)

    assert parsed["ok"] is True
    assert parsed["sdp"] == "v=0\r\n"


def test_parse_webrtc_answer_message_requires_answer_sdp():
    message = {"type": "answer", "sdp": "v=0\r\n"}

    parsed = media_service.parse_webrtc_answer_message(message)

    assert parsed["ok"] is True
    assert parsed["sdp"] == "v=0\r\n"


def test_parse_webrtc_offer_message_rejects_bad_payload():
    message = {"type": "candidate", "candidate": "x"}

    parsed = media_service.parse_webrtc_offer_message(message)

    assert parsed["ok"] is False
    assert "Expected offer" in parsed["error"]


def test_parse_webrtc_ice_message_accepts_end_of_candidates():
    message = {"type": "ice", "candidate": None, "sdpMLineIndex": 0}

    parsed = media_service.parse_webrtc_ice_message(message)

    assert parsed["ok"] is True
    assert parsed["candidate"] == ""
    assert parsed["sdp_mline_index"] == 0


def test_add_ice_candidate_ignores_empty_end_of_candidates():
    class FakeWebRTC:
        def __init__(self):
            self.calls = []

        def emit(self, *args):
            self.calls.append(args)

    session = media_service.BaseWebRTCSession(lambda _message: None)
    session._webrtc = FakeWebRTC()

    session.add_ice_candidate("", 0)

    assert session._webrtc.calls == []


def test_extract_sdp_ice_candidates_tracks_mline_index():
    sdp = (
        "v=0\r\n"
        "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
        "a=candidate:audio 1 udp 1 127.0.0.1 10000 typ host\r\n"
        "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n"
        "a=candidate:video 1 udp 1 127.0.0.1 10001 typ host\r\n"
    )

    candidates = media_service.extract_sdp_ice_candidates(sdp)

    assert candidates == [
        {"sdp_mline_index": 0, "candidate": "candidate:audio 1 udp 1 127.0.0.1 10000 typ host"},
        {"sdp_mline_index": 1, "candidate": "candidate:video 1 udp 1 127.0.0.1 10001 typ host"},
    ]


def test_should_publish_webrtc_candidate_filters_ipv6_and_docker_subnets():
    assert media_service.should_publish_webrtc_candidate(
        "candidate:1 1 udp 1 192.168.1.101 10000 typ host"
    )
    assert media_service.should_publish_webrtc_candidate("candidate:1 1 udp 1 10.10.0.15 10000 typ host")
    assert not media_service.should_publish_webrtc_candidate("candidate:1 1 udp 1 172.26.0.1 10000 typ host")
    assert not media_service.should_publish_webrtc_candidate(
        "candidate:1 1 udp 1 240b:10:760:100:a0dd:b112:1dcc:df9d 10000 typ host"
    )


def test_filter_webrtc_sdp_candidates_removes_unpublishable_candidates():
    sdp = (
        "v=0\r\n"
        "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n"
        "a=candidate:1 1 udp 1 172.26.0.1 10000 typ host\r\n"
        "a=candidate:2 1 udp 1 192.168.1.101 10001 typ host\r\n"
        "a=mid:0\r\n"
    )

    filtered = media_service.filter_webrtc_sdp_candidates(sdp)

    assert "172.26.0.1" not in filtered
    assert "192.168.1.101" in filtered
    assert "a=mid:0" in filtered


def test_ros_camera_webrtc_session_name_and_topic():
    session = media_service.RosCameraWebRTCSession("cam_high", lambda _message: None)

    assert session.camera_name == "cam_high"
    assert session.topic == "/cam_high"


def test_configure_webrtc_element_uses_max_bundle():
    class FakeWebRTC:
        def __init__(self):
            self.properties = {}

        def set_property(self, name, value):
            self.properties[name] = value

    class FakePolicy:
        MAX_BUNDLE = "max-bundle"

    webrtc = FakeWebRTC()

    media_service.configure_webrtc_element(webrtc, FakePolicy)

    assert webrtc.properties["bundle-policy"] == "max-bundle"
