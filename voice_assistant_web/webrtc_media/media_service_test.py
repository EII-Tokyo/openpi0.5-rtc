from voice_assistant_web.webrtc_media import media_service


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

    status = media_service.probe_gstreamer()

    assert status["available"] is True
    assert status["plugins"]["webrtcbin"]["available"] is True
    assert status["plugins"]["videotestsrc"]["available"] is True


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
