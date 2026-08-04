from types import SimpleNamespace

from aloha.video_encoding import probe_nvenc


def test_nvenc_probe_uses_a_supported_frame_size(monkeypatch):
    observed_commands = []

    def fake_run(command, **_kwargs):
        observed_commands.append(command)
        return SimpleNamespace(
            returncode=(
                0
                if "color=size=640x480:rate=1" in command
                else 1
            )
        )

    monkeypatch.setattr(
        "aloha.video_encoding.subprocess.run",
        fake_run,
    )
    probe_nvenc.cache_clear()
    try:
        assert probe_nvenc("ffmpeg-nvenc-probe-test")
    finally:
        probe_nvenc.cache_clear()

    assert len(observed_commands) == 1
