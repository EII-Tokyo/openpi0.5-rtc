from __future__ import annotations

from scripts import value_server


def test_detect_backend_torch(tmp_path):
    step = tmp_path / "42"
    step.mkdir()
    (step / "model.safetensors").write_bytes(b"")
    (step / "metadata.pt").write_bytes(b"")

    assert value_server._detect_backend(step, "auto") == "torch"


def test_detect_backend_jax(tmp_path):
    step = tmp_path / "8"
    step.mkdir()
    (step / "state").mkdir()

    assert value_server._detect_backend(step, "auto") == "jax"


def test_resolve_step_dir_picks_latest_numeric(tmp_path):
    (tmp_path / "1").mkdir()
    latest = tmp_path / "12"
    latest.mkdir()

    resolved = value_server._resolve_step_dir(str(tmp_path))
    assert resolved == latest
