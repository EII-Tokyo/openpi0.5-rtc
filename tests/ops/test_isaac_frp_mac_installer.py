from __future__ import annotations

import io
import os
import pathlib
import subprocess
import tarfile


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
INSTALLER = PROJECT_ROOT / "ops" / "isaac-frp" / "install-mac.sh"


def _make_bundle(bundle: pathlib.Path) -> None:
    bundle.mkdir()
    (bundle / "frps-token").write_text("test-token\n", encoding="utf-8")
    (bundle / "endpoint-secrets.env").write_text(
        "SSH_STCP_SECRET=ssh-secret\n"
        "WEBRTC_SIGNAL_STCP_SECRET=signal-secret\n"
        "WEBRTC_MEDIA_SUDP_SECRET=media-secret\n",
        encoding="utf-8",
    )


def _make_fake_archive(path: pathlib.Path) -> None:
    payload = b"#!/bin/sh\nif [ \"$1\" = --version ]; then echo 0.66.0; exit; fi\n[ \"$1\" = verify ]\n"
    info = tarfile.TarInfo("frp_0.66.0_darwin_amd64/frpc")
    info.mode = 0o755
    info.size = len(payload)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))


def test_mac_installer_renders_localhost_visitors(tmp_path: pathlib.Path) -> None:
    bundle = tmp_path / "bundle"
    target_home = tmp_path / "mac-home"
    archive = tmp_path / "frp.tar.gz"
    _make_bundle(bundle)
    _make_fake_archive(archive)
    env = {
        "PATH": "/usr/bin:/bin",
        "ISAAC_FRP_TEST_ARCH": "x86_64",
        "ISAAC_FRP_TEST_HOME": str(target_home),
        "ISAAC_FRP_ARCHIVE_PATH": str(archive),
    }

    result = subprocess.run(
        ["bash", str(INSTALLER), str(bundle)],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    config_path = target_home / ".config/frp/frpc.toml"
    config = config_path.read_text(encoding="utf-8")
    assert 'user = "eii-mac"' in config
    assert 'serverUser = "eii-103-isaac"' in config
    assert 'serverName = "isaac-ssh"' in config
    assert 'bindAddr = "127.0.0.1"' in config
    assert "bindPort = 22022" in config
    assert "bindPort = 49100" in config
    assert "bindPort = 47998" in config
    assert f'{target_home}/.config/frp/frps-token' in config
    assert config_path.stat().st_mode & 0o777 == 0o600
    assert (target_home / ".config/frp/frps-token").stat().st_mode & 0o777 == 0o600
    assert (target_home / ".local/bin/frpc").stat().st_mode & 0o777 == 0o755
    assert "ssh-secret" not in result.stdout
    assert "test-token" not in result.stdout


def test_mac_installer_rejects_unsupported_architecture(tmp_path: pathlib.Path) -> None:
    bundle = tmp_path / "bundle"
    _make_bundle(bundle)
    env = {
        "PATH": "/usr/bin:/bin",
        "ISAAC_FRP_TEST_ARCH": "powerpc",
        "ISAAC_FRP_TEST_HOME": str(tmp_path / "mac-home"),
    }

    result = subprocess.run(
        ["bash", str(INSTALLER), str(bundle)],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "unsupported Mac architecture" in result.stderr
