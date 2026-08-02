from __future__ import annotations

import pathlib
import subprocess


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
INSTALLER = PROJECT_ROOT / "ops" / "isaac-frp" / "install-103.sh"
UNIT = PROJECT_ROOT / "ops" / "isaac-frp" / "frpc.service"


def _make_source(source: pathlib.Path) -> None:
    source.mkdir()
    frpc = source / "frpc"
    frpc.write_text("#!/bin/sh\n[ \"$1\" = verify ]\n", encoding="utf-8")
    frpc.chmod(0o755)
    (source / "frpc.toml").write_text('user = "eii-103-isaac"\n', encoding="utf-8")
    (source / "frps-token").write_text("not-a-real-token\n", encoding="utf-8")


def test_installer_populates_only_declared_files_in_test_root(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "root"
    _make_source(source)

    env = {"ISAAC_FRP_TEST_ROOT": str(target), "PATH": "/usr/bin:/bin"}
    subprocess.run(
        ["bash", str(INSTALLER), str(source)],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    installed = sorted(
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file()
    )
    assert installed == [
        "etc/frp/frpc.toml",
        "etc/frp/frps-token",
        "etc/systemd/system/frpc.service",
        "usr/local/bin/frpc",
    ]
    assert (target / "usr/local/bin/frpc").stat().st_mode & 0o777 == 0o755
    assert (target / "etc/frp/frpc.toml").stat().st_mode & 0o777 == 0o640
    assert (target / "etc/frp/frps-token").stat().st_mode & 0o777 == 0o640
    assert (target / "etc/systemd/system/frpc.service").read_bytes() == UNIT.read_bytes()


def test_installer_refuses_incomplete_source(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    env = {
        "ISAAC_FRP_TEST_ROOT": str(tmp_path / "root"),
        "PATH": "/usr/bin:/bin",
    }

    result = subprocess.run(
        ["bash", str(INSTALLER), str(source)],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "missing required deployment file" in result.stderr


def test_unit_is_boot_enabled_target_and_hardened() -> None:
    text = UNIT.read_text(encoding="utf-8")

    assert "WantedBy=multi-user.target" in text
    assert "User=frp" in text
    assert "ExecStart=/usr/local/bin/frpc -c /etc/frp/frpc.toml" in text
    assert "Restart=always" in text
    assert "ProtectSystem=strict" in text
    assert "ProtectHome=true" in text
