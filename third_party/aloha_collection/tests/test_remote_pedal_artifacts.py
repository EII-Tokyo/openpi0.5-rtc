import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "deploy/foot_pedal/aloha-foot-pedal.service"
UDEV_RULE = ROOT / "deploy/foot_pedal/99-aloha-foot-pedal.rules"
ENV_EXAMPLE = ROOT / "deploy/foot_pedal/foot-pedal.env.example"
INSTALLER = ROOT / "scripts/install_foot_pedal_relay.sh"


def test_service_is_fixed_non_shell_command_with_restart():
    text = SERVICE.read_text(encoding="utf-8")
    assert "User=eii" in text
    assert "EnvironmentFile=/etc/aloha-foot-pedal.env" in text
    assert "--event-code ${PEDAL_EVENT_CODE}" in text
    assert "Restart=on-failure" in text
    assert "ExecStart=/usr/bin/python3 " in text
    assert "/bin/sh -c" not in text
    assert "bash -c" not in text


def test_udev_rule_is_scoped_to_known_receiver():
    text = UDEV_RULE.read_text(encoding="utf-8")
    assert 'ATTRS{idVendor}=="046d"' in text
    assert 'ATTRS{idProduct}=="c548"' in text
    assert 'GROUP="input"' in text
    assert 'MODE="0660"' in text


def test_environment_example_contains_no_enrolled_device_or_secret():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert text == "PEDAL_DEVICE=\nPEDAL_EVENT_CODE=48\n"
    lowered = text.lower()
    assert "password" not in lowered
    assert "token" not in lowered


def test_installer_dry_run_lists_actions_without_writes():
    result = subprocess.run(
        ["bash", str(INSTALLER), "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "/opt/aloha-foot-pedal" in result.stdout
    assert "/etc/udev/rules.d/99-aloha-foot-pedal.rules" in result.stdout
    assert "/etc/systemd/system/aloha-foot-pedal.service" in result.stdout
    assert "systemctl daemon-reload" in result.stdout
    assert "systemctl enable" not in result.stdout


def test_installer_requires_explicit_mode():
    result = subprocess.run(
        ["bash", str(INSTALLER)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--dry-run or --apply" in result.stderr


def test_installer_resolves_ssh_as_service_user():
    text = INSTALLER.read_text(encoding="utf-8")
    assert "runuser -u eii -- ssh -G aloha" in text


def test_runtime_paths_are_ignored():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/.runtime/" in text
    assert "/deploy/foot_pedal/foot-pedal.env" in text
