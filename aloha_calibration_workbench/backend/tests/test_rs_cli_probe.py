import subprocess

from calibration_workbench.models import OwnershipState
from calibration_workbench.models import ProductionProfile
from calibration_workbench.rs_cli_probe import RsEnumerateCliProbe
from calibration_workbench.rs_cli_probe import _parse_udev_properties
from calibration_workbench.rs_cli_probe import parse_rs_enumerate_output

SAMPLE_OUTPUT = """Device info:
    Name                          : Intel RealSense D405
    Serial Number                 : 218622270440
    Firmware Version              : 5.12.14.100
    Recommended Firmware Version  : 5.17.0.10
    Physical Port                 : /sys/devices/usb/video18
    Usb Type Descriptor           : 3.2

Stream Profiles supported by Color Sensor
 Supported modes:
    Color        640x480       RGB8        @ 90/60/30/15/5 Hz
    Color        1280x720      RGB8        @ 30/15/5 Hz
Device info:
    Name                          : Intel RealSense D405
    Serial Number                 : 130322270656
    Firmware Version              : 5.17.0.10
    Recommended Firmware Version  : 5.17.0.10
    Physical Port                 : /sys/devices/usb/video0
    Usb Type Descriptor           : 3.2

Stream Profiles supported by Color Sensor
 Supported modes:
    Color        640x480       YUYV        @ 90/60/30/15/5 Hz
"""


def test_parser_matches_rgb8_profile_and_preserves_firmware_evidence():
    devices = parse_rs_enumerate_output(
        SAMPLE_OUTPUT,
        ProductionProfile(width=640, height=480, fps=60, format="rgb8"),
    )

    assert len(devices) == 2
    assert devices[0].serial == "218622270440"
    assert devices[0].production_profile_supported is True
    assert devices[0].firmware == "5.12.14.100"
    assert devices[0].recommended_firmware == "5.17.0.10"
    assert devices[1].production_profile_supported is False


def test_cli_probe_uses_enumeration_only_and_never_starts_a_pipeline(tmp_path):
    executable = tmp_path / "rs-enumerate-devices"
    executable.write_text("test executable", encoding="utf-8")
    commands: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=SAMPLE_OUTPUT, stderr="")

    probe = RsEnumerateCliProbe(
        ProductionProfile(width=640, height=480, fps=60, format="rgb8"),
        executable=str(executable),
        runner=runner,
        video_node_resolver=lambda device: [f"/dev/by-serial/{device.serial}"],
        ownership_reader=lambda nodes: (OwnershipState.FREE, []),
    )

    observations = probe.enumerate()

    assert commands == [[str(executable), "--format", "full"]]
    assert all("reset" not in token and "calib" not in token for token in commands[0])
    assert observations[0].ownership is OwnershipState.FREE
    assert observations[0].video_nodes == ["/dev/by-serial/218622270440"]


def test_udev_property_parser_keeps_asic_serial_separate_from_logical_serial():
    properties = _parse_udev_properties(
        "DEVNAME=/dev/video0\nID_SERIAL_SHORT=227123070438\nID_USB_SERIAL_SHORT=227123070438\n"
    )

    assert properties["ID_USB_SERIAL_SHORT"] == "227123070438"
