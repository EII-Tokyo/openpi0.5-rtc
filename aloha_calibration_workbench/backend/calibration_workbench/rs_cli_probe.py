from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess

from .models import OwnershipState
from .models import ProductionProfile
from .preflight import CameraObservation


@dataclass(frozen=True, slots=True)
class CliDevice:
    serial: str
    product_name: str
    firmware: str
    recommended_firmware: str | None
    physical_port: str
    usb_type: str
    production_profile_supported: bool


RunCommand = Callable[[list[str]], subprocess.CompletedProcess[str]]
ProcessOwnerReader = Callable[[list[str]], list[str]]


def _default_run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True, timeout=20)


class RsEnumerateCliProbe:
    """Read device metadata through the installed librealsense enumeration CLI."""

    def __init__(
        self,
        profile: ProductionProfile,
        *,
        executable: str = "rs-enumerate-devices",
        runner: RunCommand = _default_run,
        video_node_resolver: Callable[[CliDevice], list[str]] | None = None,
        ownership_reader: Callable[[list[str]], tuple[OwnershipState, list[str]]] | None = None,
        process_signatures_by_serial: dict[str, list[str]] | None = None,
        process_owner_reader: ProcessOwnerReader | None = None,
    ):
        self._profile = profile
        self._executable = executable
        self._runner = runner
        self._video_node_resolver = video_node_resolver or (
            lambda device: _video_nodes_for_physical_port(device.physical_port)
        )
        self._ownership_reader = ownership_reader or _read_ownership
        self._process_signatures_by_serial = process_signatures_by_serial or {}
        self._process_owner_reader = process_owner_reader or _read_process_owners

    def enumerate(self) -> list[CameraObservation]:
        executable = shutil.which(self._executable) if "/" not in self._executable else self._executable
        if not executable or not Path(executable).is_file():
            raise RuntimeError("rs-enumerate-devices is unavailable")
        command = [executable, "--format", "full"]
        result = self._runner(command)
        if result.returncode != 0:
            raise RuntimeError(f"rs-enumerate-devices failed with exit code {result.returncode}")
        devices = parse_rs_enumerate_output(result.stdout, self._profile)
        observations: list[CameraObservation] = []
        for device in devices:
            nodes = self._video_node_resolver(device)
            ownership, owners = self._ownership_reader(nodes)
            process_owners = self._process_owner_reader(self._process_signatures_by_serial.get(device.serial, []))
            if process_owners:
                ownership = OwnershipState.BUSY
                owners = sorted(set(owners) | set(process_owners))
            observations.append(
                CameraObservation(
                    serial=device.serial,
                    product_name=device.product_name,
                    firmware=device.firmware,
                    recommended_firmware=device.recommended_firmware,
                    usb_type=device.usb_type,
                    physical_port=device.physical_port,
                    production_profile_supported=device.production_profile_supported,
                    ownership=ownership,
                    owner_processes=owners,
                    video_nodes=nodes,
                )
            )
        return observations


def parse_rs_enumerate_output(output: str, profile: ProductionProfile) -> list[CliDevice]:
    blocks = re.split(r"(?m)^Device info:\s*$", output)
    devices: list[CliDevice] = []
    for block in blocks[1:]:
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", maxsplit=1)
            normalized_key = " ".join(key.split())
            if normalized_key in {
                "Name",
                "Serial Number",
                "Firmware Version",
                "Recommended Firmware Version",
                "Physical Port",
                "Usb Type Descriptor",
            }:
                fields[normalized_key] = value.strip()
        serial = fields.get("Serial Number")
        if not serial:
            continue
        profile_pattern = re.compile(
            rf"(?m)^\s*Color\s+{profile.width}x{profile.height}\s+{re.escape(profile.format.upper())}"
            rf"\s+@\s+([0-9/]+)\s+Hz\s*$"
        )
        profile_match = profile_pattern.search(block)
        supported_fps = set(profile_match.group(1).split("/")) if profile_match else set()
        devices.append(
            CliDevice(
                serial=serial,
                product_name=fields.get("Name", "N/A"),
                firmware=fields.get("Firmware Version", "N/A"),
                recommended_firmware=fields.get("Recommended Firmware Version"),
                physical_port=fields.get("Physical Port", "N/A"),
                usb_type=fields.get("Usb Type Descriptor", "N/A"),
                production_profile_supported=str(profile.fps) in supported_fps,
            )
        )
    return devices


def _video_nodes_for_physical_port(physical_port: str) -> list[str]:
    anchor_name = Path(physical_port).name
    if re.fullmatch(r"video\d+", anchor_name) is None or shutil.which("udevadm") is None:
        return []
    anchor = Path("/dev") / anchor_name
    result = subprocess.run(
        ["udevadm", "info", "--query=property", f"--name={anchor}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    if result.returncode != 0:
        return []
    properties = _parse_udev_properties(result.stdout)
    asic_serial = properties.get("ID_USB_SERIAL_SHORT")
    if not asic_serial:
        return []
    nodes: set[str] = set()
    for link in Path("/dev/v4l/by-id").glob(f"*_{asic_serial}-video-index*"):
        try:
            target = link.resolve(strict=True)
        except OSError:
            continue
        if re.fullmatch(r"video\d+", target.name):
            nodes.add(str(Path("/dev") / target.name))
    return sorted(nodes, key=_video_node_number)


def _parse_udev_properties(output: str) -> dict[str, str]:
    properties: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        properties[key] = value
    return properties


def _video_node_number(path: str) -> int:
    match = re.search(r"(\d+)$", path)
    return int(match.group(1)) if match else -1


def _read_ownership(video_nodes: list[str]) -> tuple[OwnershipState, list[str]]:
    if not video_nodes or shutil.which("fuser") is None:
        return OwnershipState.UNKNOWN, []
    owners: set[str] = set()
    for node in video_nodes:
        result = subprocess.run(
            ["fuser", node],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode not in {0, 1}:
            return OwnershipState.UNKNOWN, sorted(owners)
        for token in result.stdout.split():
            if not token.isdigit():
                continue
            owners.add(f"pid={token}:{_process_name(token)}")
    if owners:
        return OwnershipState.BUSY, sorted(owners)
    return OwnershipState.FREE, []


def _process_name(pid: str) -> str:
    try:
        return (Path("/proc") / pid / "comm").read_text(encoding="utf-8").strip()[:80]
    except OSError:
        return "unknown"


def _read_process_owners(signatures: list[str]) -> list[str]:
    if not signatures:
        return []
    process_table: list[tuple[str, str, str]] = []
    for process_dir in Path("/proc").glob("[0-9]*"):
        try:
            command = (process_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
            name = (process_dir / "comm").read_text(encoding="utf-8").strip()[:80]
        except OSError:
            continue
        process_table.append((process_dir.name, name, command))
    return owners_from_process_table(process_table, signatures)


def owners_from_process_table(
    process_table: list[tuple[str, str, str]],
    signatures: list[str],
) -> list[str]:
    normalized_signatures = [signature.lower() for signature in signatures if signature]
    owners: set[str] = set()
    for pid, name, command in process_table:
        normalized_command = command.lower()
        is_camera_runtime = "realsense" in normalized_command or "realsense" in name.lower()
        if not is_camera_runtime:
            continue
        if any(signature in normalized_command for signature in normalized_signatures):
            owners.add(f"pid={pid}:{name[:80]}")
    return sorted(owners)
