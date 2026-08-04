#!/usr/bin/env python3
"""Read one enrolled USB pedal event and relay it as b to machine 103."""

from __future__ import annotations

import argparse
import glob
import os
import select
import signal
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evdev import InputDevice, ecodes

from aloha.foot_pedal_relay import (
    ForwardResult,
    PedalRelay,
    PersistentSshTransport,
    build_ssh_command,
    deduplicate_device_paths,
    open_input_devices,
)


_STOP = False


def _request_stop(signum, frame) -> None:
    global _STOP
    _STOP = True


def keyboard_candidates() -> list[Path]:
    return deduplicate_device_paths([
        Path(path)
        for path in sorted(glob.glob("/dev/input/by-path/*event-kbd"))
        if Path(path).exists()
    ])


def identify_pedal(timeout: float) -> Path:
    candidates = keyboard_candidates()
    if not candidates:
        raise RuntimeError("no /dev/input/by-path keyboard devices are readable")
    device_paths = open_input_devices(candidates, InputDevice)
    devices = [device for device, _ in device_paths]
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            readable, _, _ = select.select(
                list(devices),
                [],
                [],
                min(0.25, max(0.0, deadline - time.monotonic())),
            )
            sources = set()
            for device in readable:
                for event in device.read():
                    if (
                        event.type == ecodes.EV_KEY
                        and event.code == ecodes.KEY_B
                        and event.value == 1
                    ):
                        sources.update(
                            path
                            for opened_device, path in device_paths
                            if opened_device is device
                        )
            if len(sources) == 1:
                return sources.pop()
            if len(sources) > 1:
                raise RuntimeError("KEY_B arrived from multiple devices")
    finally:
        for device in devices:
            device.close()
    raise TimeoutError(f"no pedal KEY_B press received within {timeout:.0f}s")


def write_environment(path: Path, device: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"PEDAL_DEVICE={device}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def run_relay(device_path: Path, ssh_host: str, debounce: float, event_code: int) -> int:
    transport = PersistentSshTransport(build_ssh_command(ssh_host))
    relay = PedalRelay(
        transport,
        debounce_seconds=debounce,
        event_code=event_code,
    )
    device = None
    try:
        while not _STOP:
            now = time.monotonic()
            transport.ensure_connected(now=now)
            if device is None:
                try:
                    device = InputDevice(str(device_path))
                    print(f"pedal-relay: opened {device_path}", flush=True)
                except OSError as exc:
                    print(f"pedal-relay: device unavailable: {exc}", file=sys.stderr, flush=True)
                    time.sleep(1.0)
                    continue
            try:
                readable, _, _ = select.select([device], [], [], 0.25)
                for input_device in readable:
                    for event in input_device.read():
                        result = relay.process_event(event, now=time.monotonic())
                        if result is ForwardResult.SENT:
                            print("pedal-relay: forwarded b", flush=True)
                        elif result is ForwardResult.DROPPED:
                            print(
                                "pedal-relay: dropped b while transport unavailable",
                                file=sys.stderr,
                                flush=True,
                            )
            except OSError as exc:
                print(f"pedal-relay: device disconnected: {exc}", file=sys.stderr, flush=True)
                device.close()
                device = None
    finally:
        if device is not None:
            device.close()
        transport.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=Path)
    parser.add_argument("--ssh-host", default="aloha")
    parser.add_argument("--debounce", type=float, default=0.4)
    parser.add_argument("--event-code", type=int, default=ecodes.KEY_B)
    parser.add_argument("--enroll", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    if args.enroll:
        device = identify_pedal(args.timeout)
        if args.output:
            write_environment(args.output, device)
        print(device)
        return 0
    if args.device is None:
        parser.error("--device is required unless --enroll is used")

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    return run_relay(args.device, args.ssh_host, args.debounce, args.event_code)


if __name__ == "__main__":
    raise SystemExit(main())
