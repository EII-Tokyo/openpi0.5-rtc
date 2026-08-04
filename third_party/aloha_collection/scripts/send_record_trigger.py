#!/usr/bin/env python3
"""Send current remote pedal commands to the recorder's private socket."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aloha.remote_trigger import (
    DEFAULT_SOCKET_PATH,
    ProtocolError,
    TriggerUnavailable,
    send_command,
)


def send_one(socket_path: Path, command: str) -> bool:
    try:
        send_command(socket_path, command)
    except (ProtocolError, TriggerUnavailable) as exc:
        print(f"remote-trigger: dropped: {exc}", file=sys.stderr, flush=True)
        return False
    print("remote-trigger: sent b", file=sys.stderr, flush=True)
    return True


def stream(socket_path: Path) -> int:
    for raw_line in sys.stdin.buffer:
        if len(raw_line) > 16:
            print("remote-trigger: dropped oversized command", file=sys.stderr, flush=True)
            continue
        try:
            command = raw_line.decode("ascii", errors="strict").strip().lower()
        except UnicodeDecodeError:
            print("remote-trigger: dropped non-ASCII command", file=sys.stderr, flush=True)
            continue
        if not command:
            continue
        send_one(socket_path, command)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="b", choices=("b",))
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET_PATH)
    args = parser.parse_args()

    if args.stream:
        return stream(args.socket)
    return 0 if send_one(args.socket, args.command) else 1


if __name__ == "__main__":
    raise SystemExit(main())
