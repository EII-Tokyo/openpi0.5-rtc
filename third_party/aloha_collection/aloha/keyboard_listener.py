"""Program-lifetime terminal key listener with deterministic TTY cleanup."""

from __future__ import annotations

import select
import sys
import termios
import tty
from typing import Callable, TextIO


def run_keyboard_listener(
    program_exit,
    on_key: Callable[[str], None],
    *,
    stdin: TextIO | None = None,
    logger: Callable[[str], None] = print,
    poll_timeout: float = 0.1,
) -> None:
    """Read keys until the recorder process exits, then restore terminal state."""
    input_stream = sys.stdin if stdin is None else stdin
    if not input_stream.isatty():
        logger("[keyboard] stdin is not a TTY; keyboard hotkeys disabled for this run.")
        return

    fd = input_stream.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not program_exit.is_set():
            ready, _, _ = select.select([input_stream], [], [], poll_timeout)
            if not ready:
                continue
            key = input_stream.read(1)
            if key:
                on_key(key.lower())
    except Exception as exc:
        logger(f"[keyboard] 监听异常：{exc}")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
