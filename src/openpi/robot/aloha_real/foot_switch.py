"""Read the PCsensor pedal directly, without desktop/SSH keyboard focus."""

import fcntl
import logging
import os
import struct
import time


class FootSwitch:
    _event = struct.Struct("llHHi")  # Linux input_event

    def __init__(self, path):
        self.fd = None
        self._down = False
        if not path:
            return
        try:
            self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            # EVIOCGRAB: avoid also delivering the same b through the terminal.
            fcntl.ioctl(self.fd, 0x40044590, 1)
            logging.info("脚踏板已连接：%s；踩下等同 b", path)
        except OSError as exc:
            self.close()
            logging.warning("脚踏板不可用，仍可使用键盘 b：%s", exc)

    def poll(self):
        pressed = False
        while self.fd is not None:
            try:
                data = os.read(self.fd, self._event.size * 64)
                if not data:
                    self.close()
                    break
            except BlockingIOError:
                break
            except OSError as exc:
                logging.warning("脚踏板已断开：%s", exc)
                self.close()
                break
            for sec, usec, kind, code, value in self._event.iter_unpack(data):
                if kind == 1 and code == 48:  # EV_KEY, KEY_B
                    if value == 1 and not self._down:
                        # Discard presses queued during initialization or saving.
                        pressed |= 0 <= time.time() - (sec + usec / 1e6) < 0.5
                    if value in (0, 1):
                        self._down = bool(value)
        return "b" if pressed else None

    def close(self):
        if self.fd is not None:
            os.close(self.fd)  # Also releases EVIOCGRAB.
            self.fd = None
