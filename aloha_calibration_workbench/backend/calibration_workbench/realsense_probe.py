"""Compatibility import for the read-only librealsense CLI probe.

The capture agent intentionally does not use ``pyrealsense2.pipeline``. New code
should import :class:`RsEnumerateCliProbe` from ``rs_cli_probe`` directly.
"""

from .rs_cli_probe import RsEnumerateCliProbe

__all__ = ["RsEnumerateCliProbe"]
