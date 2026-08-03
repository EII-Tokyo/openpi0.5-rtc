"""Pure camera-frame evidence helpers for the real ALOHA cam_high stream."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from typing import Any


def frame_record(
    message: Mapping[str, object],
    *,
    receive_monotonic_ns: int,
    receive_wall_time_ns: int,
) -> dict[str, Any]:
    """Preserve source and receive timing without depending on ROS imports."""

    pixels = message.get("pixels")
    if not isinstance(pixels, bytes):
        raise TypeError("camera pixels must be bytes")
    width = int(message["width"])
    height = int(message["height"])
    if width <= 0 or height <= 0:
        raise ValueError("camera dimensions must be positive")
    for name, value in (
        ("receive_monotonic_ns", receive_monotonic_ns),
        ("receive_wall_time_ns", receive_wall_time_ns),
    ):
        if not math.isfinite(float(value)) or int(value) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    return {
        "source_stamp_ns": (
            int(message["source_stamp_ns"])
            if message.get("source_stamp_ns") is not None
            else None
        ),
        "receive_monotonic_ns": int(receive_monotonic_ns),
        "receive_wall_time_ns": int(receive_wall_time_ns),
        "sequence": (
            int(message["sequence"]) if message.get("sequence") is not None else None
        ),
        "width": width,
        "height": height,
        "encoding": str(message["encoding"]),
        "pixel_bytes": len(pixels),
        "pixel_sha256": hashlib.sha256(pixels).hexdigest(),
    }
