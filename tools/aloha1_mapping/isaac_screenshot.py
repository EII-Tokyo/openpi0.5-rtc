"""Isaac Sim 5.1 RGB capture helpers with deterministic PNG output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def look_at_orientation_wxyz(
    position: np.ndarray,
    target: np.ndarray,
    up_world: np.ndarray | None = None,
) -> np.ndarray:
    """Return the Isaac camera quaternion whose -Z axis looks at target."""
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from isaacsim.core.utils.rotations import lookat_to_quatf
    from pxr import Gf

    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if up_world is None:
        up_world = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        up_world = np.asarray(up_world, dtype=np.float64)
    up_norm = float(np.linalg.norm(up_world))
    if up_norm <= 1.0e-12:
        raise ValueError("look-at up vector must be nonzero")
    up_world = up_world / up_norm
    # The local helper aligns +Z from its first argument to its second. Isaac
    # cameras look along -Z, so target and camera position are intentionally
    # passed in this order, matching local camera_utils.py in Isaac Sim 5.1.
    quaternion = lookat_to_quatf(
        Gf.Vec3f(*target.tolist()),
        Gf.Vec3f(*position.tolist()),
        Gf.Vec3f(*up_world.tolist()),
    )
    return gf_quat_to_np_array(quaternion)


def save_camera_rgba_png(
    camera: Any,
    path: Path,
    *,
    rgba: Any | None = None,
) -> dict[str, Any]:
    """Read Camera.get_rgba() and atomically save an 8-bit RGBA PNG."""
    if rgba is None:
        rgba = camera.get_rgba()
    if rgba is None:
        raise RuntimeError("Isaac camera RGB annotator returned no data")
    pixels = np.asarray(rgba)
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise RuntimeError(f"unexpected Isaac RGBA shape: {pixels.shape}")
    if pixels.dtype != np.uint8:
        if np.issubdtype(pixels.dtype, np.floating):
            maximum = float(np.nanmax(pixels))
            scale = 255.0 if maximum <= 1.0 else 1.0
            pixels = np.clip(pixels * scale, 0.0, 255.0).astype(np.uint8)
        else:
            pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    Image.fromarray(pixels, mode="RGBA").save(temporary)
    temporary.replace(path)
    return {
        "shape": list(pixels.shape),
        "dtype": str(pixels.dtype),
        "minimum": int(pixels.min()),
        "maximum": int(pixels.max()),
    }
