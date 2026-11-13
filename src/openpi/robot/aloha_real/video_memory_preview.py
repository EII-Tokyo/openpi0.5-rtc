"""Periodically render the observation's camera histories without changing them."""

import logging
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np


def render_video_memory(images: dict, stride_seconds: float = 1.0) -> np.ndarray:
    """Return a BGR contact sheet: camera rows, oldest-to-newest frame columns."""
    names = [name for name in ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist") if name in images]
    if not names:
        raise ValueError("No camera images in observation")
    histories = [np.asarray(images[name]) for name in names]
    histories = [frames[None] if frames.ndim == 3 else frames for frames in histories]
    if any(frames.ndim != 4 or frames.shape[-1] != 3 for frames in histories):
        raise ValueError(f"Expected [T,H,W,3] RGB histories, got {[x.shape for x in histories]}")
    columns = max(len(frames) for frames in histories)
    tile_w, tile_h, label_h = 320, 240, 30
    sheet = np.full((len(names) * (tile_h + label_h), columns * tile_w, 3), 24, np.uint8)
    for row, (name, frames) in enumerate(zip(names, histories)):
        for col, rgb in enumerate(frames):
            if rgb.dtype != np.uint8:
                raise ValueError(f"Expected raw uint8 RGB observation for {name}, got {rgb.dtype}")
            scale = min(tile_w / rgb.shape[1], tile_h / rgb.shape[0])
            width, height = max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))
            bgr = cv2.cvtColor(cv2.resize(rgb, (width, height)), cv2.COLOR_RGB2BGR)
            y = row * (tile_h + label_h)
            x = col * tile_w
            label = f"{name}  t-{(len(frames) - 1 - col) * stride_seconds:g}s"
            cv2.putText(sheet, label, (x + 6, y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            left, top = x + (tile_w - width) // 2, y + label_h + (tile_h - height) // 2
            sheet[top:top + height, left:left + width] = bgr
    return sheet


class VideoMemoryPreview:
    def __init__(self):
        self.path = Path(os.getenv("ALOHA_VIDEO_MEMORY_PREVIEW_PATH", "/app/artifacts/video_memory_preview.jpg"))
        self._next_capture = 0.0
        self._worker = None

    def submit(self, images: dict, stride_seconds: float = 1.0) -> None:
        now = time.monotonic()
        if now < self._next_capture or (self._worker is not None and self._worker.is_alive()):
            return
        self._next_capture = now + 10.0
        # Camera history arrays are fresh stacks; hold references, never mutate inputs.
        snapshot = dict(images)
        self._worker = threading.Thread(target=self._save, args=(snapshot, stride_seconds), daemon=True)
        self._worker.start()

    def _save(self, images: dict, stride_seconds: float) -> None:
        try:
            sheet = render_video_memory(images, stride_seconds)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_name(f".{self.path.stem}.{os.getpid()}.tmp.jpg")
            if not cv2.imwrite(str(temp), sheet):
                raise OSError(f"Could not save {temp}")
            os.replace(temp, self.path)
            logging.info("Video memory preview: %s (oldest -> newest)", self.path)
        except Exception:
            logging.exception("Could not render video memory preview")
