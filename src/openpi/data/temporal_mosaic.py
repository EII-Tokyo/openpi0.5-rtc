"""High-camera history: four chronological 224px tiles per 448px second."""
import dataclasses

import numpy as np
from PIL import Image


@dataclasses.dataclass(frozen=True)
class HighCameraMosaicConfig:
    camera: str = "cam_high"
    history_seconds: int = 5
    sample_hz: int = 4
    tile_size: int = 224

    def __post_init__(self):
        if self.camera != "cam_high" or self.sample_hz != 4:
            raise ValueError("This layout requires cam_high and four samples per second.")
        if self.history_seconds <= 0 or self.tile_size <= 0:
            raise ValueError("History seconds and tile size must be positive.")

    @property
    def offsets_seconds(self):
        # Exclude current; each second is [start,start+1), oldest first.
        return np.arange(self.history_seconds * self.sample_hz) / self.sample_hz - self.history_seconds


def uint8_hwc(image):
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected one RGB image, got {image.shape}.")
    if image.shape[-1] != 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Floating video frames must be finite and in [0,1].")
        image = np.clip(np.rint(image * 255), 0, 255).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 or normalized float image, got {image.dtype}.")
    return image


def resize_pad(image, size):
    image = Image.fromarray(uint8_hwc(image))
    ratio = max(image.width / size, image.height / size)
    width, height = max(1, int(image.width / ratio)), max(1, int(image.height / ratio))
    image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    result = Image.new("RGB", (size, size), 0)
    result.paste(image, ((size - width) // 2, (size - height) // 2))
    return np.asarray(result)


def pack_high_history(history, current, settings):
    if len(history) != settings.history_seconds * 4:
        raise ValueError("Expected exactly four high-camera history images per second.")
    mosaics = []
    for start in range(0, len(history), 4):
        tiles = [resize_pad(frame, settings.tile_size) for frame in history[start:start + 4]]
        # Row-major time order: oldest TL, then TR, BL, newest BR.
        mosaics.append(np.concatenate((np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)), axis=0))
    mosaics.append(resize_pad(current, 2 * settings.tile_size))
    return np.stack(mosaics)
