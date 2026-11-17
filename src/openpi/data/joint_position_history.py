"""Shared causal joint-position sampling and FAST encoding for train and inference."""
from collections import deque
import dataclasses
import functools
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class JointPositionHistoryConfig:
    """An independent position-memory window; no velocities or future actions."""

    seconds: int = 5
    sample_hz: int = 10
    model_dim: int = 32
    max_tokens: int = 256
    tokenizer_path: str = "physical-intelligence/fast"
    tokenizer_revision: str = "ec4d7aa71691cac0b8bed6942be45684db2110f4"
    padding: str = "first_position"

    def __post_init__(self):
        """Bound memory and reject a mismatched tokenizer input contract."""
        if (type(self.seconds) is not int or not 1 <= self.seconds <= 10
                or type(self.sample_hz) is not int or not 1 <= self.sample_hz <= 50
                or self.num_samples > 200 or self.model_dim != 32
                or type(self.max_tokens) is not int or not 32 <= self.max_tokens <= 2048
                or self.padding != "first_position"):
            raise ValueError("Invalid joint position history settings")

    @property
    def num_samples(self):
        """Count points including the current point: 5 seconds x 10 Hz = 50."""
        return self.seconds * self.sample_hz

    @property
    def offsets_seconds(self):
        """Oldest to newest; 50 points at 10 Hz span -4.9 to 0 seconds."""
        return (np.arange(self.num_samples, dtype=np.float64) - self.num_samples + 1) / self.sample_hz


def sample_positions(timestamps, positions, now, settings):
    """Select nearest past observations, repeating the first position before episode start."""
    stamps = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(positions, dtype=np.float32)
    if (stamps.ndim != 1 or not len(stamps) or values.shape != (len(stamps), 14)
            or not np.isfinite(stamps).all() or not np.isfinite(values).all()
            or np.any(np.diff(stamps) < 0) or stamps[-1] > now + 1e-6):
        raise ValueError("Joint history requires finite, chronological, same-episode past positions")
    targets = now + settings.offsets_seconds
    right = np.clip(np.searchsorted(stamps, targets), 0, len(stamps) - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(abs(stamps[left] - targets) <= abs(stamps[right] - targets), left, right)
    return values[nearest].copy()


class PositionHistoryBuffer:
    """Robot-side observation buffer, reset only at an actual episode boundary."""

    def __init__(self, settings):
        self.settings = settings
        self._samples = deque()

    def reset(self):
        """Prevent any position from a preceding episode being reused."""
        self._samples.clear()

    def append(self, stamp, position):
        """Keep sufficient raw control-rate samples for timestamp-based resampling."""
        stamp = float(stamp)
        value = np.asarray(position, dtype=np.float32).copy()
        if value.shape != (14,) or not np.isfinite(value).all() or not np.isfinite(stamp):
            raise ValueError("Invalid observed joint position")
        if self._samples and stamp < self._samples[-1][0]:
            raise ValueError("Joint observation clock moved backwards; reset the episode")
        if self._samples and stamp == self._samples[-1][0]:
            self._samples[-1] = (stamp, value)
        else:
            self._samples.append((stamp, value))
        while len(self._samples) > 1 and self._samples[1][0] < stamp - self.settings.seconds:
            self._samples.popleft()

    def sample(self):
        """Use the exact same timestamp selection and initial padding as training."""
        if not self._samples:
            raise ValueError("No joint observations in this episode")
        stamps, values = zip(*self._samples, strict=True)
        return sample_positions(stamps, values, stamps[-1], self.settings)


@functools.lru_cache(maxsize=4)
def fast_processor(path, revision):
    """Load a pinned official processor once per worker, outside the sampling loop."""
    from transformers import AutoProcessor
    bundled = Path(__file__).resolve().parents[3] / "assets/fast-tokenizer"
    if path == "physical-intelligence/fast" and bundled.is_dir():
        if (bundled / "REVISION").read_text().strip() != revision:
            raise ValueError("Bundled FAST tokenizer revision does not match checkpoint")
        return AutoProcessor.from_pretrained(str(bundled), trust_remote_code=True, local_files_only=True)
    return AutoProcessor.from_pretrained(path, revision=revision, trust_remote_code=True)


def encode_positions(normalized_positions, settings):
    """FAST encodes a normalized, zero-padded Tx32 position trajectory."""
    values = np.asarray(normalized_positions, dtype=np.float32)
    if values.shape != (settings.num_samples, settings.model_dim) or not np.isfinite(values).all():
        raise ValueError("Invalid normalized joint history shape")
    processor = fast_processor(settings.tokenizer_path, settings.tokenizer_revision)
    tokens = np.asarray(processor(values)[0], dtype=np.int32)
    if len(tokens) > settings.max_tokens:
        raise ValueError(f"Joint history needs {len(tokens)} FAST tokens, exceeds {settings.max_tokens}; increase token budget")
    return tokens
