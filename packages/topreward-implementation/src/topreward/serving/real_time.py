from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from topreward.model import TOPRewardModel
from topreward.reward import compute_reward


@dataclass
class RealTimeConfig:
    score_interval: int = 5
    max_buffer_frames: int = 64
    subsample_factor: int = 3


class RealTimeScorer:
    """Incremental scorer for websocket frame streams."""

    def __init__(self, model: TOPRewardModel, instruction: str, config: RealTimeConfig | None = None):
        self.model = model
        self.instruction = instruction
        self.config = config or RealTimeConfig()
        self.raw_frame_buffer: list[Image.Image] = []
        self.model_frame_buffer: list[Image.Image] = []
        self.scores_history: list[float] = []

    def _to_pil(self, frame: Image.Image | np.ndarray | Any) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")

        if hasattr(frame, "detach") and hasattr(frame, "cpu"):
            frame = frame.detach().cpu().numpy()

        array = np.asarray(frame)
        if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
            array = np.transpose(array, (1, 2, 0))

        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and float(np.max(array)) <= 1.0:
                array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)

        return Image.fromarray(array).convert("RGB")

    def on_frame(self, frame: Image.Image | np.ndarray | Any) -> dict[str, float | int] | None:
        """Consume one frame and periodically emit normalized progress."""
        image = self._to_pil(frame)
        self.raw_frame_buffer.append(image)

        if len(self.raw_frame_buffer) % self.config.subsample_factor == 0:
            self.model_frame_buffer.append(image)

        if len(self.model_frame_buffer) > self.config.max_buffer_frames:
            self.model_frame_buffer = self.model_frame_buffer[-self.config.max_buffer_frames :]

        if len(self.raw_frame_buffer) % self.config.score_interval != 0:
            return None
        if not self.model_frame_buffer:
            return None

        raw_score = self._compute_current_score()
        progress = self._normalize_score(raw_score)
        return {
            "progress": progress,
            "raw_score": raw_score,
            "frame_idx": len(self.raw_frame_buffer),
            "model_frames": len(self.model_frame_buffer),
        }

    def _compute_current_score(self) -> float:
        score = compute_reward(
            self.model,
            self.model_frame_buffer,
            self.instruction,
            prefix_end=len(self.model_frame_buffer),
        )
        self.scores_history.append(score)
        return score

    def _normalize_score(self, score: float) -> float:
        if len(self.scores_history) <= 1:
            return 0.0

        r_min = float(min(self.scores_history))
        r_max = float(max(self.scores_history))
        return float((score - r_min) / (r_max - r_min + 1e-8))

    def reset(self, new_instruction: str | None = None) -> None:
        self.raw_frame_buffer = []
        self.model_frame_buffer = []
        self.scores_history = []
        if new_instruction is not None:
            self.instruction = new_instruction
