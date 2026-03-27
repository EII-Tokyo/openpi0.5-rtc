from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

from topreward.model import TOPRewardModel
from topreward.reward import compute_reward


def detect_success(
    model: TOPRewardModel,
    frames: Sequence[Image.Image],
    instruction: str,
    n_final_frames: int = 3,
    threshold: float | None = None,
    score_mode: str = "true_log_prob",
    true_token: str = "True",
    false_token: str = "False",
    max_prefix_frames: int | None = None,
) -> dict[str, float | bool | None | list[float] | list[int]]:
    """Average final-prefix log-probs to produce a success score."""
    if not frames:
        raise ValueError("frames must not be empty")
    if n_final_frames < 1:
        raise ValueError("n_final_frames must be >= 1")

    total = len(frames)
    start = max(1, total - n_final_frames + 1)
    final_prefixes = list(range(start, total + 1))

    scores = [
        compute_reward(
            model,
            frames,
            instruction,
            prefix_end=t,
            score_mode=score_mode,
            true_token=true_token,
            false_token=false_token,
            max_prefix_frames=max_prefix_frames,
        )
        for t in final_prefixes
    ]
    avg_score = float(np.mean(scores))

    return {
        "score": avg_score,
        "is_success": (avg_score > threshold) if threshold is not None else None,
        "per_frame_scores": [float(v) for v in scores],
        "final_prefixes": final_prefixes,
    }
