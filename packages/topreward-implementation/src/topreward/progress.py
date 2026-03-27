from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

from topreward.model import TOPRewardModel
from topreward.reward import compute_reward


def minmax_normalize(values: Sequence[float], eps: float = 1e-8) -> list[float]:
    """Min-max normalize values to [0, 1]."""
    if not values:
        return []

    arr = np.asarray(values, dtype=np.float64)
    v_min = float(np.min(arr))
    v_max = float(np.max(arr))
    return [float((v - v_min) / (v_max - v_min + eps)) for v in arr]


def sample_prefix_lengths(total_frames: int, k: int) -> list[int]:
    """Uniformly sample prefix endpoints from 1..T, including endpoints."""
    if total_frames < 1:
        raise ValueError("total_frames must be >= 1")
    if k < 1:
        raise ValueError("k must be >= 1")

    count = min(total_frames, k)
    sampled = np.linspace(1, total_frames, count, dtype=int)
    unique = np.unique(sampled)

    if unique[0] != 1:
        unique = np.insert(unique, 0, 1)
    if unique[-1] != total_frames:
        unique = np.append(unique, total_frames)

    return [int(v) for v in unique.tolist()]


def sample_prefix_lengths_by_stride(total_frames: int, stride: int) -> list[int]:
    """Sample prefix endpoints at a fixed frame stride, always including T."""
    if total_frames < 1:
        raise ValueError("total_frames must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    sampled = list(range(1, total_frames + 1, stride))
    if not sampled:
        sampled = [1]
    if sampled[-1] != total_frames:
        sampled.append(total_frames)
    return sampled


def estimate_progress(
    model: TOPRewardModel,
    frames: Sequence[Image.Image],
    instruction: str,
    k: int = 16,
    prefix_stride: int | None = None,
    score_mode: str = "true_log_prob",
    true_token: str = "True",
    false_token: str = "False",
    eps: float = 1e-8,
    max_prefix_frames: int | None = None,
) -> dict[str, list[float] | list[int]]:
    """Estimate per-prefix raw and normalized TOPReward progress."""
    if not frames:
        raise ValueError("frames must not be empty")

    if prefix_stride is not None:
        timestamps = sample_prefix_lengths_by_stride(len(frames), prefix_stride)
    else:
        timestamps = sample_prefix_lengths(len(frames), k)
    rewards = [
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
        for t in timestamps
    ]
    normalized = minmax_normalize(rewards, eps=eps)

    return {
        "raw_rewards": rewards,
        "normalized": normalized,
        "timestamps": timestamps,
    }
