from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from topreward.model import TOPRewardModel
from topreward.reward import compute_reward


@dataclass
class AdvantageConfig:
    tau: float = 2.0
    delta_max: float = 2.0
    good_threshold: float = 0.6
    bad_threshold: float = -0.2
    good_weight: float = 2.0
    neutral_weight: float = 1.0
    bad_weight: float = 0.1


def _minmax_normalize(values: Sequence[float], eps: float = 1e-8) -> list[float]:
    if not values:
        return []

    arr = np.asarray(values, dtype=np.float64)
    v_min = float(np.min(arr))
    v_max = float(np.max(arr))
    return [float((v - v_min) / (v_max - v_min + eps)) for v in arr]


def compute_advantages(normalized_progress: Sequence[float], config: AdvantageConfig) -> dict[str, list]:
    """Compute Eq.3 deltas and 3-tier weights from normalized progress."""
    if len(normalized_progress) < 2:
        return {
            "increments": [],
            "deltas": [],
            "tiers": [],
            "weights": [],
        }

    increments: list[float] = []
    deltas: list[float] = []
    tiers: list[str] = []
    weights: list[float] = []

    for idx in range(1, len(normalized_progress)):
        inc = float(normalized_progress[idx] - normalized_progress[idx - 1])
        delta = float(np.clip(config.tau * np.exp(inc), a_min=0.0, a_max=config.delta_max))

        if inc > config.good_threshold:
            tier = "good"
            weight = config.good_weight
        elif inc < config.bad_threshold:
            tier = "bad"
            weight = config.bad_weight
        else:
            tier = "neutral"
            weight = config.neutral_weight

        increments.append(inc)
        deltas.append(delta)
        tiers.append(tier)
        weights.append(float(weight))

    return {
        "increments": increments,
        "deltas": deltas,
        "tiers": tiers,
        "weights": weights,
    }


def compute_episode_advantages(
    model: TOPRewardModel,
    frames: Sequence[Image.Image],
    instruction: str,
    action_timestamps: Sequence[int],
    config: AdvantageConfig,
) -> dict[str, list]:
    """End-to-end advantages at provided action timestamps."""
    if not frames:
        raise ValueError("frames must not be empty")
    if not action_timestamps:
        raise ValueError("action_timestamps must not be empty")

    total_frames = len(frames)
    clamped_timestamps = [int(min(max(t, 1), total_frames)) for t in action_timestamps]

    rewards = [compute_reward(model, frames, instruction, prefix_end=t) for t in clamped_timestamps]
    normalized = _minmax_normalize(rewards)
    adv = compute_advantages(normalized, config)

    return {
        "timestamps": clamped_timestamps,
        "raw_rewards": rewards,
        "normalized_progress": normalized,
        **adv,
    }
