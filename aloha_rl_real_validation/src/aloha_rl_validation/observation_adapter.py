from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


RLINF_REQUIRED_CAMERAS: tuple[str, ...] = ("cam_high", "cam_left_wrist", "cam_right_wrist")
LOCAL_OPENPI_CAMERAS: tuple[str, ...] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")


@dataclass(frozen=True)
class AdaptedObservation:
    images: dict[str, np.ndarray]
    state: np.ndarray
    prompt: str


def adapt_real_observation_for_rlinf(
    images: Mapping[str, np.ndarray],
    state: np.ndarray,
    prompt: str,
) -> AdaptedObservation:
    missing = [name for name in RLINF_REQUIRED_CAMERAS if name not in images]
    if missing:
        raise ValueError(f"missing required RLinf cameras: {missing}")
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (14,):
        raise ValueError(f"state shape {state.shape} != (14,)")
    adapted: dict[str, np.ndarray] = {}
    for name in RLINF_REQUIRED_CAMERAS:
        img = np.asarray(images[name])
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"{name} must be HWC RGB-like image, got {img.shape}")
        adapted[name] = img
    return AdaptedObservation(adapted, state, prompt)

