from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


CONTINUOUS_JOINT_SUFFIXES = ("forearm_roll", "wrist_rotate")


def is_continuous_joint(canonical_name: str) -> bool:
    return canonical_name.endswith(CONTINUOUS_JOINT_SUFFIXES)


def nearest_equivalent_angle(target: float, reference: float) -> float:
    return float(reference + math.atan2(math.sin(target - reference), math.cos(target - reference)))


def nearest_equivalent_targets(
    raw_targets: np.ndarray,
    reference_qpos: np.ndarray,
    joint_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    raw = np.asarray(raw_targets, dtype=np.float64)
    reference = np.asarray(reference_qpos, dtype=np.float64)
    if raw.shape != reference.shape:
        raise ValueError(f"raw/reference shape mismatch: {raw.shape} vs {reference.shape}")
    if raw.shape != (len(joint_names),):
        raise ValueError(f"joint_names length {len(joint_names)} does not match target shape {raw.shape}")
    adjusted = raw.copy()
    events: list[str] = []
    for idx, name in enumerate(joint_names):
        if not is_continuous_joint(name):
            continue
        equivalent = nearest_equivalent_angle(float(raw[idx]), float(reference[idx]))
        if abs(equivalent - raw[idx]) > 1e-9:
            adjusted[idx] = equivalent
            events.append(name)
    return adjusted, events


def nearest_equivalent_sequence(
    raw_targets: np.ndarray,
    reference_qpos: np.ndarray,
    joint_names: Sequence[str],
) -> tuple[np.ndarray, dict[str, int]]:
    raw = np.asarray(raw_targets, dtype=np.float64)
    reference = np.asarray(reference_qpos, dtype=np.float64)
    if raw.shape != reference.shape:
        raise ValueError(f"raw/reference sequence shape mismatch: {raw.shape} vs {reference.shape}")
    adjusted = np.empty_like(raw)
    counts = {name: 0 for name in joint_names}
    for row_idx in range(raw.shape[0]):
        adjusted_row, events = nearest_equivalent_targets(raw[row_idx], reference[row_idx], joint_names)
        adjusted[row_idx] = adjusted_row
        for name in events:
            counts[name] += 1
    return adjusted, {name: count for name, count in counts.items() if count}

