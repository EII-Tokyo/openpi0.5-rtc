from __future__ import annotations


def target_hold_seconds(*, fps: float, steps_per_target: int = 1) -> float:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if steps_per_target <= 0:
        raise ValueError(f"steps_per_target must be positive, got {steps_per_target}")
    return steps_per_target / fps

