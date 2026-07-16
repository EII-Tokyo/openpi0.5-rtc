from __future__ import annotations

import numpy as np


def validate_50hz_timestamps(timestamps: np.ndarray, *, tolerance_seconds: float = 0.006) -> dict[str, float | bool]:
    ts = np.asarray(timestamps, dtype=np.float64)
    if ts.ndim != 1 or len(ts) < 2:
        return {"valid": False, "reason_code": "too_short", "mean_dt": float("nan"), "max_abs_dt_error": float("nan")}
    dts = np.diff(ts)
    target = 1.0 / 50.0
    errors = np.abs(dts - target)
    return {
        "valid": bool(np.max(errors) <= tolerance_seconds),
        "mean_dt": float(np.mean(dts)),
        "max_abs_dt_error": float(np.max(errors)),
        "nominal_dt": target,
    }

