from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import spearmanr


def compute_voc(predicted_values: Sequence[float], timestamps: Sequence[int]) -> float:
    """Value-Order Correlation: Spearman rank corr(timestamps, predicted_values)."""
    if len(predicted_values) != len(timestamps):
        raise ValueError(
            f"predicted_values and timestamps must have same length. Got {len(predicted_values)} and {len(timestamps)}"
        )
    if len(predicted_values) < 2:
        return float("nan")

    correlation, _ = spearmanr(np.asarray(timestamps), np.asarray(predicted_values))
    return float(correlation)
