from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def plot_progress_curve(
    timestamps: Sequence[int],
    values: Sequence[float],
    title: str = "TOPReward Progress Curve",
    output_path: str | Path | None = None,
) -> None:
    """Plot a progress curve and optionally save it."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(timestamps, values, marker="o", linewidth=2)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Normalized Progress")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)
