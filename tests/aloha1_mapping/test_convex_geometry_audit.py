from __future__ import annotations

import numpy as np

from tools.aloha1_mapping.convex_geometry_audit import convex_pair_relation


def _box(center_x: float) -> np.ndarray:
    return np.asarray(
        [
            [x + center_x, y, z]
            for x in (-0.5, 0.5)
            for y in (-0.5, 0.5)
            for z in (-0.5, 0.5)
        ],
        dtype=np.float64,
    )


def test_convex_pair_detects_overlap_and_volume() -> None:
    result = convex_pair_relation(_box(0.0), _box(0.8))

    assert result["relation"] == "OVERLAP"
    assert abs(result["signed_chebyshev_margin_m"] - 0.1) < 1.0e-10
    assert abs(result["overlap_volume_m3"] - 0.2) < 1.0e-10


def test_convex_pair_detects_separation() -> None:
    result = convex_pair_relation(_box(0.0), _box(1.2))

    assert result["relation"] == "SEPARATED"
    assert abs(result["signed_chebyshev_margin_m"] + 0.1) < 1.0e-10
    assert result["overlap_volume_m3"] == 0.0


def test_convex_pair_detects_touching() -> None:
    result = convex_pair_relation(_box(0.0), _box(1.0))

    assert result["relation"] == "TOUCHING_WITHIN_TOLERANCE"
    assert abs(result["signed_chebyshev_margin_m"]) < 1.0e-10
