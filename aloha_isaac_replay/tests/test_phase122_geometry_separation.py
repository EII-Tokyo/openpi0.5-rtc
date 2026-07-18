from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_phase122_success_failure_geometry_metrics.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_phase122_success_failure_geometry_metrics", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_separation_stats_rank_success_higher_and_lateral_lower() -> None:
    module = _load_module()
    rows = [
        {
            "label": "success",
            "path_length_m": 0.20,
            "net_displacement_m": 0.18,
            "tail_lateral_mean_m": 0.01,
            "tail_lateral_max_m": 0.02,
            "tail_progress_m": 0.04,
        },
        {
            "label": "success",
            "path_length_m": 0.18,
            "net_displacement_m": 0.16,
            "tail_lateral_mean_m": 0.015,
            "tail_lateral_max_m": 0.025,
            "tail_progress_m": 0.03,
        },
        {
            "label": "failure",
            "path_length_m": 0.09,
            "net_displacement_m": 0.08,
            "tail_lateral_mean_m": 0.04,
            "tail_lateral_max_m": 0.06,
            "tail_progress_m": 0.02,
        },
        {
            "label": "failure",
            "path_length_m": 0.10,
            "net_displacement_m": 0.07,
            "tail_lateral_mean_m": 0.035,
            "tail_lateral_max_m": 0.055,
            "tail_progress_m": 0.01,
        },
    ]

    stats = module._separation_stats(rows)

    assert stats["path_length_m"]["success_direction"] == "higher"
    assert stats["path_length_m"]["auc"] == 1.0
    assert stats["tail_lateral_mean_m"]["success_direction"] == "lower"
    assert stats["tail_lateral_mean_m"]["auc"] == 1.0
    assert stats["tail_lateral_mean_m"]["mean_gap_success_minus_failure_m"] < 0
