from __future__ import annotations

import json
from pathlib import Path

from tools.aloha1_mapping.compare_finger_tessellations import build_comparison

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESSELLATION_ROOT = (
    PROJECT_ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "viper_gripper/tessellation_determinism"
)


def test_two_fresh_tessellation_runs_match() -> None:
    report = build_comparison(
        TESSELLATION_ROOT / "run_a/manifest.json",
        TESSELLATION_ROOT / "run_b/manifest.json",
    )
    assert report["determinism_gate"] == "PASS"
    assert report["production_tessellation_gate"] == "HARD_BLOCKER"
    for mesh in report["mesh_comparisons"].values():
        assert mesh["all_fields_match"] is True
        assert mesh["run_a"]["triangle_count"] > 0
        assert mesh["run_a"]["degenerate_triangle_count"] == 0
        assert mesh["run_a"]["connected_components"] == 1


def test_saved_tessellation_report_matches_recomputed() -> None:
    expected = build_comparison(
        TESSELLATION_ROOT / "run_a/manifest.json",
        TESSELLATION_ROOT / "run_b/manifest.json",
    )
    saved = json.loads(
        (
            PROJECT_ROOT
            / "reports/aloha1_mapping/aloha_viper_finger_tessellation.json"
        ).read_text(encoding="utf-8")
    )
    assert saved == expected
