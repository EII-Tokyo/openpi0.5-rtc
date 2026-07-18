from __future__ import annotations

from pathlib import Path

import pytest

from aloha_isaac_replay.scripts.audit_table_frame_candidate import audit_table_frame


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml"


def test_phase63_table_candidate_blocks_when_base_transforms_are_not_calibrated() -> None:
    payload = audit_table_frame(CONFIG)

    assert payload["status"] == "BLOCKED_REQUIRES_MEASURED_TABLE_TO_BASE_TRANSFORM"
    assert payload["frame_status"]["T_world_table"] == "diagnostic_candidate"
    assert payload["frame_status"]["T_table_left_base"] == "not_calibrated"
    assert payload["frame_status"]["T_table_right_base"] == "not_calibrated"


def test_phase63_table_candidate_reports_table_top_geometry() -> None:
    payload = audit_table_frame(CONFIG)
    geometry = payload["table_geometry"]

    assert geometry["top_center_world"] == pytest.approx([0.593227851197621, 0.7853100288947757, -0.2971450733686908])
    assert geometry["top_corners_world"]["xmin_ymin"] == pytest.approx([
        -0.01677214880237899,
        0.4728100288947757,
        -0.2971450733686908,
    ])
    assert geometry["top_corners_world"]["xmax_ymax"] == pytest.approx([
        1.203227851197621,
        1.0978100288947758,
        -0.2971450733686908,
    ])
