from __future__ import annotations

import csv
import json
from pathlib import Path


REPORT_DIR = Path("reports/aloha_isaac_replay/action_provenance")


def test_hdf5_action_is_not_marked_as_pre_flipped() -> None:
    summary = json.loads((REPORT_DIR / "summary.json").read_text())
    assert summary["joint_flip_before_storage"] is False
    text = (REPORT_DIR / "duplicate_or_missing_transforms.md").read_text()
    assert "should not apply `adapt_to_pi` to HDF5 `action`" in text
    assert "HDF5 stores standard-space action/state" in text


def test_dimension_mapping_marks_shoulder_elbow_training_signs() -> None:
    rows = list(csv.DictReader((REPORT_DIR / "action_dimension_mapping.csv").open()))
    by_name = {row["canonical_name"]: row for row in rows}
    for name in ("left_shoulder", "left_elbow", "right_shoulder", "right_elbow"):
        assert by_name[name]["sign"] == "-1"
    for name in ("left_waist", "right_waist", "left_gripper", "right_gripper"):
        assert by_name[name]["sign"] == "1"
