from __future__ import annotations

import json
from pathlib import Path


REPORT_DIR = Path("reports/aloha_isaac_replay/action_provenance")


def test_action_source_report_records_runtime_command_source() -> None:
    text = (REPORT_DIR / "action_source.md").read_text()
    assert "runtime-emitted" in text
    assert "action[\"actions\"]" in text
    assert "final command semantics" in text
    assert "qpos is saved separately from `observation`" in text


def test_summary_records_hdf5_action_space_and_source() -> None:
    summary = json.loads((REPORT_DIR / "summary.json").read_text())
    assert "runtime action['actions']" in summary["hdf5_action_source"]
    assert summary["hdf5_action_space"] == "standard ALOHA-like 14D runtime command space"
    assert summary["absolute_or_delta"].startswith("absolute command in HDF5")
    assert summary["joint_order"] == "left 6 arm, left gripper, right 6 arm, right gripper"
