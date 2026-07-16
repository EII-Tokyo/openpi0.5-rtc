from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


AUDIT_DIR = Path("reports/aloha_isaac_replay/right_shoulder_audit")


pytestmark = pytest.mark.requires_runtime_artifacts


def _require_artifact(name: str) -> Path:
    path = AUDIT_DIR / name
    if not path.exists():
        pytest.skip(f"BLOCKED: Isaac runtime audit artifact is missing: {path}")
    return path


def test_right_shoulder_runtime_manifest_identifies_indices_consistently() -> None:
    manifest = json.loads(_require_artifact("runtime_dof_manifest.json").read_text())
    right_shoulder = [
        row for row in manifest["dofs"] if row["side"] == "right" and row["canonical_name"] == "right_shoulder"
    ][0]
    assert right_shoulder["runtime_dof_name"] == "shoulder"
    assert right_shoulder["runtime_index"] == right_shoulder["target_index"] == right_shoulder["readback_index"]
    assert right_shoulder["metric_index"] == 7
    assert right_shoulder["is_continuous_for_metrics"] is False


def test_right_shoulder_audit_summary_blocks_parameter_fitting_until_integrity_passes() -> None:
    summary = json.loads(_require_artifact("summary.json").read_text())
    assert summary["ready_for_reward"] is False
    assert summary["ready_for_rl"] is False
    assert summary["ready_for_controller_parameter_fitting"] == (
        summary["gates"]["runtime_dof_identity"] == "PASS"
        and summary["gates"]["target_readback_index_consistency"] == "PASS"
        and summary["gates"]["full_16dof_target_construction"] == "PASS"
        and summary["gates"]["right_shoulder_runtime_limit"] == "PASS"
        and summary["gates"]["gravity_off_hold"] == "PASS"
        and summary["gates"]["gravity_on_hold"] == "PASS"
        and summary["gates"]["right_shoulder_step_response"] == "PASS"
        and summary["gates"]["left_right_shoulder_symmetry"] == "PASS"
        and summary["gates"]["readback_physical_consistency"] == "PASS"
    )


def test_right_shoulder_step_response_files_are_machine_readable() -> None:
    for name in ("right_shoulder_step_response.csv", "left_shoulder_step_response.csv"):
        with _require_artifact(name).open() as f:
            rows = list(csv.DictReader(f))
        assert rows, name
        assert {"phase", "step", "target", "qpos", "qvel", "position_error", "limit_violation"}.issubset(rows[0])
