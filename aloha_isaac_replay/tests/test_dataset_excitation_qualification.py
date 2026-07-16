from __future__ import annotations

import csv
import json
from pathlib import Path


REPORT_DIR = Path("reports/aloha_isaac_replay/controller_system_id")


def test_dataset_excitation_distribution_reports_insufficient_right_arm_data() -> None:
    summary = json.loads((REPORT_DIR / "controller_validation_summary.json").read_text())
    assert summary["full_dataset_scanned"] == 330
    assert summary["no_actor_likely"] == 52
    assert summary["new_right_arm_candidates_distribution"] == 3
    assert summary["new_right_arm_candidates_strict"] == 2
    assert summary["all_local_hdf5_scanned"] == 1659
    assert summary["all_local_hdf5_right_arm_candidates"] == 86
    assert summary["all_local_hdf5_right_arm_strict"] == 30
    assert summary["lerobot_human_episodes"] == 643
    assert summary["lerobot_human_right_arm_candidates"] == 132
    assert summary["right_arm_id_data_collection_required"] is False
    assert summary["right_arm_dataset_ready_offline"] is True
    # Dataset availability is not enough; Isaac runtime identity/hold artifacts are still blocked.
    assert summary["ready_for_right_arm_controller_id"] is False
    assert summary["ready_for_reward"] is False
    assert summary["ready_for_rl"] is False


def test_right_arm_candidate_csv_contains_real_no_actor_warmup_rows() -> None:
    with (REPORT_DIR / "right_arm_id_candidates.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert all(row["source_bucket"] == "warmup" for row in rows)
    assert all(row["no_actor_likely"] == "True" for row in rows)
    assert {row["usable_right_arm_id"] for row in rows} == {"True"}


def test_runtime_artifacts_are_blocked_not_faked() -> None:
    summary = json.loads((REPORT_DIR / "controller_validation_summary.json").read_text())
    assert summary["runtime_artifact_tests"] == "BLOCKED"
    assert summary["gates"]["runtime_dof_identity"] == "BLOCKED"
    assert summary["gates"]["gravity_off_hold"] == "BLOCKED"
    assert summary["gates"]["gravity_on_hold"] == "BLOCKED"
    assert summary["gates"]["readback_physical_consistency"] == "BLOCKED"
