from __future__ import annotations

import csv

from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _write_csv


def test_passive_contact_csv_writer_preserves_late_diagnostic_columns(tmp_path) -> None:
    path = tmp_path / "contact.csv"
    _write_csv(
        path,
        [
            {"phase": "settle", "step": 0, "object_center_x": 0.0},
            {"phase": "close", "step": 0, "tracking_controlled_max_abs_error": 0.12},
        ],
    )

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert "tracking_controlled_max_abs_error" in rows[0]
    assert rows[1]["tracking_controlled_max_abs_error"] == "0.12"
