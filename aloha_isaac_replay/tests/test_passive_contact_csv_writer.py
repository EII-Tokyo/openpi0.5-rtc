from __future__ import annotations

import csv

from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _summarize_contact_pairs
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


def test_contact_summary_classifies_diagnostic_support_contacts() -> None:
    object_path = "/World/object"
    finger_path = "/World/left_finger"
    support_path = "/World/support"

    summary = _summarize_contact_pairs(
        contact_pair_rows=[
            {
                "phase": "close",
                "step": 1,
                "type_name": "CONTACT_FOUND",
                "collider0": f"{object_path}/body",
                "collider1": support_path,
                "sorted_pair": [f"{object_path}/body", support_path],
            },
            {
                "phase": "close",
                "step": 2,
                "type_name": "CONTACT_FOUND",
                "collider0": support_path,
                "collider1": f"{finger_path}/proxy",
                "sorted_pair": [support_path, f"{finger_path}/proxy"],
            },
            {
                "phase": "close",
                "step": 3,
                "type_name": "CONTACT_FOUND",
                "collider0": f"{object_path}/body",
                "collider1": f"{finger_path}/proxy",
                "sorted_pair": [f"{object_path}/body", f"{finger_path}/proxy"],
            },
        ],
        object_path=object_path,
        expected_finger_paths=[finger_path],
        diagnostic_contact_paths=[support_path],
    )

    support_summary = summary["diagnostic_contact_summaries"][support_path]
    assert summary["target_contact_pair_found"] is True
    assert support_summary["contact_pair_count"] == 2
    assert support_summary["object_contact_pair_count"] == 1
    assert support_summary["expected_finger_contact_pair_count"] == 1
    assert support_summary["other_contact_pair_count"] == 0
