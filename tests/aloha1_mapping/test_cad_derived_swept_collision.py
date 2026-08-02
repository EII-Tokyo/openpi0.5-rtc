from __future__ import annotations

from tools.aloha1_mapping.cad_derived_swept_collision import deterministic_sweep_signature
from tools.aloha1_mapping.cad_derived_swept_collision import summarize_swept_samples


def test_swept_summary_requires_every_waypoint_to_be_clear() -> None:
    clear = {
        "sample_id": "sample_01",
        "waypoint_count": 3,
        "unexpected_overlap_waypoint_count": 0,
        "finite": True,
    }
    assert summarize_swept_samples([clear])["status"] == "PASS"
    blocked = {
        **clear,
        "sample_id": "sample_02",
        "unexpected_overlap_waypoint_count": 1,
    }
    summary = summarize_swept_samples([clear, blocked])
    assert summary["status"] == "FAIL"
    assert summary["failed_sample_ids"] == ["sample_02"]


def test_sweep_signature_is_order_stable_and_sensitive() -> None:
    records = [
        {
            "sample_id": "sample_02",
            "status": "PASS",
            "waypoint_count": 4,
            "unexpected_pairs": [],
        },
        {
            "sample_id": "sample_01",
            "status": "PASS",
            "waypoint_count": 3,
            "unexpected_pairs": [],
        },
    ]
    first = deterministic_sweep_signature(records)
    assert first == deterministic_sweep_signature(list(reversed(records)))
    records[0]["waypoint_count"] = 5
    assert deterministic_sweep_signature(records) != first
