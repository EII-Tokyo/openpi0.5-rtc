from __future__ import annotations

from tools.finalize_aloha1_cad_derived_five_pose_review import aggregate_visual_reviews


def _results() -> dict[str, object]:
    samples = []
    for index in range(5):
        signature = f"signature-{index}"
        samples.append(
            {
                "sample_id": f"sample_{index + 1:02d}",
                "primary": {
                    "machine_status": "PASS",
                    "evidence_status": "PASS",
                    "deterministic_signature": signature,
                    "video_count": 2,
                },
                "collider_repeat": {
                    "machine_status": "PASS",
                    "evidence_status": "PASS",
                    "deterministic_signature": signature,
                    "collision_record_count": 24,
                },
            }
        )
    return {
        "machine_status": "PASS",
        "machine_pass_count": 5,
        "evidence_pass_count": 5,
        "samples": samples,
        "task8": "NOT_RUN",
    }


def _reviews() -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"sample_{index + 1:02d}",
            "status": "PARTIAL",
            "machine_status": "PASS",
            "visual_model_review": "PASS",
            "user_confirmation": "NOT_RUN",
            "deterministic_signature": f"signature-{index}",
            "primary_action_video": {
                "raw": {"sha256": f"raw-{index}"},
                "annotated": {"sha256": f"annotated-{index}"},
            },
            "collision_screenshot_evidence": {
                "status": "PASS",
                "records": [{"visual_model_review": "PASS"}] * 24,
            },
        }
        for index in range(5)
    ]


def test_five_pose_visual_review_stays_partial_before_user_confirmation() -> None:
    report = aggregate_visual_reviews(_results(), _reviews())

    assert report["machine_status"] == "PASS"
    assert report["visual_model_review"] == "PASS"
    assert report["status"] == "PARTIAL"
    assert report["user_confirmation"] == "NOT_RUN"
    assert report["promotion_status"] == "AWAITING_USER_VIDEO_CONFIRMATION"
    assert report["task8"] == "NOT_RUN"
    assert len(report["samples"]) == 5


def test_visual_review_rejects_a_signature_mismatch() -> None:
    reviews = _reviews()
    reviews[2]["deterministic_signature"] = "wrong"

    report = aggregate_visual_reviews(_results(), reviews)

    assert report["status"] == "FAIL"
    assert report["visual_model_review"] == "FAIL"
    assert report["samples"][2]["status"] == "FAIL"
