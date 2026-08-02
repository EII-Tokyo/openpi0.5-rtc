from __future__ import annotations

from tools.build_aloha1_five_pose_initialization_finger_safety_closure import build_closure


def test_closure_keeps_runtime_pass_separate_from_unpromoted_candidate() -> None:
    report = build_closure(
        historical={"status": "PASS", "user_confirmation": "PASS"},
        runtime={
            "machine_status": "PASS",
            "fresh_process_count": 10,
            "task8": "NOT_RUN",
            "samples": [
                {
                    "sample_id": f"sample_{index:02d}",
                        "primary": {
                            "machine_status": "PASS",
                            "deterministic_signature": f"sig-{index}",
                            "initialization_signature": f"init-{index}",
                            "initialization_contract_status": "PASS",
                            "finger_safety_status": "PASS",
                        "finger_safety_violation_count": 0,
                    },
                        "collider_repeat": {
                            "machine_status": "PASS",
                            "deterministic_signature": f"sig-{index}",
                            "initialization_signature": f"init-{index}",
                            "initialization_contract_status": "PASS",
                        "finger_safety_status": "PASS",
                        "finger_safety_violation_count": 0,
                    },
                }
                for index in range(1, 6)
            ],
        },
        screenshot_review={
            "status": "PASS",
            "capture_record_count": 120,
            "image_record_count": 240,
            "task8": "NOT_RUN",
        },
        semantics={
            "status": "PASS",
            "limit_semantics_status": "VERIFIED_USD_LIMIT_DEFECT",
            "pair_collision_support_status": "INCONCLUSIVE",
            "candidate_created": True,
            "candidate": {
                "status": "CREATED_NOT_PROMOTED",
                "verification_status": "PASS",
                "pair_collision_authored": False,
            },
            "task8": "NOT_RUN",
        },
        negative_controls={
            "status": "PASS",
            "control_count": 4,
            "task8": "NOT_RUN",
        },
        prior_task7={"status": "PARTIAL", "asset_promotion": "FAIL"},
        physics_root_cause={
            "status": "PARTIAL",
            "remaining_real_blockers": ["PHYSICS_CANDIDATE_NOT_PROMOTED"],
            "task8": "NOT_RUN",
        },
    )

    assert report["runtime_grasp_outcome"] == "PASS"
    assert report["attempt10_finger_safety"] == "PASS"
    assert report["task7"] == "PARTIAL"
    assert report["task8"] == "NOT_RUN"
    assert report["final_default_promotion"] == "NOT_PROMOTED"
    assert report["physical_pair_collision_candidate"] == "NOT_AUTHORED_INCONCLUSIVE"
    assert "FINGER_SOURCE_LIMIT_SESSION_LAYER_NOT_PROMOTED" in report[
        "remaining_real_blockers"
    ]
    assert "PHYSICS_CANDIDATE_NOT_PROMOTED" in report["remaining_real_blockers"]


def test_closure_fails_attempt10_on_any_finger_safety_violation() -> None:
    runtime = {
        "machine_status": "PASS",
        "fresh_process_count": 10,
        "task8": "NOT_RUN",
        "samples": [],
    }
    for index in range(1, 6):
        signature = f"sig-{index}"
        runtime["samples"].append(
            {
                "sample_id": f"sample_{index:02d}",
                "primary": {
                    "machine_status": "PASS",
                    "deterministic_signature": signature,
                    "initialization_signature": signature,
                    "initialization_contract_status": "PASS",
                    "finger_safety_status": "FAIL" if index == 4 else "PASS",
                    "finger_safety_violation_count": 1 if index == 4 else 0,
                },
                "collider_repeat": {
                    "machine_status": "PASS",
                    "deterministic_signature": signature,
                    "initialization_signature": signature,
                    "initialization_contract_status": "PASS",
                    "finger_safety_status": "PASS",
                    "finger_safety_violation_count": 0,
                },
            }
        )

    report = build_closure(
        historical={"status": "PASS", "user_confirmation": "PASS"},
        runtime=runtime,
        screenshot_review={
            "status": "PASS",
            "capture_record_count": 120,
            "image_record_count": 240,
            "task8": "NOT_RUN",
        },
        semantics={
            "status": "PASS",
            "limit_semantics_status": "VERIFIED_USD_LIMIT_DEFECT",
            "pair_collision_support_status": "INCONCLUSIVE",
            "candidate_created": True,
            "candidate": {
                "status": "CREATED_NOT_PROMOTED",
                "verification_status": "PASS",
                "pair_collision_authored": False,
            },
            "task8": "NOT_RUN",
        },
        negative_controls={"status": "PASS", "control_count": 4, "task8": "NOT_RUN"},
        prior_task7={"status": "PARTIAL", "asset_promotion": "FAIL"},
        physics_root_cause={
            "status": "PARTIAL",
            "remaining_real_blockers": [],
            "task8": "NOT_RUN",
        },
    )

    assert report["attempt10_finger_safety"] == "FAIL"
    assert report["task7"] == "FAIL"
