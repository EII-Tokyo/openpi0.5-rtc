from tools.aloha1_mapping.signal_correspondence_screenshots import merge_capture_documents


def _document(robot: str) -> dict:
    captures = []
    for index, phase in enumerate(
        (
            "home_reference",
            "small_up_start",
            "small_up_max",
            "small_down_return",
            "waist_positive",
            "waist_negative",
        )
    ):
        captures.append(
            {
                "capture_id": f"{robot}_{phase}",
                "robot": robot,
                "phase": phase,
                "camera": {
                    "position_world_m": [1.0, 2.0, 3.0],
                    "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "resolution": [1280, 900],
                    "projections": {
                        "robot_visual": {},
                        "driven_joint_visual": {},
                        "end_effector": {},
                        "home_end_effector": {},
                    },
                },
                "raw_sha256": f"{index:064x}",
            }
        )
    return {
        "status": "PASS",
        "capture_count": 6,
        "expected_capture_count": 6,
        "selected_robots": [robot],
        "captures": captures,
        "stage": {
            "absolute_path": "/tmp/stage.usda",
            "sha256_before": "a" * 64,
            "sha256_after": "a" * 64,
            "immutable": True,
        },
    }


def test_merge_capture_documents_requires_two_complete_robot_sets() -> None:
    merged = merge_capture_documents(
        _document("follower_left"),
        _document("follower_right"),
    )

    assert merged["status"] == "PASS"
    assert merged["capture_count"] == 12
    assert merged["expected_capture_count"] == 12
    assert merged["robots"] == ["follower_left", "follower_right"]
    assert merged["fixed_camera_within_robot"] == {
        "follower_left": True,
        "follower_right": True,
    }
