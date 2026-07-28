from pathlib import Path

from tools.aloha1_mapping.workcell_config import build_workcell_plan

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_workcell_plan_keeps_calibration_unknowns_machine_readable() -> None:
    plan = build_workcell_plan(PROJECT_ROOT, enable_leaders=False)

    assert plan["status"] == "PARTIAL"
    assert [robot["name"] for robot in plan["robots"]] == [
        "follower_left",
        "follower_right",
    ]
    assert all(
        robot["transform_status"] == "TEMPORARY_SEPARATION_ONLY"
        for robot in plan["robots"]
    )
    assert "follower_mount_relation" in plan["hard_blockers"]
    assert all(
        item["collision_enabled"] is False
        for item in plan["workcell_objects"]
    )


def test_camera_and_observation_contract_preserve_source_order() -> None:
    plan = build_workcell_plan(PROJECT_ROOT, enable_leaders=False)

    assert [camera["name"] for camera in plan["cameras"]] == [
        "cam_high",
        "cam_low",
        "cam_left_wrist",
        "cam_right_wrist",
    ]
    assert all(
        camera["calibration_status"] == "CALIBRATION_PENDING"
        for camera in plan["cameras"]
    )
    assert plan["observation"]["image_shape_hwc"] == [480, 640, 3]
    assert plan["observation"]["camera_order"] == [
        "cam_high",
        "cam_low",
        "cam_left_wrist",
        "cam_right_wrist",
    ]
