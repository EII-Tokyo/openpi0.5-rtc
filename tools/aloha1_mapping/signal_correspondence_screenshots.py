"""Machine checks for ALOHA1 signal-correspondence screenshots."""

from __future__ import annotations

from typing import Any

PHASES = (
    "home_reference",
    "small_up_start",
    "small_up_max",
    "small_down_return",
    "waist_positive",
    "waist_negative",
)
ROBOTS = ("follower_left", "follower_right")
REQUIRED_PROJECTIONS = {
    "robot_visual",
    "driven_joint_visual",
    "end_effector",
    "home_end_effector",
}


def _camera_signature(record: dict[str, Any]) -> tuple[Any, ...]:
    camera = record["camera"]
    return (
        tuple(camera["position_world_m"]),
        tuple(camera["orientation_wxyz"]),
        tuple(camera["resolution"]),
    )


def merge_capture_documents(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    """Merge independent fresh Isaac capture processes with strict checks."""
    documents = {
        "follower_left": left,
        "follower_right": right,
    }
    captures = []
    fixed_cameras = {}
    for robot in ROBOTS:
        document = documents[robot]
        if document["status"] != "PASS":
            raise ValueError(f"{robot} capture document is not PASS")
        if document["selected_robots"] != [robot]:
            raise ValueError(f"{robot} process scope mismatch")
        robot_captures = document["captures"]
        if len(robot_captures) != 6:
            raise ValueError(f"{robot} must have six captures")
        if [item["phase"] for item in robot_captures] != list(PHASES):
            raise ValueError(f"{robot} phase order mismatch")
        if any(item["robot"] != robot for item in robot_captures):
            raise ValueError(f"{robot} record identity mismatch")
        for item in robot_captures:
            if not REQUIRED_PROJECTIONS.issubset(item["camera"]["projections"]):
                raise ValueError(f"{item['capture_id']} lacks required projections")
        fixed_cameras[robot] = len({_camera_signature(item) for item in robot_captures}) == 1
        captures.extend(robot_captures)

    stages = [documents[robot]["stage"] for robot in ROBOTS]
    if any(not stage["immutable"] for stage in stages):
        raise ValueError("source Stage changed during capture")
    if (
        len(
            {
                (
                    stage["absolute_path"],
                    stage["sha256_before"],
                    stage["sha256_after"],
                )
                for stage in stages
            }
        )
        != 1
    ):
        raise ValueError("left/right captures do not share a frozen Stage")
    status = "PASS" if len(captures) == 12 and all(fixed_cameras.values()) else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "capture_count": len(captures),
        "expected_capture_count": 12,
        "robots": list(ROBOTS),
        "phase_order": list(PHASES),
        "captures": captures,
        "fixed_camera_within_robot": fixed_cameras,
        "stage": stages[0],
        "source_process_documents": documents,
    }
