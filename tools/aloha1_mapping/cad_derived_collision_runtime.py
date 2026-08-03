"""Pure contracts for CAD-derived ALOHA collision runtime validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.signal_correspondence import HOME_ARM

ROBOTS = ("follower_left", "follower_right")
OVERLAP_CLASSES = {
    "NONE",
    "CAD_ASSEMBLY_INTERFACE_EXPECTED",
    "ADJACENT_JOINT_INTERFACE_EXPECTED",
    "UNEXPECTED_SELF_COLLISION",
    "UNEXPECTED_ENVIRONMENT_COLLISION",
    "ENVIRONMENT_CONTACT_REQUIRES_RUNTIME_EFFECT_REVIEW",
    "CROSS_FOLLOWER_COLLISION",
}


def canonical_pair(first: str, second: str) -> tuple[str, str]:
    """Return a stable unordered pair."""

    return tuple(sorted((str(first), str(second))))


def _robot_for_path(path: str) -> str | None:
    for robot in ROBOTS:
        if path == f"/World/{robot}" or path.startswith(f"/World/{robot}/"):
            return robot
    return None


def classify_overlap_pair(
    *,
    actor0: str,
    actor1: str,
    collider0: str,
    collider1: str,
    adjacent_body_pairs: Iterable[tuple[str, str]],
    cad_assembly_interface_pairs: Iterable[tuple[str, str]] = (),
    relation: str,
    overlap_volume_m3: float,
) -> dict[str, Any]:
    """Classify one measured collider relation without hiding table contacts."""

    actor_pair = canonical_pair(actor0, actor1)
    collider_pair = canonical_pair(collider0, collider1)
    adjacent = {canonical_pair(*pair) for pair in adjacent_body_pairs}
    assembly_interfaces = {canonical_pair(*pair) for pair in cad_assembly_interface_pairs}
    robot0 = _robot_for_path(actor0)
    robot1 = _robot_for_path(actor1)
    is_overlap = relation == "OVERLAP" and overlap_volume_m3 > 0.0

    if not is_overlap:
        classification = "NONE"
        allowed: bool | None = True
    elif actor0 == actor1 or actor_pair in assembly_interfaces:
        classification = "CAD_ASSEMBLY_INTERFACE_EXPECTED"
        allowed = True
    elif actor_pair in adjacent:
        classification = "ADJACENT_JOINT_INTERFACE_EXPECTED"
        allowed = True
    elif robot0 is not None and robot0 == robot1:
        classification = "UNEXPECTED_SELF_COLLISION"
        allowed = False
    elif robot0 is not None and robot1 is not None:
        classification = "CROSS_FOLLOWER_COLLISION"
        allowed = False
    elif (robot0 is None) != (robot1 is None):
        robot_path = actor0 if robot0 is not None else actor1
        environment_path = actor1 if robot0 is not None else actor0
        if environment_path.endswith("/user_confirmed_table") and robot_path.endswith(
            ("_finger_link", "_gripper_link")
        ):
            classification = "ENVIRONMENT_CONTACT_REQUIRES_RUNTIME_EFFECT_REVIEW"
            allowed = None
        else:
            classification = "UNEXPECTED_ENVIRONMENT_COLLISION"
            allowed = False
    else:
        classification = "UNEXPECTED_ENVIRONMENT_COLLISION"
        allowed = False

    if classification not in OVERLAP_CLASSES:
        raise AssertionError(f"unregistered overlap class: {classification}")
    return {
        "actor_pair": list(actor_pair),
        "collider_pair": list(collider_pair),
        "relation": relation,
        "overlap_volume_m3": float(overlap_volume_m3),
        "classification": classification,
        "allowed": allowed,
    }


def load_frozen_pose_manifest(path: Path) -> list[dict[str, Any]]:
    """Load the exact five accepted initial arm starts plus project home."""

    payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if "samples" in payload:
        samples = payload["samples"]
        sample_source = "FROZEN_USER_ACCEPTED_FIVE_POSE_RUNTIME"
    else:
        samples = payload.get("selected_samples", [])
        sample_source = "FROZEN_CAD_COLLISION_REPLAN_PREFLIGHT"
    if len(samples) != 5:
        raise ValueError("frozen five-pose report must contain exactly five samples")
    records = [
        {
            "pose_id": "home_reference",
            "candidate_index": None,
            "arm_q_rad": [float(value) for value in HOME_ARM],
            "source": "PROJECT_FROZEN_HOME_REFERENCE",
        }
    ]
    expected_ids = [f"sample_{index:02d}" for index in range(1, 6)]
    actual_ids = [str(sample.get("sample_id")) for sample in samples]
    if actual_ids != expected_ids:
        raise ValueError(f"frozen sample order drift: {actual_ids}")
    for sample in samples:
        values = [float(value) for value in sample["initial_arm_q_rad"]]
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"invalid arm start: {sample['sample_id']}")
        records.append(
            {
                "pose_id": str(sample["sample_id"]),
                "candidate_index": int(sample["candidate_index"]),
                "arm_q_rad": values,
                "source": sample_source,
                **(
                    {
                        "bottle_line_yaw_deg": float(sample["bottle_line_yaw_deg"]),
                        "initial_ee_position_world_m": [
                            float(value) for value in sample["initial_ee_position_world_m"]
                        ],
                    }
                    if "bottle_line_yaw_deg" in sample
                    else {}
                ),
            }
        )
    return records


def summarize_static_validation(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize static gates while keeping unresolved table effects visible."""

    failures = []
    partials = []
    for record in records:
        pose_id = str(record["pose_id"])
        if not bool(record.get("finite")):
            failures.append({"pose_id": pose_id, "gate": "finite"})
        if not bool(record.get("within_joint_limits")):
            failures.append({"pose_id": pose_id, "gate": "within_joint_limits"})
        if int(record.get("unexpected_overlap_count", 0)):
            failures.append({"pose_id": pose_id, "gate": "unexpected_overlap"})
        if int(record.get("nonfinite_contact_count", 0)):
            failures.append({"pose_id": pose_id, "gate": "finite_contact"})
        jump = float(record.get("first_frame_jump_max_abs_rad", math.inf))
        jump_gate = float(record.get("first_frame_jump_gate_rad", 0.02))
        if not math.isfinite(jump) or jump > jump_gate:
            failures.append({"pose_id": pose_id, "gate": "first_frame_jump"})
        if int(record.get("unresolved_environment_contact_count", 0)):
            partials.append({"pose_id": pose_id, "gate": "environment_contact_effect"})
    status = "FAIL" if failures else "PARTIAL" if partials else "PASS"
    return {
        "status": status,
        "pose_count": len(records),
        "failure_count": len(failures),
        "partial_count": len(partials),
        "failures": failures,
        "partials": partials,
    }


def _rounded(value: Any) -> float:
    return round(float(value), 9)


def canonical_runtime_signature(report: Mapping[str, Any]) -> str:
    """Hash deterministic collision evidence, independent of record ordering."""

    poses = []
    for pose in report.get("poses", []):
        overlaps = [
            {
                "actor_pair": list(canonical_pair(*item["actor_pair"])),
                "collider_pair": list(canonical_pair(*item["collider_pair"])),
                "classification": item["classification"],
                "overlap_volume_m3": _rounded(item["overlap_volume_m3"]),
            }
            for item in pose.get("overlaps", [])
        ]
        overlaps.sort(
            key=lambda item: (
                item["classification"],
                item["actor_pair"],
                item["collider_pair"],
            )
        )
        poses.append(
            {
                "pose_id": pose["pose_id"],
                "status": pose.get("status"),
                "first_frame_jump_max_abs_rad": _rounded(pose.get("first_frame_jump_max_abs_rad", 0.0)),
                "overlaps": overlaps,
            }
        )
    poses.sort(key=lambda item: item["pose_id"])
    encoded = json.dumps(poses, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
