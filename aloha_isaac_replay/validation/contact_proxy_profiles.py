from __future__ import annotations

from copy import deepcopy
from typing import Any

CONTACT_PROXY_PROFILES: dict[str, dict[str, Any]] = {
    "legacy_puppet": {
        "description": "Legacy ALOHA1 proxy paths authored under /puppet_* runtime roots.",
        "stage_units_in_meters": 0.01,
        "stage_up_axis": "Y",
        "robot_roots": {
            "left": "/puppet_left_vx300s",
            "right": "/puppet_right_vx300s",
        },
        "finger_dof_names": {
            "left": {"left_finger": "left_finger", "right_finger": "right_finger"},
            "right": {"left_finger": "left_finger", "right_finger": "right_finger"},
        },
        "finger_proxy_paths": {
            "left": {
                "left_finger": "/puppet_left_vx300s/puppet_left_left_finger_link/bbox_collision_proxy",
                "right_finger": "/puppet_left_vx300s/puppet_left_right_finger_link/bbox_collision_proxy",
                "articulation": "/puppet_left_vx300s/root_joint",
            },
            "right": {
                "left_finger": "/puppet_right_vx300s/puppet_right_left_finger_link/bbox_collision_proxy",
                "right_finger": "/puppet_right_vx300s/puppet_right_right_finger_link/bbox_collision_proxy",
                "articulation": "/puppet_right_vx300s/root_joint",
            },
        },
    },
    "scene_base_link": {
        "description": "Trossen/Menagerie scene paths where each arm lives under /scene/<side>_base_link.",
        "stage_units_in_meters": 1.0,
        "stage_up_axis": "Z",
        "robot_roots": {
            "left": "/scene/left_base_link",
            "right": "/scene/right_base_link",
        },
        "finger_dof_names": {
            "left": {"left_finger": "left_left_finger", "right_finger": "left_right_finger"},
            "right": {"left_finger": "right_left_finger", "right_finger": "right_right_finger"},
        },
        "finger_proxy_paths": {
            "left": {
                "left_finger": "/scene/left_base_link/left_left_finger_link/bbox_collision_proxy",
                "right_finger": "/scene/left_base_link/left_right_finger_link/bbox_collision_proxy",
                "articulation": "/scene/left_base_link/left_base_link",
            },
            "right": {
                "left_finger": "/scene/right_base_link/right_left_finger_link/bbox_collision_proxy",
                "right_finger": "/scene/right_base_link/right_right_finger_link/bbox_collision_proxy",
                "articulation": "/scene/right_base_link/right_base_link",
            },
        },
    },
}


def contact_proxy_profile_names() -> tuple[str, ...]:
    return tuple(CONTACT_PROXY_PROFILES)


def _profile(profile_name: str) -> dict[str, Any]:
    try:
        return CONTACT_PROXY_PROFILES[profile_name]
    except KeyError as exc:
        available = ", ".join(contact_proxy_profile_names())
        raise ValueError(f"unknown contact proxy profile {profile_name!r}; available profiles: {available}") from exc


def resolve_contact_proxy_paths(profile_name: str) -> dict[str, dict[str, str]]:
    return deepcopy(_profile(profile_name)["finger_proxy_paths"])


def finger_dof_names_for_side(profile_name: str, side: str) -> dict[str, str]:
    try:
        return deepcopy(_profile(profile_name)["finger_dof_names"][side])
    except KeyError as exc:
        raise ValueError(f"contact proxy profile {profile_name!r} has no side {side!r}") from exc


def robot_root_for_side(profile_name: str, side: str) -> str | None:
    root = _profile(profile_name)["robot_roots"].get(side)
    return str(root) if root is not None else None


def side_from_rigid_body_path(profile_name: str, rigid_body_path: str) -> str:
    for side in ("left", "right"):
        robot_root = robot_root_for_side(profile_name, side)
        if robot_root and (rigid_body_path == robot_root or rigid_body_path.startswith(robot_root + "/")):
            return side
    return "unknown"


def proxy_path_for_rigid_body(profile_name: str, rigid_body_path: str) -> str:
    _profile(profile_name)
    return f"{rigid_body_path}/bbox_collision_proxy"


def contact_proxy_namespace_roots(paths: dict[str, dict[str, str]]) -> list[str]:
    roots: set[str] = set()
    for side_paths in paths.values():
        for prim_path in side_paths.values():
            parts = [part for part in str(prim_path).split("/") if part]
            if parts:
                roots.add(parts[0])
    return sorted(roots)


def stage_units_in_meters_for_profile(profile_name: str) -> float:
    return float(_profile(profile_name)["stage_units_in_meters"])


def stage_up_axis_for_profile(profile_name: str) -> str:
    return str(_profile(profile_name)["stage_up_axis"])
