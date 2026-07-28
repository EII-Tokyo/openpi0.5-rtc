"""Workcell, camera, and observation interface plan for ALOHA 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

CAMERA_ORDER = [
    "cam_high",
    "cam_low",
    "cam_left_wrist",
    "cam_right_wrist",
]


def build_workcell_plan(
    project_root: Path,
    *,
    enable_leaders: bool,
) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    follower_root = root / "assets/Trossen/ALOHA1/1.0/follower_vx300s"
    robots = []
    for name, translation in (
        ("follower_left", [-0.4, 0.0, 0.0]),
        ("follower_right", [0.4, 0.0, 0.0]),
    ):
        usd = (
            follower_root
            / name
            / "configuration"
            / f"{name}_debug_acceleration_drive.usd"
        )
        if not usd.is_file():
            raise FileNotFoundError(f"debug follower profile unavailable: {usd}")
        robots.append(
            {
                "name": name,
                "usd": str(usd.resolve()),
                "translation_m": translation,
                "rotation_rpy_rad": [0.0, 0.0, 0.0],
                "transform_status": "TEMPORARY_SEPARATION_ONLY",
                "physics_claim_allowed": False,
            }
        )
    leaders = []
    leader_root = root / "assets/Trossen/ALOHA1/1.0/leader_wx250s"
    for name, translation in (
        ("leader_left", [-0.4, -0.5, 0.0]),
        ("leader_right", [0.4, -0.5, 0.0]),
    ):
        usd = leader_root / name / f"{name}.usd"
        if enable_leaders and not usd.is_file():
            raise FileNotFoundError(f"enabled leader unavailable: {usd}")
        leaders.append(
            {
                "name": name,
                "usd": str(usd.resolve()),
                "translation_m": translation,
                "rotation_rpy_rad": [0.0, 0.0, 0.0],
                "transform_status": "TEMPORARY_SEPARATION_ONLY",
            }
        )
    cameras = [
        {
            "name": name,
            "logical_role": (
                "fixed_third_person"
                if name in {"cam_high", "cam_low"}
                else "wrist"
            ),
            "resolution_wh": [640, 480],
            "resolution_status": "CONTROL_CODE_INTERFACE_CONTRACT",
            "frame_rate_hz": None,
            "intrinsics": None,
            "distortion_policy": None,
            "mounting_extrinsics": None,
            "calibration_status": "CALIBRATION_PENDING",
            "render_eligible": False,
        }
        for name in CAMERA_ORDER
    ]
    return {
        "schema_version": 1,
        "status": "PARTIAL",
        "stage": str(
            (
                root
                / "assets/Trossen/ALOHA1/1.0/workcell/aloha1_workcell.usd"
            ).resolve()
        ),
        "coordinate_convention": (
            "temporary local interface frame; Z=0 is a schematic base plane, "
            "not a measured world/table calibration"
        ),
        "enable_leaders": enable_leaders,
        "leader_variant_default": "enabled" if enable_leaders else "disabled",
        "robots": robots,
        "leaders": leaders,
        "workcell_objects": [
            {
                "name": "table",
                "status": "VISUAL_REFERENCE_ONLY",
                "collision_enabled": False,
                "dimensions_m": [1.2192, 0.7490, 0.0200],
                "source": (
                    "aloha_isaac_rebuild parameter_registry manufacturer "
                    "reference; base-to-table relation remains missing"
                ),
            },
            {
                "name": "frame",
                "status": "SEMANTIC_PLACEHOLDER",
                "collision_enabled": False,
                "dimensions_m": None,
            },
            {
                "name": "camera_mounts",
                "status": "SEMANTIC_PLACEHOLDER",
                "collision_enabled": False,
                "dimensions_m": None,
            },
            {
                "name": "pipe_fixture",
                "status": "CALIBRATION_PENDING",
                "collision_enabled": False,
                "dimensions_m": None,
            },
            {
                "name": "bottle",
                "status": "CALIBRATION_PENDING",
                "collision_enabled": False,
                "dimensions_m": None,
            },
        ],
        "cameras": cameras,
        "observation": {
            "camera_order": list(CAMERA_ORDER),
            "image_shape_hwc": [480, 640, 3],
            "image_dtype": "uint8",
            "color_semantics": (
                "RGB interface target; existing historical datasets may carry "
                "BGR-valued JPEG data and require dataset-specific handling"
            ),
            "state_order_reference": "configs/aloha1_joint_map.yaml",
            "state_shape": [14],
            "action_shape": [14],
        },
        "hard_blockers": {
            "follower_mount_relation": (
                "measured left/right base transforms and orientations"
            ),
            "tabletop_to_base_relation": (
                "measured tabletop plane relative to both follower bases"
            ),
            "pipe_fixture": "measured pipe pose and collision geometry",
            "bottle": "measured bottle geometry, mass, and inertia",
            "camera_calibration": (
                "intrinsics, distortion/cropping, frame rate, and extrinsics "
                "for all four cameras"
            ),
        },
    }
