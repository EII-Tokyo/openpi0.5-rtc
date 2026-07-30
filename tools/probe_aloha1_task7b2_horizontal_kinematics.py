#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Probe follower-left Lula/URDF/USD correspondence in local Isaac Sim 5.1."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
from importlib.metadata import version
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import tomllib
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STAGE_SHA256 = "2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c"
EXPECTED_URDF_SHA256 = "d9e4b32723ee71dfce26fb4e78546cfcfef147b2d7dbf5e53e3620e3d8aa96bd"
EXPECTED_MOTION_GENERATION_VERSION = "8.0.26"
EXPECTED_ISAAC_SIM_VERSION = "5.1.0.0"
EXPECTED_KIT_VERSION = "107.3.3"
EXPECTED_PHYSX_VERSION = "107.3.26"
EXPECTED_CSPACE = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]
DEFAULT_Q = np.asarray([0.0, -0.96, 1.16, 0.0, -0.3, 0.0])
ARTICULATION_PATH = "/World/follower_left/vx300s_left/root_joint"
BASE_LINK_PATH = "/World/follower_left/vx300s_left/follower_left_base_link"
END_EFFECTOR_PATH = "/World/follower_left/vx300s_left/follower_left_ee_gripper_link"
END_EFFECTOR_FRAME = "follower_left_ee_gripper_link"
TRANSLATION_GATE_M = 0.001
ROTATION_GATE_RAD = 0.005
EXPECTED_EPISODE_SHA256 = "f073a21c6a790e738e36085d791482924a82832ca6d80cece04a26353b9fc745"
EXPECTED_JOINT_MAP_SHA256 = "f56be097d859f7361b804705af6659e0d51d9e480d1c721a60040ab787530308"
EXPECTED_BOTTLE_USD_SHA256 = "16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e"
SUPPLIER_CAD_CLEARANCE_REPORT = ROOT / "reports/aloha1_mapping/aloha1_supplier_cad_grasp_clearance.json"
EXPECTED_SUPPLIER_CAD_CLEARANCE_REPORT_SHA256 = "9f23974af362dc92134a38633180360bfff8b54bc0a5eaefae8032e2240b91bc"
SUPPLIER_CAD_CLEARANCE_SCREENSHOT_REVIEW = (
    ROOT / "reports/aloha1_mapping/aloha1_supplier_cad_grasp_clearance_screenshot_review.json"
)
EXPECTED_SUPPLIER_CAD_CLEARANCE_SCREENSHOT_REVIEW_SHA256 = (
    "c7097b05654a3966c976690e5f0f79c3b2be69eaa51727f430998efae2bbe0f3"
)


@dataclass(frozen=True)
class LiftOnsetDetection:
    lift_onset_frame: int
    threshold: float
    baseline_median: float
    baseline_mad: float
    raw_baseline_median: float
    raw_baseline_mad: float
    candidates: tuple[dict[str, Any], ...]


def detect_lift_onset(
    *,
    frames: np.ndarray,
    delta_z: np.ndarray,
    z_positions: np.ndarray,
    close_command_start_frame: int,
    readback_response_start_frame: int,
) -> LiftOnsetDetection:
    """Detect two-step positive FK lift after gripper readback begins."""
    frame_values = np.asarray(frames, dtype=np.int64)
    delta_values = np.asarray(delta_z, dtype=np.float64)
    z_values = np.asarray(z_positions, dtype=np.float64)
    if (
        frame_values.ndim != 1
        or delta_values.shape != frame_values.shape
        or z_values.shape != frame_values.shape
        or frame_values.size < 3
    ):
        raise ValueError("frames, delta_z, and z_positions must be equal 1-D arrays")
    if not np.isfinite(delta_values).all() or not np.isfinite(z_values).all():
        raise ValueError("lift-onset inputs must be finite")
    baseline = delta_values[frame_values <= close_command_start_frame]
    if baseline.size < 2:
        raise ValueError("lift-onset baseline must contain at least two values")
    raw_baseline_median = float(np.median(baseline))
    raw_baseline_mad = float(np.median(np.abs(baseline - raw_baseline_median)))
    upward_noise = np.maximum(baseline, 0.0)
    baseline_median = float(np.median(upward_noise))
    baseline_mad = float(np.median(np.abs(upward_noise - baseline_median)))
    scale = max(1.0, float(np.max(np.abs(delta_values))))
    threshold = baseline_median + 5.0 * baseline_mad
    if baseline_mad == 0.0:
        threshold += float(np.finfo(np.float64).eps * scale)

    candidates: list[dict[str, Any]] = []
    selected: int | None = None
    for index in range(frame_values.size - 1):
        frame = int(frame_values[index])
        if frame <= readback_response_start_frame:
            continue
        two_above = bool(delta_values[index] > threshold and delta_values[index + 1] > threshold)
        previous_z = z_values[index - 1] if index > 0 else z_values[index]
        cumulative_to_end = float(z_values[-1] - previous_z)
        positive_cumulative = cumulative_to_end > 0.0
        accepted = two_above and positive_cumulative
        candidates.append(
            {
                "frame": frame,
                "delta_z_m": float(delta_values[index]),
                "next_delta_z_m": float(delta_values[index + 1]),
                "two_consecutive_above_threshold": two_above,
                "cumulative_z_to_end_m": cumulative_to_end,
                "positive_cumulative_z_to_end": positive_cumulative,
                "accepted": accepted,
            }
        )
        if accepted and selected is None:
            selected = frame
    if selected is None:
        raise ValueError("no FK lift onset satisfies the two-step criterion")
    return LiftOnsetDetection(
        lift_onset_frame=selected,
        threshold=threshold,
        baseline_median=baseline_median,
        baseline_mad=baseline_mad,
        raw_baseline_median=raw_baseline_median,
        raw_baseline_mad=raw_baseline_mad,
        candidates=tuple(candidates),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _rotation_residual_rad(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(cosine))


def _mesh_points_world(
    stage: Any,
    prim_path: str,
    usd_geom: Any,
    gf_module: Any,
) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    mesh = usd_geom.Mesh(prim)
    if not prim.IsValid() or not mesh:
        raise RuntimeError(f"missing mesh prim: {prim_path}")
    points = mesh.GetPointsAttr().Get()
    if not points:
        raise RuntimeError(f"mesh has no points: {prim_path}")
    transform = usd_geom.XformCache().GetLocalToWorldTransform(prim)
    result = np.asarray(
        [
            transform.Transform(
                gf_module.Vec3d(
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                )
            )
            for point in points
        ],
        dtype=np.float64,
    )
    if not np.isfinite(result).all():
        raise RuntimeError(f"mesh world points are non-finite: {prim_path}")
    return result


def _bottle_collision_points_local(
    bottle_usd: Path,
    usd_module: Any,
    usd_geom: Any,
    gf_module: Any,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    stage = usd_module.Stage.Open(str(bottle_usd))
    if stage is None:
        raise RuntimeError(f"cannot open Bottle500 USD: {bottle_usd}")
    root = stage.GetPrimAtPath("/Bottle500")
    if not root.IsValid():
        raise RuntimeError("Bottle500 USD is missing /Bottle500")
    cache = usd_geom.XformCache()
    root_to_world = cache.GetLocalToWorldTransform(root)
    world_to_root = root_to_world.GetInverse()
    points: list[np.ndarray] = []
    meshes: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        if not prim.GetPath().pathString.startswith("/Bottle500/Collisions/"):
            continue
        mesh = usd_geom.Mesh(prim)
        if not mesh:
            continue
        authored_points = mesh.GetPointsAttr().Get()
        if not authored_points:
            continue
        mesh_to_world = cache.GetLocalToWorldTransform(prim)
        mesh_to_root = mesh_to_world * world_to_root
        transformed = np.asarray(
            [
                mesh_to_root.Transform(
                    gf_module.Vec3d(
                        float(point[0]),
                        float(point[1]),
                        float(point[2]),
                    )
                )
                for point in authored_points
            ],
            dtype=np.float64,
        )
        points.append(transformed)
        contact_offset = prim.GetAttribute("physxCollision:contactOffset").Get()
        rest_offset = prim.GetAttribute("physxCollision:restOffset").Get()
        meshes.append(
            {
                "prim_path": prim.GetPath().pathString,
                "point_count": int(transformed.shape[0]),
                "contact_offset_authored_m": (float(contact_offset) if contact_offset is not None else None),
                "rest_offset_authored_m": (float(rest_offset) if rest_offset is not None else None),
            }
        )
    if len(meshes) != 41:
        raise RuntimeError(f"expected 41 Bottle500 collision meshes, found {len(meshes)}")
    combined = np.concatenate(points, axis=0)
    extents = np.max(combined, axis=0) - np.min(combined, axis=0)
    if (
        not np.isfinite(extents).all()
        or np.any(extents <= 0.0)
        or not math.isclose(float(extents[2]), 0.206, abs_tol=2e-6)
        or np.any(extents[:2] > 0.068 + 2e-6)
    ):
        raise RuntimeError(f"unexpected Bottle500 collision extents: {extents}")
    return combined, meshes


def _closest_opposing_points(
    left_points: np.ndarray,
    right_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    differences = left_points[:, None, :] - right_points[None, :, :]
    squared = np.einsum("ijk,ijk->ij", differences, differences)
    left_index, right_index = np.unravel_index(
        int(np.argmin(squared)),
        squared.shape,
    )
    return (
        left_points[left_index],
        right_points[right_index],
        float(math.sqrt(float(squared[left_index, right_index]))),
    )


def _solve_adaptive_linear_ik(
    *,
    solver: Any,
    frame_name: str,
    start_position: np.ndarray,
    end_position: np.ndarray,
    orientation_wxyz: np.ndarray,
    start_q: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    velocity_limits: np.ndarray,
    physics_dt: float,
    phase: str,
    position_tolerance: float,
    orientation_tolerance: float,
    maximum_segments: int = 512,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    segment_count = 1
    while segment_count <= maximum_segments:
        previous = np.asarray(start_q, dtype=np.float64)
        waypoints: list[dict[str, Any]] = []
        attempt_status = "PASS"
        failure_reason = None
        for segment in range(1, segment_count + 1):
            fraction = segment / segment_count
            target = start_position + fraction * (end_position - start_position)
            solution, success = solver.compute_inverse_kinematics(
                frame_name=frame_name,
                target_position=target,
                target_orientation=orientation_wxyz,
                warm_start=previous,
                position_tolerance=position_tolerance,
                orientation_tolerance=orientation_tolerance,
            )
            solution = np.asarray(solution, dtype=np.float64)
            finite = bool(np.isfinite(solution).all())
            within_limits = bool(finite and np.all(solution >= lower_limits) and np.all(solution <= upper_limits))
            delta = solution - previous if finite else np.full(6, np.nan)
            velocity_ok = bool(finite and np.all(np.abs(delta) <= velocity_limits * physics_dt))
            waypoint_status = "PASS" if success and finite and within_limits and velocity_ok else "FAIL"
            waypoints.append(
                {
                    "phase": phase,
                    "segment": segment,
                    "segment_count": segment_count,
                    "fraction": fraction,
                    "target_position_world_m": [float(value) for value in target],
                    "target_orientation_world_wxyz": [float(value) for value in orientation_wxyz],
                    "joint_positions_rad": [float(value) for value in solution],
                    "joint_delta_rad": [float(value) for value in delta],
                    "solver_success": bool(success),
                    "finite": finite,
                    "within_limits": within_limits,
                    "velocity_ok": velocity_ok,
                    "status": waypoint_status,
                }
            )
            if waypoint_status != "PASS":
                attempt_status = "FAIL"
                failure_reason = {
                    "segment": segment,
                    "solver_success": bool(success),
                    "finite": finite,
                    "within_limits": within_limits,
                    "velocity_ok": velocity_ok,
                }
                break
            previous = solution
        attempts.append(
            {
                "segment_count": segment_count,
                "status": attempt_status,
                "failure_reason": failure_reason,
            }
        )
        if attempt_status == "PASS":
            return waypoints, {
                "status": "PASS",
                "selected_segment_count": segment_count,
                "attempts": attempts,
            }
        segment_count *= 2
    return [], {
        "status": "FAIL",
        "selected_segment_count": None,
        "attempts": attempts,
    }


def _source_inventory(
    *,
    extension_root: Path,
    core_api_root: Path,
    validation_root: Path,
) -> dict[str, Any]:
    paths = {
        "extension_toml": extension_root / "config/extension.toml",
        "articulation_kinematics_solver": (
            extension_root / "isaacsim/robot_motion/motion_generation/articulation_kinematics_solver.py"
        ),
        "lula_kinematics": (extension_root / "isaacsim/robot_motion/motion_generation/lula/kinematics.py"),
        "nvidia_test_kinematics": (extension_root / "isaacsim/robot_motion/motion_generation/tests/test_kinematics.py"),
        "physics_context": (core_api_root / "isaacsim/core/api/physics_context/physics_context.py"),
        "physics_rules": (validation_root / "isaacsim/asset/validation/physics_rules.py"),
    }
    records = {}
    for name, path in paths.items():
        resolved = path.resolve(strict=True)
        records[name] = {
            "path": str(resolved),
            "sha256": _sha256(resolved),
        }
    extension_data = tomllib.loads(paths["extension_toml"].read_text())
    records["extension_toml"]["declared_version"] = extension_data["package"]["version"]
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        type=Path,
        default=(
            ROOT / "assets/Trossen/ALOHA1/1.0/diagnostics/"
            "table_support_alignment/1.0/"
            "aloha1_table_support_aligned_workcell.usda"
        ),
    )
    parser.add_argument(
        "--descriptor",
        type=Path,
        default=ROOT / "configs/aloha1_lula_follower_left.yaml",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=ROOT / "generated/urdf/follower_left.urdf",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    stage_path = args.stage.resolve(strict=True)
    descriptor_path = args.descriptor.resolve(strict=True)
    urdf_path = args.urdf.resolve(strict=True)
    output_path = args.output.resolve()

    stage_hash_before = _sha256(stage_path)
    urdf_hash = _sha256(urdf_path)
    if stage_hash_before != EXPECTED_STAGE_SHA256:
        raise RuntimeError(f"frozen Stage SHA-256 mismatch: {stage_hash_before}")
    if urdf_hash != EXPECTED_URDF_SHA256:
        raise RuntimeError(f"frozen URDF SHA-256 mismatch: {urdf_hash}")

    isaacsim_module = importlib.import_module("isaacsim")
    SimulationApp = isaacsim_module.SimulationApp  # noqa: N806
    app = SimulationApp({"headless": args.headless})

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "FAIL",
        "hard_blockers": [],
        "task8": "NOT_RUN",
    }
    try:
        import carb
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices
        from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats
        from isaacsim.core.utils.stage import open_stage
        from isaacsim.core.utils.xforms import get_world_pose
        from isaacsim.robot_motion.motion_generation.articulation_kinematics_solver import ArticulationKinematicsSolver
        from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver
        import omni.kit.app
        import omni.physx
        from omni.physx import get_physx_simulation_interface
        import omni.usd
        from pxr import Gf
        from pxr import PhysicsSchemaTools
        from pxr import PhysxSchema
        from pxr import Usd
        from pxr import UsdGeom
        import yaml

        from tools.aloha1_mapping.aloha_kinematics_reference import SOURCE_CLASS as POE_SOURCE_CLASS
        from tools.aloha1_mapping.aloha_kinematics_reference import SOURCE_FILE as POE_SOURCE_FILE
        from tools.aloha1_mapping.aloha_kinematics_reference import SOURCE_SHA256 as POE_SOURCE_SHA256
        from tools.aloha1_mapping.aloha_kinematics_reference import fk_space
        from tools.aloha1_mapping.episode18_grasp_window import detect_gripper_phases
        from tools.aloha1_mapping.episode18_grasp_window import load_episode_window
        from tools.aloha1_mapping.grasp_frame_contract import derive_urdf_fixed_transform
        from tools.aloha1_mapping.horizontal_bottle_geometry import canonical_bottle_axis
        from tools.aloha1_mapping.horizontal_bottle_geometry import derive_horizontal_support_placement
        from tools.aloha1_mapping.horizontal_bottle_geometry import evaluate_geometry
        from tools.aloha1_mapping.horizontal_bottle_geometry import point_on_axis
        from tools.aloha1_mapping.horizontal_bottle_geometry import shortest_arc_rotation
        from tools.aloha1_mapping.horizontal_bottle_geometry import transform_points
        from tools.aloha1_mapping.supplier_cad_grasp_frame import load_verified_clearance_grasp_frame

        extension_root = (
            Path(importlib.import_module("isaacsim.robot_motion.motion_generation").__path__[0]).resolve().parents[2]
        )
        extensions_root = extension_root.parent
        core_api_root = extensions_root / "isaacsim.core.api"
        validation_root = extensions_root / "isaacsim.asset.validation"
        sources = _source_inventory(
            extension_root=extension_root,
            core_api_root=core_api_root,
            validation_root=validation_root,
        )
        source_version = sources["extension_toml"]["declared_version"]

        manager = omni.kit.app.get_app().get_extension_manager()
        motion_extension_id = manager.get_enabled_extension_id("isaacsim.robot_motion.motion_generation")
        physx_extension_id = manager.get_enabled_extension_id("omni.physx")
        motion_extension = manager.get_extension_dict(motion_extension_id) if motion_extension_id else None
        physx_extension = manager.get_extension_dict(physx_extension_id) if physx_extension_id else None
        motion_runtime_version = motion_extension.get("package", {}).get("version") if motion_extension else None
        physx_runtime_version = physx_extension.get("package", {}).get("version") if physx_extension else None
        if physx_runtime_version is None:
            physx_module_path = Path(next(iter(omni.physx.__path__))).resolve()
            extension_directory = next(
                (parent for parent in physx_module_path.parents if parent.name.startswith("omni.physx-")),
                None,
            )
            version_match = re.match(r"omni\.physx-([^+]+)", extension_directory.name) if extension_directory else None
            physx_runtime_version = version_match.group(1) if version_match else None
        if source_version != EXPECTED_MOTION_GENERATION_VERSION:
            raise RuntimeError(f"motion-generation source version mismatch: {source_version}")
        if (
            motion_runtime_version is not None
            and str(motion_runtime_version).split("+", maxsplit=1)[0] != EXPECTED_MOTION_GENERATION_VERSION
        ):
            raise RuntimeError(f"motion-generation runtime version mismatch: {motion_runtime_version}")
        isaac_runtime_version = version("isaacsim")
        kit_runtime_version = str(carb.tokens.get_tokens_interface().resolve("${kit_version}")).split("+", maxsplit=1)[
            0
        ]
        normalized_physx_version = (
            str(physx_runtime_version).split("+", maxsplit=1)[0] if physx_runtime_version else None
        )
        expected_runtime = {
            "isaac_sim": EXPECTED_ISAAC_SIM_VERSION,
            "kit": EXPECTED_KIT_VERSION,
            "physx": EXPECTED_PHYSX_VERSION,
        }
        actual_runtime = {
            "isaac_sim": isaac_runtime_version,
            "kit": kit_runtime_version,
            "physx": normalized_physx_version,
        }
        if actual_runtime != expected_runtime:
            raise RuntimeError(f"Isaac runtime boundary mismatch: expected={expected_runtime}, actual={actual_runtime}")

        solver = LulaKinematicsSolver(
            robot_description_path=str(descriptor_path),
            urdf_path=str(urdf_path),
        )
        solver_joint_names = list(solver.get_joint_names())
        solver_frames = list(solver.get_all_frame_names())
        if solver_joint_names != EXPECTED_CSPACE:
            raise RuntimeError(f"Lula cspace mismatch: {solver_joint_names}")
        if END_EFFECTOR_FRAME not in solver_frames:
            raise RuntimeError(f"missing Lula frame: {END_EFFECTOR_FRAME}")

        if not open_stage(str(stage_path)):
            raise RuntimeError(f"failed to open frozen Stage: {stage_path}")
        world = World(
            stage_units_in_meters=1.0,
            backend="numpy",
            device="cpu",
            physics_dt=1.0 / 60.0,
            rendering_dt=1.0 / 60.0,
        )
        physics_context = world.get_physics_context()
        physics_context.set_solve_articulation_contact_last(True)
        solve_contact_last_readback = physics_context.get_solve_articulation_contact_last()
        articulation = SingleArticulation(
            prim_path=ARTICULATION_PATH,
            name="task7b2_horizontal_kinematics_follower_left",
            reset_xform_properties=False,
        )
        world.scene.add(articulation)
        world.reset()
        runtime_dof_order = list(articulation.dof_names)
        expected_runtime_order = [
            *EXPECTED_CSPACE,
            "gripper",
            "left_finger",
            "right_finger",
        ]
        if runtime_dof_order != expected_runtime_order:
            raise RuntimeError(f"runtime DOF order mismatch: {runtime_dof_order}")

        cases = [
            ("approved_home", DEFAULT_Q.copy()),
            (
                "validated_waist_positive",
                DEFAULT_Q + np.asarray([0.05, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ),
            ("approved_home_return", DEFAULT_Q.copy()),
        ]
        fk_records = []
        for case_name, arm_q in cases:
            full_q = np.asarray(
                articulation.get_joint_positions(),
                dtype=np.float64,
            )
            full_q[:6] = arm_q
            full_q[6] = 0.0
            full_q[7] = 0.057
            full_q[8] = -0.057
            articulation.set_joint_positions(full_q)
            app.update()

            readback = np.asarray(
                articulation.get_joint_positions(),
                dtype=np.float64,
            )
            base_position, base_orientation = get_world_pose(BASE_LINK_PATH)
            usd_position, usd_orientation = get_world_pose(END_EFFECTOR_PATH)
            solver.set_robot_base_pose(
                np.asarray(base_position, dtype=np.float64),
                np.asarray(base_orientation, dtype=np.float64),
            )
            lula_position, lula_rotation = solver.compute_forward_kinematics(
                END_EFFECTOR_FRAME,
                readback[:6],
            )
            usd_rotation = quats_to_rot_matrices(np.asarray(usd_orientation, dtype=np.float64))
            base_rotation = quats_to_rot_matrices(np.asarray(base_orientation, dtype=np.float64))
            world_from_base = np.eye(4, dtype=np.float64)
            world_from_base[:3, :3] = base_rotation
            world_from_base[:3, 3] = np.asarray(
                base_position,
                dtype=np.float64,
            )
            world_from_interbotix_poe = world_from_base @ fk_space(readback[:6])
            poe_position = world_from_interbotix_poe[:3, 3]
            poe_rotation = world_from_interbotix_poe[:3, :3]
            translation_residual = float(np.linalg.norm(np.asarray(lula_position) - np.asarray(usd_position)))
            rotation_residual = _rotation_residual_rad(
                np.asarray(lula_rotation),
                np.asarray(usd_rotation),
            )
            poe_lula_translation_residual = float(np.linalg.norm(poe_position - np.asarray(lula_position)))
            poe_lula_rotation_residual = _rotation_residual_rad(
                poe_rotation,
                np.asarray(lula_rotation),
            )
            poe_usd_translation_residual = float(np.linalg.norm(poe_position - np.asarray(usd_position)))
            poe_usd_rotation_residual = _rotation_residual_rad(
                poe_rotation,
                np.asarray(usd_rotation),
            )
            case_status = (
                "PASS"
                if translation_residual <= TRANSLATION_GATE_M
                and rotation_residual <= ROTATION_GATE_RAD
                and poe_lula_translation_residual <= TRANSLATION_GATE_M
                and poe_lula_rotation_residual <= ROTATION_GATE_RAD
                and poe_usd_translation_residual <= TRANSLATION_GATE_M
                and poe_usd_rotation_residual <= ROTATION_GATE_RAD
                else "FAIL"
            )
            fk_records.append(
                {
                    "case": case_name,
                    "status": case_status,
                    "arm_target_rad": [float(value) for value in arm_q],
                    "arm_readback_rad": [float(value) for value in readback[:6]],
                    "finger_readback_m": [
                        float(readback[7]),
                        float(readback[8]),
                    ],
                    "base_position_world_m": [float(value) for value in base_position],
                    "base_orientation_world_wxyz": [float(value) for value in base_orientation],
                    "usd_position_world_m": [float(value) for value in usd_position],
                    "usd_orientation_world_wxyz": [float(value) for value in usd_orientation],
                    "lula_position_world_m": [float(value) for value in lula_position],
                    "lula_rotation_world": [[float(value) for value in row] for row in lula_rotation],
                    "interbotix_poe_position_world_m": [float(value) for value in poe_position],
                    "interbotix_poe_rotation_world": [[float(value) for value in row] for row in poe_rotation],
                    "translation_residual_m": translation_residual,
                    "rotation_angle_residual_rad": rotation_residual,
                    "interbotix_poe_to_lula_translation_residual_m": (poe_lula_translation_residual),
                    "interbotix_poe_to_lula_rotation_residual_rad": (poe_lula_rotation_residual),
                    "interbotix_poe_to_usd_translation_residual_m": (poe_usd_translation_residual),
                    "interbotix_poe_to_usd_rotation_residual_rad": (poe_usd_rotation_residual),
                }
            )

        correspondence_pass = all(record["status"] == "PASS" for record in fk_records)
        if not correspondence_pass:
            report["hard_blockers"].append("HARD_BLOCKER_LULA_USD_FRAME_CORRESPONDENCE")

        episode_path = Path("/home/eii/project/bottles_data/episode_18.hdf5").resolve(strict=True)
        episode_window = load_episode_window(
            episode_path,
            208,
            244,
            expected_sha256=EXPECTED_EPISODE_SHA256,
        )
        gripper_phases = detect_gripper_phases(
            episode_window.action[:, 6],
            episode_window.qpos[:, 6],
            first_frame=208,
        )
        solver.set_robot_base_pose(
            np.zeros(3, dtype=np.float64),
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
        episode_records = []
        episode_positions: list[np.ndarray] = []
        episode_rotations: list[np.ndarray] = []
        previous_position: np.ndarray | None = None
        for index, frame_value in enumerate(episode_window.frames):
            qpos_arm = episode_window.qpos[index, :6]
            action_arm = episode_window.action[index, :6]
            position, rotation = solver.compute_forward_kinematics(
                END_EFFECTOR_FRAME,
                qpos_arm,
            )
            position = np.asarray(position, dtype=np.float64)
            rotation = np.asarray(rotation, dtype=np.float64)
            orientation = rot_matrices_to_quats(rotation)
            delta = np.zeros(3, dtype=np.float64) if previous_position is None else position - previous_position
            episode_positions.append(position)
            episode_rotations.append(rotation)
            episode_records.append(
                {
                    "frame": int(frame_value),
                    "qpos_arm_6d": [float(value) for value in qpos_arm],
                    "action_arm_6d": [float(value) for value in action_arm],
                    "ee_position_robot_base_m": [float(value) for value in position],
                    "ee_orientation_wxyz": [float(value) for value in orientation],
                    "ee_delta_m": [float(value) for value in delta],
                    "ee_delta_z_m": float(delta[2]),
                    "gripper_action": float(episode_window.action[index, 6]),
                    "gripper_qpos": float(episode_window.qpos[index, 6]),
                }
            )
            previous_position = position
        episode_position_array = np.asarray(episode_positions)
        episode_delta_z = np.asarray(
            [record["ee_delta_z_m"] for record in episode_records],
            dtype=np.float64,
        )
        lift_detection = detect_lift_onset(
            frames=episode_window.frames,
            delta_z=episode_delta_z,
            z_positions=episode_position_array[:, 2],
            close_command_start_frame=(gripper_phases.close_command_start_frame),
            readback_response_start_frame=(gripper_phases.readback_response_start_frame),
        )
        lift_index = int(np.flatnonzero(episode_window.frames == lift_detection.lift_onset_frame)[0])

        full_q = np.asarray(
            articulation.get_joint_positions(),
            dtype=np.float64,
        )
        full_q[:6] = episode_window.qpos[lift_index, :6]
        full_q[6] = 0.0
        full_q[7] = 0.057
        full_q[8] = -0.057
        articulation.set_joint_positions(full_q)
        app.update()
        onset_readback = np.asarray(
            articulation.get_joint_positions(),
            dtype=np.float64,
        )
        base_position, base_orientation = get_world_pose(BASE_LINK_PATH)
        current_ee_position, current_ee_orientation = get_world_pose(END_EFFECTOR_PATH)
        base_position_array = np.asarray(base_position, dtype=np.float64)
        base_orientation_array = np.asarray(
            base_orientation,
            dtype=np.float64,
        )
        base_rotation = quats_to_rot_matrices(base_orientation_array)
        exact_lift_q = np.asarray(
            episode_window.qpos[lift_index, :6],
            dtype=np.float64,
        )
        world_from_base = np.eye(4, dtype=np.float64)
        world_from_base[:3, :3] = base_rotation
        world_from_base[:3, 3] = base_position_array
        world_from_ee_exact = world_from_base @ fk_space(exact_lift_q)
        target_world_rotation = world_from_ee_exact[:3, :3]
        target_world_orientation = rot_matrices_to_quats(target_world_rotation)
        solver.set_robot_base_pose(
            np.zeros(3, dtype=np.float64),
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
        readback_position_base, readback_rotation_base = solver.compute_forward_kinematics(
            END_EFFECTOR_FRAME,
            onset_readback[:6],
        )
        predicted_world_position = base_position_array + base_rotation @ np.asarray(
            readback_position_base, dtype=np.float64
        )
        onset_usd_translation_residual = float(
            np.linalg.norm(predicted_world_position - np.asarray(current_ee_position, dtype=np.float64))
        )
        onset_usd_rotation_residual = _rotation_residual_rad(
            base_rotation @ np.asarray(readback_rotation_base, dtype=np.float64),
            quats_to_rot_matrices(np.asarray(current_ee_orientation, dtype=np.float64)),
        )
        if onset_usd_translation_residual > TRANSLATION_GATE_M or onset_usd_rotation_residual > ROTATION_GATE_RAD:
            raise RuntimeError(
                "episode lift-onset FK does not match composed USD pose: "
                f"translation={onset_usd_translation_residual}, "
                f"rotation={onset_usd_rotation_residual}"
            )

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("no composed Stage after open_stage")
        table_path = "/World/environment/worldBody/user_confirmed_table"
        table_prim = stage.GetPrimAtPath(table_path)
        if not table_prim.IsValid():
            raise RuntimeError(f"missing user-confirmed table: {table_path}")
        table_bound = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_],
        ).ComputeWorldBound(table_prim)
        table_top_z = float(table_bound.ComputeAlignedBox().GetMax()[2])

        left_finger_collider = (
            "/World/follower_left/vx300s_left/"
            "follower_left_left_finger_link/collisions/"
            "diagnostic_supplier_cad_left_finger/mesh"
        )
        right_finger_collider = (
            "/World/follower_left/vx300s_left/"
            "follower_left_right_finger_link/collisions/"
            "diagnostic_supplier_cad_right_finger/mesh"
        )
        left_points = _mesh_points_world(
            stage,
            left_finger_collider,
            UsdGeom,
            Gf,
        )
        right_points = _mesh_points_world(
            stage,
            right_finger_collider,
            UsdGeom,
            Gf,
        )
        rejected_left_point, rejected_right_point, rejected_open_gap = _closest_opposing_points(
            left_points, right_points
        )
        clearance_frame = load_verified_clearance_grasp_frame(
            clearance_report_path=SUPPLIER_CAD_CLEARANCE_REPORT,
            screenshot_review_path=(SUPPLIER_CAD_CLEARANCE_SCREENSHOT_REVIEW),
            expected_clearance_sha256=(EXPECTED_SUPPLIER_CAD_CLEARANCE_REPORT_SHA256),
            expected_screenshot_sha256=(EXPECTED_SUPPLIER_CAD_CLEARANCE_SCREENSHOT_REVIEW_SHA256),
        )
        gripper_link_from_contact = np.asarray(
            clearance_frame["reference_from_grasp"],
            dtype=np.float64,
        )
        gripper_link_from_ee = derive_urdf_fixed_transform(
            urdf_path,
            source_link="follower_left_gripper_link",
            target_link=END_EFFECTOR_FRAME,
        )
        ee_from_contact = np.linalg.inv(gripper_link_from_ee) @ gripper_link_from_contact
        world_from_contact = world_from_ee_exact @ ee_from_contact
        contact_midpoint = world_from_contact[:3, 3]
        gripper_line_world = world_from_ee_exact[:3, :3] @ np.asarray(
            clearance_frame["finger_line_axis_reference"],
            dtype=np.float64,
        )
        gripper_line_xy = gripper_line_world.copy()
        gripper_line_xy[2] = 0.0
        bottle_axis_world = canonical_bottle_axis(gripper_line_xy)
        left_contact_reference = np.asarray(
            clearance_frame["contact_points_reference_m"]["left"],
            dtype=np.float64,
        )
        right_contact_reference = np.asarray(
            clearance_frame["contact_points_reference_m"]["right"],
            dtype=np.float64,
        )
        left_open_reference = left_contact_reference + np.asarray(
            [
                0.0,
                0.057 - clearance_frame["finger_targets_m"]["left_finger"],
                0.0,
            ],
            dtype=np.float64,
        )
        right_open_reference = right_contact_reference + np.asarray(
            [
                0.0,
                -0.057 - clearance_frame["finger_targets_m"]["right_finger"],
                0.0,
            ],
            dtype=np.float64,
        )
        open_gap = float(np.linalg.norm(right_open_reference - left_open_reference))

        bottle_usd_path = (ROOT / "assets/bottle_500ml/isaac/bottle_500ml_sim.usd").resolve(strict=True)
        if _sha256(bottle_usd_path) != EXPECTED_BOTTLE_USD_SHA256:
            raise RuntimeError("Bottle500 USD SHA-256 mismatch")
        bottle_points_local, bottle_collision_records = _bottle_collision_points_local(
            bottle_usd_path,
            Usd,
            UsdGeom,
            Gf,
        )
        authored_offsets = [
            record["contact_offset_authored_m"]
            for record in bottle_collision_records
            if record["contact_offset_authored_m"] is not None
        ]
        for finger_path in (left_finger_collider, right_finger_collider):
            value = stage.GetPrimAtPath(finger_path).GetAttribute("physxCollision:contactOffset").Get()
            if value is not None:
                authored_offsets.append(float(value))
        setup_gap_m = max(authored_offsets, default=0.0)
        bottle_rotation = shortest_arc_rotation(
            source=[0.0, 0.0, 1.0],
            target=bottle_axis_world,
        )
        placement = derive_horizontal_support_placement(
            local_collision_points=bottle_points_local,
            rotation=bottle_rotation,
            grasp_center_world_xy=contact_midpoint[:2],
            grasp_coordinate_m=0.069,
            table_top_z=table_top_z,
            setup_gap_m=setup_gap_m,
            axis_a_local=[0.0, 0.0, 0.0],
            axis_b_local=[0.0, 0.0, 0.206],
        )
        bottle_points_world = transform_points(
            bottle_points_local,
            placement.matrix,
        )
        bottle_grasp_point = point_on_axis(
            placement.a_world,
            placement.axis_unit,
            0.069,
        )
        geometry_gate = evaluate_geometry(
            axis_unit=placement.axis_unit,
            table_normal=[0.0, 0.0, 1.0],
            gripper_line=gripper_line_xy,
            approach_delta=[0.0, 0.0, -1.0],
            axis_vertical_angle_gate_deg=1.0,
            gripper_perpendicular_gate_deg=3.0,
            approach_direction_gate_deg=3.0,
        )
        current_ee_position_array = world_from_ee_exact[:3, 3].copy()
        link_to_contact_midpoint = contact_midpoint - current_ee_position_array
        grasp_ee_position = np.asarray(bottle_grasp_point, dtype=np.float64) - link_to_contact_midpoint
        finger_points = np.concatenate([left_points, right_points], axis=0)
        finger_relative_min_z = float(np.min(finger_points[:, 2] - current_ee_position_array[2]))
        bottle_top_z = float(np.max(bottle_points_world[:, 2]))
        pregrasp_ee_position = grasp_ee_position.copy()
        pregrasp_ee_position[2] = bottle_top_z + setup_gap_m - finger_relative_min_z
        if pregrasp_ee_position[2] <= grasp_ee_position[2]:
            raise RuntimeError("derived pregrasp is not above the final grasp pose")
        lift_distance_m = float(episode_position_array[-1, 2] - episode_position_array[lift_index, 2])
        if lift_distance_m <= 0.0:
            raise RuntimeError("episode-derived lift distance is not positive")
        lift_ee_position = grasp_ee_position + np.asarray([0.0, 0.0, lift_distance_m])

        joint_map_path = (ROOT / "configs/aloha1_joint_map.yaml").resolve(strict=True)
        if _sha256(joint_map_path) != EXPECTED_JOINT_MAP_SHA256:
            raise RuntimeError("joint-map SHA-256 mismatch")
        joint_map = yaml.safe_load(joint_map_path.read_text())
        arm_dofs = joint_map["robots"]["follower_left"]["dofs"][:6]
        lower_limits = np.asarray(
            [record["position_limit"]["lower"] for record in arm_dofs],
            dtype=np.float64,
        )
        upper_limits = np.asarray(
            [record["position_limit"]["upper"] for record in arm_dofs],
            dtype=np.float64,
        )
        velocity_limits = np.asarray(
            [record["velocity_limit"] for record in arm_dofs],
            dtype=np.float64,
        )
        solver.set_robot_base_pose(
            base_position_array,
            base_orientation_array,
        )
        physics_dt = 1.0 / 60.0
        ik_waypoints: list[dict[str, Any]] = []
        ik_phase_summaries = {}
        phase_specs = [
            (
                "move_to_pregrasp",
                current_ee_position_array,
                pregrasp_ee_position,
            ),
            (
                "vertical_descent",
                pregrasp_ee_position,
                grasp_ee_position,
            ),
            (
                "vertical_lift",
                grasp_ee_position,
                lift_ee_position,
            ),
        ]
        previous_q = episode_window.qpos[lift_index, :6].copy()
        ik_status = "PASS"
        for phase_name, start_position, end_position in phase_specs:
            phase_waypoints, phase_summary = _solve_adaptive_linear_ik(
                solver=solver,
                frame_name=END_EFFECTOR_FRAME,
                start_position=start_position,
                end_position=end_position,
                orientation_wxyz=target_world_orientation,
                start_q=previous_q,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                velocity_limits=velocity_limits,
                physics_dt=physics_dt,
                phase=phase_name,
                position_tolerance=0.001,
                orientation_tolerance=0.005,
            )
            ik_phase_summaries[phase_name] = phase_summary
            if phase_summary["status"] != "PASS":
                ik_status = "FAIL"
                break
            ik_waypoints.extend(phase_waypoints)
            previous_q = np.asarray(
                phase_waypoints[-1]["joint_positions_rad"],
                dtype=np.float64,
            )
        if ik_status != "PASS":
            report["hard_blockers"].append("HARD_BLOCKER_HORIZONTAL_GRASP_IK_FEASIBILITY")

        descriptor_hash = _sha256(descriptor_path)
        kinematics_sections = {
            "bindings": {
                "episode": {
                    "path": str(episode_path),
                    "sha256": EXPECTED_EPISODE_SHA256,
                    "frames_inclusive": [208, 244],
                },
                "urdf": {
                    "path": str(urdf_path),
                    "sha256": urdf_hash,
                },
                "descriptor": {
                    "path": str(descriptor_path),
                    "sha256": descriptor_hash,
                },
                "joint_map": {
                    "path": str(joint_map_path),
                    "sha256": EXPECTED_JOINT_MAP_SHA256,
                },
                "stage": {
                    "path": str(stage_path),
                    "sha256": stage_hash_before,
                },
                "bottle_usd": {
                    "path": str(bottle_usd_path),
                    "sha256": EXPECTED_BOTTLE_USD_SHA256,
                },
                "articulation_path": ARTICULATION_PATH,
                "base_frame": "follower_left_base_link",
                "end_effector_frame": END_EFFECTOR_FRAME,
            },
            "episode_fk": {
                "status": "PASS",
                "record_count": len(episode_records),
                "records": episode_records,
                "lift_onset_usd_translation_residual_m": (onset_usd_translation_residual),
                "lift_onset_usd_rotation_residual_rad": (onset_usd_rotation_residual),
                "lift_onset_requested_qpos_arm_6d": [float(value) for value in episode_window.qpos[lift_index, :6]],
                "lift_onset_runtime_readback_arm_6d": [float(value) for value in onset_readback[:6]],
                "lift_onset_requested_readback_error_rad": [
                    float(value) for value in (onset_readback[:6] - episode_window.qpos[lift_index, :6])
                ],
            },
            "lift_detection": {
                "status": "PASS",
                "lift_onset_frame": lift_detection.lift_onset_frame,
                "threshold_m": lift_detection.threshold,
                "baseline_median_m": lift_detection.baseline_median,
                "baseline_mad_m": lift_detection.baseline_mad,
                "raw_baseline_median_m": (lift_detection.raw_baseline_median),
                "raw_baseline_mad_m": lift_detection.raw_baseline_mad,
                "directional_baseline": ("median(max(delta_z,0)) + 5 * MAD(max(delta_z,0))"),
                "directional_baseline_reason": (
                    "frames 208-222 contain intentional downward approach; negative descent is not upward lift noise"
                ),
                "close_command_start_frame": (gripper_phases.close_command_start_frame),
                "readback_response_start_frame": (gripper_phases.readback_response_start_frame),
                "candidates": list(lift_detection.candidates),
                "lift_distance_m": lift_distance_m,
            },
            "placement": {
                "status": ("PASS" if geometry_gate["status"] == "PASS" else "FAIL"),
                "source": ("FROZEN_SUPPLIER_CAD_CLEARANCE_FRAME_AND_EXACT_EPISODE18_POE"),
                "table": {
                    "prim_path": table_path,
                    "top_z_world_m": table_top_z,
                },
                "contact_offset_setup_gap": {
                    "value_m": setup_gap_m,
                    "authored_value_count": len(authored_offsets),
                    "status": (
                        "RUNTIME_AUTHORED_READBACK" if authored_offsets else "UNAUTHORED_READBACK_NONE_ZERO_SETUP"
                    ),
                },
                "supplier_cad_finger_geometry": {
                    "ee_frame": END_EFFECTOR_FRAME,
                    "left_collider": left_finger_collider,
                    "right_collider": right_finger_collider,
                    "left_point_count": int(left_points.shape[0]),
                    "right_point_count": int(right_points.shape[0]),
                    "open_contact_region_gap_m": open_gap,
                    "contact_midpoint_world_m": [float(value) for value in contact_midpoint],
                    "method": ("USER_APPROVED_COMPLETE_GRIPPER_CLEARANCE_FRAME"),
                    "clearance_report": (clearance_frame["clearance_report"]),
                    "screenshot_gate": clearance_frame["screenshot_gate"],
                    "rejected_method": ("MINIMUM_COLLIDER_VERTEX_DISTANCE"),
                    "rejected_closest_left_world_m": [float(value) for value in rejected_left_point],
                    "rejected_closest_right_world_m": [float(value) for value in rejected_right_point],
                    "rejected_closest_gap_m": rejected_open_gap,
                },
                "bottle_collision_meshes": bottle_collision_records,
                "bottle_collision_envelope": {
                    "combined_point_count": int(bottle_points_local.shape[0]),
                    "combined_extents_m": [
                        float(value)
                        for value in (np.max(bottle_points_local, axis=0) - np.min(bottle_points_local, axis=0))
                    ],
                    "cad_maximum_diameter_m": 0.068,
                    "collision_is_exact_cad": False,
                },
                "bottle_axis": {
                    "status": geometry_gate["status"],
                    "a_world_m": [float(value) for value in placement.a_world],
                    "b_world_m": [float(value) for value in placement.b_world],
                    "unit_world": [float(value) for value in placement.axis_unit],
                    "length_m": 0.206,
                    "grasp_coordinate_m": 0.069,
                    "grasp_point_world_m": [float(value) for value in bottle_grasp_point],
                    "lowest_point_world_z_m": (placement.lowest_point_world_z),
                    "lowest_point_to_table_gap_m": (placement.lowest_point_world_z - table_top_z),
                },
                "geometry_gate": geometry_gate,
                "placement_matrix": [[float(value) for value in row] for row in placement.matrix],
                "target_poses": {
                    "pregrasp_ee_position_world_m": [float(value) for value in pregrasp_ee_position],
                    "grasp_ee_position_world_m": [float(value) for value in grasp_ee_position],
                    "lift_ee_position_world_m": [float(value) for value in lift_ee_position],
                    "orientation_world_wxyz": [float(value) for value in target_world_orientation],
                    "approach_direction_world": [0.0, 0.0, -1.0],
                    "lift_direction_world": [0.0, 0.0, 1.0],
                },
            },
            "ik": {
                "status": ik_status,
                "solver": "LulaKinematicsSolver",
                "position_tolerance_m": 0.001,
                "orientation_tolerance_rad": 0.005,
                "physics_dt_s": physics_dt,
                "velocity_limit_source": str(joint_map_path),
                "joint_order": EXPECTED_CSPACE,
                "joint_lower_limits_rad": [float(value) for value in lower_limits],
                "joint_upper_limits_rad": [float(value) for value in upper_limits],
                "joint_velocity_limits_rad_s": [float(value) for value in velocity_limits],
                "phase_summaries": ik_phase_summaries,
                "waypoints": ik_waypoints,
                "diagnostic_only": True,
            },
        }

        stage_hash_after = _sha256(stage_path)
        stage_immutable = stage_hash_after == stage_hash_before
        report.update(
            {
                "status": (
                    "PASS"
                    if correspondence_pass and stage_immutable and solve_contact_last_readback is True
                    else "FAIL"
                ),
                "runtime": {
                    "isaac_sim": isaac_runtime_version,
                    "kit": kit_runtime_version,
                    "physx": normalized_physx_version,
                    "python": platform.python_version(),
                    "motion_generation_extension": (
                        str(motion_runtime_version).split("+", maxsplit=1)[0]
                        if motion_runtime_version
                        else source_version
                    ),
                },
                "stage": {
                    "path": str(stage_path),
                    "sha256_before": stage_hash_before,
                    "sha256_after": stage_hash_after,
                    "immutable": stage_immutable,
                    "root_prim": "/World",
                    "articulation_path": ARTICULATION_PATH,
                    "base_link_path": BASE_LINK_PATH,
                    "end_effector_path": END_EFFECTOR_PATH,
                },
                "inputs": {
                    "descriptor_path": str(descriptor_path),
                    "descriptor_sha256": _sha256(descriptor_path),
                    "urdf_path": str(urdf_path),
                    "urdf_sha256": urdf_hash,
                },
                "official_api_evidence": {
                    "source": "MCPJungle Gateway NVIDIA official Isaac capability",
                    "status": "QUERIED_AND_LOCAL_SOURCE_CROSS_CHECKED",
                    "symbols": [
                        (
                            "isaacsim.robot_motion.motion_generation."
                            "articulation_kinematics_solver."
                            "ArticulationKinematicsSolver."
                            "compute_inverse_kinematics"
                        ),
                        ("isaacsim.robot_motion.motion_generation.lula.kinematics.LulaKinematicsSolver"),
                        ("isaacsim.core.api.physics_context.PhysicsContext.set_solve_articulation_contact_last"),
                        "PhysxSchema.PhysxContactReportAPI",
                        "PhysicsSchemaTools.intToSdfPath",
                    ],
                },
                "local_source_inventory": sources,
                "api_readback": {
                    "motion_generation_extension_id": motion_extension_id,
                    "physx_extension_id": physx_extension_id,
                    "lula_constructor_signature": str(inspect.signature(LulaKinematicsSolver)),
                    "lula_fk_signature": str(inspect.signature(LulaKinematicsSolver.compute_forward_kinematics)),
                    "articulation_ik_signature": str(
                        inspect.signature(ArticulationKinematicsSolver.compute_inverse_kinematics)
                    ),
                    "contact_report_schema_attributes": list(
                        PhysxSchema.PhysxContactReportAPI.GetSchemaAttributeNames()
                    ),
                    "int_to_sdf_path_callable": callable(PhysicsSchemaTools.intToSdfPath),
                    "get_contact_report_callable": callable(get_physx_simulation_interface().get_contact_report),
                    "subscribe_contact_report_events_callable": callable(
                        get_physx_simulation_interface().subscribe_contact_report_events
                    ),
                    "solve_articulation_contact_last": (solve_contact_last_readback),
                },
                "constructor_probe": {
                    "status": "PASS",
                    "expected_cspace": EXPECTED_CSPACE,
                    "solver_cspace": solver_joint_names,
                    "end_effector_frame": END_EFFECTOR_FRAME,
                    "end_effector_frame_present": True,
                    "all_frame_count": len(solver_frames),
                },
                "runtime_dof_order": runtime_dof_order,
                "fk_correspondence": {
                    "status": "PASS" if correspondence_pass else "FAIL",
                    "translation_gate_m": TRANSLATION_GATE_M,
                    "rotation_gate_rad": ROTATION_GATE_RAD,
                    "source_contract": {
                        "interbotix_product": POE_SOURCE_CLASS,
                        "source_file": POE_SOURCE_FILE,
                        "source_sha256": POE_SOURCE_SHA256,
                        "frame": END_EFFECTOR_FRAME,
                    },
                    "cases": fk_records,
                },
                **kinematics_sections,
            }
        )
        report["status"] = (
            "PASS"
            if report["status"] == "PASS" and geometry_gate["status"] == "PASS" and ik_status == "PASS"
            else "PARTIAL"
            if correspondence_pass and stage_immutable
            else "FAIL"
        )
        _atomic_json(output_path, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "output": str(output_path),
                    "constructor_probe": report["constructor_probe"]["status"],
                    "fk_correspondence": report["fk_correspondence"]["status"],
                    "hard_blockers": report["hard_blockers"],
                },
                sort_keys=True,
            )
        )
        return 0 if report["status"] == "PASS" else 1
    except Exception as exc:
        report["status"] = "FAIL"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        if not report["hard_blockers"]:
            report["hard_blockers"].append("HARD_BLOCKER_KINEMATICS_PROBE_EXECUTION")
        _atomic_json(output_path, report)
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "output": str(output_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "hard_blockers": report["hard_blockers"],
                },
                sort_keys=True,
            )
        )
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
