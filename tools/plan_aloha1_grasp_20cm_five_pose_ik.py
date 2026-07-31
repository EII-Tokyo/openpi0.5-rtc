#!/usr/bin/env python3
"""Freeze five diverse Bottle500/arm-start records using Isaac Sim 5.1."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np
import yaml

from tools.aloha1_mapping.task7a_swept_collision import classify_contact_observation
from tools.aloha1_mapping.task7a_swept_collision import classify_contact_pair

ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION = "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    return parser.parse_args()


def _resolve_record_path(record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve(strict=True)


def freeze_preflight_records(
    records: Sequence[Mapping[str, Any]],
    *,
    required: int,
) -> list[dict[str, Any]]:
    """Freeze passing preflight records without consulting runtime outcomes."""

    count = int(required)
    if count < 1:
        raise ValueError("required must be positive")
    selected = [
        copy.deepcopy(dict(record))
        for record in records
        if record.get("preflight_status") == "PASS"
    ][:count]
    if len(selected) != count:
        raise ValueError(
            f"required {count} preflight passes, found {len(selected)}"
        )
    return selected


def classify_preflight_contacts(
    contacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the user-confirmed Task 7A contact policy to setup contacts."""

    records: list[dict[str, Any]] = []
    unresolved_or_nonfinite = 0
    forbidden_physical = 0
    allowed_physical = 0
    for contact in contacts:
        actor0 = str(
            contact.get("actor0_path")
            or contact.get("collider0_path")
            or ""
        )
        actor1 = str(
            contact.get("actor1_path")
            or contact.get("collider1_path")
            or ""
        )
        try:
            separation = float(contact["separation_m"])
            impulse = float(contact["impulse_ns"])
        except (KeyError, TypeError, ValueError):
            separation = math.nan
            impulse = math.nan
        finite = math.isfinite(separation) and math.isfinite(impulse)
        pair = classify_contact_pair(actor0, actor1, adjacent_body_pairs=())
        observed = classify_contact_observation(
            base_classification=str(pair["classification"]),
            base_allowed=bool(pair["allowed"]),
            minimum_separation_m=separation if finite else None,
            maximum_impulse_norm_n_s=impulse,
        )
        physical = bool(observed["physical_contact"])
        allowed = bool(observed["allowed"])
        if not finite:
            unresolved_or_nonfinite += 1
        elif physical and allowed:
            allowed_physical += 1
        elif physical:
            forbidden_physical += 1
        records.append(
            {
                "actor0_path": actor0,
                "actor1_path": actor1,
                "separation_m": separation,
                "impulse_ns": impulse,
                "finite": finite,
                "physical_contact": physical,
                "allowed": allowed,
                "classification": observed["classification"],
                "geometric_classification": observed[
                    "geometric_classification"
                ],
            }
        )
    status = (
        "PASS"
        if unresolved_or_nonfinite == 0 and forbidden_physical == 0
        else "FAIL"
    )
    unique_records = {
        (
            record["actor0_path"],
            record["actor1_path"],
            record["classification"],
            record["allowed"],
        ): record
        for record in records
    }
    return {
        "status": status,
        "contact_record_count": len(records),
        "unique_contact_pair_count": len(unique_records),
        "allowed_physical_contact_count": allowed_physical,
        "forbidden_physical_contact_count": forbidden_physical,
        "unresolved_or_nonfinite_count": unresolved_or_nonfinite,
        "minimum_separation_m": min(
            (
                float(record["separation_m"])
                for record in records
                if record["finite"]
            ),
            default=None,
        ),
        "maximum_impulse_ns": max(
            (
                float(record["impulse_ns"])
                for record in records
                if record["finite"]
            ),
            default=0.0,
        ),
        "unique_records": list(unique_records.values()),
    }


def _matrix_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    from tools.validate_aloha1_task7b2_horizontal_grasp import _rotation_matrix_to_quaternion_wxyz

    return _rotation_matrix_to_quaternion_wxyz(matrix)


def _condense_ik(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": str(result["status"]),
        "failure_phase": result.get("failure_phase"),
        "source": result.get("source"),
        "classification": result.get("classification"),
        "pregrasp_position_world_m": result.get(
            "pregrasp_position_world_m"
        ),
        "grasp_position_world_m": result.get(
            "grasp_position_world_m"
        ),
        "lift_position_world_m": result.get(
            "lift_position_world_m"
        ),
        "target_orientation_world_wxyz": result.get(
            "target_orientation_world_wxyz"
        ),
        "phase_summaries": result.get("phase_summaries"),
        "waypoint_count": len(result.get("waypoints", [])),
    }


def _transform_points(
    points: np.ndarray,
    world_from_object: np.ndarray,
) -> np.ndarray:
    return (
        points @ world_from_object[:3, :3].T
        + world_from_object[:3, 3]
    )


def _runtime_validate_initial_pose(
    *,
    app: Any,
    runtime_profile: Mapping[str, Any],
    stage_path: Path,
    stage_hash: str,
    artifact_root: Path,
    world_from_object: np.ndarray,
    initial_arm_q_rad: np.ndarray,
    initial_pose_hold_frames: int,
    readback_tolerance_rad: float,
    first_frame_jump_tolerance_rad: float,
) -> dict[str, Any]:
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.utils.stage import open_stage

    from tools.aloha1_mapping.grasp_20cm_controller import Phase
    from tools.aloha1_mapping.grasp_20cm_isaac_bindings import IsaacGrasp20cmBindings
    from tools.aloha1_mapping.grasp_20cm_runtime import sha256_file
    from tools.aloha1_mapping.grasp_20cm_runtime import validate_composed_stage

    if not open_stage(str(stage_path)):
        raise RuntimeError(f"failed to reopen approved Stage: {stage_path}")
    stage = get_current_stage()
    config = runtime_profile["config"]
    validate_composed_stage(
        stage=stage,
        expected_root_prim=str(config["stage"]["root_prim"]),
        required_prims=[
            str(config["stage"]["articulation_prim"]),
            str(config["stage"]["table_prim"]),
        ],
    )
    if sha256_file(stage_path) != stage_hash:
        raise RuntimeError("approved Stage hash changed before candidate setup")
    bindings = IsaacGrasp20cmBindings(
        app=app,
        profile=runtime_profile,
        artifact_root=artifact_root,
        delegate_readback={
            "path": "/app/useFabricSceneDelegate",
            "requested": False,
            "effective": False,
            "purpose": "HEADLESS_FIVE_POSE_INITIAL_STATE_PREFLIGHT",
        },
        bottle_world_from_object=world_from_object.tolist(),
        initial_arm_q_rad=initial_arm_q_rad.tolist(),
        initial_pose_hold_frames=initial_pose_hold_frames,
        capture_collider_evidence=False,
    )
    contacts: list[dict[str, Any]] = []
    q_records: list[np.ndarray] = []
    observations = []
    try:
        bindings._phase = Phase.SETUP_KINEMATIC  # noqa: SLF001
        bindings.apply_phase_target(Phase.SETUP_KINEMATIC)
        for frame in range(1, initial_pose_hold_frames + 1):
            bindings.world.step(render=False)
            observation = bindings.read_observation(
                frame=frame,
                time_s=frame * bindings.dt,
            )
            observations.append(observation)
            telemetry = bindings.telemetry[-1]
            q_records.append(
                np.asarray(
                    telemetry["joint_readback"],
                    dtype=np.float64,
                )
            )
            contacts.extend(telemetry["contacts"])
            if frame < initial_pose_hold_frames:
                bindings.apply_phase_target(Phase.SETUP_KINEMATIC)
        final_q = q_records[-1]
        arm_indices = np.asarray(
            bindings.arm_dof_indices,
            dtype=np.int64,
        )
        target = np.asarray(
            bindings.initial_command,
            dtype=np.float64,
        )
        maximum_arm_error = max(
            float(np.max(np.abs(record[arm_indices] - target[arm_indices])))
            for record in q_records
        )
        first_frame_jump = float(
            bindings.initial_pose_evidence["first_frame_jump_rad"]
        )
        contact_report = classify_preflight_contacts(contacts)
        persistent_penetration = any(
            observation.persistent_penetration
            for observation in observations
        )
        finite = bool(
            all(observation.finite_state for observation in observations)
            and np.isfinite(np.asarray(q_records)).all()
        )
        gates = {
            "hold_frame_count": (
                bindings._initial_pose_hold_observed_frames  # noqa: SLF001
                == initial_pose_hold_frames
            ),
            "setup_complete": bool(bindings._setup_complete),  # noqa: SLF001
            "readback": maximum_arm_error <= readback_tolerance_rad,
            "first_frame_jump": (
                first_frame_jump <= first_frame_jump_tolerance_rad
            ),
            "finite": finite,
            "contact_policy": contact_report["status"] == "PASS",
            "persistent_penetration": not persistent_penetration,
            "stage_hash": sha256_file(stage_path) == stage_hash,
        }
        return {
            "status": "PASS" if all(gates.values()) else "FAIL",
            "gates": gates,
            "dof_order": list(bindings.articulation.dof_names),
            "initial_arm_q_target_rad": target[arm_indices].tolist(),
            "initial_arm_q_readback_final_rad": final_q[
                arm_indices
            ].tolist(),
            "maximum_arm_readback_error_rad": maximum_arm_error,
            "readback_tolerance_rad": readback_tolerance_rad,
            "first_frame_jump_rad": first_frame_jump,
            "first_frame_jump_tolerance_rad": (
                first_frame_jump_tolerance_rad
            ),
            "hold_frames_required": initial_pose_hold_frames,
            "hold_frames_observed": (
                bindings._initial_pose_hold_observed_frames  # noqa: SLF001
            ),
            "initial_ee_position_world_m": bindings.initial_pose_evidence[
                "initial_ee_position_world_m"
            ],
            "initial_ee_orientation_world_wxyz": (
                bindings.initial_pose_evidence[
                    "initial_ee_orientation_world_wxyz"
                ]
            ),
            "initial_collision": contact_report,
            "persistent_penetration": persistent_penetration,
            "finite": finite,
        }
    finally:
        subscription = getattr(bindings, "contact_subscription", None)
        if subscription is not None and hasattr(subscription, "unsubscribe"):
            subscription.unsubscribe()


def main() -> int:
    args = _parse_args()
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(ROOT))

    from tools.aloha1_mapping.grasp_20cm_runtime import load_and_verify_config
    from tools.aloha1_mapping.grasp_20cm_runtime import sha256_file

    config_path = args.config.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 2:
        raise RuntimeError("unsupported five-pose config schema")
    if config.get("classification") != CLASSIFICATION:
        raise RuntimeError("unexpected five-pose classification")
    if config["boundaries"]["task8"] != "NOT_RUN":
        raise RuntimeError("Task 8 boundary changed")

    frozen_paths: dict[str, Path] = {}
    for name, record in config["frozen_inputs"].items():
        path = _resolve_record_path(record)
        actual = sha256_file(path)
        if actual != str(record["sha256"]):
            raise RuntimeError(
                f"frozen input hash mismatch for {name}: {actual}"
            )
        frozen_paths[str(name)] = path
    runtime_profile = load_and_verify_config(
        frozen_paths["runtime_config"],
        project_root=ROOT,
    )
    stage_path = frozen_paths["approved_stage"]
    stage_hash_before = sha256_file(stage_path)
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "create_new_stage": False,
            "width": 640,
            "height": 360,
        }
    )
    report: dict[str, Any]
    try:
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.stage import open_stage
        from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver

        from tools.aloha1_mapping.grasp_20cm_five_pose_ik import apply_frozen_bottle_transform
        from tools.aloha1_mapping.grasp_20cm_five_pose_ik import canonical_five_pose_signature
        from tools.aloha1_mapping.grasp_20cm_five_pose_ik import derive_sample_geometry
        from tools.aloha1_mapping.grasp_20cm_five_pose_ik import line_yaw_distance_deg
        from tools.aloha1_mapping.grasp_20cm_five_pose_ik import place_bottle_center_and_yaw
        from tools.aloha1_mapping.grasp_20cm_five_pose_ik import sample_bottle_center_yaw_candidates
        from tools.aloha1_mapping.grasp_20cm_five_pose_ik import sample_initial_arm_joint_candidates
        from tools.aloha1_mapping.grasp_20cm_isaac_bindings import IsaacGrasp20cmBindings
        from tools.aloha1_mapping.grasp_20cm_runtime import validate_composed_stage
        from tools.aloha1_mapping.grasp_20cm_sampling import extend_profile_for_clearance_lift
        from tools.validate_aloha1_task7b2_horizontal_grasp import _solve_settled_bottle_runtime_ik
        from tools.validate_aloha1_task7b2_horizontal_grasp import _world_bounds

        if not open_stage(str(stage_path)):
            raise RuntimeError(f"failed to open approved Stage: {stage_path}")
        stage = get_current_stage()
        runtime_config = runtime_profile["config"]
        validate_composed_stage(
            stage=stage,
            expected_root_prim=str(runtime_config["stage"]["root_prim"]),
            required_prims=[
                str(runtime_config["stage"]["articulation_prim"]),
                str(runtime_config["stage"]["table_prim"]),
            ],
        )
        nominal_bindings = IsaacGrasp20cmBindings(
            app=app,
            profile=runtime_profile,
            artifact_root=artifact_root / "nominal_probe",
            delegate_readback={
                "path": "/app/useFabricSceneDelegate",
                "requested": False,
                "effective": False,
                "purpose": "HEADLESS_FIVE_POSE_GEOMETRY_PROBE",
            },
            capture_collider_evidence=False,
        )
        dof_order = list(nominal_bindings.articulation.dof_names)
        properties = nominal_bindings.articulation.dof_properties
        lower = np.asarray(properties["lower"][:6], dtype=np.float64)
        upper = np.asarray(properties["upper"][:6], dtype=np.float64)
        joint_map = yaml.safe_load(
            frozen_paths["joint_map"].read_text(encoding="utf-8")
        )
        mapped_dofs = joint_map["robots"]["follower_left"]["dofs"][:6]
        mapped_order = [str(record["name"]) for record in mapped_dofs]
        mapped_lower = np.asarray(
            [record["isaac_runtime"]["lower"] for record in mapped_dofs],
            dtype=np.float64,
        )
        mapped_upper = np.asarray(
            [record["isaac_runtime"]["upper"] for record in mapped_dofs],
            dtype=np.float64,
        )
        joint_map_gate = bool(
            dof_order[:6] == mapped_order
            and np.allclose(lower, mapped_lower, rtol=0.0, atol=1.0e-6)
            and np.allclose(upper, mapped_upper, rtol=0.0, atol=1.0e-6)
        )
        if not joint_map_gate:
            raise RuntimeError("runtime arm limits do not match frozen joint map")

        table_bounds = _world_bounds(
            stage,
            str(runtime_config["stage"]["table_prim"]),
        )
        left_base_bounds = _world_bounds(
            stage,
            "/World/follower_left/vx300s_left/follower_left_base_link",
        )
        right_base_bounds = _world_bounds(
            stage,
            "/World/follower_right/vx300s_right/follower_right_base_link",
        )
        free_min = np.asarray(
            [
                max(
                    table_bounds["minimum"][0],
                    left_base_bounds["maximum"][0],
                ),
                table_bounds["minimum"][1],
            ],
            dtype=np.float64,
        )
        free_max = np.asarray(
            [
                min(
                    table_bounds["maximum"][0],
                    right_base_bounds["minimum"][0],
                ),
                table_bounds["maximum"][1],
            ],
            dtype=np.float64,
        )
        collision_points = np.asarray(
            nominal_bindings.bottle_collision_points_local,
            dtype=np.float64,
        )
        center_local = np.asarray(
            config["geometry"]["bottle_geometric_center_local_m"],
            dtype=np.float64,
        )
        conservative_radius = float(
            np.max(np.linalg.norm(collision_points - center_local, axis=1))
        )
        center_bounds = {
            "minimum": (free_min + conservative_radius).tolist(),
            "maximum": (free_max - conservative_radius).tolist(),
        }
        if not (
            center_bounds["minimum"][0] < 0.0
            < center_bounds["maximum"][0]
            and center_bounds["minimum"][1] < 0.0
            < center_bounds["maximum"][1]
        ):
            raise RuntimeError(
                "conservative Bottle500 center region lacks required signs"
            )
        nominal_world_object = np.asarray(
            nominal_bindings.task_profile["kinematics"]["placement"][
                "placement_matrix"
            ],
            dtype=np.float64,
        )
        object_gripper = np.asarray(
            nominal_bindings.task_profile["kinematics"]["placement"][
                "target_poses"
            ]["object_from_gripper"],
            dtype=np.float64,
        )
        axis_a_local = np.asarray(
            config["geometry"]["bottle_cad_axis"]["a_local_m"],
            dtype=np.float64,
        )
        axis_b_local = np.asarray(
            config["geometry"]["bottle_cad_axis"]["b_local_m"],
            dtype=np.float64,
        )
        nominal_geometry = derive_sample_geometry(
            world_from_object=nominal_world_object,
            a_local_m=axis_a_local,
            b_local_m=axis_b_local,
            object_from_gripper=object_gripper,
        )
        nominal_yaw = float(nominal_geometry["line_yaw_deg"])
        base_position = np.asarray(
            nominal_bindings.base_position,
            dtype=np.float64,
        )
        base_orientation = np.asarray(
            nominal_bindings.base_orientation,
            dtype=np.float64,
        )
        fk_solver = LulaKinematicsSolver(
            robot_description_path=str(
                nominal_bindings.profile["frozen_inputs"][
                    "lula_descriptor"
                ]["absolute_path"]
            ),
            urdf_path=str(
                nominal_bindings.task_profile["inputs"][
                    "follower_left_urdf"
                ]
            ),
        )
        fk_solver.set_robot_base_pose(base_position, base_orientation)
        formal_profile = extend_profile_for_clearance_lift(
            nominal_bindings.task_profile,
            target_clearance_m=float(
                runtime_config["target"]["clearance_m"]
            ),
            hold_drop_gate_m=float(
                runtime_config["target"]["hold_drop_gate_m"]
            ),
        )
        nominal_subscription = getattr(
            nominal_bindings,
            "contact_subscription",
            None,
        )
        if (
            nominal_subscription is not None
            and hasattr(nominal_subscription, "unsubscribe")
        ):
            nominal_subscription.unsubscribe()
        candidate_count = int(config["sampling"]["candidate_count"])
        seed = int(config["sampling"]["seed"])
        q_candidates = sample_initial_arm_joint_candidates(
            lower_limits=lower,
            upper_limits=upper,
            seed=seed,
            count=candidate_count,
        )
        bottle_candidates = [
            sample_bottle_center_yaw_candidates(
                center_xy_bounds=center_bounds,
                yaw_domain_deg=config["sampling"][
                    "bottle_line_yaw_domain_deg"
                ],
                seed=seed,
                count=candidate_count,
                formal_sample_index=sample_index,
            )
            for sample_index in range(5)
        ]
        selected: list[dict[str, Any]] = []
        candidate_results: list[dict[str, Any]] = []
        failure_counts: dict[str, int] = {}
        table_top_z = float(table_bounds["maximum"][2])
        yaw_gate = float(
            config["gates"]["minimum_bottle_line_yaw_separation_deg"]
        )
        ee_gate = float(config["gates"]["minimum_initial_ee_separation_m"])
        axis_gate = float(
            config["gates"]["axis_to_table_normal_tolerance_deg"]
        )
        gap_gate = float(
            config["gates"]["support_contact_latch_clearance_m"]
        )
        readback_gate = float(
            config["gates"]["initial_arm_readback_tolerance_rad"]
        )
        jump_gate = float(
            config["gates"]["first_frame_jump_tolerance_rad"]
        )
        hold_frames = int(config["runtime"]["initial_pose_hold_frames"])
        end_effector_frame = formal_profile["config"]["robot"][
            "end_effector_frame"
        ]

        for candidate_index in range(candidate_count):
            if len(selected) == int(config["sampling"]["formal_sample_count"]):
                break
            sample_index = len(selected)
            sample_id = f"sample_{sample_index + 1:02d}"
            bottle_candidate = bottle_candidates[sample_index][
                candidate_index
            ]
            desired_yaw = float(
                bottle_candidate["bottle_line_yaw_deg"]
            )
            world_object = place_bottle_center_and_yaw(
                nominal_world_from_object=nominal_world_object,
                geometric_center_local_m=center_local,
                desired_center_xy_m=bottle_candidate[
                    "bottle_center_xy_m"
                ],
                yaw_delta_rad=math.radians(desired_yaw - nominal_yaw),
            )
            geometry = derive_sample_geometry(
                world_from_object=world_object,
                a_local_m=axis_a_local,
                b_local_m=axis_b_local,
                object_from_gripper=object_gripper,
            )
            center_world = (
                world_object[:3, :3] @ center_local
                + world_object[:3, 3]
            )
            world_points = _transform_points(
                collision_points,
                world_object,
            )
            bottle_min = world_points.min(axis=0)
            bottle_max = world_points.max(axis=0)
            inside = bool(
                np.all(bottle_min[:2] >= free_min)
                and np.all(bottle_max[:2] <= free_max)
            )
            lowest_gap = float(bottle_min[2] - table_top_z)
            axis_horizontal = bool(
                abs(float(geometry["axis_to_world_z_deg"]) - 90.0)
                <= axis_gate
            )
            centerline = bool(
                sample_index not in {0, 3}
                or abs(float(center_world[0]))
                <= float(config["gates"]["bottle_centerline_residual_m"])
            )
            q = q_candidates[candidate_index]
            ee_position, ee_rotation = fk_solver.compute_forward_kinematics(
                end_effector_frame,
                q,
            )
            ee_position = np.asarray(ee_position, dtype=np.float64)
            ee_rotation = np.asarray(ee_rotation, dtype=np.float64)
            ee_orientation = _matrix_quaternion_wxyz(ee_rotation)
            finite_fk = bool(
                np.isfinite(ee_position).all()
                and np.isfinite(ee_rotation).all()
                and np.isfinite(q).all()
            )
            ee_above_table = bool(
                finite_fk and float(ee_position[2]) > table_top_z
            )
            yaw_margin = min(
                (
                    line_yaw_distance_deg(
                        float(geometry["line_yaw_deg"]),
                        float(record["bottle_line_yaw_deg"]),
                    )
                    for record in selected
                ),
                default=math.inf,
            )
            ee_margin = min(
                (
                    float(
                        np.linalg.norm(
                            ee_position
                            - np.asarray(
                                record[
                                    "initial_ee_position_world_m"
                                ],
                                dtype=np.float64,
                            )
                        )
                    )
                    for record in selected
                ),
                default=math.inf,
            )
            geometry_pass = bool(
                inside
                and abs(lowest_gap) <= gap_gate
                and axis_horizontal
                and centerline
                and finite_fk
                and ee_above_table
                and yaw_margin + 1.0e-12 >= yaw_gate
                and ee_margin + 1.0e-12 >= ee_gate
            )
            failure = None
            if not geometry_pass:
                failure = "geometry_fk_or_diversity"
                ik: dict[str, Any] = {
                    "status": "NOT_RUN_GEOMETRY_GATE"
                }
                runtime_initial: dict[str, Any] = {
                    "status": "NOT_RUN_GEOMETRY_GATE"
                }
            else:
                candidate_profile = apply_frozen_bottle_transform(
                    formal_profile,
                    world_from_object=world_object,
                )
                ik = _solve_settled_bottle_runtime_ik(
                    candidate_profile,
                    base_position=base_position,
                    base_orientation=base_orientation,
                    bottle_state={
                        "position_world_m": world_object[:3, 3].tolist(),
                        "orientation_wxyz": _matrix_quaternion_wxyz(
                            world_object[:3, :3]
                        ).tolist(),
                    },
                    current_ee_position=ee_position,
                    current_ee_orientation=ee_orientation,
                    current_arm_q=q,
                )
                if ik["status"] != "PASS":
                    failure = "ik"
                    runtime_initial = {"status": "NOT_RUN_IK_GATE"}
                else:
                    runtime_initial = _runtime_validate_initial_pose(
                        app=app,
                        runtime_profile=runtime_profile,
                        stage_path=stage_path,
                        stage_hash=stage_hash_before,
                        artifact_root=(
                            artifact_root
                            / "candidate_initial_state"
                            / f"candidate_{candidate_index:03d}"
                        ),
                        world_from_object=world_object,
                        initial_arm_q_rad=q,
                        initial_pose_hold_frames=hold_frames,
                        readback_tolerance_rad=readback_gate,
                        first_frame_jump_tolerance_rad=jump_gate,
                    )
                    if runtime_initial["status"] != "PASS":
                        failure = "runtime_initial_state"
            record = {
                "sample_id": sample_id,
                "candidate_index": candidate_index,
                "seed": seed,
                "bottle_geometric_center_world_m": center_world.tolist(),
                "bottle_line_yaw_deg": float(geometry["line_yaw_deg"]),
                "world_from_object": world_object.tolist(),
                "a_world_m": geometry["a_world_m"],
                "b_world_m": geometry["b_world_m"],
                "axis_unit_world": geometry["axis_unit_world"],
                "axis_to_world_z_deg": geometry[
                    "axis_to_world_z_deg"
                ],
                "lowest_point_to_table_gap_m": lowest_gap,
                "bottle_xy_bounds_world_m": {
                    "minimum": bottle_min[:2].tolist(),
                    "maximum": bottle_max[:2].tolist(),
                },
                "full_bottle_inside_free_surface": inside,
                "centerline_residual_m": (
                    abs(float(center_world[0]))
                    if sample_index in {0, 3}
                    else None
                ),
                "initial_arm_q_rad": q.tolist(),
                "initial_ee_position_world_m": ee_position.tolist(),
                "initial_ee_orientation_world_wxyz": (
                    ee_orientation.tolist()
                ),
                "minimum_prior_yaw_separation_deg": yaw_margin,
                "minimum_prior_ee_distance_m": ee_margin,
                "joint_limits": {
                    "lower_rad": lower.tolist(),
                    "upper_rad": upper.tolist(),
                    "within_limits": bool(
                        np.all(q >= lower) and np.all(q <= upper)
                    ),
                },
                "geometry_gates": {
                    "inside_free_surface": inside,
                    "support_gap": abs(lowest_gap) <= gap_gate,
                    "axis_horizontal": axis_horizontal,
                    "centerline": centerline,
                    "finite_fk": finite_fk,
                    "ee_above_table": ee_above_table,
                    "yaw_diversity": yaw_margin + 1.0e-12 >= yaw_gate,
                    "ee_diversity": ee_margin + 1.0e-12 >= ee_gate,
                },
                "ik": (
                    ik if ik.get("status") == "PASS" else _condense_ik(ik)
                ),
                "initial_runtime": runtime_initial,
                "initial_collision": runtime_initial.get(
                    "initial_collision",
                    {"status": "NOT_RUN"},
                ),
                "preflight_status": (
                    "PASS"
                    if failure is None
                    else "FAIL"
                ),
                "failure_gate": failure,
            }
            candidate_results.append(
                record
                if record["preflight_status"] == "PASS"
                else {
                    **record,
                    "ik": _condense_ik(ik) if "status" in ik else ik,
                }
            )
            if failure is None:
                selected.append(record)
            else:
                failure_counts[failure] = failure_counts.get(failure, 0) + 1

        frozen = freeze_preflight_records(
            selected,
            required=int(config["sampling"]["formal_sample_count"]),
        )
        signature = canonical_five_pose_signature(frozen)
        stage_hash_after = sha256_file(stage_path)
        final_stage = get_current_stage()
        report = {
            "schema_version": 1,
            "status": (
                "PASS"
                if len(frozen)
                == int(config["sampling"]["formal_sample_count"])
                and stage_hash_after == stage_hash_before
                else "FAIL"
            ),
            "classification": CLASSIFICATION,
            "runtime": {
                "isaac_sim": "5.1.0.0",
                "kit": "107.3.3",
                "physx": "107.3.26",
                "ik": "LOCAL_LULA_5_1_WITH_WARM_START_AND_FK_READBACK",
            },
            "config": {
                "absolute_path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "stage": {
                "absolute_path": str(stage_path),
                "sha256_before": stage_hash_before,
                "sha256_after": stage_hash_after,
                "root_prim": str(final_stage.GetDefaultPrim().GetPath()),
                "sublayers": list(
                    final_stage.GetRootLayer().subLayerPaths
                ),
            },
            "joint_map_runtime_gate": {
                "status": "PASS" if joint_map_gate else "FAIL",
                "dof_order": dof_order,
                "mapped_arm_order": mapped_order,
                "runtime_lower_rad": lower.tolist(),
                "runtime_upper_rad": upper.tolist(),
                "mapped_lower_rad": mapped_lower.tolist(),
                "mapped_upper_rad": mapped_upper.tolist(),
            },
            "legal_geometry": {
                "table_bounds_world_m": table_bounds,
                "left_base_bounds_world_m": left_base_bounds,
                "right_base_bounds_world_m": right_base_bounds,
                "free_surface_xy_m": {
                    "minimum": free_min.tolist(),
                    "maximum": free_max.tolist(),
                },
                "conservative_cad_center_radius_m": conservative_radius,
                "candidate_center_xy_bounds_m": center_bounds,
                "derivation": (
                    "COMPOSED_TABLE_AND_BASE_AABBS_PLUS_MAXIMUM_"
                    "CAD_COLLISION_POINT_DISTANCE_FROM_GEOMETRIC_CENTER"
                ),
            },
            "candidate_count_generated": candidate_count,
            "candidate_count_preflighted": len(candidate_results),
            "failure_gate_counts": failure_counts,
            "selected_sample_count": len(frozen),
            "candidate_results": candidate_results,
            "selected_samples": frozen,
            "minimum_pairwise_yaw_separation_deg": min(
                line_yaw_distance_deg(
                    float(first["bottle_line_yaw_deg"]),
                    float(second["bottle_line_yaw_deg"]),
                )
                for index, first in enumerate(frozen)
                for second in frozen[index + 1 :]
            ),
            "minimum_pairwise_initial_ee_distance_m": min(
                float(
                    np.linalg.norm(
                        np.asarray(
                            first["initial_ee_position_world_m"],
                            dtype=np.float64,
                        )
                        - np.asarray(
                            second["initial_ee_position_world_m"],
                            dtype=np.float64,
                        )
                    )
                )
                for index, first in enumerate(frozen)
                for second in frozen[index + 1 :]
            ),
            "deterministic_signature": signature,
            "semantics": {
                "formal_samples_frozen_before_runtime_grasp_outcomes": True,
                "runtime_status_used_for_selection": False,
                "sample_01_and_04_centerline": "BOTTLE_CAD_GEOMETRIC_CENTER_WORLD_X_ZERO",
                "bottle_rotation": "WORLD_Z_LINE_YAW_WITH_HORIZONTAL_ROLL_PRESERVED",
                "object_from_gripper": "UNCHANGED_VALIDATED_T_O_G",
            },
            "boundaries": {
                **config["boundaries"],
                "task8": "NOT_RUN",
            },
        }
    except Exception:
        report = {
            "schema_version": 1,
            "status": "FAIL",
            "classification": CLASSIFICATION,
            "reason": "exception",
            "exception": traceback.format_exc(limit=40)[-20000:],
            "stage": {
                "absolute_path": str(stage_path),
                "sha256_before": stage_hash_before,
                "sha256_after": sha256_file(stage_path),
            },
            "task8": "NOT_RUN",
        }
    finally:
        _atomic_json(args.output.resolve(), report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "output": str(args.output.resolve()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        app.close()
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
