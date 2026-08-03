#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Validate static coverage, overlaps, and first-frame stability in Isaac 5.1."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.cad_derived_collision_runtime import canonical_runtime_signature
from tools.aloha1_mapping.cad_derived_collision_runtime import classify_overlap_pair
from tools.aloha1_mapping.cad_derived_collision_runtime import load_frozen_pose_manifest
from tools.aloha1_mapping.cad_derived_collision_runtime import summarize_static_validation
from tools.aloha1_mapping.convex_geometry_audit import convex_pair_relation

ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    ROOT / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_derived_full_body_colliders/1.0/"
    "aloha1_cad_derived_full_body_collider_diagnostic.usda"
)
GRIPPER_DECOMP_STAGE = (
    ROOT / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_derived_full_body_colliders/1.0/"
    "aloha1_cad_derived_full_body_collider_gripper_decomposition_diagnostic.usda"
)
STAGE_REPORT = ROOT / "reports/aloha1_mapping/aloha1_cad_derived_collider_stage.json"
TASK8_CANDIDATE_REPORT = ROOT / "reports/aloha1_mapping/aloha1_task8_collider_lod_candidate.json"
FIVE_POSE_REPORT = ROOT / "reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_results_downward_contact_gate_v5.json"
OUTPUT_COVERAGE = ROOT / "reports/aloha1_mapping/aloha1_cad_derived_collision_coverage.json"
OUTPUT_OVERLAP = ROOT / "reports/aloha1_mapping/aloha1_cad_derived_initial_overlap.json"
ARTICULATION_PATHS = {
    "follower_left": "/World/follower_left/vx300s_left/root_joint",
    "follower_right": "/World/follower_right/vx300s_right/root_joint",
}
GRIPPER_STATES = {
    "open": (0.057, -0.057),
    "partially_closed": (0.035, -0.035),
    "closed": (0.021, -0.021),
    "maximum_legal_aperture": (0.057, -0.057),
}
FIRST_FRAME_JUMP_GATE_RAD = 0.020
OVERLAP_TOLERANCE_M = 1.0e-7


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument(
        "--profile",
        choices=(
            "compound_hull",
            "gripper_decomposition",
            "task8_fidelity",
            "task8_throughput",
        ),
        default="compound_hull",
    )
    parser.add_argument("--coverage", type=Path, default=OUTPUT_COVERAGE)
    parser.add_argument("--overlap", type=Path, default=OUTPUT_OVERLAP)
    parser.add_argument("--repeat-index", type=int, default=1)
    parser.add_argument("--pose-source", type=Path, default=FIVE_POSE_REPORT)
    parser.add_argument(
        "--scan-preflight-candidates",
        action="store_true",
        help="Scan every downward/task-IK-valid candidate in pose-source.",
    )
    parser.add_argument(
        "--enable-self-collision-diagnostic",
        action="store_true",
        help=(
            "Enable articulation self-collision only in the anonymous session "
            "layer; never use this mode for the default first-frame gate."
        ),
    )
    return parser.parse_args()


def _expected_source_counts(profile: str) -> dict[str, int]:
    cad_derived = 28 if profile == "task8_throughput" else 34
    return {
        "CAD_DERIVED": cad_derived,
        "SUPPLIER_CAD_FINGER": 4,
        "IMPORTER_BASELINE_FALLBACK": 4,
    }


def _load_pose_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.scan_preflight_candidates:
        return load_frozen_pose_manifest(args.pose_source)
    payload = json.loads(args.pose_source.resolve(strict=True).read_text(encoding="utf-8"))
    records = []
    for candidate in payload.get("candidate_results", []):
        values = candidate.get("initial_arm_q_rad")
        if not isinstance(values, list) or len(values) != 6:
            continue
        if candidate.get("initial_tool_orientation_gate", {}).get("status") != "PASS":
            continue
        if candidate.get("initial_task_space_ik", {}).get("status") != "PASS":
            continue
        records.append(
            {
                "pose_id": f"candidate_{int(candidate['candidate_index']):03d}",
                "candidate_index": int(candidate["candidate_index"]),
                "arm_q_rad": [float(value) for value in values],
                "source": "FROZEN_DOWNWARD_TASK_IK_VALID_CANDIDATE",
                "bottle_line_yaw_deg": float(candidate["bottle_line_yaw_deg"]),
                "initial_ee_position_world_m": [float(value) for value in candidate["initial_ee_position_world_m"]],
                "preflight_status": candidate["preflight_status"],
                "preflight_failure_gate": candidate.get("failure_gate"),
            }
        )
    if not records:
        raise ValueError("no downward/task-IK-valid candidates found")
    return records


def _runtime_versions(app: Any) -> dict[str, str]:
    import carb

    manager = app.get_extension_manager()
    physx_id = manager.get_enabled_extension_id("omni.physx")
    record = manager.get_extension_dict(physx_id) if physx_id else {}
    physx = record.get("package", {}).get("version", "")
    return {
        "isaac_sim": version("isaacsim"),
        "kit": str(carb.tokens.get_tokens_interface().resolve("${kit_version}")).split("+", maxsplit=1)[0],
        "physx": str(physx).split("+", maxsplit=1)[0],
    }


def _iter_instance_proxies(stage: Any) -> Any:
    from pxr import Usd

    return Usd.PrimRange(
        stage.GetPseudoRoot(),
        Usd.TraverseInstanceProxies(),
    )


def _rigid_body_owner(prim: Any) -> str:
    from pxr import UsdPhysics

    cursor = prim
    while cursor and cursor.IsValid() and not cursor.IsPseudoRoot():
        if cursor.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(cursor.GetPath())
        cursor = cursor.GetParent()
    return "/"


def _world_points(prim: Any) -> np.ndarray:
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    if not points:
        raise ValueError(f"mesh has no points: {prim.GetPath()}")
    matrix = np.asarray(
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0),
        dtype=np.float64,
    )
    local = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack([local, np.ones(len(local))])
    # Gf matrices use row-vector multiplication semantics.
    return (homogeneous @ matrix)[:, :3]


def _active_mesh_colliders(stage: Any) -> list[dict[str, Any]]:
    from pxr import UsdGeom
    from pxr import UsdPhysics

    records = []
    for prim in _iter_instance_proxies(stage):
        path = str(prim.GetPath())
        if not path.startswith(("/World/follower_left/", "/World/follower_right/")):
            continue
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if enabled is False:
            continue
        points = _world_points(prim)
        approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        records.append(
            {
                "collider_path": path,
                "actor_path": _rigid_body_owner(prim),
                "approximation": str(approximation),
                "point_count": len(points),
                "points_world_m": points,
                "aabb_min_world_m": points.min(axis=0).tolist(),
                "aabb_max_world_m": points.max(axis=0).tolist(),
                "source_kind": (
                    "CAD_DERIVED"
                    if "/cad_derived_collisions/" in path
                    else "SUPPLIER_CAD_FINGER"
                    if "diagnostic_supplier_cad_" in path
                    else "IMPORTER_BASELINE_FALLBACK"
                ),
            }
        )
    return sorted(records, key=lambda record: record["collider_path"])


def _adjacent_body_pairs(stage: Any) -> set[tuple[str, str]]:
    from pxr import UsdPhysics

    pairs = set()
    for prim in stage.Traverse():
        joint = UsdPhysics.Joint(prim)
        if not joint:
            continue
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if body0 and body1:
            pairs.add(tuple(sorted((str(body0[0]), str(body1[0])))))
    return pairs


def _documented_assembly_interface_pairs() -> set[tuple[str, str]]:
    """Return only pairs backed by supplier CAD or explicit gripper topology."""

    pairs = set()
    for side in ("left", "right"):
        root = f"/World/follower_{side}/vx300s_{side}/follower_{side}"
        gripper = f"{root}_gripper_link"
        pairs.update(
            {
                tuple(sorted((gripper, f"{root}_left_finger_link"))),
                tuple(sorted((gripper, f"{root}_right_finger_link"))),
                tuple(sorted((gripper, f"{root}_gripper_prop_link"))),
            }
        )
    return pairs


def _aabb_overlap(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_min = np.asarray(first["aabb_min_world_m"])
    first_max = np.asarray(first["aabb_max_world_m"])
    second_min = np.asarray(second["aabb_min_world_m"])
    second_max = np.asarray(second["aabb_max_world_m"])
    return bool(np.all(first_max >= second_min) and np.all(second_max >= first_min))


def _numerical_overlaps(
    colliders: Sequence[dict[str, Any]],
    adjacent: set[tuple[str, str]],
    assembly_interfaces: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    records = []
    for index, first in enumerate(colliders):
        robot = first["collider_path"].split("/")[2]
        for second in colliders[index + 1 :]:
            if second["collider_path"].split("/")[2] != robot:
                continue
            if first["actor_path"] == second["actor_path"]:
                continue
            if not _aabb_overlap(first, second):
                continue
            if "convexDecomposition" in {
                first["approximation"],
                second["approximation"],
            }:
                # A single-hull LP would recreate the exact over-envelope this
                # profile is intended to diagnose. PhysX contact readback and
                # first-frame displacement remain the runtime acceptance gates.
                continue
            try:
                relation = convex_pair_relation(
                    first["points_world_m"],
                    second["points_world_m"],
                    tolerance_m=OVERLAP_TOLERANCE_M,
                )
            except BaseException as exc:
                records.append(
                    {
                        "actor_pair": sorted([first["actor_path"], second["actor_path"]]),
                        "collider_pair": sorted([first["collider_path"], second["collider_path"]]),
                        "classification": "NUMERICAL_QUERY_FAILED",
                        "allowed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "overlap_volume_m3": math.nan,
                    }
                )
                continue
            classified = classify_overlap_pair(
                actor0=first["actor_path"],
                actor1=second["actor_path"],
                collider0=first["collider_path"],
                collider1=second["collider_path"],
                adjacent_body_pairs=adjacent,
                cad_assembly_interface_pairs=assembly_interfaces,
                relation=relation["relation"],
                overlap_volume_m3=relation["overlap_volume_m3"],
            )
            if classified["classification"] != "NONE":
                classified["signed_chebyshev_margin_m"] = relation["signed_chebyshev_margin_m"]
                classified["intersection_vertex_count"] = relation["intersection_vertex_count"]
                records.append(classified)
    return sorted(
        records,
        key=lambda record: (
            record["classification"],
            record["actor_pair"],
            record["collider_pair"],
        ),
    )


def _serialize_contacts(headers: Sequence[Any], data: Sequence[Any]) -> list[dict[str, Any]]:
    from pxr import PhysicsSchemaTools

    def path(value: Any) -> str:
        return str(PhysicsSchemaTools.intToSdfPath(value))

    records = []
    for header in headers:
        contacts = []
        start = int(header.contact_data_offset)
        end = start + int(header.num_contact_data)
        for index in range(start, end):
            item = data[index]
            impulse = np.asarray(item.impulse, dtype=np.float64)
            contacts.append(
                {
                    "position_world_m": [float(value) for value in item.position],
                    "normal": [float(value) for value in item.normal],
                    "impulse_n_s": impulse.tolist(),
                    "impulse_norm_n_s": float(np.linalg.norm(impulse)),
                    "separation_m": float(item.separation),
                }
            )
        records.append(
            {
                "event_type": str(header.type),
                "actor0": path(header.actor0),
                "actor1": path(header.actor1),
                "collider0": path(header.collider0),
                "collider1": path(header.collider1),
                "contacts": contacts,
            }
        )
    return records


def _install_contact_reports_and_self_collision(stage: Any, *, enable_self_collision: bool) -> dict[str, Any]:
    from pxr import PhysxSchema
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdPhysics

    layer = Sdf.Layer.CreateAnonymous("cad_collision_static_session.usda")
    stage.GetSessionLayer().subLayerPaths.append(layer.identifier)
    previous = stage.GetEditTarget()
    stage.SetEditTarget(Usd.EditTarget(layer))
    bodies = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
            bodies.append(str(prim.GetPath()))
    self_collision = {}
    for robot, path in ARTICULATION_PATHS.items():
        prim = stage.GetPrimAtPath(path)
        api = PhysxSchema.PhysxArticulationAPI(prim)
        attr = api.GetEnabledSelfCollisionsAttr()
        before = attr.Get()
        if enable_self_collision:
            api.CreateEnabledSelfCollisionsAttr().Set(True)  # noqa: FBT003
        self_collision[robot] = {
            "attribute": attr.GetName(),
            "before": before,
            "diagnostic_session_value": attr.Get(),
            "changed_in_session": enable_self_collision,
        }
    stage.SetEditTarget(previous)
    return {
        "session_only": True,
        "session_layer_identifier": layer.identifier,
        "contact_report_body_count": len(bodies),
        "self_collision": self_collision,
        "self_collision_diagnostic_enabled": enable_self_collision,
        "final_policy_modified": False,
    }


def _apply_full_state(
    articulation: Any,
    arm_q: Sequence[float],
    finger: tuple[float, float],
) -> np.ndarray:
    from isaacsim.core.utils.types import ArticulationAction

    order = list(articulation.dof_names)
    expected = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
        "left_finger",
        "right_finger",
    ]
    if order != expected:
        raise ValueError(f"explicit DOF order drift: {order}")
    values = np.asarray(articulation.get_joint_positions(), dtype=np.float32)
    for suffix, target in zip(
        ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"),
        arm_q,
        strict=True,
    ):
        values[order.index(suffix)] = float(target)
    values[order.index("left_finger")] = finger[0]
    values[order.index("right_finger")] = finger[1]
    articulation.set_joint_positions(values)
    articulation.set_joint_velocities(np.zeros_like(values))
    articulation.get_articulation_controller().apply_action(ArticulationAction(joint_positions=values))
    return values.astype(np.float64)


def main(args: argparse.Namespace) -> int:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import open_stage
    import omni.kit.app
    from omni.physx import get_physx_interface
    from omni.physx import get_physx_simulation_interface
    import omni.usd

    stage_report = json.loads(STAGE_REPORT.read_text(encoding="utf-8"))
    if args.profile == "compound_hull":
        expected_stage = STAGE
        expected_hash = stage_report["root_layer"]["sha256"]
        expected_physics_hash = stage_report["physics_layer"]["sha256"]
    elif args.profile == "gripper_decomposition":
        expected_stage = GRIPPER_DECOMP_STAGE
        variant = stage_report["gripper_decomposition_diagnostic_variant"]
        expected_hash = variant["root_layer"]["sha256"]
        expected_physics_hash = variant["physics_layer"]["sha256"]
    else:
        task8_candidate = json.loads(TASK8_CANDIDATE_REPORT.read_text(encoding="utf-8"))
        layer_key = (
            "throughput_profile"
            if args.profile == "task8_throughput"
            else "fidelity_profile"
        )
        expected_stage = Path(
            task8_candidate["layers"][layer_key]["absolute_path"]
        )
        expected_hash = task8_candidate["layers"][layer_key]["sha256"]
        variant = stage_report["gripper_decomposition_diagnostic_variant"]
        expected_physics_hash = variant["physics_layer"]["sha256"]
    stage_path = args.stage.resolve(strict=True)
    hash_before = _sha256(stage_path)
    if stage_path != expected_stage.resolve(strict=True) or hash_before != expected_hash:
        raise ValueError("diagnostic Stage path/hash is not the frozen Phase-4 input")
    geometry_hash = _sha256(Path(stage_report["geometry_layer"]["absolute_path"]))
    physics_path = (
        Path(stage_report["physics_layer"]["absolute_path"])
        if args.profile == "compound_hull"
        else Path(
            stage_report["gripper_decomposition_diagnostic_variant"]["physics_layer"][
                "absolute_path"
            ]
        )
    )
    if geometry_hash != stage_report["geometry_layer"]["sha256"]:
        raise ValueError("diagnostic geometry sublayer hash drift")
    if _sha256(physics_path) != expected_physics_hash:
        raise ValueError("diagnostic physics sublayer hash drift")
    if not open_stage(str(stage_path)):
        raise RuntimeError(f"failed to open {stage_path}")
    app = omni.kit.app.get_app()
    for _ in range(20):
        app.update()
    stage = omni.usd.get_context().get_stage()
    runtime = _runtime_versions(app)
    expected_runtime = {
        "isaac_sim": "5.1.0.0",
        "kit": "107.3.3",
        "physx": "107.3.26",
    }
    if runtime != expected_runtime:
        raise RuntimeError(f"runtime version drift: {runtime}")

    session = _install_contact_reports_and_self_collision(
        stage,
        enable_self_collision=args.enable_self_collision_diagnostic,
    )
    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
    )
    articulations = {}
    for robot, path in ARTICULATION_PATHS.items():
        articulation = SingleArticulation(
            prim_path=path,
            name=f"cad_collision_static_{robot}",
            reset_xform_properties=False,
        )
        world.scene.add(articulation)
        articulations[robot] = articulation

    current_contacts: list[dict[str, Any]] = []

    def on_contact(headers: Sequence[Any], data: Sequence[Any]) -> None:
        current_contacts.extend(_serialize_contacts(headers, data))

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    physx_interface = get_physx_interface()
    world.reset()
    pose_manifest = _load_pose_records(args)
    home_arm_q = load_frozen_pose_manifest(FIVE_POSE_REPORT)[0]["arm_q_rad"]
    adjacent = _adjacent_body_pairs(stage)
    assembly_interfaces = _documented_assembly_interface_pairs()
    pose_records = []
    coverage_snapshot = None
    gripper_states = {"open": GRIPPER_STATES["open"]} if args.scan_preflight_candidates else GRIPPER_STATES
    for pose in pose_manifest:
        for gripper_state, finger in gripper_states.items():
            current_contacts.clear()
            world.reset()
            injected = {}
            for robot, articulation in articulations.items():
                arm_q = pose["arm_q_rad"] if robot == "follower_left" else home_arm_q
                injected[robot] = _apply_full_state(articulation, arm_q, finger)
            physx_interface.update_transformations(True, True, False, False)  # noqa: FBT003
            colliders = _active_mesh_colliders(stage)
            if coverage_snapshot is None:
                coverage_snapshot = colliders
            overlaps = _numerical_overlaps(
                colliders,
                adjacent,
                assembly_interfaces,
            )
            before = {
                robot: np.asarray(articulation.get_joint_positions(), dtype=np.float64)
                for robot, articulation in articulations.items()
            }
            world.step(render=False)
            after = {
                robot: np.asarray(articulation.get_joint_positions(), dtype=np.float64)
                for robot, articulation in articulations.items()
            }
            arm_jump_by_robot = {
                robot: float(np.max(np.abs(after[robot][:6] - before[robot][:6]))) for robot in articulations
            }
            finger_jump_by_robot = {
                robot: float(np.max(np.abs(after[robot][7:9] - before[robot][7:9]))) for robot in articulations
            }
            finite_contacts = [
                contact
                for event in current_contacts
                for contact in event["contacts"]
                if all(
                    math.isfinite(value)
                    for value in (
                        contact["separation_m"],
                        contact["impulse_norm_n_s"],
                        *contact["position_world_m"],
                        *contact["normal"],
                    )
                )
            ]
            contact_count = sum(len(event["contacts"]) for event in current_contacts)
            unexpected = [item for item in overlaps if item.get("allowed") is False]
            unresolved_environment = [item for item in overlaps if item.get("allowed") is None]
            limits_ok = all(
                bool(
                    np.all(injected[robot] >= articulation.dof_properties["lower"] - 1.0e-7)
                    and np.all(injected[robot] <= articulation.dof_properties["upper"] + 1.0e-7)
                )
                for robot, articulation in articulations.items()
            )
            record = {
                **pose,
                "state_id": f"{pose['pose_id']}:{gripper_state}",
                "gripper_state": gripper_state,
                "finger_targets_m": list(finger),
                "finite": all(np.all(np.isfinite(value)) for value in after.values()),
                "within_joint_limits": limits_ok,
                "collider_count": len(colliders),
                "overlaps": overlaps,
                "unexpected_overlap_count": len(unexpected),
                "unresolved_environment_contact_count": len(unresolved_environment),
                "first_frame_arm_jump_by_robot_rad": arm_jump_by_robot,
                "first_frame_jump_max_abs_rad": max(arm_jump_by_robot.values()),
                "first_frame_jump_gate_rad": FIRST_FRAME_JUMP_GATE_RAD,
                "first_frame_finger_jump_by_robot_m": finger_jump_by_robot,
                "first_frame_finger_jump_gate": ("RECORDED_NOT_GATED_NO_FROZEN_OFFICIAL_TOLERANCE"),
                "contact_events": current_contacts.copy(),
                "contact_point_count": contact_count,
                "nonfinite_contact_count": contact_count - len(finite_contacts),
            }
            record["status"] = summarize_static_validation([record])["status"]
            pose_records.append(record)

    summary = summarize_static_validation(pose_records)
    signature = canonical_runtime_signature({"poses": pose_records})
    hash_after = _sha256(stage_path)
    if hash_after != hash_before:
        raise RuntimeError("diagnostic Stage changed during native validation")

    coverage_records = []
    assert coverage_snapshot is not None
    for record in coverage_snapshot:
        output = {key: value for key, value in record.items() if key != "points_world_m"}
        coverage_records.append(output)
    source_counts = {
        kind: sum(record["source_kind"] == kind for record in coverage_records)
        for kind in ("CAD_DERIVED", "SUPPLIER_CAD_FINGER", "IMPORTER_BASELINE_FALLBACK")
    }
    coverage_status = "PASS" if source_counts == _expected_source_counts(args.profile) else "FAIL"
    coverage = {
        "schema_version": 1,
        "status": coverage_status,
        "scope": "ISOLATED_DIAGNOSTIC_ONLY_NOT_FINAL",
        "profile": args.profile,
        "pose_source": {
            "absolute_path": str(args.pose_source.resolve(strict=True)),
            "sha256": _sha256(args.pose_source),
            "scan_preflight_candidates": args.scan_preflight_candidates,
        },
        "runtime": runtime,
        "stage": {"absolute_path": str(stage_path), "sha256": hash_before},
        "collider_count": len(coverage_records),
        "source_kind_counts": source_counts,
        "colliders": coverage_records,
        "source_or_final_asset_modified": False,
        "task8": (
            "AUTHORIZED_IN_PROGRESS"
            if args.profile.startswith("task8_")
            else "NOT_RUN"
        ),
    }
    overlap = {
        "schema_version": 1,
        "status": summary["status"],
        "scope": "CAD_DERIVED_STATIC_OVERLAP_AND_FIRST_FRAME_DIAGNOSTIC",
        "profile": args.profile,
        "pose_source": {
            "absolute_path": str(args.pose_source.resolve(strict=True)),
            "sha256": _sha256(args.pose_source),
            "scan_preflight_candidates": args.scan_preflight_candidates,
        },
        "runtime": runtime,
        "repeat_index": args.repeat_index,
        "stage": {
            "absolute_path": str(stage_path),
            "sha256_before": hash_before,
            "sha256_after": hash_after,
        },
        "session_diagnostics": session,
        "physics_frequency_hz": 60,
        "first_frame_jump_gate_rad": FIRST_FRAME_JUMP_GATE_RAD,
        "first_frame_jump_scope": "SIX_ARM_DOF_ONLY",
        "numerical_overlap_method": "CONVEX_HULL_HALFSPACE_LP_WORLD_TRANSFORMED_AUTHORED_POINTS",
        "unsafe_scene_query_overlap_shape_retried": False,
        "poses": pose_records,
        "summary": summary,
        "deterministic_signature": signature,
        "contact_subscription_created": subscription is not None,
        "documented_assembly_interface_pairs": [list(pair) for pair in sorted(assembly_interfaces)],
        "source_or_final_asset_modified": False,
        "task8": (
            "AUTHORIZED_IN_PROGRESS"
            if args.profile.startswith("task8_")
            else "NOT_RUN"
        ),
    }
    args.coverage.resolve().write_text(
        json.dumps(coverage, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.overlap.resolve().write_text(
        json.dumps(overlap, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": overlap["status"],
                "coverage_status": coverage_status,
                "state_count": len(pose_records),
                "signature": signature,
                "overlap_report": str(args.overlap.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    world.stop()
    return 0 if coverage_status == "PASS" and overlap["status"] == "PASS" else 2


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "create_new_stage": False,
            "disable_viewport_updates": True,
        }
    )
    exit_code = 1
    try:
        exit_code = main(_parse_args())
    except BaseException:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
