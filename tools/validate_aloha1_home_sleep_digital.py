#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Run the frozen ALOHA1 Home-Sleep-Home trajectory in Isaac Sim 5.1."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.home_sleep_correspondence import ARM_JOINT_ORDER
from tools.aloha1_mapping.home_sleep_correspondence import command_index_for_physics_frame
from tools.aloha1_mapping.home_sleep_correspondence import count_follower_articulation_roots
from tools.aloha1_mapping.home_sleep_correspondence import digital_runtime_signature
from tools.aloha1_mapping.home_sleep_correspondence import manifest_initial_terminal_arm
from tools.aloha1_mapping.home_sleep_correspondence import validate_digital_preflight
from tools.aloha1_mapping.home_sleep_correspondence import values_within_float32_limits

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "reports/aloha1_mapping/aloha1_home_sleep_command_manifest.json"
DEFAULT_STAGE = ROOT / (
    "assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/"
    "aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_z_up_meters_diagnostic.usda"
)
DEFAULT_FINGER_LAYER = ROOT / (
    "assets/Trossen/ALOHA1/1.0/diagnostics/finger_limit_pair_collision_candidate/1.0/"
    "configuration/finger_source_limits.usda"
)
ARTICULATION_PATHS = {
    "follower_left": "/World/follower_left/vx300s_left/root_joint",
    "follower_right": "/World/follower_right/vx300s_right/root_joint",
}
EXPECTED_DOF_ORDER = [*ARM_JOINT_ORDER, "gripper", "left_finger", "right_finger"]
EXPECTED_RUNTIME = {
    "isaac_sim": "5.1.0.0",
    "kit": "107.3.3",
    "physx": "107.3.26",
}
# Frozen before this trajectory from the existing Task 7A structure gate. The
# hardware quantization cross-check is 2*pi/4096 = 0.0015339808 rad; the larger
# pre-existing 0.02 rad gate remains conservative and is not selected from this run.
POSITION_GATE_RAD = 0.02
STATIONARY_GATE_RAD_OR_M = 0.02
IMPULSE_NUMERICAL_FLOOR_N_S = 1.0e-9


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--stage-sha256", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--finger-limit-layer", type=Path, default=DEFAULT_FINGER_LAYER)
    parser.add_argument("--finger-limit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--start-monotonic-ns", type=int)
    parser.add_argument("--realtime-pacing", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _runtime_versions(app: Any) -> dict[str, str]:
    import carb

    manager = app.get_extension_manager()
    physx_id = manager.get_enabled_extension_id("omni.physx")
    physx_record = manager.get_extension_dict(physx_id) if physx_id else {}
    return {
        "isaac_sim": version("isaacsim"),
        "kit": str(carb.tokens.get_tokens_interface().resolve("${kit_version}")).split("+", maxsplit=1)[0],
        "physx": str(physx_record.get("package", {}).get("version", "")).split("+", maxsplit=1)[0],
    }


def _stage_composition(stage: Any) -> dict[str, Any]:
    from pxr import UsdGeom

    authored_references = [
        {
            "prim_path": str(prim.GetPath()),
            "references": str(prim.GetMetadata("references")),
        }
        for prim in stage.Traverse()
        if prim.HasAuthoredReferences()
    ]
    root = stage.GetRootLayer()
    file_layers = sorted(str(Path(layer.realPath).resolve()) for layer in stage.GetUsedLayers() if layer.realPath)
    return {
        "default_prim": str(stage.GetDefaultPrim().GetPath()),
        "root_layer": str(Path(root.realPath).resolve()),
        "root_sublayers": list(root.subLayerPaths),
        "used_file_layers": file_layers,
        "authored_references": authored_references,
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
    }


def _install_session_layers(stage: Any, finger_layer: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    from pxr import PhysxSchema
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdPhysics

    session = stage.GetSessionLayer()
    finger_identifier = str(finger_layer.resolve(strict=True))
    session.subLayerPaths.append(finger_identifier)
    limit_layer = None
    limit_readback = []
    override = manifest.get("diagnostic_limit_override")
    if override:
        if override.get("classification") != ("DIAGNOSTIC_ONLY_RUNTIME_ALIGNMENT_NOT_FINAL_ASSET"):
            raise ValueError("unapproved diagnostic limit override classification")
        limit_layer = Sdf.Layer.CreateAnonymous("aloha1_runtime_sleep_diagnostic_limits.usda")
        session.subLayerPaths.insert(0, limit_layer.identifier)
        old_target = stage.GetEditTarget()
        stage.SetEditTarget(Usd.EditTarget(limit_layer))
        for change in override["changes"]:
            joint_name = str(change["joint_name"])
            joint_path = f"/World/follower_left/vx300s_left/joints/{joint_name}"
            prim = stage.GetPrimAtPath(joint_path)
            if not prim or not prim.IsA(UsdPhysics.RevoluteJoint):
                raise ValueError(f"missing follower_left revolute joint: {joint_path}")
            joint = UsdPhysics.RevoluteJoint(prim)
            value_rad = float(change["diagnostic_value_rad"])
            value_degrees = math.degrees(value_rad)
            if change["bound"] == "lower":
                joint.GetLowerLimitAttr().Set(value_degrees)
                readback_degrees = float(joint.GetLowerLimitAttr().Get())
            elif change["bound"] == "upper":
                joint.GetUpperLimitAttr().Set(value_degrees)
                readback_degrees = float(joint.GetUpperLimitAttr().Get())
            else:
                raise ValueError(f"unsupported limit bound: {change['bound']}")
            limit_readback.append(
                {
                    "joint_path": joint_path,
                    "joint_name": joint_name,
                    "bound": str(change["bound"]),
                    "authored_value_rad": value_rad,
                    "authored_value_degrees": value_degrees,
                    "usd_readback_degrees": readback_degrees,
                    "classification": "DIAGNOSTIC_ONLY_RUNTIME_ALIGNMENT",
                }
            )
        stage.SetEditTarget(old_target)
    report_layer = Sdf.Layer.CreateAnonymous("aloha1_home_sleep_contact_reports.usda")
    session.subLayerPaths.insert(0, report_layer.identifier)
    old_target = stage.GetEditTarget()
    stage.SetEditTarget(Usd.EditTarget(report_layer))
    report_bodies = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
        report_bodies.append(str(prim.GetPath()))
    stage.SetEditTarget(old_target)
    return {
        "finger_layer": finger_identifier,
        "diagnostic_limit_layer": limit_layer.identifier if limit_layer else None,
        "diagnostic_limit_readback": limit_readback,
        "contact_report_layer": report_layer.identifier,
        "contact_report_bodies": report_bodies,
    }


def _serialize_contacts(headers: Sequence[Any], data: Sequence[Any]) -> list[dict[str, Any]]:
    from pxr import PhysicsSchemaTools

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
                "actor0": str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                "actor1": str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
                "collider0": str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                "collider1": str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                "contacts": contacts,
            }
        )
    return records


def _apply_targets(articulation: Any, positions: Sequence[float], indices: Sequence[int]) -> None:
    from isaacsim.core.utils.types import ArticulationAction

    articulation.get_articulation_controller().apply_action(
        ArticulationAction(
            joint_positions=np.asarray(positions, dtype=np.float32),
            joint_indices=np.asarray(indices, dtype=np.int32),
        )
    )


def _finite_json_vector(values: Any) -> list[float]:
    result = [float(value) for value in np.asarray(values, dtype=np.float64)]
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError(f"non-finite runtime vector: {result}")
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":")) if isinstance(value, list | dict) else value
                    for key, value in row.items()
                }
            )


def _summarize_rows(
    rows: list[dict[str, Any]],
    *,
    manifest: Mapping[str, Any],
    limits: Mapping[str, list[list[float]]],
    initial_stationary: Mapping[str, list[float]],
) -> dict[str, Any]:
    left = np.asarray([row["left_q"] for row in rows], dtype=np.float64)
    targets = np.asarray([row["target_arm_q"] for row in rows], dtype=np.float64)
    right = np.asarray([row["right_q"] for row in rows], dtype=np.float64)
    finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    lower = np.asarray(limits["follower_left"][0], dtype=np.float64)
    upper = np.asarray(limits["follower_left"][1], dtype=np.float64)
    legal = all(values_within_float32_limits(row, lower, upper) for row in left.tolist())
    right_start = np.asarray(initial_stationary["follower_right"], dtype=np.float64)
    left_gripper_start = np.asarray(initial_stationary["follower_left_gripper"], dtype=np.float64)
    right_drift = float(np.max(np.abs(right - right_start[None, :])))
    left_gripper_drift = float(np.max(np.abs(left[:, 6:] - left_gripper_start[None, :])))

    segment_names = sorted({str(row["segment"]) for row in rows})
    endpoints = []
    directions = []
    for segment in segment_names:
        indices = [i for i, row in enumerate(rows) if row["segment"] == segment]
        if not indices:
            continue
        first, last = indices[0], indices[-1]
        if segment.endswith("_hold"):
            hold_count = min(12, len(indices))
            hold_readback = left[indices[-hold_count:], :6]
            hold_target = targets[last]
            maximum = float(np.max(np.abs(hold_readback - hold_target[None, :])))
            endpoints.append(
                {
                    "segment": segment,
                    "maximum_abs_error_rad": maximum,
                    "status": "PASS" if maximum <= POSITION_GATE_RAD else "FAIL",
                }
            )
        elif "_to_" in segment:
            command_delta = targets[last] - targets[first]
            readback_delta = left[last, :6] - left[first, :6]
            active = np.abs(command_delta) > (2.0 * math.pi / 4096.0)
            matches = np.sign(readback_delta[active]) == np.sign(command_delta[active])
            directions.append(
                {
                    "segment": segment,
                    "active_joint_count": int(active.sum()),
                    "command_delta_rad": command_delta.tolist(),
                    "readback_delta_rad": readback_delta.tolist(),
                    "status": "PASS" if bool(matches.all()) else "FAIL",
                }
            )

    physical_contacts = []
    minimum_separation = math.inf
    maximum_impulse = 0.0
    for row in rows:
        for pair in row["contacts"]:
            for point in pair["contacts"]:
                impulse = float(point["impulse_norm_n_s"])
                separation = float(point["separation_m"])
                maximum_impulse = max(maximum_impulse, impulse)
                minimum_separation = min(minimum_separation, separation)
                if impulse > IMPULSE_NUMERICAL_FLOOR_N_S:
                    physical_contacts.append(
                        {
                            "physics_frame": row["physics_frame"],
                            "collider0": pair["collider0"],
                            "collider1": pair["collider1"],
                            "impulse_norm_n_s": impulse,
                            "separation_m": separation,
                        }
                    )
    _, terminal_arm = manifest_initial_terminal_arm(dict(manifest))
    terminal_label = str(manifest.get("terminal_pose_label", "home"))
    final_terminal_error = float(np.max(np.abs(left[-1, :6] - np.asarray(terminal_arm, dtype=np.float64))))
    gates = {
        "finite_readback": finite,
        "legal_limits": legal,
        "directions": bool(directions) and all(item["status"] == "PASS" for item in directions),
        "endpoints": bool(endpoints) and all(item["status"] == "PASS" for item in endpoints),
        "follower_right_stationary": right_drift <= STATIONARY_GATE_RAD_OR_M,
        "grippers_stationary": left_gripper_drift <= STATIONARY_GATE_RAD_OR_M,
        "no_impulse_carrying_contact": not physical_contacts,
        "three_cycles_complete": int(rows[-1]["cycle"]) == 3,
    }
    gates["final_home" if terminal_label == "home" else "final_terminal"] = final_terminal_error <= POSITION_GATE_RAD
    signature_payload = {
        "command_signature": manifest["command_signature"],
        "rows": [
            {
                "physics_frame": row["physics_frame"],
                "command_index": row["command_index"],
                "left_q": [round(value, 9) for value in row["left_q"]],
                "left_qd": [round(value, 9) for value in row["left_qd"]],
                "right_q": [round(value, 9) for value in row["right_q"]],
                "contact_pairs": [[pair["collider0"], pair["collider1"]] for pair in row["contacts"]],
            }
            for row in rows
        ],
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "position_gate_rad": POSITION_GATE_RAD,
        "position_gate_provenance": {
            "existing_task7a_gate_rad": 0.02,
            "exact_motor_resolution_pulse_per_rev": 4096,
            "one_pulse_rad": 2.0 * math.pi / 4096.0,
            "selection": "MAX_EXISTING_PREDECLARED_TASK7A_GATE_AND_ONE_PULSE",
        },
        "right_max_drift_rad_or_m": right_drift,
        "left_gripper_max_drift_rad_or_m": left_gripper_drift,
        "terminal_pose_label": terminal_label,
        "final_terminal_max_error_rad": final_terminal_error,
        **({"final_home_max_error_rad": final_terminal_error} if terminal_label == "home" else {}),
        "endpoint_results": endpoints,
        "direction_results": directions,
        "contact": {
            "event_point_count": sum(len(pair["contacts"]) for row in rows for pair in row["contacts"]),
            "impulse_carrying_point_count": len(physical_contacts),
            "maximum_impulse_n_s": maximum_impulse,
            "minimum_separation_m": (minimum_separation if math.isfinite(minimum_separation) else None),
            "first_impulse_contacts": physical_contacts[:20],
        },
        "normalized_numeric_signature": digital_runtime_signature(signature_payload),
    }


def main(args: argparse.Namespace) -> int:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.core.utils.xforms import get_world_pose
    import omni.kit.app
    from omni.physx import get_physx_simulation_interface
    import omni.usd
    from pxr import UsdPhysics

    from tools.run_aloha1_home_sleep_isaac_worker import frame_deadline_ns
    from tools.run_aloha1_home_sleep_isaac_worker import frame_lateness_status
    from tools.run_aloha1_home_sleep_isaac_worker import wait_until_monotonic_ns

    started = time.monotonic()
    stage_path = args.stage.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    finger_path = args.finger_limit_layer.resolve(strict=True)
    stage_hash_before = _sha256(stage_path)
    manifest_hash_before = _sha256(manifest_path)
    finger_hash_before = _sha256(finger_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not open_stage(str(stage_path)):
        raise RuntimeError(f"failed to open frozen Stage: {stage_path}")
    app = omni.kit.app.get_app()
    for _ in range(20):
        app.update()
    stage = omni.usd.get_context().get_stage()
    composition = _stage_composition(stage)
    session = _install_session_layers(stage, finger_path, manifest)

    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=1.0 / int(manifest["physics_rate_hz"]),
        rendering_dt=1.0 / int(manifest["physics_rate_hz"]),
    )
    world.get_physics_context().set_solve_articulation_contact_last(True)
    articulations = {}
    for robot, path in ARTICULATION_PATHS.items():
        articulation = SingleArticulation(
            prim_path=path,
            name=f"home_sleep_{args.repeat_index}_{robot}",
            reset_xform_properties=False,
        )
        world.scene.add(articulation)
        articulations[robot] = articulation
    world.reset()

    dof_orders = {robot: list(item.dof_names) for robot, item in articulations.items()}
    properties = {robot: item.dof_properties.copy() for robot, item in articulations.items()}
    runtime = _runtime_versions(app)
    required_paths = ["/World", "/World/PhysicsScene", *ARTICULATION_PATHS.values()]
    initial_arm_tuple, terminal_arm_tuple = manifest_initial_terminal_arm(manifest)
    initial_arm = np.asarray(initial_arm_tuple, dtype=np.float32)
    left_initial = np.asarray(articulations["follower_left"].get_joint_positions(), dtype=np.float32)
    right_initial = np.asarray(articulations["follower_right"].get_joint_positions(), dtype=np.float32)
    left_full_initial = left_initial.copy()
    left_full_initial[:6] = initial_arm
    articulations["follower_left"].set_joint_positions(left_full_initial)
    articulations["follower_left"].set_joint_velocities(np.zeros_like(left_full_initial))
    _apply_targets(articulations["follower_left"], left_full_initial[:8], range(8))
    _apply_targets(articulations["follower_right"], right_initial[:8], range(8))
    world.step(render=False)
    left_first = np.asarray(articulations["follower_left"].get_joint_positions(), dtype=np.float64)
    first_frame_arm_jump = float(np.max(np.abs(left_first[:6] - left_full_initial[:6])))
    first_frame_gripper_jump = float(np.max(np.abs(left_first[6:] - left_full_initial[6:])))
    # Establish the manifest-declared robot-local initial pose before the
    # formal 37-second command stream begins.
    for _ in range(29):
        _apply_targets(articulations["follower_left"], left_full_initial[:8], range(8))
        _apply_targets(articulations["follower_right"], right_initial[:8], range(8))
        world.step(render=False)
    left_settled = np.asarray(articulations["follower_left"].get_joint_positions(), dtype=np.float32)
    right_settled = np.asarray(articulations["follower_right"].get_joint_positions(), dtype=np.float32)

    limits: dict[str, list[list[float]]] = {}
    for robot, prop in properties.items():
        limits[robot] = [
            [float(row["lower"]) for row in prop],
            [float(row["upper"]) for row in prop],
        ]
    initial_legal = bool(
        np.isfinite(initial_arm).all()
        and np.all(initial_arm >= np.asarray(limits["follower_left"][0][:6]))
        and np.all(initial_arm <= np.asarray(limits["follower_left"][1][:6]))
    )
    all_articulation_roots = [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    articulation_roots = count_follower_articulation_roots(all_articulation_roots)
    preflight_contract = {
        "runtime_versions_match": runtime == EXPECTED_RUNTIME,
        "stage_hash_match": stage_hash_before == args.stage_sha256,
        "manifest_hash_match": manifest_hash_before == args.manifest_sha256,
        "default_prim": composition["default_prim"],
        "root_prim_valid": bool(stage.GetPrimAtPath("/World")),
        "required_prims_valid": all(bool(stage.GetPrimAtPath(path)) for path in required_paths),
        "articulation_count": len(articulation_roots),
        "dof_order_match": all(order == EXPECTED_DOF_ORDER for order in dof_orders.values()),
        "finger_limit_hash_match": finger_hash_before == args.finger_limit_sha256,
        "home_finite_and_legal": initial_legal,
        "first_frame_arm_stable": first_frame_arm_jump <= POSITION_GATE_RAD,
        "stationary_scope_declared": manifest["stationary_scope"]
        == {
            "follower_right": True,
            "follower_left_gripper": True,
            "follower_right_gripper": True,
        },
        "source_hashes_immutable": True,
        "final_default_asset_modified": bool(manifest["final_default_asset_modified"]),
    }
    preflight = validate_digital_preflight(preflight_contract)
    preflight.update(
        {
            "runtime": runtime,
            "stage_composition": composition,
            "session_only_layers": session,
            "articulation_roots": articulation_roots,
            "all_schema_articulation_roots": all_articulation_roots,
            "dof_orders": dof_orders,
            "limits": limits,
            "first_frame_jump_rad_or_m": float(np.max(np.abs(left_first - left_full_initial))),
            "first_frame_arm_jump_rad": first_frame_arm_jump,
            "first_frame_gripper_jump_rad_or_m": first_frame_gripper_jump,
            "left_target_before_first_frame": left_full_initial.astype(np.float64).tolist(),
            "initial_pose_label": str(manifest.get("initial_pose_label", "home")),
            "initial_arm_rad": list(initial_arm_tuple),
            "terminal_pose_label": str(manifest.get("terminal_pose_label", "home")),
            "terminal_arm_rad": list(terminal_arm_tuple),
            "initial_finite_and_legal": initial_legal,
            "left_readback_after_first_frame": left_first.tolist(),
            "initialization_settle_frames_not_in_formal_trajectory": 30,
            "solve_articulation_contact_last": bool(world.get_physics_context().get_solve_articulation_contact_last()),
        }
    )

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    contact_events: list[dict[str, Any]] = []
    scheduler: dict[str, Any] = {
        "mode": ("ABSOLUTE_MONOTONIC_NO_BURST" if args.realtime_pacing else "UNPACED_DETERMINISTIC_PHYSICS"),
        "status": "NOT_REQUESTED" if not args.realtime_pacing else "PENDING",
        "requested_start_monotonic_ns": args.start_monotonic_ns,
        "first_applied_monotonic_ns": None,
        "start_skew_ns": None,
        "maximum_lateness_ns": None,
        "completed_physics_frames": 0,
        "burst_catchup_used": False,
    }

    def on_contact(headers: Sequence[Any], data: Sequence[Any]) -> None:
        contact_events.extend(_serialize_contacts(headers, data))

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    if not args.preflight_only and preflight["status"] == "PASS":
        if args.realtime_pacing and args.start_monotonic_ns is None:
            raise ValueError("--realtime-pacing requires --start-monotonic-ns")
        samples = manifest["samples"]
        physics_hz = int(manifest["physics_rate_hz"])
        command_hz = int(manifest["command_rate_hz"])
        total_frames = math.ceil(len(samples) * physics_hz / command_hz)
        scheduler["expected_physics_frames"] = total_frames
        scheduler["physics_rate_hz"] = physics_hz
        frozen_left_gripper = left_settled[6:]
        frozen_right = right_settled.copy()
        for physics_frame in range(total_frames):
            applied_monotonic_ns = None
            frame_lateness_ns = None
            if args.realtime_pacing:
                frame_deadline = frame_deadline_ns(
                    int(args.start_monotonic_ns),
                    frame_index=physics_frame,
                    physics_rate_hz=physics_hz,
                )
                applied_monotonic_ns = wait_until_monotonic_ns(frame_deadline)
                frame_lateness_ns = applied_monotonic_ns - frame_deadline
                if scheduler["first_applied_monotonic_ns"] is None:
                    scheduler["first_applied_monotonic_ns"] = applied_monotonic_ns
                    scheduler["start_skew_ns"] = frame_lateness_ns
                previous_maximum = scheduler["maximum_lateness_ns"]
                scheduler["maximum_lateness_ns"] = (
                    frame_lateness_ns if previous_maximum is None else max(int(previous_maximum), frame_lateness_ns)
                )
                pacing_status = frame_lateness_status(frame_lateness_ns, physics_rate_hz=physics_hz)
                if pacing_status != "ON_TIME":
                    scheduler["status"] = pacing_status
                    scheduler["aborted_physics_frame"] = physics_frame
                    break
            command_index = command_index_for_physics_frame(
                physics_frame,
                physics_hz=physics_hz,
                command_hz=command_hz,
                sample_count=len(samples),
            )
            sample = samples[command_index]
            target = np.asarray(sample["q_rad"], dtype=np.float32)
            _apply_targets(articulations["follower_left"], target, range(6))
            _apply_targets(articulations["follower_left"], frozen_left_gripper[:2], (6, 7))
            _apply_targets(articulations["follower_right"], frozen_right[:8], range(8))
            contact_events.clear()
            world.step(render=not args.headless)
            left_q = _finite_json_vector(articulations["follower_left"].get_joint_positions())
            left_qd = _finite_json_vector(articulations["follower_left"].get_joint_velocities())
            right_q = _finite_json_vector(articulations["follower_right"].get_joint_positions())
            right_qd = _finite_json_vector(articulations["follower_right"].get_joint_velocities())
            ee_position, ee_orientation = get_world_pose("/World/follower_left/vx300s_left/follower_left_gripper_link")
            rows.append(
                {
                    "physics_frame": physics_frame,
                    "physics_time_s": (physics_frame + 1) / physics_hz,
                    "physics_dt_s": 1.0 / physics_hz,
                    "command_index": command_index,
                    "nominal_command_time_s": int(sample["time_ns"]) / 1.0e9,
                    "scheduler_phase_error_s": physics_frame / physics_hz - int(sample["time_ns"]) / 1.0e9,
                    "scheduler_applied_monotonic_ns": applied_monotonic_ns,
                    "scheduler_frame_lateness_ns": frame_lateness_ns,
                    "cycle": int(sample["cycle"]),
                    "segment": str(sample["segment"]),
                    "target_arm_q": target.astype(np.float64).tolist(),
                    "left_q": left_q,
                    "left_qd": left_qd,
                    "right_q": right_q,
                    "right_qd": right_qd,
                    "left_ee_position_world_m": _finite_json_vector(ee_position),
                    "left_ee_orientation_world_wxyz": _finite_json_vector(ee_orientation),
                    "contacts": list(contact_events),
                }
            )
        scheduler["completed_physics_frames"] = len(rows)
        if len(rows) == total_frames:
            if args.realtime_pacing:
                scheduler["status"] = "PASS"
            summary = _summarize_rows(
                rows,
                manifest=manifest,
                limits=limits,
                initial_stationary={
                    "follower_right": frozen_right.astype(np.float64).tolist(),
                    "follower_left_gripper": frozen_left_gripper.astype(np.float64).tolist(),
                },
            )
        if rows:
            _write_csv(args.telemetry.resolve(), rows)

    del subscription
    stage_hash_after = _sha256(stage_path)
    manifest_hash_after = _sha256(manifest_path)
    finger_hash_after = _sha256(finger_path)
    immutable = {
        "stage": stage_hash_after == stage_hash_before,
        "manifest": manifest_hash_after == manifest_hash_before,
        "finger_limit_layer": finger_hash_after == finger_hash_before,
    }
    if not all(immutable.values()):
        preflight["status"] = "FAIL"
        preflight["failed_gates"].append("source_hashes_immutable_after_run")
        if summary is not None:
            summary["status"] = "FAIL"
            summary["gates"]["source_hashes_immutable"] = False
    if scheduler["status"] == "ABORTED_DEADLINE_MISS":
        status = "ABORTED_DEADLINE_MISS"
    else:
        status = preflight["status"] if summary is None else summary["status"]
    report = {
        "schema_version": 1,
        "status": status,
        "classification": (
            "DIGITAL_PREFLIGHT_ONLY"
            if args.preflight_only
            else (
                "DIGITAL_SLEEP_HOME_SLEEP_THREE_CYCLE"
                if manifest.get("sequence_kind") == "SLEEP_HOME_SLEEP"
                else "DIGITAL_HOME_SLEEP_THREE_CYCLE"
            )
        ),
        "repeat_index": args.repeat_index,
        "run_id": args.run_id,
        "runtime_pid": os.getpid(),
        "wall_time_s": time.monotonic() - started,
        "runtime": runtime,
        "manifest": {
            "absolute_path": str(manifest_path),
            "sha256_before": manifest_hash_before,
            "sha256_after": manifest_hash_after,
            "command_signature": manifest["command_signature"],
            "manifest_signature": manifest["manifest_signature"],
        },
        "stage": {
            "absolute_path": str(stage_path),
            "sha256_before": stage_hash_before,
            "sha256_after": stage_hash_after,
        },
        "finger_limit_layer": {
            "absolute_path": str(finger_path),
            "sha256_before": finger_hash_before,
            "sha256_after": finger_hash_after,
        },
        "preflight": preflight,
        "summary": summary,
        "telemetry": {
            "absolute_path": str(args.telemetry.resolve()),
            "row_count": len(rows),
            "sha256": _sha256(args.telemetry) if rows else None,
        },
        "scheduler": scheduler,
        "immutability": immutable,
        "source_or_final_asset_modified": False,
        "real_execution_authorized": False,
        "task8": "COMPLETE_WITH_NO_PROMOTION",
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "repeat_index": args.repeat_index,
                "telemetry_rows": len(rows),
                "numeric_signature": (summary["normalized_numeric_signature"] if summary else None),
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    world.stop()
    return 0 if status == "PASS" else 2


def run() -> int:
    from isaacsim import SimulationApp

    args = _parse_args()
    app = SimulationApp(
        {
            "headless": bool(args.headless),
            "create_new_stage": False,
            "disable_viewport_updates": bool(args.headless),
        }
    )
    exit_code = 1
    try:
        exit_code = main(args)
    except BaseException:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
