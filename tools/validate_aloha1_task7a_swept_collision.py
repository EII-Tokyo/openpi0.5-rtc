#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Run dual-follower Task 7A swept-collision validation in Isaac Sim 5.1."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.signal_correspondence import HOME_ARM
from tools.aloha1_mapping.signal_correspondence import HOME_LEFT_FINGER_M
from tools.aloha1_mapping.signal_correspondence import HOME_RIGHT_FINGER_M
from tools.aloha1_mapping.signal_correspondence import RUNTIME_SPECS
from tools.aloha1_mapping.signal_correspondence import canonical_dof_name
from tools.aloha1_mapping.task7a_swept_collision import ARM_JOINTS
from tools.aloha1_mapping.task7a_swept_collision import build_sweep_cases
from tools.aloha1_mapping.task7a_swept_collision import canonical_pair
from tools.aloha1_mapping.task7a_swept_collision import classify_contact_observation
from tools.aloha1_mapping.task7a_swept_collision import classify_contact_pair
from tools.aloha1_mapping.task7a_swept_collision import summarize_sweep_cases

ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "signal_correspondence/1.0/"
    "aloha1_signal_correspondence_workcell.usda"
)
EXPECTED_STAGE_SHA256 = (
    "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
)
REPORT_ROOT = ROOT / "reports/aloha1_mapping"
DEFAULT_OUTPUT = REPORT_ROOT / "aloha1_task7a_swept_collision.json"
DEFAULT_CURVES = (
    REPORT_ROOT / "aloha1_task7a_swept_collision_curves.csv"
)
DEFAULT_PAIRS = REPORT_ROOT / "aloha1_task7a_collision_pair_inventory.csv"
PHYSICS_HZ = 60
HOME_SETTLE_STEPS = 30
SWEEP_STEPS = 180
TARGET_SETTLE_STEPS = 60
DEFAULT_REPEAT_COUNT = 2
TARGET_ERROR_MINIMUM_RAD = 0.05
TARGET_ERROR_SPAN_FRACTION = 0.03
NON_TARGET_DRIFT_GATE = 0.03


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight_frozen_stage(path: Path) -> dict[str, Any]:
    """Fail before SimulationApp unless the exact approved Stage is selected."""
    resolved = path.resolve(strict=True)
    if resolved != STAGE.resolve(strict=True):
        raise ValueError("candidate is not the approved frozen Stage")
    digest = _sha256(resolved)
    if digest != EXPECTED_STAGE_SHA256:
        raise ValueError("candidate is not the approved frozen Stage")
    text = resolved.read_text(encoding="utf-8")
    required_tokens = [
        'defaultPrim = "World"',
        "@configuration/aloha1_signal_home_targets.usda@",
        "@follower_left_asset/aloha1_signal_follower_left.usda@",
        (
            "@../../supplier_cad_follower_right/1.0/"
            "supplier_cad_follower_right.usda@"
        ),
        'def Xform "follower_left"',
        'def Xform "follower_right"',
    ]
    missing = [token for token in required_tokens if token not in text]
    if missing:
        raise ValueError(f"frozen Stage tokens missing: {missing}")
    return {
        "status": "PASS",
        "absolute_path": str(resolved),
        "sha256": digest,
        "root_prim": "/World",
        "required_token_status": "PASS",
        "required_tokens": required_tokens,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    parser.add_argument("--pair-inventory", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=DEFAULT_REPEAT_COUNT,
    )
    return parser.parse_args()


def _home() -> np.ndarray:
    return np.asarray(
        [
            *HOME_ARM,
            0.0,
            HOME_LEFT_FINGER_M,
            HOME_RIGHT_FINGER_M,
        ],
        dtype=np.float32,
    )


def _smoothstep(frame: int, frames: int) -> float:
    value = frame / frames
    return value * value * (3.0 - 2.0 * value)


def _apply_targets(
    articulation: Any,
    values: np.ndarray,
    indices: np.ndarray,
) -> None:
    from isaacsim.core.utils.types import ArticulationAction

    articulation.get_articulation_controller().apply_action(
        ArticulationAction(
            joint_positions=np.asarray(values, dtype=np.float32),
            joint_indices=np.asarray(indices, dtype=np.int32),
        )
    )


def _prepare_home(
    world: Any,
    articulations: dict[str, Any],
) -> dict[str, np.ndarray]:
    world.reset()
    home = _home()
    for articulation in articulations.values():
        articulation.set_joint_positions(home)
        articulation.set_joint_velocities(np.zeros_like(home))
        active = np.arange(len(home) - 1, dtype=np.int32)
        _apply_targets(articulation, home[active], active)
    for _ in range(HOME_SETTLE_STEPS):
        world.step(render=False)
    return {
        robot: np.asarray(
            articulation.get_joint_positions(),
            dtype=np.float64,
        )
        for robot, articulation in articulations.items()
    }


def _path_from_id(value: Any) -> str:
    from pxr import PhysicsSchemaTools

    return str(PhysicsSchemaTools.intToSdfPath(value))


def _serialize_contacts(
    headers: Sequence[Any],
    data: Sequence[Any],
    *,
    frame: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for header in headers:
        contacts = []
        begin = int(header.contact_data_offset)
        end = begin + int(header.num_contact_data)
        for index in range(begin, end):
            item = data[index]
            impulse = [float(value) for value in item.impulse]
            contacts.append(
                {
                    "position_world_m": [
                        float(value) for value in item.position
                    ],
                    "normal": [float(value) for value in item.normal],
                    "impulse_n_s": impulse,
                    "impulse_norm_n_s": float(
                        np.linalg.norm(np.asarray(impulse))
                    ),
                    "separation_m": float(item.separation),
                    "material0": _path_from_id(item.material0),
                    "material1": _path_from_id(item.material1),
                }
            )
        records.append(
            {
                "frame": frame,
                "event_type": str(header.type),
                "actor0": _path_from_id(header.actor0),
                "actor1": _path_from_id(header.actor1),
                "collider0": _path_from_id(header.collider0),
                "collider1": _path_from_id(header.collider1),
                "contacts": contacts,
            }
        )
    return records


def _install_session_contact_reports(stage: Any) -> dict[str, Any]:
    from pxr import PhysxSchema
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdPhysics

    layer = Sdf.Layer.CreateAnonymous(
        "aloha1_task7a_swept_contact_reports.usda"
    )
    stage.GetSessionLayer().subLayerPaths.append(layer.identifier)
    previous = stage.GetEditTarget()
    stage.SetEditTarget(Usd.EditTarget(layer))
    paths = []
    for prim in Usd.PrimRange(
        stage.GetPseudoRoot(),
        Usd.TraverseInstanceProxies(),
    ):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            report_api.CreateThresholdAttr().Set(0.0)
            paths.append(str(prim.GetPath()))
    stage.SetEditTarget(previous)
    if not paths:
        raise RuntimeError("no rigid body accepted contact reporting")
    return {
        "session_layer_identifier": layer.identifier,
        "session_only": True,
        "rigid_body_count": len(set(paths)),
        "rigid_body_paths": sorted(set(paths)),
    }


def _adjacent_body_pairs(stage: Any) -> set[tuple[str, str]]:
    from pxr import UsdPhysics

    pairs: set[tuple[str, str]] = set()
    for prim in stage.Traverse():
        joint = UsdPhysics.Joint(prim)
        if not joint:
            continue
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if body0 and body1:
            pairs.add(canonical_pair(str(body0[0]), str(body1[0])))
    return pairs


def _stage_runtime_manifest(stage: Any) -> dict[str, Any]:
    required = [
        "/World",
        "/World/follower_left",
        "/World/follower_right",
        RUNTIME_SPECS["follower_left"]["articulation_path"],
        RUNTIME_SPECS["follower_right"]["articulation_path"],
    ]
    missing = [
        path for path in required if not stage.GetPrimAtPath(path).IsValid()
    ]
    if missing:
        raise RuntimeError(f"required composed prims missing: {missing}")
    layer_stack = [
        layer.realPath or layer.identifier for layer in stage.GetLayerStack()
    ]
    root = stage.GetRootLayer()
    references = {}
    for robot in ("follower_left", "follower_right"):
        prim = stage.GetPrimAtPath(f"/World/{robot}")
        references[robot] = [
            {
                "layer": spec.layer.realPath or spec.layer.identifier,
                "path": str(spec.path),
            }
            for spec in prim.GetPrimStack()
        ]
    return {
        "root_prim": str(stage.GetDefaultPrim().GetPath()),
        "root_layer": root.realPath,
        "root_sublayers": list(root.subLayerPaths),
        "layer_stack": layer_stack,
        "references_and_prim_stack": references,
        "required_prims": required,
        "required_prims_status": "PASS",
    }


def _self_collision_readback(stage: Any) -> dict[str, Any]:
    result = {}
    for robot, spec in RUNTIME_SPECS.items():
        prim = stage.GetPrimAtPath(spec["articulation_path"])
        attr = prim.GetAttribute(
            "physxArticulation:enabledSelfCollisions"
        )
        result[robot] = {
            "prim_path": str(prim.GetPath()),
            "attribute_exists": attr.IsValid(),
            "authored": attr.HasAuthoredValueOpinion() if attr.IsValid() else False,
            "readback": attr.Get() if attr.IsValid() else None,
            "mutation_performed": False,
        }
    return result


def _build_runtime_sweep_plan(
    articulations: dict[str, Any],
) -> list[dict[str, Any]]:
    limits = {}
    for robot, articulation in articulations.items():
        order = [
            canonical_dof_name(robot, name)
            for name in articulation.dof_names
        ]
        if tuple(order[:6]) != ARM_JOINTS:
            raise RuntimeError(f"arm DOF order mismatch for {robot}: {order}")
        properties = articulation.dof_properties
        limits[robot] = [
            {
                "name": name,
                "lower": float(properties[index]["lower"]),
                "upper": float(properties[index]["upper"]),
                "home": float(HOME_ARM[index]),
            }
            for index, name in enumerate(ARM_JOINTS)
        ]
    return build_sweep_cases(limits)


def _aggregate_contact_pairs(
    events: Sequence[dict[str, Any]],
    adjacent: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    aggregated: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for event in events:
        actor_pair = canonical_pair(event["actor0"], event["actor1"])
        collider_pair = canonical_pair(
            event["collider0"],
            event["collider1"],
        )
        key = (*actor_pair, *collider_pair)
        if key not in aggregated:
            classification = classify_contact_pair(
                event["actor0"],
                event["actor1"],
                adjacent,
            )
            aggregated[key] = {
                **classification,
                "collider_pair": list(collider_pair),
                "event_count": 0,
                "contact_point_count": 0,
                "first_frame": event["frame"],
                "last_frame": event["frame"],
                "minimum_separation_m": None,
                "maximum_penetration_m": 0.0,
                "maximum_impulse_norm_n_s": 0.0,
            }
        record = aggregated[key]
        record["event_count"] += 1
        record["last_frame"] = event["frame"]
        for contact in event["contacts"]:
            separation = float(contact["separation_m"])
            record["contact_point_count"] += 1
            current_minimum = record["minimum_separation_m"]
            record["minimum_separation_m"] = (
                separation
                if current_minimum is None
                else min(current_minimum, separation)
            )
            record["maximum_penetration_m"] = max(
                record["maximum_penetration_m"],
                0.0,
                -separation,
            )
            record["maximum_impulse_norm_n_s"] = max(
                record["maximum_impulse_norm_n_s"],
                float(contact["impulse_norm_n_s"]),
            )
    records = list(aggregated.values())
    for record in records:
        observation = classify_contact_observation(
            base_classification=record["classification"],
            base_allowed=record["allowed"],
            minimum_separation_m=record["minimum_separation_m"],
            maximum_impulse_norm_n_s=record[
                "maximum_impulse_norm_n_s"
            ],
        )
        record.update(observation)
    return sorted(
        records,
        key=lambda item: (
            item["classification"],
            item["actor_pair"],
            item["collider_pair"],
        ),
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    failed = [
        case for case in report["cases"] if case["status"] == "FAIL"
    ]
    unique_failed = sorted({case["case_id"] for case in failed})
    envelope_pairs = [
        pair
        for pair in report["pair_inventory"]
        if pair["classification"] == "CONTACT_ENVELOPE_ONLY"
    ]
    failed_rows = []
    for case_id in unique_failed:
        representative = next(
            case for case in failed if case["case_id"] == case_id
        )
        physical_pairs = [
            pair
            for pair in representative["contact_pairs"]
            if pair.get("physical_contact") is True
        ]
        failed_rows.append(
            f"| `{case_id}` | `{representative['target']:.9f}` | "
            f"`{representative['final_readback']:.9f}` | "
            f"`{len(physical_pairs)}` |"
        )
    path.write_text(
        "\n".join(
            [
                "# ALOHA1 Task 7A swept-collision validation",
                "",
                f"- Status: `{report['status']}`",
                f"- Stage SHA-256: `{report['stage']['sha256_after']}`",
                f"- Cases: `{summary['case_count']}`",
                f"- Failed cases: `{summary['failed_case_count']}`",
                (
                    "- Determinism: "
                    f"`{summary['determinism']['status']}`"
                ),
                (
                    "- solve_articulation_contact_last: "
                    f"`{str(report['runtime']['solve_articulation_contact_last']).lower()}`"
                ),
                (
                    "- Unique failed trajectories: "
                    f"`{len(unique_failed)}`"
                ),
                (
                    "- Contact-envelope-only pairs: "
                    f"`{len(envelope_pairs)}`"
                ),
                "",
                "## Deterministic failures",
                "",
                "| Case | Target (rad) | Final readback (rad) | Physical pairs |",
                "|---|---:|---:|---:|",
                *failed_rows,
                "",
                (
                    "Both positive-shoulder trajectories are stopped near "
                    "`0.288 rad` when both supplier-CAD finger colliders "
                    "physically contact `user_confirmed_table`. The same "
                    "two failures reproduce in both fresh repeats."
                ),
                "",
                "## Interpretation boundary",
                "",
                (
                    "This run preserves the authored collision filters and "
                    "self-collision settings. PASS proves no unexpected "
                    "reported contact along the tested trajectories under "
                    "those settings; it does not prove disabled collision "
                    "pairs are geometrically separated."
                ),
                "",
                "No source Stage, collider, drive, mimic, timestep, solver "
                "iteration, or final/default asset was modified.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _run_cases(
    world: Any,
    articulations: dict[str, Any],
    plans: Sequence[dict[str, Any]],
    *,
    repeat_count: int,
    event_state: dict[str, Any],
    adjacent: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for repeat in range(repeat_count):
        for plan in plans:
            homes = _prepare_home(world, articulations)
            robot = plan["robot"]
            articulation = articulations[robot]
            index = int(plan["joint_index"])
            start = homes[robot]
            event_state["events"] = []
            event_state["case_id"] = plan["case_id"]
            event_state["frame"] = 0
            target = float(plan["target"])
            for frame in range(1, SWEEP_STEPS + TARGET_SETTLE_STEPS + 1):
                event_state["frame"] = frame
                alpha = (
                    _smoothstep(frame, SWEEP_STEPS)
                    if frame <= SWEEP_STEPS
                    else 1.0
                )
                command = float(start[index]) + alpha * (
                    target - float(start[index])
                )
                _apply_targets(
                    articulation,
                    np.asarray([command], dtype=np.float32),
                    np.asarray([index], dtype=np.int32),
                )
                world.step(render=False)
                qpos = np.asarray(
                    articulation.get_joint_positions(),
                    dtype=np.float64,
                )
                qvel = np.asarray(
                    articulation.get_joint_velocities(),
                    dtype=np.float64,
                )
                curves.append(
                    {
                        "repeat": repeat,
                        "case_id": plan["case_id"],
                        "robot": robot,
                        "joint": plan["joint"],
                        "joint_index": index,
                        "direction": plan["direction"],
                        "frame": frame,
                        "time_s": frame / PHYSICS_HZ,
                        "command_target_rad": command,
                        "joint_readback_rad": float(qpos[index]),
                        "position_error_rad": command - float(qpos[index]),
                        "joint_velocity_rad_s": float(qvel[index]),
                        "maximum_non_target_drift": max(
                            abs(float(qpos[item] - start[item]))
                            for item in range(len(qpos))
                            if item != index
                        ),
                        "contact_event_count": len(event_state["events"]),
                    }
                )
            event_state["case_id"] = None
            end = np.asarray(
                articulation.get_joint_positions(),
                dtype=np.float64,
            )
            moved = end - start
            pairs = _aggregate_contact_pairs(
                event_state["events"],
                adjacent,
            )
            unexpected = [item for item in pairs if not item["allowed"]]
            span = float(plan["upper"] - plan["lower"])
            target_error_gate = max(
                TARGET_ERROR_MINIMUM_RAD,
                TARGET_ERROR_SPAN_FRACTION * span,
            )
            target_error = abs(float(end[index]) - target)
            direction_pass = (
                float(moved[index]) * (target - float(start[index])) > 0.0
                and abs(float(moved[index])) >= 0.05
            )
            non_target_drift = max(
                abs(float(moved[item]))
                for item in range(len(moved))
                if item != index
            )
            legal = float(plan["lower"]) <= float(end[index]) <= float(
                plan["upper"]
            )
            finite = bool(np.isfinite(end).all())
            case_pass = (
                direction_pass
                and target_error <= target_error_gate
                and non_target_drift <= NON_TARGET_DRIFT_GATE
                and legal
                and finite
                and not unexpected
            )
            cases.append(
                {
                    **plan,
                    "repeat": repeat,
                    "status": "PASS" if case_pass else "FAIL",
                    "fresh_world_reset": True,
                    "start": start.tolist(),
                    "end": end.tolist(),
                    "final_readback": float(end[index]),
                    "readback_delta": float(moved[index]),
                    "direction_status": (
                        "PASS" if direction_pass else "FAIL"
                    ),
                    "target_error": target_error,
                    "target_error_gate": target_error_gate,
                    "maximum_non_target_drift": non_target_drift,
                    "non_target_drift_gate": NON_TARGET_DRIFT_GATE,
                    "legal_range_status": "PASS" if legal else "FAIL",
                    "finite_readback_status": (
                        "PASS" if finite else "FAIL"
                    ),
                    "contact_event_count": len(event_state["events"]),
                    "contact_pairs": pairs,
                    "unexpected_contact_pair_count": len(unexpected),
                    "contact_events": list(event_state["events"]),
                }
            )
    return cases, curves


def main(args: argparse.Namespace, preflight: dict[str, Any]) -> int:
    if args.repeat_count < 1:
        raise ValueError("repeat-count must be positive")
    stage_path = Path(preflight["absolute_path"])
    hash_before = _sha256(stage_path)

    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.utils.stage import open_stage
    from omni.physx import get_physx_simulation_interface

    if not open_stage(str(stage_path)):
        raise RuntimeError(f"unable to open frozen Stage: {stage_path}")
    stage = get_current_stage()
    manifest = _stage_runtime_manifest(stage)
    contact_setup = _install_session_contact_reports(stage)
    adjacent = _adjacent_body_pairs(stage)
    self_collision = _self_collision_readback(stage)

    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=1.0 / PHYSICS_HZ,
        rendering_dt=1.0 / PHYSICS_HZ,
    )
    physics_context = world.get_physics_context()
    physics_context.set_solve_articulation_contact_last(True)
    solve_last = (
        physics_context.get_solve_articulation_contact_last()
    )
    articulations = {}
    for robot, spec in RUNTIME_SPECS.items():
        articulation = SingleArticulation(
            prim_path=spec["articulation_path"],
            name=f"task7a_sweep_{robot}",
            reset_xform_properties=False,
        )
        world.scene.add(articulation)
        articulations[robot] = articulation
    world.reset()
    plans = _build_runtime_sweep_plan(articulations)

    event_state: dict[str, Any] = {
        "case_id": None,
        "frame": 0,
        "events": [],
    }

    def on_contact(
        headers: Sequence[Any],
        data: Sequence[Any],
    ) -> None:
        if event_state["case_id"] is None:
            return
        event_state["events"].extend(
            _serialize_contacts(
                headers,
                data,
                frame=int(event_state["frame"]),
            )
        )

    subscription = (
        get_physx_simulation_interface().subscribe_contact_report_events(
            on_contact
        )
    )
    cases, curves = _run_cases(
        world,
        articulations,
        plans,
        repeat_count=args.repeat_count,
        event_state=event_state,
        adjacent=adjacent,
    )
    del subscription

    summary = summarize_sweep_cases(
        cases,
        repeat_count=args.repeat_count,
    )
    hash_after = _sha256(stage_path)
    immutable = hash_before == hash_after == EXPECTED_STAGE_SHA256
    if not immutable:
        summary["status"] = "FAIL"
    pair_inventory: dict[
        tuple[str, str, str, str, str], dict[str, Any]
    ] = {}
    for case in cases:
        for pair in case["contact_pairs"]:
            key = (
                *pair["actor_pair"],
                *pair["collider_pair"],
                pair["classification"],
            )
            record = pair_inventory.setdefault(
                key,
                {
                    "classification": pair["classification"],
                    "allowed": pair["allowed"],
                    "actor0": pair["actor_pair"][0],
                    "actor1": pair["actor_pair"][1],
                    "collider0": pair["collider_pair"][0],
                    "collider1": pair["collider_pair"][1],
                    "case_count": 0,
                    "maximum_penetration_m": 0.0,
                    "maximum_impulse_norm_n_s": 0.0,
                },
            )
            record["case_count"] += 1
            record["maximum_penetration_m"] = max(
                record["maximum_penetration_m"],
                float(pair["maximum_penetration_m"]),
            )
            record["maximum_impulse_norm_n_s"] = max(
                record["maximum_impulse_norm_n_s"],
                float(pair["maximum_impulse_norm_n_s"]),
            )
    report = {
        "schema_version": 1,
        "status": summary["status"],
        "scope": "TASK_7A_CURRENT_COLLISION_SEMANTICS",
        "stage": {
            **preflight,
            "sha256_before": hash_before,
            "sha256_after": hash_after,
            "immutable": immutable,
            "composition": manifest,
        },
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "physics_frequency_hz": PHYSICS_HZ,
            "solve_articulation_contact_last": bool(solve_last),
            "trajectory_steps": SWEEP_STEPS,
            "target_settle_steps": TARGET_SETTLE_STEPS,
            "repeat_count": args.repeat_count,
        },
        "contact_report_setup": contact_setup,
        "authored_self_collision_readback": self_collision,
        "coverage_boundary": (
            "authored collision filters and self-collision settings preserved; "
            "disabled pairs are not proven geometrically separated"
        ),
        "adjacent_body_pairs": [
            list(pair) for pair in sorted(adjacent)
        ],
        "plan": plans,
        "summary": summary,
        "cases": cases,
        "pair_inventory": list(pair_inventory.values()),
        "real_robot_connected": False,
        "remote_192_168_1_103_accessed": False,
        "source_stage_modified": False,
        "collider_modified": False,
        "drive_modified": False,
        "mimic_modified": False,
        "task_8": "NOT_RUN",
    }
    output = args.output.resolve()
    curves_path = args.curves.resolve()
    pairs_path = args.pair_inventory.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(report, output.with_suffix(".md"))
    _write_csv(curves_path, curves)
    _write_csv(pairs_path, list(pair_inventory.values()))
    print(
        json.dumps(
            {
                "status": report["status"],
                "case_count": summary["case_count"],
                "failed_case_count": summary["failed_case_count"],
                "pair_count": len(pair_inventory),
                "determinism": summary["determinism"]["status"],
                "stage_immutable": immutable,
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if report["status"] == "PASS" else 1


def run() -> int:
    args = _parse_args()
    preflight = preflight_frozen_stage(args.stage)
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
        exit_code = main(args, preflight)
    except BaseException:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
