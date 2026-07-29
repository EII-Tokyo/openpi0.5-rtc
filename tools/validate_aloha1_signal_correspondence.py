#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Run Task 7A one-joint and small up/down validation in Isaac Sim 5.1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.signal_correspondence import ACTIVE_ONE_JOINT_TESTS
from tools.aloha1_mapping.signal_correspondence import HOME_ARM
from tools.aloha1_mapping.signal_correspondence import HOME_LEFT_FINGER_M
from tools.aloha1_mapping.signal_correspondence import HOME_RIGHT_FINGER_M
from tools.aloha1_mapping.signal_correspondence import RUNTIME_SPECS
from tools.aloha1_mapping.signal_correspondence import build_signal_mapping_plan
from tools.aloha1_mapping.signal_correspondence import build_small_up_down_targets
from tools.aloha1_mapping.signal_correspondence import canonical_dof_name
from tools.aloha1_mapping.signal_correspondence import classify_task7a_status

ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    ROOT / "assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda"
)
REPORT_ROOT = ROOT / "reports/aloha1_mapping"
ARTIFACT_ROOT = ROOT / ".codex/artifacts/20260729-aloha1-signal-correspondence"
PHYSICS_HZ = 60
ONE_JOINT_DELTA_RAD = 0.05
ONE_JOINT_STEPS = 45
HOME_SETTLE_STEPS = 30
UP_DOWN_STEPS = 60
UP_DOWN_REPEATS = 3
ONE_JOINT_REPEATS = 2
OFFICIAL_RULES_REPORT = REPORT_ROOT / "aloha1_signal_correspondence_official_rules.json"
SWEPT_COLLISION_REPORT = (
    REPORT_ROOT / "aloha1_task7a_swept_collision.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _smoothstep(frame: int, frames: int) -> float:
    value = frame / frames
    return value * value * (3.0 - 2.0 * value)


def _home(robot: str) -> np.ndarray:
    values = [
        *HOME_ARM,
        0.0,
        HOME_LEFT_FINGER_M,
        HOME_RIGHT_FINGER_M,
    ]
    return np.asarray(values, dtype=np.float32)


def _active_drive_indices(robot: str) -> np.ndarray:
    count = len(RUNTIME_SPECS[robot]["runtime_expected_order"])
    return np.arange(count - 1, dtype=np.int32)


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
    for robot, articulation in articulations.items():
        home = _home(robot)
        articulation.set_joint_positions(home)
        articulation.set_joint_velocities(np.zeros_like(home))
        indices = _active_drive_indices(robot)
        _apply_targets(articulation, home[indices], indices)
    for _ in range(HOME_SETTLE_STEPS):
        world.step(render=False)
    return {
        robot: np.asarray(
            articulation.get_joint_positions(),
            dtype=np.float64,
        )
        for robot, articulation in articulations.items()
    }


def _eef_position(robot: str) -> list[float]:
    from isaacsim.core.utils.xforms import get_world_pose

    position, _ = get_world_pose(RUNTIME_SPECS[robot]["end_effector_path"])
    return [float(value) for value in position]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _inspect_drive_mimic_structure(stage_path: Path) -> dict[str, Any]:
    from pxr import PhysxSchema
    from pxr import Usd
    from pxr import UsdPhysics

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"unable to inspect Stage: {stage_path}")
    robots = {}
    for robot, spec in RUNTIME_SPECS.items():
        joint_root = str(Path(spec["articulation_path"]).parent / "joints")
        records = []
        for name in spec["runtime_expected_order"]:
            prim = stage.GetPrimAtPath(f"{joint_root}/{name}")
            axis = "linear" if prim.IsA(UsdPhysics.PrismaticJoint) else "angular"
            drive = UsdPhysics.DriveAPI(prim, axis)
            has_drive = bool(drive) and bool(drive.GetTypeAttr().Get())
            mimic = prim.HasAPI(PhysxSchema.PhysxMimicJointAPI)
            physx_joint = PhysxSchema.PhysxJointAPI(prim)
            max_velocity = physx_joint.GetMaxJointVelocityAttr().Get()
            stiffness = float(drive.GetStiffnessAttr().Get()) if has_drive else 0.0
            damping = float(drive.GetDampingAttr().Get()) if has_drive else 0.0
            max_force = float(drive.GetMaxForceAttr().Get()) if has_drive else None
            legal = (
                (mimic and not has_drive and stiffness == 0.0 and damping == 0.0)
                or (not mimic and has_drive and max_force is not None and np.isfinite(max_force) and max_force > 0.0)
            ) and (max_velocity is not None and np.isfinite(float(max_velocity)) and float(max_velocity) > 0.0)
            records.append(
                {
                    "name": name,
                    "prim_path": str(prim.GetPath()),
                    "has_drive": has_drive,
                    "drive_type": (str(drive.GetTypeAttr().Get()) if has_drive else None),
                    "mimic": mimic,
                    "stiffness": stiffness,
                    "damping": damping,
                    "max_force": max_force,
                    "max_velocity": float(max_velocity),
                    "status": "PASS" if legal else "FAIL",
                }
            )
        robots[robot] = {
            "status": (
                "PASS"
                if all(item["status"] == "PASS" for item in records) and sum(item["mimic"] for item in records) == 1
                else "FAIL"
            ),
            "dofs": records,
        }
    status = "PASS" if all(item["status"] == "PASS" for item in robots.values()) else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "stage": str(stage_path),
        "stage_sha256": _sha256(stage_path),
        "robots": robots,
        "official_rule_results_suppressed": False,
    }


def _run_structure_checks(
    world: Any,
    articulations: dict[str, Any],
    properties: dict[str, Any],
) -> dict[str, Any]:
    world.reset()
    initial = {
        name: np.asarray(
            articulation.get_joint_positions(),
            dtype=np.float64,
        )
        for name, articulation in articulations.items()
    }
    world.step(render=False)
    first = {
        name: np.asarray(
            articulation.get_joint_positions(),
            dtype=np.float64,
        )
        for name, articulation in articulations.items()
    }
    homes = _prepare_home(world, articulations)
    robots = {}
    for robot, articulation in articulations.items():
        order = list(articulation.dof_names)
        active = _active_drive_indices(robot)
        expected_home = _home(robot)
        max_first_jump = float(np.max(np.abs(first[robot] - initial[robot])))
        maximum_target_error = float(np.max(np.abs(homes[robot][active] - expected_home[active])))
        mimic_error = abs(
            float(homes[robot][order.index("right_finger")]) + float(homes[robot][order.index("left_finger")])
        )
        props = properties[robot]
        finite_active_limits = all(
            np.isfinite(float(props[index]["maxVelocity"]))
            and float(props[index]["maxVelocity"]) > 0.0
            and np.isfinite(float(props[index]["maxEffort"]))
            and float(props[index]["maxEffort"]) > 0.0
            for index in active
        )
        passed = (
            max_first_jump <= 0.02 and maximum_target_error <= 0.02 and mimic_error <= 0.001 and finite_active_limits
        )
        robots[robot] = {
            "status": "PASS" if passed else "FAIL",
            "articulation_path": RUNTIME_SPECS[robot]["articulation_path"],
            "dof_order": order,
            "first_frame_jump_rad_or_m": max_first_jump,
            "first_frame_jump_tolerance": 0.02,
            "home_target_readback_max_error": maximum_target_error,
            "home_target_readback_tolerance": 0.02,
            "mimic_home_error_m": mimic_error,
            "finite_positive_active_max_velocity_and_force": (finite_active_limits),
        }
    return {
        "schema_version": 1,
        "status": (
            "PASS" if len(articulations) == 2 and all(item["status"] == "PASS" for item in robots.values()) else "FAIL"
        ),
        "articulation_count": len(articulations),
        "expected_articulation_count": 2,
        "robots": robots,
    }


def _run_one_joint(
    world: Any,
    articulations: dict[str, Any],
    *,
    robot: str,
    properties: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    articulation = articulations[robot]
    runtime_names = list(articulation.dof_names)
    canonical = [canonical_dof_name(robot, item) for item in runtime_names]
    cases = []
    curves: list[dict[str, Any]] = []
    for repeat in range(ONE_JOINT_REPEATS):
        for joint in ACTIVE_ONE_JOINT_TESTS:
            index = canonical.index(joint)
            magnitude = 0.002 if joint == "left_finger" else ONE_JOINT_DELTA_RAD
            for direction, requested_delta in (
                ("negative", -magnitude),
                ("positive", magnitude),
            ):
                homes = _prepare_home(world, articulations)
                start = homes[robot]
                lower = float(properties[index]["lower"])
                upper = float(properties[index]["upper"])
                limit_margin = magnitude * 0.1
                target = float(
                    np.clip(
                        start[index] + requested_delta,
                        lower + limit_margin,
                        upper - limit_margin,
                    )
                )
                command_delta = target - float(start[index])
                case_steps = 90 if joint == "left_finger" else ONE_JOINT_STEPS
                for frame in range(1, case_steps + 1):
                    alpha = _smoothstep(frame, case_steps)
                    command = float(start[index]) + alpha * command_delta
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
                    eef = _eef_position(robot)
                    curves.append(
                        {
                            "robot": robot,
                            "repeat": repeat,
                            "test": f"{joint}_{direction}",
                            "joint": joint,
                            "joint_index": index,
                            "direction": direction,
                            "frame": frame,
                            "time_s": frame / PHYSICS_HZ,
                            "unit": ("m" if joint == "left_finger" else "rad"),
                            "command_target": command,
                            "joint_readback": float(qpos[index]),
                            "position_error": command - float(qpos[index]),
                            "joint_velocity": float(qvel[index]),
                            "end_effector_x_m": eef[0],
                            "end_effector_y_m": eef[1],
                            "end_effector_z_m": eef[2],
                        }
                    )
                end = np.asarray(
                    articulation.get_joint_positions(),
                    dtype=np.float64,
                )
                moved = end - start
                readback_delta = float(moved[index])
                minimum_readback = 1.0e-4 if joint == "left_finger" else 0.005
                direction_pass = readback_delta * command_delta > 0.0 and abs(readback_delta) >= minimum_readback
                target_error = abs(float(end[index]) - target)
                excluded = {index}
                mimic_error = None
                mimic_readback = None
                mimic_pass = True
                if joint == "left_finger":
                    mimic_index = canonical.index("right_finger")
                    excluded.add(mimic_index)
                    mimic_readback = float(end[mimic_index])
                    mimic_error = abs(mimic_readback + float(end[index]))
                    mimic_pass = mimic_error <= 0.001
                drift = max(
                    (abs(float(moved[item])) for item in range(len(moved)) if item not in excluded),
                    default=0.0,
                )
                legal = lower <= float(end[index]) <= upper
                case_pass = direction_pass and target_error <= 0.02 and drift <= 0.01 and legal and mimic_pass
                cases.append(
                    {
                        "status": "PASS" if case_pass else "FAIL",
                        "robot": robot,
                        "repeat": repeat,
                        "joint": joint,
                        "isaac_dof_name": runtime_names[index],
                        "isaac_dof_index": index,
                        "direction": direction,
                        "unit": ("m" if joint == "left_finger" else "rad"),
                        "requested_delta": requested_delta,
                        "requested_delta_rad": requested_delta,
                        "command_delta_rad": command_delta,
                        "readback_delta_rad": readback_delta,
                        "target_rad": target,
                        "readback_rad": float(end[index]),
                        "position_error_rad": target_error,
                        "maximum_non_target_drift": drift,
                        "mimic_readback": mimic_readback,
                        "mimic_error": mimic_error,
                        "mimic_status": ("PASS" if mimic_pass else "FAIL"),
                        "legal_range": legal,
                        "fresh_world_reset": True,
                        "start": start.tolist(),
                        "end": end.tolist(),
                    }
                )
    signatures = []
    for repeat in range(ONE_JOINT_REPEATS):
        payload = [
            {
                "joint": item["joint"],
                "direction": item["direction"],
                "status": item["status"],
                "readback_delta": round(item["readback_delta_rad"], 6),
                "end": [round(float(value), 6) for value in item["end"]],
            }
            for item in cases
            if item["repeat"] == repeat
        ]
        signatures.append(
            hashlib.sha256(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        )
    deterministic = len(set(signatures)) == 1
    status = "PASS" if all(item["status"] == "PASS" for item in cases) and deterministic else "FAIL"
    return (
        {
            "schema_version": 1,
            "status": status,
            "robot": robot,
            "stage": str(STAGE.resolve()),
            "stage_sha256": _sha256(STAGE),
            "articulation_path": RUNTIME_SPECS[robot]["articulation_path"],
            "runtime_dof_order": runtime_names,
            "canonical_dof_order": canonical,
            "case_count": len(cases),
            "cases": cases,
            "determinism": {
                "status": "PASS" if deterministic else "FAIL",
                "repeat_count": ONE_JOINT_REPEATS,
                "signatures": signatures,
            },
            "task_8": "NOT_RUN",
            "real_robot_connected": False,
        },
        curves,
    )


def _run_up_down(
    world: Any,
    articulations: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_plan = build_small_up_down_targets()
    rows: list[dict[str, Any]] = []
    runs = []
    signatures = []
    for repeat in range(UP_DOWN_REPEATS):
        homes = _prepare_home(world, articulations)
        home_eef = {robot: _eef_position(robot) for robot in articulations}
        phase_records = {
            robot: {
                "home": {
                    "qpos": homes[robot].tolist(),
                    "end_effector_position_m": home_eef[robot],
                }
            }
            for robot in articulations
        }
        for phase, final_delta in (("small_up", -0.08), ("return_home", 0.0)):
            phase_start = {
                robot: np.asarray(
                    articulation.get_joint_positions(),
                    dtype=np.float64,
                )
                for robot, articulation in articulations.items()
            }
            for frame in range(1, UP_DOWN_STEPS + 1):
                alpha = _smoothstep(frame, UP_DOWN_STEPS)
                for robot, articulation in articulations.items():
                    canonical = [canonical_dof_name(robot, name) for name in articulation.dof_names]
                    index = canonical.index("shoulder")
                    desired = float(HOME_ARM[1] + final_delta)
                    command = float(phase_start[robot][index]) + alpha * (desired - float(phase_start[robot][index]))
                    _apply_targets(
                        articulation,
                        np.asarray([command], dtype=np.float32),
                        np.asarray([index], dtype=np.int32),
                    )
                world.step(render=False)
                for robot, articulation in articulations.items():
                    canonical = [canonical_dof_name(robot, name) for name in articulation.dof_names]
                    index = canonical.index("shoulder")
                    qpos = np.asarray(
                        articulation.get_joint_positions(),
                        dtype=np.float64,
                    )
                    qvel = np.asarray(
                        articulation.get_joint_velocities(),
                        dtype=np.float64,
                    )
                    eef = _eef_position(robot)
                    desired = float(HOME_ARM[1] + final_delta)
                    rows.append(
                        {
                            "repeat": repeat,
                            "robot": robot,
                            "phase": phase,
                            "frame": frame,
                            "simulation_time_s": frame / PHYSICS_HZ,
                            "joint": "shoulder",
                            "isaac_dof_index": index,
                            "command_target_rad": desired,
                            "joint_readback_rad": float(qpos[index]),
                            "position_error_rad": (desired - float(qpos[index])),
                            "joint_velocity_rad_s": float(qvel[index]),
                            "end_effector_x_m": eef[0],
                            "end_effector_y_m": eef[1],
                            "end_effector_z_m": eef[2],
                            "delta_z_from_home_m": (eef[2] - home_eef[robot][2]),
                        }
                    )
            for robot, articulation in articulations.items():
                canonical = [canonical_dof_name(robot, name) for name in articulation.dof_names]
                index = canonical.index("shoulder")
                qpos = np.asarray(
                    articulation.get_joint_positions(),
                    dtype=np.float64,
                )
                eef = _eef_position(robot)
                phase_records[robot][phase] = {
                    "qpos": qpos.tolist(),
                    "shoulder_target_rad": float(HOME_ARM[1] + final_delta),
                    "shoulder_readback_rad": float(qpos[index]),
                    "end_effector_position_m": eef,
                    "delta_z_from_home_m": eef[2] - home_eef[robot][2],
                }
        run_robots = {}
        for robot in articulations:
            up = phase_records[robot]["small_up"]
            returned = phase_records[robot]["return_home"]
            up_pass = up["delta_z_from_home_m"] >= 0.005
            return_pass = abs(returned["delta_z_from_home_m"]) <= 0.01
            target_pass = abs(up["shoulder_readback_rad"] - up["shoulder_target_rad"]) <= 0.02
            run_robots[robot] = {
                "status": ("PASS" if up_pass and return_pass and target_pass else "FAIL"),
                "phases": phase_records[robot],
                "end_effector_z_direction": ("PASS" if up_pass else "FAIL"),
                "return_to_home": "PASS" if return_pass else "FAIL",
                "target_readback": "PASS" if target_pass else "FAIL",
                "minimum_table_clearance_method": ("END_EFFECTOR_Z_MINUS_USER_CONFIRMED_TABLE_TOP_ONLY"),
                "minimum_end_effector_clearance_m": min(
                    phase_records[robot][phase]["end_effector_position_m"][2] + 0.09090000152587889
                    for phase in ("home", "small_up", "return_home")
                ),
            }
        run_status = "PASS" if all(item["status"] == "PASS" for item in run_robots.values()) else "FAIL"
        runs.append(
            {
                "repeat": repeat,
                "status": run_status,
                "fresh_world_reset": True,
                "robots": run_robots,
            }
        )
        signature_payload = {
            robot: {
                phase: {
                    "qpos": [round(float(value), 6) for value in phase_records[robot][phase]["qpos"]],
                    "eef": [round(float(value), 6) for value in phase_records[robot][phase]["end_effector_position_m"]],
                }
                for phase in ("home", "small_up", "return_home")
            }
            for robot in articulations
        }
        signatures.append(
            hashlib.sha256(
                json.dumps(
                    signature_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        )
    deterministic = len(set(signatures)) == 1
    status = "PASS" if all(run["status"] == "PASS" for run in runs) and deterministic else "FAIL"
    return (
        {
            "schema_version": 1,
            "status": status,
            "scope": "DIGITAL_ONLY_NO_REAL_ROBOT_CONNECTION",
            "stage": str(STAGE.resolve()),
            "stage_sha256": _sha256(STAGE),
            "physics_frequency_hz": PHYSICS_HZ,
            "target_plan": target_plan,
            "runs": runs,
            "determinism": {
                "status": "PASS" if deterministic else "FAIL",
                "repeat_count": UP_DOWN_REPEATS,
                "signatures": signatures,
            },
            "collision_gate": {
                "status": "PARTIAL",
                "reason": (
                    "end-effector/table clearance is numeric; full swept-link contact monitoring remains pending"
                ),
            },
            "dynamic_response": "RECORDED_NOT_SIM2REAL_CALIBRATED",
            "task_8": "NOT_RUN",
            "real_robot_connected": False,
        },
        rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=STAGE)
    args = parser.parse_args()
    stage_path = args.stage.resolve(strict=True)
    if stage_path != STAGE.resolve():
        raise ValueError("this frozen run only accepts the user-confirmed baseline Stage")
    stage_hash_before = _sha256(stage_path)

    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import open_stage

    if not open_stage(str(stage_path)):
        raise RuntimeError(f"failed to open Stage: {stage_path}")
    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=1.0 / PHYSICS_HZ,
        rendering_dt=1.0 / PHYSICS_HZ,
    )
    world.get_physics_context().set_solve_articulation_contact_last(True)
    articulations = {}
    for robot, spec in RUNTIME_SPECS.items():
        articulation = SingleArticulation(
            prim_path=spec["articulation_path"],
            name=f"signal_validation_{robot}",
            reset_xform_properties=False,
        )
        world.scene.add(articulation)
        articulations[robot] = articulation
    world.reset()

    mapping = build_signal_mapping_plan(ROOT)
    inventory_checks = {}
    properties = {}
    for robot, articulation in articulations.items():
        order = list(articulation.dof_names)
        expected = RUNTIME_SPECS[robot]["runtime_expected_order"]
        inventory_checks[robot] = {
            "status": "PASS" if order == expected else "FAIL",
            "expected_order": expected,
            "runtime_order": order,
            "articulation_path": RUNTIME_SPECS[robot]["articulation_path"],
        }
        properties[robot] = articulation.dof_properties.copy()
    if not all(item["status"] == "PASS" for item in inventory_checks.values()):
        raise RuntimeError(f"runtime order mismatch: {inventory_checks}")

    structure = _run_structure_checks(
        world,
        articulations,
        properties,
    )
    drive_mimic = _inspect_drive_mimic_structure(stage_path)
    structure["drive_mimic_structure"] = drive_mimic
    structure_path = REPORT_ROOT / "aloha1_signal_task7a_structure_validation.json"
    structure_path.write_text(
        json.dumps(structure, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    one_joint_reports = {}
    for robot in articulations:
        report, rows = _run_one_joint(
            world,
            articulations,
            robot=robot,
            properties=properties[robot],
        )
        json_path = REPORT_ROOT / f"aloha1_{robot}_one_joint_validation.json"
        csv_path = REPORT_ROOT / f"aloha1_{robot}_one_joint_curves.csv"
        report["curve_csv"] = str(csv_path.resolve())
        json_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_csv(csv_path, rows)
        one_joint_reports[robot] = report

    up_down, up_down_rows = _run_up_down(world, articulations)
    up_down_json = REPORT_ROOT / "aloha1_digital_up_down_motion.json"
    up_down_csv = REPORT_ROOT / "aloha1_digital_up_down_motion_curves.csv"
    up_down["curve_csv"] = str(up_down_csv.resolve())
    up_down_json.write_text(
        json.dumps(up_down, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(up_down_csv, up_down_rows)

    stage_hash_after = _sha256(stage_path)
    immutable = stage_hash_before == stage_hash_after
    mapping["runtime_inventory"] = inventory_checks
    mapping["stage"] = {
        "absolute_path": str(stage_path),
        "sha256_before": stage_hash_before,
        "sha256_after": stage_hash_after,
        "immutable": immutable,
    }
    mapping["status"] = (
        "PASS"
        if immutable
        and all(report["status"] == "PASS" for report in one_joint_reports.values())
        and up_down["status"] == "PASS"
        else "FAIL"
    )
    mapping_path = REPORT_ROOT / "aloha1_joint_mapping_validation.json"
    mapping_path.write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    official_rules = json.loads(OFFICIAL_RULES_REPORT.resolve(strict=True).read_text(encoding="utf-8"))
    swept_collision = json.loads(
        SWEPT_COLLISION_REPORT.resolve(strict=True).read_text(
            encoding="utf-8"
        )
    )
    if swept_collision["stage"]["sha256_after"] != stage_hash_after:
        raise RuntimeError("swept-collision report Stage hash mismatch")
    task7a_status = classify_task7a_status(
        mapping_status=mapping["status"],
        structure_status=structure["status"],
        drive_mimic_status=drive_mimic["status"],
        small_up_down_status=up_down["status"],
        swept_collision_status=swept_collision["status"],
        official_task7a_status=official_rules["task7a_applicable_status"],
    )
    summary = {
        "schema_version": 1,
        "status": task7a_status,
        "task_7a": {
            "status": task7a_status,
            "structure_and_runtime_order": (
                "PASS" if all(item["status"] == "PASS" for item in inventory_checks.values()) else "FAIL"
            ),
            "joint_mapping": mapping["status"],
            "follower_left_one_joint": one_joint_reports["follower_left"]["status"],
            "follower_right_one_joint": one_joint_reports["follower_right"]["status"],
            "small_up_down": up_down["status"],
            "drive_mimic_structure": drive_mimic["status"],
            "initial_target_readback_first_frame": structure["status"],
            "official_rules": official_rules["task7a_applicable_status"],
            "official_rules_unsuppressed_status": official_rules["official_status"],
            "collision_swept_path": swept_collision["status"],
            "reason_if_partial": (
                "NVIDIA official PhysicsRules/RobotRules findings remain "
                "unsuppressed; runtime signal gates are separately PASS"
            ),
            "reason_if_fail": (
                "deterministic positive-shoulder sweeps on both followers "
                "produce physical CAD-finger/table contact before the "
                "authored upper joint target"
                if swept_collision["status"] == "FAIL"
                else None
            ),
        },
        "task_7b": {
            "status": "NOT_RUN",
            "deferred": [
                "bottle_static_hold",
                "friction_calibration",
                "full_simready",
            ],
        },
        "task_8": "NOT_RUN",
        "real_robot_connected": False,
        "remote_192_168_1_103_accessed": False,
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "solve_articulation_contact_last": True,
        },
        "stage": mapping["stage"],
        "reports": {
            "joint_mapping": str(mapping_path.resolve()),
            "follower_left_one_joint": str((REPORT_ROOT / "aloha1_follower_left_one_joint_validation.json").resolve()),
            "follower_right_one_joint": str(
                (REPORT_ROOT / "aloha1_follower_right_one_joint_validation.json").resolve()
            ),
            "digital_up_down": str(up_down_json.resolve()),
            "structure": str(structure_path.resolve()),
            "official_rules": str(OFFICIAL_RULES_REPORT.resolve()),
            "swept_collision": str(SWEPT_COLLISION_REPORT.resolve()),
        },
    }
    summary_path = REPORT_ROOT / "aloha1_task7a_7b_validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "mapping": mapping["status"],
                "left_one_joint": one_joint_reports["follower_left"]["status"],
                "right_one_joint": one_joint_reports["follower_right"]["status"],
                "up_down": up_down["status"],
                "summary": str(summary_path.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if summary["status"] in {"PASS", "PARTIAL"} else 1


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
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
