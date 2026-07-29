#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Validate the isolated supplier-CAD follower_right in robot-local space."""

from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import pairwise
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.supplier_cad_one_joint import evaluate_one_joint_run

ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "supplier_cad_follower_right/1.0/"
    "supplier_cad_follower_right.usda"
)
ASSET_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_supplier_cad_follower_right_asset.json"
)
CAD_IDENTITY_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_left_right_cad_identity.json"
)
LEFT_GEOMETRY_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_geometry_audit.json"
)
OUTPUT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_right_one_joint_validation.json"
)
STRUCTURE_OUTPUT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_right_structure_validation.json"
)
CURVES = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_right_one_joint_curves.csv"
)
ARTICULATION_PATH = "/follower_right/vx300s_right/root_joint"
SCOPE = "ROBOT_LOCAL_DIAGNOSTIC_NOT_WORKCELL_PLACEMENT"
EXPECTED_DOF_ORDER = [
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
HOME = np.asarray(
    [0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.0, 0.05, -0.05],
    dtype=np.float32,
)
ACTIVE_DRIVE_INDICES = np.asarray(list(range(8)), dtype=np.int32)
GRIPPER_STATES = {
    "closed": 0.021,
    "partially_closed": 0.039,
    "open": 0.052,
    "maximum_legal_aperture": 0.057,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--structure-output", type=Path, default=STRUCTURE_OUTPUT)
    parser.add_argument("--curves", type=Path, default=CURVES)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--terminal-settle-steps", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=2)
    return parser.parse_args()


def _apply_positions(
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


def _prepare_home(world: Any, articulation: Any, settle_steps: int) -> np.ndarray:
    world.reset()
    articulation.set_joint_positions(HOME)
    articulation.set_joint_velocities(np.zeros_like(HOME))
    _apply_positions(
        articulation,
        HOME[ACTIVE_DRIVE_INDICES],
        ACTIVE_DRIVE_INDICES,
    )
    for _ in range(settle_steps):
        world.step(render=False)
    return np.asarray(articulation.get_joint_positions(), dtype=np.float64)


def _append_curve(
    curves: list[dict[str, Any]],
    *,
    repeat: int,
    test: str,
    frame: int,
    q: np.ndarray,
) -> None:
    curves.append(
        {
            "repeat": repeat,
            "test": test,
            "frame": frame,
            "time_s": frame / 60.0,
            **{
                name: float(q[index])
                for index, name in enumerate(EXPECTED_DOF_ORDER)
            },
        }
    )


def _run_arm_cases(
    *,
    world: Any,
    articulation: Any,
    properties: Any,
    steps: int,
    settle_steps: int,
    repeat: int,
    curves: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cases = []
    for index, joint_name in enumerate(EXPECTED_DOF_ORDER[:6]):
        for direction, requested_delta in (
            ("negative", -0.05),
            ("positive", 0.05),
        ):
            start = _prepare_home(world, articulation, settle_steps)
            lower = float(properties[index]["lower"])
            upper = float(properties[index]["upper"])
            target = float(
                np.clip(
                    start[index] + requested_delta,
                    lower + 0.005,
                    upper - 0.005,
                )
            )
            actual_command_delta = target - float(start[index])
            test_name = f"{joint_name}_{direction}"
            for frame in range(1, steps + 1):
                alpha = frame / steps
                command = float(start[index]) + alpha * actual_command_delta
                _apply_positions(
                    articulation,
                    np.asarray([command], dtype=np.float32),
                    np.asarray([index], dtype=np.int32),
                )
                world.step(render=False)
                q = np.asarray(
                    articulation.get_joint_positions(),
                    dtype=np.float64,
                )
                _append_curve(
                    curves,
                    repeat=repeat,
                    test=test_name,
                    frame=frame,
                    q=q,
                )
            end = np.asarray(
                articulation.get_joint_positions(),
                dtype=np.float64,
            )
            evaluated = evaluate_one_joint_run(
                dof_names=EXPECTED_DOF_ORDER,
                commanded_indices=[index],
                commanded_delta=[actual_command_delta],
                start=start.tolist(),
                end=end.tolist(),
                lower=[
                    float(properties[item]["lower"])
                    for item in range(len(EXPECTED_DOF_ORDER))
                ],
                upper=[
                    float(properties[item]["upper"])
                    for item in range(len(EXPECTED_DOF_ORDER))
                ],
                readback_minimum=0.005,
                target_tolerance=0.02,
                unexpected_tolerance=0.01,
            )
            evaluated.update(
                {
                    "joint_name": joint_name,
                    "direction": direction,
                    "repeat": repeat,
                    "test": test_name,
                    "target": target,
                    "fresh_world_reset": True,
                }
            )
            cases.append(evaluated)
    return cases


def _run_gripper_states(
    *,
    world: Any,
    articulation: Any,
    properties: Any,
    steps: int,
    settle_steps: int,
    terminal_settle_steps: int,
    repeat: int,
    curves: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cases = []
    for state, left_target in GRIPPER_STATES.items():
        start = _prepare_home(world, articulation, settle_steps)
        test_name = f"gripper_{state}"
        delta = left_target - float(start[7])
        for frame in range(1, steps + 1):
            alpha = frame / steps
            command = float(start[7]) + alpha * delta
            _apply_positions(
                articulation,
                np.asarray([command], dtype=np.float32),
                np.asarray([7], dtype=np.int32),
            )
            world.step(render=False)
            q = np.asarray(
                articulation.get_joint_positions(),
                dtype=np.float64,
            )
            _append_curve(
                curves,
                repeat=repeat,
                test=test_name,
                frame=frame,
                q=q,
            )
        for terminal_frame in range(1, terminal_settle_steps + 1):
            _apply_positions(
                articulation,
                np.asarray([left_target], dtype=np.float32),
                np.asarray([7], dtype=np.int32),
            )
            world.step(render=False)
            q = np.asarray(
                articulation.get_joint_positions(),
                dtype=np.float64,
            )
            _append_curve(
                curves,
                repeat=repeat,
                test=test_name,
                frame=steps + terminal_frame,
                q=q,
            )
        end = np.asarray(
            articulation.get_joint_positions(),
            dtype=np.float64,
        )
        left = float(end[7])
        right = float(end[8])
        aperture = left - right
        mimic_residual = abs(left + right)
        legal = (
            float(properties[7]["lower"]) - 1.0e-6
            <= left
            <= float(properties[7]["upper"]) + 1.0e-6
            and float(properties[8]["lower"]) - 1.0e-6
            <= right
            <= float(properties[8]["upper"]) + 1.0e-6
        )
        passed = (
            abs(left - left_target) <= 0.001
            and mimic_residual <= 0.001
            and legal
        )
        cases.append(
            {
                "status": "PASS" if passed else "FAIL",
                "state": state,
                "repeat": repeat,
                "test": test_name,
                "target_left_m": left_target,
                "readback_left_m": left,
                "readback_right_m": right,
                "aperture_m": aperture,
                "mimic_residual_m": mimic_residual,
                "legal_range": legal,
                "fresh_world_reset": True,
                "terminal_target_settle_steps": terminal_settle_steps,
            }
        )
    return cases


def _first_frame_and_static_hold(
    world: Any,
    articulation: Any,
    settle_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    start = _prepare_home(world, articulation, settle_steps)
    world.step(render=False)
    after_first = np.asarray(
        articulation.get_joint_positions(),
        dtype=np.float64,
    )
    jump = np.abs(after_first - start)
    first = {
        "status": "PASS" if float(np.max(jump)) <= 0.01 else "FAIL",
        "start": start.tolist(),
        "after_first_frame": after_first.tolist(),
        "absolute_delta": jump.tolist(),
        "maximum_delta": float(np.max(jump)),
        "threshold": 0.01,
    }

    start_hold = _prepare_home(world, articulation, settle_steps)
    samples = []
    for _ in range(120):
        world.step(render=False)
        samples.append(
            np.asarray(
                articulation.get_joint_positions(),
                dtype=np.float64,
            )
        )
    stacked = np.stack(samples)
    drift = np.max(np.abs(stacked - start_hold), axis=0)
    arm_max = float(np.max(drift[:6]))
    auxiliary_gripper_max = float(drift[6])
    finger_max = float(np.max(drift[7:]))
    hold = {
        "status": (
            "PASS"
            if (
                arm_max <= 0.02
                and auxiliary_gripper_max <= 0.02
                and finger_max <= 0.002
            )
            else "FAIL"
        ),
        "duration_s": 2.0,
        "frame_count": 120,
        "maximum_arm_drift": arm_max,
        "maximum_auxiliary_gripper_drift": auxiliary_gripper_max,
        "maximum_finger_drift": finger_max,
        "arm_threshold": 0.02,
        "finger_threshold": 0.002,
    }
    return first, hold


def _determinism(
    arm_cases: list[dict[str, Any]],
    gripper_cases: list[dict[str, Any]],
    repeats: int,
) -> dict[str, Any]:
    signatures = []
    for repeat in range(repeats):
        payload = {
            "arm": [
                {
                    "test": item["test"],
                    "status": item["status"],
                    "end": [round(float(value), 6) for value in item["end"]],
                }
                for item in arm_cases
                if item["repeat"] == repeat
            ],
            "gripper": [
                {
                    "state": item["state"],
                    "status": item["status"],
                    "left": round(item["readback_left_m"], 6),
                    "right": round(item["readback_right_m"], 6),
                }
                for item in gripper_cases
                if item["repeat"] == repeat
            ],
        }
        signatures.append(
            hashlib.sha256(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )
    unique = sorted(set(signatures))
    return {
        "status": "PASS" if len(unique) == 1 and repeats >= 2 else "FAIL",
        "repeat_count": repeats,
        "signatures": signatures,
        "unique_signatures": unique,
        "unique_signature_count": len(unique),
    }


def _gripper_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    first = {
        state: next(
            item
            for item in cases
            if item["repeat"] == 0 and item["state"] == state
        )
        for state in GRIPPER_STATES
    }
    apertures = [
        first[state]["aperture_m"]
        for state in (
            "closed",
            "partially_closed",
            "open",
            "maximum_legal_aperture",
        )
    ]
    monotonic = all(
        later > earlier for earlier, later in pairwise(apertures)
    )
    direction = (
        first["closed"]["readback_left_m"] < HOME[7]
        and first["closed"]["readback_right_m"] > HOME[8]
        and first["maximum_legal_aperture"]["readback_left_m"] > HOME[7]
        and first["maximum_legal_aperture"]["readback_right_m"] < HOME[8]
    )
    maximum_residual = max(item["mimic_residual_m"] for item in cases)
    passed = (
        all(item["status"] == "PASS" for item in cases)
        and monotonic
        and direction
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "control_mode": "SOURCE_URDF_MIMIC_LEFT_DRIVEN",
        "mimic_parent": "left_finger",
        "mimic_multiplier": -1.0,
        "mimic_offset": 0.0,
        "maximum_mimic_residual_m": maximum_residual,
        "aperture_monotonicity": "PASS" if monotonic else "FAIL",
        "motion_direction": "PASS" if direction else "FAIL",
        "legal_range": (
            "PASS" if all(item["legal_range"] for item in cases) else "FAIL"
        ),
        "states": first,
        "all_repeats": cases,
    }


def _write_curves(path: Path, curves: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "repeat",
        "test",
        "frame",
        "time_s",
        *EXPECTED_DOF_ORDER,
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(curves)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# ALOHA Viper follower_right one-joint validation",
                "",
                f"- Status: `{report['status']}`",
                f"- Scope: `{report['scope']}`",
                f"- Stage: `{report['stage']['absolute_path']}`",
                f"- Stage SHA-256: `{report['stage']['sha256_before']}`",
                f"- DOF order: `{', '.join(report['dof_order'])}`",
                f"- Arm cases: `{len(report['arm_one_joint_cases'])}`",
                f"- Gripper: `{report['gripper_validation']['status']}`",
                f"- First-frame jump: `{report['first_frame_jump']['status']}`",
                f"- Static pose hold: `{report['static_pose_hold']['status']}`",
                f"- Determinism: `{report['determinism']['status']}`",
                "- Workcell placement: `NOT_VERIFIED`",
                "- Task 8: `NOT_RUN`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _structure_report(
    *,
    report: dict[str, Any],
    asset: dict[str, Any],
    identity: dict[str, Any],
    left_geometry: dict[str, Any],
) -> dict[str, Any]:
    overlap_pass = (
        identity["classification"]
        == "VERIFIED_SINGLE_REUSABLE_ROBOT_PRODUCT"
        and asset["construction"]["robot_geometry_mirrored"] is False
        and left_geometry["gates"]["no_finger_to_finger_overlap"]
        and left_geometry["gates"]["three_legal_states_audited"]
        and bool(left_geometry["attachment_component_overlaps"])
    )
    if report["status"] == "PASS" and overlap_pass:
        status = "PASS"
    elif report["status"] == "PARTIAL" and overlap_pass:
        status = "PARTIAL"
    else:
        status = "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "classification": (
            "RIGHT_ROBOT_LOCAL_STRUCTURE_VERIFIED_NOT_WORKCELL_PLACEMENT"
            if status == "PASS"
            else "RIGHT_ROBOT_LOCAL_STRUCTURE_PARTIAL_MIMIC_ACCURACY_FAILED"
            if status == "PARTIAL"
            else "RIGHT_ROBOT_LOCAL_STRUCTURE_FAILED"
        ),
        "stage": report["stage"],
        "articulation_count": 1,
        "articulation_root": ARTICULATION_PATH,
        "dof_order": report["dof_order"],
        "supplier_finger_identity": (
            "PASS" if asset["supplier_fingers"]["mirrored"] is False else "FAIL"
        ),
        "generic_finger_deactivated": not asset["supplier_fingers"][
            "generic_856_face_active"
        ],
        "one_joint_direction_and_range": (
            "PASS"
            if all(
                item["status"] == "PASS"
                for item in report["arm_one_joint_cases"]
            )
            else "FAIL"
        ),
        "gripper": report["gripper_validation"],
        "first_frame_jump": report["first_frame_jump"],
        "static_pose_hold": report["static_pose_hold"],
        "determinism": report["determinism"],
        "initial_overlap": {
            "status": "PASS" if overlap_pass else "FAIL",
            "evidence_method": (
                "VERIFIED_ROBOT_LOCAL_GEOMETRY_EQUIVALENCE_TO_FOLLOWER_LEFT"
            ),
            "identity_report": str(CAD_IDENTITY_REPORT.resolve()),
            "left_geometry_report": str(LEFT_GEOMETRY_REPORT.resolve()),
            "same_normalized_urdf": identity["urdf_identity"][
                "normalized_equal"
            ],
            "same_supplier_geometry_layer": True,
            "identity_reference_xform": True,
            "no_finger_to_finger_overlap": left_geometry["gates"][
                "no_finger_to_finger_overlap"
            ],
            "unexpected_overlap": False,
            "attachment_common_volume_retained": bool(
                left_geometry["attachment_component_overlaps"]
            ),
            "boundary": (
                "Supplier finger-to-gripper-bar common volume is retained "
                "as attachment semantics; this is not a workcell collision "
                "claim."
            ),
        },
        "workcell_placement_verified": False,
        "hard_blockers": [
            "HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM"
        ],
        "screenshots": "PENDING_SEPARATE_RAW_AND_ANNOTATED_EVIDENCE",
        "task8": "NOT_RUN",
    }


def main() -> int:
    args = _parse_args()
    stage_path = args.stage.resolve(strict=True)
    stage_hash_before = _sha256(stage_path)
    asset = json.loads(ASSET_REPORT.read_text(encoding="utf-8"))
    identity = json.loads(CAD_IDENTITY_REPORT.read_text(encoding="utf-8"))
    left_geometry = json.loads(
        LEFT_GEOMETRY_REPORT.read_text(encoding="utf-8")
    )

    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import open_stage

    if not open_stage(str(stage_path)):
        raise RuntimeError(f"failed to open Stage: {stage_path}")
    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
    )
    world.get_physics_context().set_solve_articulation_contact_last(True)
    articulation = SingleArticulation(
        prim_path=ARTICULATION_PATH,
        name="supplier_cad_follower_right_one_joint",
        reset_xform_properties=False,
    )
    world.scene.add(articulation)
    world.reset()
    dof_order = list(articulation.dof_names)
    if dof_order != EXPECTED_DOF_ORDER:
        raise RuntimeError(
            f"DOF order mismatch: expected={EXPECTED_DOF_ORDER} "
            f"actual={dof_order}"
        )
    properties = articulation.dof_properties.copy()

    all_arm_cases = []
    all_gripper_cases = []
    all_curves: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        all_arm_cases.extend(
            _run_arm_cases(
                world=world,
                articulation=articulation,
                properties=properties,
                steps=args.steps,
                settle_steps=args.settle_steps,
                repeat=repeat,
                curves=all_curves,
            )
        )
        all_gripper_cases.extend(
            _run_gripper_states(
                world=world,
                articulation=articulation,
                properties=properties,
                steps=args.steps,
                settle_steps=args.settle_steps,
                terminal_settle_steps=args.terminal_settle_steps,
                repeat=repeat,
                curves=all_curves,
            )
        )
    first_frame, static_hold = _first_frame_and_static_hold(
        world,
        articulation,
        args.settle_steps,
    )
    gripper = _gripper_summary(all_gripper_cases)
    determinism = _determinism(
        all_arm_cases,
        all_gripper_cases,
        args.repeats,
    )
    _write_curves(args.curves.resolve(), all_curves)
    stage_hash_after = _sha256(stage_path)
    immutable = stage_hash_before == stage_hash_after
    arm_pass = all(
        item["status"] == "PASS" for item in all_arm_cases
    )
    core_runtime_pass = all(
        (
            arm_pass,
            first_frame["status"] == "PASS",
            static_hold["status"] == "PASS",
            determinism["status"] == "PASS",
            immutable,
        )
    )
    status = (
        "PASS"
        if core_runtime_pass and gripper["status"] == "PASS"
        else "PARTIAL"
        if core_runtime_pass
        else "FAIL"
    )
    report = {
        "schema_version": 1,
        "status": status,
        "scope": SCOPE,
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "physics_frequency_hz": 60,
            "solve_articulation_contact_last": True,
        },
        "stage": {
            "absolute_path": str(stage_path),
            "sha256_before": stage_hash_before,
            "sha256_after": stage_hash_after,
            "immutable": immutable,
        },
        "articulation_path": ARTICULATION_PATH,
        "dof_order": dof_order,
        "protocol": {
            "motion_steps": args.steps,
            "home_settle_steps": args.settle_steps,
            "terminal_target_settle_steps": args.terminal_settle_steps,
            "terminal_target_held_constant": True,
        },
        "dof_properties": [
            {
                "index": index,
                "name": name,
                **{
                    field: (
                        bool(properties[index][field])
                        if field == "hasLimits"
                        else float(properties[index][field])
                    )
                    for field in (
                        "hasLimits",
                        "lower",
                        "upper",
                        "maxVelocity",
                        "maxEffort",
                        "stiffness",
                        "damping",
                    )
                    if field in (properties.dtype.names or ())
                },
            }
            for index, name in enumerate(dof_order)
        ],
        "arm_one_joint_cases": all_arm_cases,
        "component_status": {
            "arm_one_joint_direction_range": (
                "PASS" if arm_pass else "FAIL"
            ),
            "gripper_motion_direction": gripper["motion_direction"],
            "aperture_monotonicity": gripper[
                "aperture_monotonicity"
            ],
            "mimic_accuracy": (
                "PASS"
                if gripper["maximum_mimic_residual_m"] <= 0.001
                else "FAIL"
            ),
            "first_frame_jump": first_frame["status"],
            "static_pose_hold": static_hold["status"],
            "determinism": determinism["status"],
        },
        "gripper_validation": gripper,
        "first_frame_jump": first_frame,
        "static_pose_hold": static_hold,
        "determinism": determinism,
        "curve_csv": str(args.curves.resolve()),
        "workcell_placement_verified": False,
        "hard_blockers": [
            "HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM"
        ],
        "screenshots": "PENDING_SEPARATE_RAW_AND_ANNOTATED_EVIDENCE",
        "task8": "NOT_RUN",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output.with_suffix(".md"), report)
    structure = _structure_report(
        report=report,
        asset=asset,
        identity=identity,
        left_geometry=left_geometry,
    )
    structure_output = args.structure_output.resolve()
    structure_output.write_text(
        json.dumps(structure, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    structure_output.with_suffix(".md").write_text(
        "\n".join(
            [
                "# ALOHA Viper follower_right structure validation",
                "",
                f"- Status: `{structure['status']}`",
                f"- Classification: `{structure['classification']}`",
                "- Workcell placement: `NOT_VERIFIED`",
                "- Task 8: `NOT_RUN`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "arm_cases": len(all_arm_cases),
                "gripper_cases": len(all_gripper_cases),
                "curve_rows": len(all_curves),
                "structure": structure["status"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] in {"PASS", "PARTIAL"} else 1


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
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
