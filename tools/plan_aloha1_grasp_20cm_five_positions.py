#!/usr/bin/env python3
"""Select five reproducible Bottle500 XY positions using local Lula IK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_source(record: dict[str, Any]) -> Path:
    path = Path(str(record["path"]))
    return (path if path.is_absolute() else ROOT / path).resolve(strict=True)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(f"empty telemetry: {path}")
    return records


def _matrix_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    from tools.validate_aloha1_task7b2_horizontal_grasp import _rotation_matrix_to_quaternion_wxyz

    return _rotation_matrix_to_quaternion_wxyz(matrix)


def _condense_ik(result: dict[str, Any]) -> dict[str, Any]:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument(
        "--additional-lift-margin-m",
        type=float,
        default=0.0,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(ROOT))
    from tools.aloha1_mapping.grasp_20cm_runtime import load_and_verify_config
    from tools.aloha1_mapping.grasp_20cm_runtime import sha256_file
    from tools.aloha1_mapping.grasp_20cm_runtime import validate_composed_stage
    from tools.aloha1_mapping.grasp_20cm_sampling import derive_legal_offset_bounds
    from tools.aloha1_mapping.grasp_20cm_sampling import extend_profile_for_clearance_lift
    from tools.aloha1_mapping.grasp_20cm_sampling import sample_candidate_offsets

    sampling_config_path = args.config.resolve(strict=True)
    if (
        not np.isfinite(args.additional_lift_margin_m)
        or args.additional_lift_margin_m < 0.0
    ):
        raise ValueError(
            "additional lift margin must be finite and non-negative"
        )
    sampling = yaml.safe_load(
        sampling_config_path.read_text(encoding="utf-8")
    )
    if sampling.get("schema_version") != 1:
        raise RuntimeError("unsupported sampling config schema")
    source_paths: dict[str, Path] = {}
    for name, record in sampling["sources"].items():
        path = _resolve_source(record)
        actual = sha256_file(path)
        if actual != str(record["sha256"]):
            raise RuntimeError(
                f"sampling source hash mismatch for {name}: {actual}"
            )
        source_paths[str(name)] = path

    runtime_profile = load_and_verify_config(
        source_paths["grasp_runtime_config"],
        project_root=ROOT,
    )
    stage_path = Path(
        runtime_profile["frozen_inputs"]["stage"]["absolute_path"]
    )
    stage_hash_before = sha256_file(stage_path)

    table_report = json.loads(
        source_paths["table_support_alignment"].read_text(
            encoding="utf-8"
        )
    )
    accepted_report = json.loads(
        source_paths["accepted_single_position_runtime"].read_text(
            encoding="utf-8"
        )
    )
    accepted_telemetry = _load_jsonl(
        source_paths["accepted_single_position_telemetry"]
    )
    first_telemetry = accepted_telemetry[0]
    table_bounds = table_report["diagnostic_stage"][
        "table_aabb_world_m"
    ]
    stacks = table_report["alignment"]["support_stacks"]
    nominal_bounds = first_telemetry["bottle"]["collision_bounds"]
    legal = derive_legal_offset_bounds(
        table_xy_bounds={
            "minimum": table_bounds["minimum"][:2],
            "maximum": table_bounds["maximum"][:2],
        },
        left_base_aabb=stacks["follower_left"][
            "base_aabb_world_m"
        ],
        right_base_aabb=stacks["follower_right"][
            "base_aabb_world_m"
        ],
        nominal_bottle_xy_bounds={
            "minimum": nominal_bounds["minimum"][:2],
            "maximum": nominal_bounds["maximum"][:2],
        },
    )
    candidates = sample_candidate_offsets(
        offset_xy_bounds=legal["offset_xy_m"],
        seed=int(sampling["sampling"]["seed"]),
        count=int(sampling["sampling"]["candidate_count"]),
    )

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "create_new_stage": False,
            "width": 640,
            "height": 360,
        }
    )
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any]
    try:
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.stage import open_stage

        from tools.aloha1_mapping.grasp_20cm_isaac_bindings import IsaacGrasp20cmBindings

        if not open_stage(str(stage_path)):
            raise RuntimeError(f"failed to open Stage: {stage_path}")
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
        bindings = IsaacGrasp20cmBindings(
            app=app,
            profile=runtime_profile,
            artifact_root=artifact_root,
            delegate_readback={
                "path": "/app/useFabricSceneDelegate",
                "requested": False,
                "effective": False,
                "purpose": "HEADLESS_LULA_PREFLIGHT",
            },
            additional_lift_margin_m=float(
                args.additional_lift_margin_m
            ),
        )
        first_open_index = next(
            index
            for index, record in enumerate(accepted_telemetry)
            if record["phase"] == "OPEN_PREGRASP"
        )
        if first_open_index < 1:
            raise RuntimeError(
                "accepted telemetry has no pre-open SETTLE state"
            )
        ik_start_record = accepted_telemetry[first_open_index - 1]
        if ik_start_record["phase"] != "SETTLE":
            raise RuntimeError(
                "accepted IK start does not immediately follow SETTLE"
            )
        current_q = np.asarray(
            ik_start_record["joint_readback"][:6],
            dtype=np.float64,
        )
        from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver

        fk_solver = LulaKinematicsSolver(
            robot_description_path=str(
                bindings.profile["frozen_inputs"][
                    "lula_descriptor"
                ]["absolute_path"]
            ),
            urdf_path=str(
                bindings.task_profile["inputs"]["follower_left_urdf"]
            ),
        )
        fk_solver.set_robot_base_pose(
            np.asarray(bindings.base_position, dtype=np.float64),
            np.asarray(bindings.base_orientation, dtype=np.float64),
        )
        ee_position, ee_rotation = (
            fk_solver.compute_forward_kinematics(
                bindings.task_profile["config"]["robot"][
                    "end_effector_frame"
                ],
                current_q,
            )
        )
        ee_orientation = _matrix_quaternion_wxyz(
            np.asarray(ee_rotation, dtype=np.float64)
        )
        accepted_world_from_object = np.asarray(
            accepted_report["runtime"]["ik"]["world_from_object"],
            dtype=np.float64,
        )
        nominal_position = accepted_world_from_object[:3, 3]
        nominal_orientation = _matrix_quaternion_wxyz(
            accepted_world_from_object[:3, :3]
        )
        formal_profile = extend_profile_for_clearance_lift(
            bindings.task_profile,
            target_clearance_m=float(
                runtime_profile["config"]["target"]["clearance_m"]
            ),
            hold_drop_gate_m=float(
                runtime_profile["config"]["target"]["hold_drop_gate_m"]
            ),
            additional_lift_margin_m=float(
                args.additional_lift_margin_m
            ),
        )
        nominal_ik = bindings._solve_settled_ik(  # noqa: SLF001
            formal_profile,
            base_position=np.asarray(
                bindings.base_position,
                dtype=np.float64,
            ),
            base_orientation=np.asarray(
                bindings.base_orientation,
                dtype=np.float64,
            ),
            bottle_state={
                "position_world_m": nominal_position.tolist(),
                "orientation_wxyz": nominal_orientation.tolist(),
            },
            current_ee_position=np.asarray(
                ee_position,
                dtype=np.float64,
            ),
            current_ee_orientation=np.asarray(
                ee_orientation,
                dtype=np.float64,
            ),
            current_arm_q=current_q,
        )
        if nominal_ik["status"] != "PASS":
            raise RuntimeError(
                "nominal positive-control IK did not reproduce PASS: "
                f"{nominal_ik.get('failure_phase')}"
            )
        results: list[dict[str, Any]] = []
        selected: list[dict[str, Any]] = []
        required = int(
            sampling["sampling"]["required_passing_positions"]
        )
        for candidate in candidates:
            offset = np.asarray(
                candidate["offset_xy_m"],
                dtype=np.float64,
            )
            bottle_position = nominal_position.copy()
            bottle_position[:2] += offset
            bottle_min = np.asarray(
                nominal_bounds["minimum"][:2],
                dtype=np.float64,
            ) + offset
            bottle_max = np.asarray(
                nominal_bounds["maximum"][:2],
                dtype=np.float64,
            ) + offset
            free_min = np.asarray(
                legal["free_surface_xy"]["minimum"],
                dtype=np.float64,
            )
            free_max = np.asarray(
                legal["free_surface_xy"]["maximum"],
                dtype=np.float64,
            )
            inside = bool(
                np.all(bottle_min >= free_min)
                and np.all(bottle_max <= free_max)
            )
            if not inside:
                raise RuntimeError(
                    "generated candidate escaped derived legal envelope"
                )
            ik = bindings._solve_settled_ik(  # noqa: SLF001
                formal_profile,
                base_position=np.asarray(
                    bindings.base_position,
                    dtype=np.float64,
                ),
                base_orientation=np.asarray(
                    bindings.base_orientation,
                    dtype=np.float64,
                ),
                bottle_state={
                    "position_world_m": bottle_position.tolist(),
                    "orientation_wxyz": nominal_orientation.tolist(),
                },
                current_ee_position=np.asarray(
                    ee_position,
                    dtype=np.float64,
                ),
                current_ee_orientation=np.asarray(
                    ee_orientation,
                    dtype=np.float64,
                ),
                current_arm_q=current_q,
            )
            record = {
                **candidate,
                "bottle_position_world_m": bottle_position.tolist(),
                "bottle_xy_bounds_world_m": {
                    "minimum": bottle_min.tolist(),
                    "maximum": bottle_max.tolist(),
                },
                "full_bottle_inside_free_surface": inside,
                "ik": _condense_ik(ik),
                "selected": False,
            }
            results.append(record)
            if ik["status"] == "PASS":
                record["selected"] = True
                record["position_id"] = f"position_{len(selected) + 1:02d}"
                selected.append(record)
                if len(selected) == required:
                    break

        report = {
            "schema_version": 1,
            "status": "PASS" if len(selected) == required else "FAIL",
            "classification": (
                "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
            ),
            "runtime": {
                "isaac_sim": "5.1.0.0",
                "kit": "107.3.3",
                "physx": "107.3.26",
                "ik": "LULA_LOCAL_5_1_RUNTIME",
            },
            "sampling_config": {
                "absolute_path": str(sampling_config_path),
                "sha256": sha256_file(sampling_config_path),
            },
            "stage": {
                "absolute_path": str(stage_path),
                "sha256_before": stage_hash_before,
                "sha256_after": sha256_file(stage_path),
                "root_prim": str(stage.GetDefaultPrim().GetPath()),
                "sublayers": list(
                    stage.GetRootLayer().subLayerPaths
                ),
            },
            "legal_geometry": legal,
            "nominal_positive_control_ik": _condense_ik(nominal_ik),
            "formal_lift_distance_m": float(
                formal_profile["formal_lift_distance_m"]
            ),
            "formal_lift_derivation": str(
                formal_profile["formal_lift_derivation"]
            ),
            "additional_lift_margin_m": float(
                formal_profile["additional_lift_margin_m"]
            ),
            "ik_start_state": {
                "source": (
                    "ACCEPTED_SINGLE_POSITION_LAST_SETTLE_READBACK_"
                    "PLUS_LOCAL_LULA_FK"
                ),
                "physics_frame": int(ik_start_record["frame"]),
                "joint_readback_arm_rad": current_q.tolist(),
                "ee_position_world_m": np.asarray(
                    ee_position,
                    dtype=np.float64,
                ).tolist(),
                "ee_orientation_world_wxyz": ee_orientation.tolist(),
            },
            "candidate_count_generated": len(candidates),
            "candidate_count_preflighted": len(results),
            "required_passing_positions": required,
            "selected_position_count": len(selected),
            "candidate_results": results,
            "selected_positions": selected,
            "semantics": {
                "randomized": "BOTTLE_INITIAL_WORLD_XY_TRANSLATION_ONLY",
                "bottle_rotation": "UNCHANGED",
                "object_from_gripper": "UNCHANGED_VARIANT_B",
                "selection": sampling["sampling"]["selection"],
                "formal_physics_required_after_preflight": True,
            },
            "boundaries": {
                **sampling["boundaries"],
                "task8": "NOT_RUN",
            },
        }
        if report["stage"]["sha256_after"] != stage_hash_before:
            report["status"] = "FAIL"
            report["failure"] = "approved_stage_hash_changed"
    except Exception:
        report = {
            "schema_version": 1,
            "status": "FAIL",
            "reason": "exception",
            "exception": traceback.format_exc(limit=30)[-16000:],
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
            )
        )
        app.close()
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
