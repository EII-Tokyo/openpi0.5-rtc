#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Probe follower-left Lula/URDF/USD correspondence in local Isaac Sim 5.1."""

from __future__ import annotations

import argparse
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
EXPECTED_STAGE_SHA256 = (
    "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
)
EXPECTED_URDF_SHA256 = (
    "d9e4b32723ee71dfce26fb4e78546cfcfef147b2d7dbf5e53e3620e3d8aa96bd"
)
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
END_EFFECTOR_PATH = (
    "/World/follower_left/vx300s_left/follower_left_gripper_link"
)
END_EFFECTOR_FRAME = "follower_left_gripper_link"
TRANSLATION_GATE_M = 0.001
ROTATION_GATE_RAD = 0.005


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


def _source_inventory(
    *,
    extension_root: Path,
    core_api_root: Path,
    validation_root: Path,
) -> dict[str, Any]:
    paths = {
        "extension_toml": extension_root / "config/extension.toml",
        "articulation_kinematics_solver": (
            extension_root
            / "isaacsim/robot_motion/motion_generation/"
            "articulation_kinematics_solver.py"
        ),
        "lula_kinematics": (
            extension_root
            / "isaacsim/robot_motion/motion_generation/lula/kinematics.py"
        ),
        "nvidia_test_kinematics": (
            extension_root
            / "isaacsim/robot_motion/motion_generation/tests/"
            "test_kinematics.py"
        ),
        "physics_context": (
            core_api_root
            / "isaacsim/core/api/physics_context/physics_context.py"
        ),
        "physics_rules": (
            validation_root / "isaacsim/asset/validation/physics_rules.py"
        ),
    }
    records = {}
    for name, path in paths.items():
        resolved = path.resolve(strict=True)
        records[name] = {
            "path": str(resolved),
            "sha256": _sha256(resolved),
        }
    extension_data = tomllib.loads(paths["extension_toml"].read_text())
    records["extension_toml"]["declared_version"] = extension_data["package"][
        "version"
    ]
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        type=Path,
        default=(
            ROOT
            / "assets/Trossen/ALOHA1/1.0/diagnostics/"
            "signal_correspondence/1.0/"
            "aloha1_signal_correspondence_workcell.usda"
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
        raise RuntimeError(
            f"frozen Stage SHA-256 mismatch: {stage_hash_before}"
        )
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
        from isaacsim.core.utils.stage import open_stage
        from isaacsim.core.utils.xforms import get_world_pose
        from isaacsim.robot_motion.motion_generation.articulation_kinematics_solver import ArticulationKinematicsSolver
        from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver
        import omni.kit.app
        import omni.physx
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools
        from pxr import PhysxSchema

        extension_root = Path(
            importlib.import_module(
                "isaacsim.robot_motion.motion_generation"
            ).__path__[0]
        ).resolve().parents[2]
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
        motion_extension_id = manager.get_enabled_extension_id(
            "isaacsim.robot_motion.motion_generation"
        )
        physx_extension_id = manager.get_enabled_extension_id("omni.physx")
        motion_extension = (
            manager.get_extension_dict(motion_extension_id)
            if motion_extension_id
            else None
        )
        physx_extension = (
            manager.get_extension_dict(physx_extension_id)
            if physx_extension_id
            else None
        )
        motion_runtime_version = (
            motion_extension.get("package", {}).get("version")
            if motion_extension
            else None
        )
        physx_runtime_version = (
            physx_extension.get("package", {}).get("version")
            if physx_extension
            else None
        )
        if physx_runtime_version is None:
            physx_module_path = Path(next(iter(omni.physx.__path__))).resolve()
            extension_directory = next(
                (
                    parent
                    for parent in physx_module_path.parents
                    if parent.name.startswith("omni.physx-")
                ),
                None,
            )
            version_match = (
                re.match(r"omni\.physx-([^+]+)", extension_directory.name)
                if extension_directory
                else None
            )
            physx_runtime_version = (
                version_match.group(1) if version_match else None
            )
        if source_version != EXPECTED_MOTION_GENERATION_VERSION:
            raise RuntimeError(
                "motion-generation source version mismatch: "
                f"{source_version}"
            )
        if (
            motion_runtime_version is not None
            and str(motion_runtime_version).split("+", maxsplit=1)[0]
            != EXPECTED_MOTION_GENERATION_VERSION
        ):
            raise RuntimeError(
                "motion-generation runtime version mismatch: "
                f"{motion_runtime_version}"
            )
        isaac_runtime_version = version("isaacsim")
        kit_runtime_version = str(
            carb.tokens.get_tokens_interface().resolve("${kit_version}")
        ).split("+", maxsplit=1)[0]
        normalized_physx_version = (
            str(physx_runtime_version).split("+", maxsplit=1)[0]
            if physx_runtime_version
            else None
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
            raise RuntimeError(
                "Isaac runtime boundary mismatch: "
                f"expected={expected_runtime}, actual={actual_runtime}"
            )

        solver = LulaKinematicsSolver(
            robot_description_path=str(descriptor_path),
            urdf_path=str(urdf_path),
        )
        solver_joint_names = list(solver.get_joint_names())
        solver_frames = list(solver.get_all_frame_names())
        if solver_joint_names != EXPECTED_CSPACE:
            raise RuntimeError(
                f"Lula cspace mismatch: {solver_joint_names}"
            )
        if END_EFFECTOR_FRAME not in solver_frames:
            raise RuntimeError(
                f"missing Lula frame: {END_EFFECTOR_FRAME}"
            )

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
        solve_contact_last_readback = (
            physics_context.get_solve_articulation_contact_last()
        )
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
            raise RuntimeError(
                f"runtime DOF order mismatch: {runtime_dof_order}"
            )

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
            usd_rotation = quats_to_rot_matrices(
                np.asarray(usd_orientation, dtype=np.float64)
            )
            translation_residual = float(
                np.linalg.norm(
                    np.asarray(lula_position) - np.asarray(usd_position)
                )
            )
            rotation_residual = _rotation_residual_rad(
                np.asarray(lula_rotation),
                np.asarray(usd_rotation),
            )
            case_status = (
                "PASS"
                if translation_residual <= TRANSLATION_GATE_M
                and rotation_residual <= ROTATION_GATE_RAD
                else "FAIL"
            )
            fk_records.append(
                {
                    "case": case_name,
                    "status": case_status,
                    "arm_target_rad": [float(value) for value in arm_q],
                    "arm_readback_rad": [
                        float(value) for value in readback[:6]
                    ],
                    "finger_readback_m": [
                        float(readback[7]),
                        float(readback[8]),
                    ],
                    "base_position_world_m": [
                        float(value) for value in base_position
                    ],
                    "base_orientation_world_wxyz": [
                        float(value) for value in base_orientation
                    ],
                    "usd_position_world_m": [
                        float(value) for value in usd_position
                    ],
                    "usd_orientation_world_wxyz": [
                        float(value) for value in usd_orientation
                    ],
                    "lula_position_world_m": [
                        float(value) for value in lula_position
                    ],
                    "lula_rotation_world": [
                        [float(value) for value in row]
                        for row in lula_rotation
                    ],
                    "translation_residual_m": translation_residual,
                    "rotation_angle_residual_rad": rotation_residual,
                }
            )

        correspondence_pass = all(
            record["status"] == "PASS" for record in fk_records
        )
        if not correspondence_pass:
            report["hard_blockers"].append(
                "HARD_BLOCKER_LULA_USD_FRAME_CORRESPONDENCE"
            )

        stage_hash_after = _sha256(stage_path)
        stage_immutable = stage_hash_after == stage_hash_before
        report.update(
            {
                "status": (
                    "PASS"
                    if correspondence_pass
                    and stage_immutable
                    and solve_contact_last_readback is True
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
                        (
                            "isaacsim.robot_motion.motion_generation.lula."
                            "kinematics.LulaKinematicsSolver"
                        ),
                        (
                            "isaacsim.core.api.physics_context.PhysicsContext."
                            "set_solve_articulation_contact_last"
                        ),
                        "PhysxSchema.PhysxContactReportAPI",
                        "PhysicsSchemaTools.intToSdfPath",
                    ],
                },
                "local_source_inventory": sources,
                "api_readback": {
                    "motion_generation_extension_id": motion_extension_id,
                    "physx_extension_id": physx_extension_id,
                    "lula_constructor_signature": str(
                        inspect.signature(LulaKinematicsSolver)
                    ),
                    "lula_fk_signature": str(
                        inspect.signature(
                            LulaKinematicsSolver.compute_forward_kinematics
                        )
                    ),
                    "articulation_ik_signature": str(
                        inspect.signature(
                            ArticulationKinematicsSolver.compute_inverse_kinematics
                        )
                    ),
                    "contact_report_schema_attributes": list(
                        PhysxSchema.PhysxContactReportAPI.GetSchemaAttributeNames()
                    ),
                    "int_to_sdf_path_callable": callable(
                        PhysicsSchemaTools.intToSdfPath
                    ),
                    "get_contact_report_callable": callable(
                        get_physx_simulation_interface().get_contact_report
                    ),
                    "subscribe_contact_report_events_callable": callable(
                        get_physx_simulation_interface().subscribe_contact_report_events
                    ),
                    "solve_articulation_contact_last": (
                        solve_contact_last_readback
                    ),
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
                    "cases": fk_records,
                },
            }
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
            report["hard_blockers"].append(
                "HARD_BLOCKER_LULA_USD_FRAME_CORRESPONDENCE"
            )
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
