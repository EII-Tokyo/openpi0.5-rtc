#!/usr/bin/env python3
"""Run the Isaac Sim 5.1 horizontal Bottle500 grasp diagnostic.

This is a session-only diagnostic.  It never saves the source Stage, changes
the final collider, or promotes a candidate visual recording.
"""

# Isaac Sim 5.1.0.0 / Kit 107.3.3 / PhysX 107.3.26 only.
# ruff: noqa: FBT003, PERF401, PLC0415

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any

import numpy as np
from PIL import Image
import yaml

from tools.aloha1_mapping.task7b2_horizontal_grasp import canonical_horizontal_signature
from tools.aloha1_mapping.task7b2_horizontal_grasp import evaluate_horizontal_trial
from tools.aloha1_mapping.task7b2_horizontal_grasp import render_horizontal_markdown
from tools.aloha1_mapping.task7b2_horizontal_grasp import summarize_horizontal_trials

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/aloha1_task7b2_horizontal_grasp.yaml"
DEFAULT_OUTPUT = (
    ROOT / "reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp.json"
)
DEFAULT_TRIALS = (
    ROOT
    / "reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp_trials.jsonl"
)
DEFAULT_MARKDOWN = (
    ROOT / "reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp.md"
)
DEFAULT_ARTIFACT_ROOT = (
    ROOT / ".codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/runtime"
)
KINEMATICS_REPORT = (
    ROOT / "reports/aloha1_mapping/aloha1_task7b2_horizontal_kinematics.json"
)
LULA_DESCRIPTOR = ROOT / "configs/aloha1_lula_follower_left.yaml"

VIDEO_VIEWS = ("overview", "gripper_closeup")
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
PHASE_ORDER = (
    "setup_kinematic",
    "release_dynamic",
    "support_settle",
    "open_pregrasp",
    "vertical_descent",
    "bilateral_contact",
    "closing_preload",
    "vertical_lift",
    "support_clear",
    "hold_end",
)


def main() -> int:
    args = _parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("capture resolution must be positive")
    profile = _load_profile(args.config)
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    isaacsim_module = importlib.import_module("isaacsim")
    app = isaacsim_module.SimulationApp(
        {
            "headless": True,
            "width": int(args.width),
            "height": int(args.height),
        }
    )
    trials: list[dict[str, Any]] = []
    exit_code = 1
    try:
        runtime = _verify_runtime_versions(profile["config"])
        for trial_index in range(args.repeats):
            trials.append(
                _run_trial(
                    app,
                    profile,
                    trial_index=trial_index,
                    artifact_root=artifact_root,
                    capture_video_frames=bool(args.capture_video_frames),
                    resolution=(int(args.width), int(args.height)),
                )
            )
        summary = summarize_horizontal_trials(
            [trial["metrics"] for trial in trials]
        )
        physical_status = (
            trials[0]["status"]
            if len(trials) == 1
            else summary["status"]
        )
        report = {
            "schema_version": 2,
            "status": "PARTIAL" if len(trials) < 20 else summary["status"],
            "physical_trial_status": physical_status,
            "conclusion": (
                "SMOKE_PHYSICAL_PASS_ACCEPTANCE_NOT_RUN"
                if len(trials) == 1 and physical_status == "PASS"
                else "HORIZONTAL_PICKUP_NOT_VERIFIED"
                if physical_status != "PASS"
                else "HORIZONTAL_PICKUP_VERIFIED"
            ),
            "runtime": runtime,
            "command": [sys.executable, *sys.argv],
            "environment_allowlist": {
                key: os.environ.get(key)
                for key in ("OMNI_KIT_ACCEPT_EULA", "PYTHONPATH", "DISPLAY")
                if key in os.environ
            },
            "config": {
                "absolute_path": str(profile["path"]),
                "sha256": profile["sha256"],
            },
            "frozen_inputs": {
                name: {
                    "absolute_path": str(path),
                    "sha256": profile["hashes"][name],
                }
                for name, path in profile["inputs"].items()
            },
            "summary": summary,
            "trials": trials,
            "boundaries": {
                "source_assets_modified": False,
                "default_configuration_modified": False,
                "final_collider_modified": False,
                "task8": "NOT_RUN",
            },
        }
        _atomic_json(args.output.resolve(), report)
        _atomic_jsonl(args.trials_output.resolve(), trials)
        args.markdown.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.markdown.resolve().write_text(
            render_horizontal_markdown(summary),
            encoding="utf-8",
        )
        print(
            "ALOHA1_HORIZONTAL_GRASP_TERMINAL "
            + json.dumps(
                {
                    "status": report["status"],
                    "physical_trial_status": physical_status,
                    "report": str(args.output.resolve()),
                    "trial_count": len(trials),
                },
                sort_keys=True,
            )
        )
        exit_code = 0
    except Exception as error:
        report = {
            "schema_version": 2,
            "status": "FAIL",
            "physical_trial_status": "FAIL",
            "conclusion": "RUNTIME_ERROR",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "trials": trials,
            "boundaries": {"task8": "NOT_RUN"},
        }
        _atomic_json(args.output.resolve(), report)
        print(
            "ALOHA1_HORIZONTAL_GRASP_TERMINAL "
            + json.dumps(
                {
                    "status": "FAIL",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "report": str(args.output.resolve()),
                },
                sort_keys=True,
            )
        )
    finally:
        app.close()
    return exit_code


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                    default=_json_default,
                )
                + "\n"
            )
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trials-output", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--capture-video-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    return parser.parse_args()


def _resolve_source(root: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve(strict=True)


def _load_profile(config_path: Path) -> dict[str, Any]:
    path = config_path.resolve(strict=True)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if int(config["schema_version"]) != 2:
        raise RuntimeError("horizontal grasp config schema mismatch")
    if config["task_geometry"] != "HORIZONTAL_DYNAMIC_TABLE_SUPPORTED":
        raise RuntimeError("horizontal grasp geometry contract mismatch")

    frozen = config["frozen_inputs"]
    input_specs = {
        "task7a_stage": frozen["task7a_stage"],
        "project_bottle_cad": frozen["project_bottle_cad"],
        "project_bottle_usd": frozen["project_bottle_usd"],
        "follower_left_urdf": frozen["follower_left_urdf"],
        "joint_map": frozen["joint_map"],
        "task7b_static_hold_report": frozen["task7b_static_hold_report"],
        "episode18": frozen["episode18"],
    }
    inputs = {
        name: _resolve_source(ROOT, str(spec["path"]))
        for name, spec in input_specs.items()
    }
    inputs["kinematics_report"] = KINEMATICS_REPORT.resolve(strict=True)
    inputs["lula_descriptor"] = LULA_DESCRIPTOR.resolve(strict=True)
    hashes = {name: _sha256(source) for name, source in inputs.items()}
    mismatches = {
        name: {
            "expected": str(input_specs[name]["sha256"]),
            "actual": hashes[name],
        }
        for name in input_specs
        if hashes[name] != str(input_specs[name]["sha256"])
    }
    if mismatches:
        raise RuntimeError(
            "frozen input hash mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )

    kinematics = json.loads(
        inputs["kinematics_report"].read_text(encoding="utf-8")
    )
    if kinematics.get("status") != "PASS":
        raise RuntimeError("horizontal kinematics report is not PASS")
    stage_record = kinematics.get("stage", {})
    if (
        Path(stage_record.get("path", "")).resolve()
        != inputs["task7a_stage"]
        or stage_record.get("sha256_after") != hashes["task7a_stage"]
        or not stage_record.get("immutable")
    ):
        raise RuntimeError("kinematics report does not bind frozen Stage")
    if kinematics.get("ik", {}).get("status") != "PASS":
        raise RuntimeError("kinematics report IK gate is not PASS")
    return {
        "path": path,
        "sha256": _sha256(path),
        "config": config,
        "inputs": inputs,
        "hashes": hashes,
        "kinematics": kinematics,
    }


def _smoothstep(value: float) -> float:
    clipped = min(max(float(value), 0.0), 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _command_positions(articulation: Any, target: np.ndarray) -> None:
    from isaacsim.core.utils.types import ArticulationAction

    articulation.apply_action(
        ArticulationAction(joint_positions=np.asarray(target, dtype=np.float32))
    )


def _world_bounds(stage: Any, prim_path: str) -> dict[str, list[float]]:
    from pxr import Usd
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"missing prim for world bounds: {prim_path}")
    bound = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
    ).ComputeWorldBound(prim)
    aligned = bound.ComputeAlignedBox()
    return {
        "minimum": [float(value) for value in aligned.GetMin()],
        "maximum": [float(value) for value in aligned.GetMax()],
    }


def _path_from_id(value: Any) -> str:
    from pxr import PhysicsSchemaTools

    return str(PhysicsSchemaTools.intToSdfPath(value))


def _serialize_contacts(
    headers: Sequence[Any],
    data: Sequence[Any],
    *,
    frame: int,
    time_s: float,
    phase: str,
    dt: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for header in headers:
        actor0 = _path_from_id(header.actor0)
        actor1 = _path_from_id(header.actor1)
        collider0 = _path_from_id(header.collider0)
        collider1 = _path_from_id(header.collider1)
        start = int(header.contact_data_offset)
        end = start + int(header.num_contact_data)
        for index in range(start, end):
            item = data[index]
            impulse = np.asarray(item.impulse, dtype=np.float64)
            records.append(
                {
                    "event_type": str(header.type),
                    "frame": frame,
                    "time_s": time_s,
                    "phase": phase,
                    "actor0_path": actor0,
                    "actor1_path": actor1,
                    "collider0_path": collider0,
                    "collider1_path": collider1,
                    "position_world_m": [
                        float(value) for value in item.position
                    ],
                    "normal_world": [float(value) for value in item.normal],
                    "impulse_ns": float(np.linalg.norm(impulse)),
                    "impulse_vector_ns": [
                        float(value) for value in impulse
                    ],
                    "estimated_normal_force_n": float(
                        np.linalg.norm(impulse) / dt
                    ),
                    "separation_m": float(item.separation),
                    "material0_path": _path_from_id(item.material0),
                    "material1_path": _path_from_id(item.material1),
                }
            )
    return records


def _pair_text(contact: Mapping[str, Any]) -> str:
    return "\n".join(
        str(contact.get(key, ""))
        for key in (
            "actor0_path",
            "actor1_path",
            "collider0_path",
            "collider1_path",
        )
    )


def _physical_contacts(
    contacts: Sequence[Mapping[str, Any]],
    *,
    tokens: Sequence[str],
) -> list[Mapping[str, Any]]:
    return [
        contact
        for contact in contacts
        if all(token in _pair_text(contact) for token in tokens)
        and float(contact["separation_m"]) <= 0.0
    ]


def _bottle_state(bottle: Any) -> dict[str, Any]:
    position, orientation = bottle.get_world_pose()
    linear = np.asarray(bottle.get_linear_velocity(), dtype=np.float64)
    angular = np.asarray(bottle.get_angular_velocity(), dtype=np.float64)
    return {
        "position_world_m": np.asarray(position, dtype=np.float64).tolist(),
        "orientation_wxyz": np.asarray(
            orientation, dtype=np.float64
        ).tolist(),
        "linear_velocity_world_m_s": linear.tolist(),
        "angular_velocity_world_rad_s": angular.tolist(),
        "vertical_velocity_m_s": float(linear[2]),
        "angular_speed_rad_s": float(np.linalg.norm(angular)),
    }


def _rotation_matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
        scale = math.sqrt(
            1.0
            + matrix[index, index]
            - matrix[next_index, next_index]
            - matrix[last_index, last_index]
        ) * 2.0
        xyz = np.zeros(3, dtype=np.float64)
        xyz[index] = 0.25 * scale
        xyz[next_index] = (
            matrix[next_index, index] + matrix[index, next_index]
        ) / scale
        xyz[last_index] = (
            matrix[last_index, index] + matrix[index, last_index]
        ) / scale
        w = (
            matrix[last_index, next_index] - matrix[next_index, last_index]
        ) / scale
        quaternion = np.asarray([w, *xyz], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def _look_at_quaternion(
    camera_position: np.ndarray,
    target_position: np.ndarray,
) -> np.ndarray:
    from isaacsim.core.utils.rotations import rot_matrices_to_quats

    forward = target_position - camera_position
    forward /= np.linalg.norm(forward)
    camera_z = -forward
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(up, camera_z))) > 0.98:
        up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    camera_x = np.cross(up, camera_z)
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    rotation = np.column_stack([camera_x, camera_y, camera_z])
    return np.asarray(rot_matrices_to_quats(rotation), dtype=np.float64)


def _create_material(
    stage: Any,
    path: str,
    *,
    friction: float,
    restitution: float,
) -> Any:
    from pxr import PhysxSchema
    from pxr import UsdPhysics
    from pxr import UsdShade

    material = UsdShade.Material.Define(stage, path)
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(friction)
    physics.CreateDynamicFrictionAttr(friction)
    physics.CreateRestitutionAttr(restitution)
    physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx.CreateFrictionCombineModeAttr("average")
    physx.CreateRestitutionCombineModeAttr("average")
    return material


def _bind_material(prim: Any, material: Any, *, strong: bool) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material,
        bindingStrength=(
            UsdShade.Tokens.strongerThanDescendants
            if strong
            else UsdShade.Tokens.weakerThanDescendants
        ),
        materialPurpose="physics",
    )


def _create_session_bottle(
    stage: Any,
    profile: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    from pxr import Gf
    from pxr import PhysxSchema
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    config = profile["config"]
    bottle_path = str(config["bottle"]["session_path"])
    product_prim = str(
        config["frozen_inputs"]["project_bottle_usd"]["reference_prim"]
    )
    session_root = str(Path(bottle_path).parent).replace("\\", "/")
    stage.DefinePrim(session_root, "Scope")
    bottle = UsdGeom.Xform.Define(stage, bottle_path)
    if not bottle.GetPrim().GetReferences().AddReference(
        str(profile["inputs"]["project_bottle_usd"]),
        product_prim,
    ):
        raise RuntimeError("failed to reference /Bottle500 product")

    placement = np.asarray(
        profile["kinematics"]["placement"]["placement_matrix"],
        dtype=np.float64,
    )
    quaternion = _rotation_matrix_to_quaternion_wxyz(placement[:3, :3])
    bottle.AddTranslateOp().Set(Gf.Vec3d(*placement[:3, 3]))
    bottle.AddOrientOp().Set(
        Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*quaternion[1:]))
    )
    bottle_prim = bottle.GetPrim()
    collision_prims = [
        prim
        for prim in Usd.PrimRange(bottle_prim)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    expected_count = int(
        config["frozen_inputs"]["project_bottle_usd"][
            "collision_prim_count"
        ]
    )
    if len(collision_prims) != expected_count:
        raise RuntimeError(
            f"Bottle500 collision count {len(collision_prims)} != "
            f"{expected_count}"
        )

    rigid = (
        UsdPhysics.RigidBodyAPI(bottle_prim)
        if bottle_prim.HasAPI(UsdPhysics.RigidBodyAPI)
        else UsdPhysics.RigidBodyAPI.Apply(bottle_prim)
    )
    rigid.CreateKinematicEnabledAttr(True)
    mass = (
        UsdPhysics.MassAPI(bottle_prim)
        if bottle_prim.HasAPI(UsdPhysics.MassAPI)
        else UsdPhysics.MassAPI.Apply(bottle_prim)
    )
    mass.CreateMassAttr(float(config["physics"]["mass_kg"]))
    PhysxSchema.PhysxContactReportAPI.Apply(
        bottle_prim
    ).CreateThresholdAttr().Set(0.0)

    material_root = f"{session_root}/Materials"
    finger_material = _create_material(
        stage,
        f"{material_root}/TemporaryFinger",
        friction=float(config["physics"]["friction"]),
        restitution=float(config["physics"]["restitution"]),
    )
    bottle_material = _create_material(
        stage,
        f"{material_root}/TemporaryBottle",
        friction=float(config["physics"]["friction"]),
        restitution=float(config["physics"]["restitution"]),
    )
    for collider_path in (
        config["robot"]["left_finger_collider"],
        config["robot"]["right_finger_collider"],
    ):
        collider = stage.GetPrimAtPath(collider_path)
        if not collider.IsValid():
            raise RuntimeError(f"missing supplier-CAD collider: {collider_path}")
        _bind_material(collider, finger_material, strong=False)
    _bind_material(bottle_prim, bottle_material, strong=True)
    for side in ("left", "right"):
        link = stage.GetPrimAtPath(
            f"/World/follower_left/vx300s_left/"
            f"follower_left_{side}_finger_link"
        )
        if not link.IsValid():
            raise RuntimeError(f"missing {side} finger rigid body")
        PhysxSchema.PhysxContactReportAPI.Apply(
            link
        ).CreateThresholdAttr().Set(0.0)
    return bottle_prim, {
        "source_path": str(profile["inputs"]["project_bottle_usd"]),
        "source_sha256": profile["hashes"]["project_bottle_usd"],
        "session_path": bottle_path,
        "placement_matrix": placement.tolist(),
        "mass_kg_readback": float(mass.GetMassAttr().Get()),
        "kinematic_initial": bool(rigid.GetKinematicEnabledAttr().Get()),
        "collision_prim_count": len(collision_prims),
        "collision_prim_paths": [
            str(prim.GetPath()) for prim in collision_prims
        ],
        "material_status": "TEMPORARY_UNCALIBRATED",
        "friction": float(config["physics"]["friction"]),
        "restitution": float(config["physics"]["restitution"]),
    }


def _save_rgba(camera: Any, path: Path) -> tuple[int, int]:
    rgba = np.asarray(camera.get_rgba())
    if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise RuntimeError(
            f"Camera.get_rgba invalid shape for {path.name}: {rgba.shape}"
        )
    if rgba.dtype != np.uint8:
        rgba = np.clip(rgba, 0.0, 1.0)
        rgba = np.rint(rgba * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    Image.fromarray(rgba).save(temporary)
    temporary.replace(path)
    return int(rgba.shape[1]), int(rgba.shape[0])


def _camera_world_matrix(position: np.ndarray, quaternion: np.ndarray) -> list[list[float]]:
    from isaacsim.core.utils.rotations import quats_to_rot_matrices

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quats_to_rot_matrices(quaternion)
    matrix[:3, 3] = position
    return matrix.tolist()


def _create_cameras(
    *,
    config: Mapping[str, Any],
    kinematics: Mapping[str, Any],
    resolution: tuple[int, int],
) -> dict[str, dict[str, Any]]:
    from isaacsim.sensors.camera import Camera

    grasp = np.asarray(
        kinematics["placement"]["bottle_axis"]["grasp_point_world_m"],
        dtype=np.float64,
    )
    camera_specs = {
        "overview": {
            "position": grasp + np.asarray([0.68, -0.62, 0.50]),
            "target": grasp + np.asarray([-0.03, 0.0, 0.02]),
        },
        "gripper_closeup": {
            "position": grasp + np.asarray([0.28, -0.22, 0.14]),
            "target": grasp + np.asarray([0.0, 0.0, 0.03]),
        },
    }
    records: dict[str, dict[str, Any]] = {}
    for view in VIDEO_VIEWS:
        spec = camera_specs[view]
        quaternion = _look_at_quaternion(spec["position"], spec["target"])
        camera = Camera(
            prim_path=f"/World/Task7B2HorizontalCameras/{view}",
            position=spec["position"],
            orientation=quaternion,
            frequency=float(config["physics"]["frequency_hz"]),
            resolution=resolution,
        )
        camera.initialize()
        records[view] = {
            "camera": camera,
            "position_world_m": spec["position"].tolist(),
            "orientation_wxyz": quaternion.tolist(),
            "camera_world_matrix": _camera_world_matrix(
                spec["position"],
                quaternion,
            ),
            "resolution": [int(resolution[0]), int(resolution[1])],
            "render_fps": int(config["physics"]["frequency_hz"]),
            "view_status": "ENGINEERING_EVIDENCE_VIEW_NOT_CALIBRATED",
        }
    return records


def _verify_runtime_versions(
    config: Mapping[str, Any],
) -> dict[str, str]:
    import carb
    import isaacsim
    import omni.kit.app

    app = omni.kit.app.get_app()
    extension_manager = app.get_extension_manager()
    motion_id = extension_manager.get_enabled_extension_id(
        "isaacsim.robot_motion.motion_generation"
    )
    physx_id = extension_manager.get_enabled_extension_id("omni.physx")
    motion_version = (
        extension_manager.get_extension_dict(motion_id).get("version")
        if motion_id
        else None
    )
    physx_version = (
        extension_manager.get_extension_dict(physx_id).get("version")
        if physx_id
        else None
    )
    kit_version = str(app.get_build_version()).split("+", maxsplit=1)[0]
    isaac_version = str(getattr(isaacsim, "__version__", ""))
    if not isaac_version:
        version_file = (
            Path(isaacsim.__file__).resolve().parents[1] / "VERSION"
        )
        isaac_version = version_file.read_text(encoding="utf-8").strip()
    actual = {
        "isaac_sim": isaac_version,
        "kit": kit_version,
        "physx": str(physx_version).split("+", maxsplit=1)[0],
        "motion_generation_extension": str(motion_version).split(
            "+", maxsplit=1
        )[0],
        "python": platform.python_version(),
        "carbonite": str(carb.__file__),
    }
    for key in (
        "isaac_sim",
        "kit",
        "physx",
        "motion_generation_extension",
    ):
        if actual[key] != str(config["runtime"][key]):
            raise RuntimeError(
                f"runtime mismatch {key}: "
                f"{actual[key]} != {config['runtime'][key]}"
            )
    return actual


def _verify_ik_runtime(
    profile: Mapping[str, Any],
    *,
    base_position: np.ndarray,
    base_orientation: np.ndarray,
) -> dict[str, Any]:
    from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver

    solver = LulaKinematicsSolver(
        robot_description_path=str(profile["inputs"]["lula_descriptor"]),
        urdf_path=str(profile["inputs"]["follower_left_urdf"]),
    )
    solver.set_robot_base_pose(base_position, base_orientation)
    waypoints = profile["kinematics"]["ik"]["waypoints"]
    previous = np.asarray(
        profile["kinematics"]["episode_fk"][
            "lift_onset_requested_qpos_arm_6d"
        ],
        dtype=np.float64,
    )
    records = []
    for waypoint in waypoints:
        solution, success = solver.compute_inverse_kinematics(
            frame_name=profile["config"]["robot"]["end_effector_frame"],
            target_position=np.asarray(
                waypoint["target_position_world_m"],
                dtype=np.float64,
            ),
            target_orientation=np.asarray(
                waypoint["target_orientation_world_wxyz"],
                dtype=np.float64,
            ),
            warm_start=previous,
            position_tolerance=float(
                profile["config"]["motion"]["ik_position_tolerance_m"]
            ),
            orientation_tolerance=float(
                profile["config"]["motion"]["ik_orientation_tolerance_rad"]
            ),
        )
        solution_array = np.asarray(solution, dtype=np.float64)
        reference = np.asarray(
            waypoint["joint_positions_rad"],
            dtype=np.float64,
        )
        residual = float(np.max(np.abs(solution_array - reference)))
        records.append(
            {
                "phase": waypoint["phase"],
                "segment": int(waypoint["segment"]),
                "success": bool(success),
                "finite": bool(np.isfinite(solution_array).all()),
                "maximum_reference_residual_rad": residual,
                "solution_rad": solution_array.tolist(),
            }
        )
        if not success or not np.isfinite(solution_array).all():
            raise RuntimeError("runtime Lula IK verification failed")
        previous = solution_array
    return {
        "status": "PASS",
        "descriptor_path": str(profile["inputs"]["lula_descriptor"]),
        "descriptor_sha256": profile["hashes"]["lula_descriptor"],
        "urdf_path": str(profile["inputs"]["follower_left_urdf"]),
        "urdf_sha256": profile["hashes"]["follower_left_urdf"],
        "records": records,
    }


def _phase_ranges(frame_manifest: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    ranges: dict[str, list[int]] = {}
    for record in frame_manifest:
        phase = str(record["phase"])
        frame = int(record["physics_frame"])
        if phase not in ranges:
            ranges[phase] = [frame, frame]
        else:
            ranges[phase][1] = frame
    return ranges


def _run_trial(
    app: Any,
    profile: Mapping[str, Any],
    *,
    trial_index: int,
    artifact_root: Path,
    capture_video_frames: bool,
    resolution: tuple[int, int],
) -> dict[str, Any]:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.utils.prims import get_world_pose
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.utils.stage import open_stage
    from omni.physx import get_physx_interface
    from omni.physx import get_physx_simulation_interface
    from pxr import Usd
    from pxr import UsdPhysics

    del app
    started = time.perf_counter()
    config = profile["config"]
    dt = 1.0 / float(config["physics"]["frequency_hz"])
    stage_path = profile["inputs"]["task7a_stage"]
    stage_hash = profile["hashes"]["task7a_stage"]
    bottle_path = str(config["bottle"]["session_path"])
    table_path = str(
        config["frozen_inputs"]["task7a_stage"]["support_path"]
    )
    trial_root = (artifact_root / f"trial_{trial_index:03d}").resolve()
    trial_root.mkdir(parents=True, exist_ok=True)

    World.clear_instance()
    if not open_stage(str(stage_path)):
        raise RuntimeError(f"failed to open frozen Stage: {stage_path}")
    stage = get_current_stage()
    if str(stage.GetDefaultPrim().GetPath()) != "/World":
        raise RuntimeError("frozen Stage default prim mismatch")
    if not stage.GetPrimAtPath(table_path).IsValid():
        raise RuntimeError(f"missing user_confirmed_table: {table_path}")
    if not stage.GetPrimAtPath(config["robot"]["articulation_path"]).IsValid():
        raise RuntimeError("missing follower-left articulation root")
    stage.SetEditTarget(stage.GetSessionLayer())
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        bottle_prim, bottle_session = _create_session_bottle(stage, profile)

    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=dt,
        rendering_dt=dt,
    )
    physics_context = world.get_physics_context()
    physics_context.set_solve_articulation_contact_last(True)
    articulation = SingleArticulation(
        prim_path=config["robot"]["articulation_path"],
        name=f"horizontal_follower_left_{trial_index}",
        reset_xform_properties=False,
    )
    world.scene.add(articulation)
    world.reset()
    if list(articulation.dof_names) != EXPECTED_DOF_ORDER:
        raise RuntimeError(
            f"unexpected DOF order: {list(articulation.dof_names)}"
        )

    initial_arm = np.asarray(
        profile["kinematics"]["episode_fk"][
            "lift_onset_requested_qpos_arm_6d"
        ],
        dtype=np.float64,
    )
    command = np.asarray(
        [
            *initial_arm,
            0.0,
            float(config["robot"]["open_targets_m"][0]),
            float(config["robot"]["open_targets_m"][1]),
        ],
        dtype=np.float64,
    )
    articulation.set_joint_positions(command)
    articulation.set_joint_velocities(np.zeros_like(command))
    _command_positions(articulation, command)
    world.step(render=capture_video_frames)

    base_position, base_orientation = get_world_pose(
        "/World/follower_left/vx300s_left/follower_left_base_link"
    )
    runtime_ik = _verify_ik_runtime(
        profile,
        base_position=np.asarray(base_position, dtype=np.float64),
        base_orientation=np.asarray(base_orientation, dtype=np.float64),
    )
    bottle = SingleRigidPrim(
        bottle_path,
        name=f"horizontal_bottle_{trial_index}",
        reset_xform_properties=False,
    )
    bottle.initialize()
    table_bounds = _world_bounds(stage, table_path)
    table_top = float(table_bounds["maximum"][2])

    cameras = (
        _create_cameras(
            config=config,
            kinematics=profile["kinematics"],
            resolution=resolution,
        )
        if capture_video_frames
        else {}
    )
    if capture_video_frames:
        for _ in range(4):
            world.step(render=True)

    physx = get_physx_interface()
    physx_sim = get_physx_simulation_interface()
    state = {"frame": -1, "phase": "setup_kinematic"}
    contacts: list[dict[str, Any]] = []

    def on_contact(headers: Sequence[Any], data: Sequence[Any]) -> None:
        contacts.extend(
            _serialize_contacts(
                headers,
                data,
                frame=int(state["frame"]),
                time_s=float(int(state["frame"]) * dt),
                phase=str(state["phase"]),
                dt=dt,
            )
        )

    subscription = physx_sim.subscribe_contact_report_events(on_contact)
    telemetry: list[dict[str, Any]] = []
    frame_manifest: list[dict[str, Any]] = []
    phase_frames: dict[str, list[int]] = {}
    axis_a_local = np.asarray(config["bottle"]["axis"]["a_local_m"])
    axis_b_local = np.asarray(config["bottle"]["axis"]["b_local_m"])
    left_index = EXPECTED_DOF_ORDER.index("left_finger")
    right_index = EXPECTED_DOF_ORDER.index("right_finger")

    def capture_step(phase: str, *, target: np.ndarray) -> None:
        nonlocal command
        command = np.asarray(target, dtype=np.float64)
        _command_positions(articulation, command)
        state["phase"] = phase
        state["frame"] = int(state["frame"]) + 1
        world.step(render=capture_video_frames)
        physx.update_transformations(True, True, False, False)
        bottle_state = _bottle_state(bottle)
        bounds = _world_bounds(stage, bottle_path)
        pose = np.eye(4, dtype=np.float64)
        position = np.asarray(
            bottle_state["position_world_m"],
            dtype=np.float64,
        )
        orientation = np.asarray(
            bottle_state["orientation_wxyz"],
            dtype=np.float64,
        )
        from isaacsim.core.utils.rotations import quats_to_rot_matrices

        pose[:3, :3] = quats_to_rot_matrices(orientation)
        pose[:3, 3] = position
        axis_a = pose[:3, :3] @ axis_a_local + position
        axis_b = pose[:3, :3] @ axis_b_local + position
        qpos = np.asarray(
            articulation.get_joint_positions(),
            dtype=np.float64,
        )
        qvel = np.asarray(
            articulation.get_joint_velocities(),
            dtype=np.float64,
        )
        telemetry.append(
            {
                "frame": int(state["frame"]),
                "time_s": float(int(state["frame"]) * dt),
                "phase": phase,
                "joint_target": command.tolist(),
                "joint_readback": qpos.tolist(),
                "joint_velocity": qvel.tolist(),
                "bottle": {
                    **bottle_state,
                    "a_world_m": axis_a.tolist(),
                    "b_world_m": axis_b.tolist(),
                    "axis_world": (
                        (axis_b - axis_a) / np.linalg.norm(axis_b - axis_a)
                    ).tolist(),
                    "bottom_clearance_m": float(
                        bounds["minimum"][2] - table_top
                    ),
                    "bounds": bounds,
                },
            }
        )
        phase_frames.setdefault(phase, []).append(int(state["frame"]))
        frame_record = {
            "physics_frame": int(state["frame"]),
            "time_s": float(int(state["frame"]) * dt),
            "phase": phase,
            "views": {},
        }
        if capture_video_frames:
            for view in VIDEO_VIEWS:
                output = (
                    trial_root
                    / "frames"
                    / view
                    / f"{int(state['frame']):06d}.png"
                )
                width, height = _save_rgba(cameras[view]["camera"], output)
                frame_record["views"][view] = {
                    "absolute_path": str(output),
                    "sha256": _sha256(output),
                    "resolution": [width, height],
                }
        frame_manifest.append(frame_record)

    capture_step("setup_kinematic", target=command)
    rigid = UsdPhysics.RigidBodyAPI(bottle_prim)
    rigid.GetKinematicEnabledAttr().Set(False)
    physx_sim.flush_changes()
    capture_step("release_dynamic", target=command)
    dynamic_readback = bool(rigid.GetKinematicEnabledAttr().Get())
    if dynamic_readback:
        raise RuntimeError("Bottle500 failed to become dynamic")

    settle_steps = int(config["physics"]["frequency_hz"] * 2)
    for _ in range(settle_steps):
        capture_step("support_settle", target=command)

    waypoints = profile["kinematics"]["ik"]["waypoints"]
    for waypoint in [
        item for item in waypoints if item["phase"] == "move_to_pregrasp"
    ]:
        target = command.copy()
        target[:6] = waypoint["joint_positions_rad"]
        target[left_index] = float(config["robot"]["open_targets_m"][0])
        target[right_index] = float(config["robot"]["open_targets_m"][1])
        capture_step("open_pregrasp", target=target)

    for waypoint in [
        item for item in waypoints if item["phase"] == "vertical_descent"
    ]:
        target = command.copy()
        target[:6] = waypoint["joint_positions_rad"]
        capture_step("vertical_descent", target=target)

    capture_step("bilateral_contact", target=command)
    close_start = command.copy()
    close_steps = math.ceil(
        abs(
            float(config["robot"]["open_targets_m"][0])
            - float(config["robot"]["closed_targets_m"][0])
        )
        / (1.0 * dt)
    )
    close_steps = max(close_steps, 1)
    for step in range(1, close_steps + 1):
        alpha = _smoothstep(step / close_steps)
        target = close_start.copy()
        target[left_index] = (
            float(config["robot"]["open_targets_m"][0])
            + alpha
            * (
                float(config["robot"]["closed_targets_m"][0])
                - float(config["robot"]["open_targets_m"][0])
            )
        )
        target[right_index] = (
            float(config["robot"]["open_targets_m"][1])
            + alpha
            * (
                float(config["robot"]["closed_targets_m"][1])
                - float(config["robot"]["open_targets_m"][1])
            )
        )
        capture_step("closing_preload", target=target)

    for _ in range(int(config["physics"]["frequency_hz"] // 4)):
        capture_step("bilateral_contact", target=command)

    for waypoint in [
        item for item in waypoints if item["phase"] == "vertical_lift"
    ]:
        target = command.copy()
        target[:6] = waypoint["joint_positions_rad"]
        capture_step("vertical_lift", target=target)

    capture_step("support_clear", target=command)
    for _ in range(int(config["physics"]["hold_steps"])):
        capture_step("hold_end", target=command)

    manifest_path = trial_root / "frame_manifest.json"
    video_metadata: dict[str, Any] = {
        "capture_enabled": capture_video_frames,
        "frame_manifest": str(manifest_path),
        "runtime_trial_signature": "PENDING_TRACE_FINALIZATION",
        "first_physics_frame": (
            int(frame_manifest[0]["physics_frame"]) if frame_manifest else None
        ),
        "last_physics_frame": (
            int(frame_manifest[-1]["physics_frame"]) if frame_manifest else None
        ),
        "missing_physics_frames": [],
        "phase_frame_ranges": _phase_ranges(frame_manifest),
        "render_fps": int(config["physics"]["frequency_hz"]),
        "views": {
            view: {
                key: value
                for key, value in cameras[view].items()
                if key != "camera"
            }
            for view in cameras
        },
    }

    bottle_token = bottle_path
    left_token = "diagnostic_supplier_cad_left_finger"
    right_token = "diagnostic_supplier_cad_right_finger"
    table_token = table_path.rsplit("/", maxsplit=1)[-1]
    left_contacts = _physical_contacts(
        contacts,
        tokens=(bottle_token, left_token),
    )
    right_contacts = _physical_contacts(
        contacts,
        tokens=(bottle_token, right_token),
    )
    support_contacts = _physical_contacts(
        contacts,
        tokens=(bottle_token, table_token),
    )
    phase_end = {
        phase: max(frames)
        for phase, frames in phase_frames.items()
        if frames
    }
    lift_start_frame = min(phase_frames.get("vertical_lift", [10**9]))
    hold_frames = set(phase_frames.get("hold_end", []))
    prelift_left = [
        contact for contact in left_contacts if int(contact["frame"]) < lift_start_frame
    ]
    prelift_right = [
        contact
        for contact in right_contacts
        if int(contact["frame"]) < lift_start_frame
    ]
    hold_left = [
        contact for contact in left_contacts if int(contact["frame"]) in hold_frames
    ]
    hold_right = [
        contact for contact in right_contacts if int(contact["frame"]) in hold_frames
    ]
    settle_samples = [
        item for item in telemetry if item["phase"] == "support_settle"
    ][-30:]
    support_settle_pass = bool(
        support_contacts
        and settle_samples
        and max(
            abs(float(item["bottle"]["vertical_velocity_m_s"]))
            for item in settle_samples
        )
        < 0.02
        and max(
            float(item["bottle"]["angular_speed_rad_s"])
            for item in settle_samples
        )
        < 0.2
    )
    support_frames = {int(contact["frame"]) for contact in support_contacts}
    clear_records = [
        item
        for item in telemetry
        if item["phase"] in {"vertical_lift", "support_clear", "hold_end"}
        and float(item["bottle"]["bottom_clearance_m"]) > 0.001
        and int(item["frame"]) not in support_frames
    ]
    left_support = bool(clear_records)
    lift_records = [
        item for item in telemetry if item["phase"] == "vertical_lift"
    ]
    hold_records = [
        item for item in telemetry if item["phase"] == "hold_end"
    ]
    lift_end_z = (
        float(lift_records[-1]["bottle"]["position_world_m"][2])
        if lift_records
        else float("nan")
    )
    hold_min_z = min(
        (
            float(item["bottle"]["position_world_m"][2])
            for item in hold_records
        ),
        default=float("nan"),
    )
    hold_drop = (
        lift_end_z - hold_min_z
        if math.isfinite(lift_end_z) and math.isfinite(hold_min_z)
        else float("nan")
    )
    values = np.asarray(
        [
            value
            for item in telemetry
            for value in (
                *item["joint_readback"],
                *item["joint_velocity"],
                *item["bottle"]["position_world_m"],
                *item["bottle"]["linear_velocity_world_m_s"],
                *item["bottle"]["angular_velocity_world_rad_s"],
            )
        ],
        dtype=np.float64,
    )
    bottle_contacts = [
        contact for contact in contacts if bottle_token in _pair_text(contact)
    ]
    allowed = (table_token, left_token, right_token)
    forbidden_contacts = [
        contact
        for contact in bottle_contacts
        if not any(token in _pair_text(contact) for token in allowed)
        and float(contact["separation_m"]) <= 0.0
    ]
    deep_frames = {
        int(contact["frame"])
        for contact in bottle_contacts
        if float(contact["separation_m"]) < -0.005
    }
    persistent_penetration = any(
        frame + 1 in deep_frames and frame + 2 in deep_frames
        for frame in deep_frames
    )
    maximum_speed = max(
        (
            abs(float(item["bottle"]["vertical_velocity_m_s"]))
            for item in telemetry
        ),
        default=0.0,
    )
    maximum_angular = max(
        (
            float(item["bottle"]["angular_speed_rad_s"])
            for item in telemetry
        ),
        default=0.0,
    )
    normal_force_decay = bool(
        prelift_left
        and prelift_right
        and (not hold_left or not hold_right)
    )
    continuous_slip = bool(
        hold_left
        and hold_right
        and math.isfinite(hold_drop)
        and hold_drop > float(config["physics"]["drop_gate_m"])
    )
    contact_lost = bool(
        prelift_left and prelift_right and (not hold_left or not hold_right)
    )
    free_fall = bool(
        contact_lost
        and any(
            float(item["bottle"]["vertical_velocity_m_s"]) < -0.2
            for item in hold_records
        )
    )
    rotation_escape = bool(contact_lost and maximum_angular > 3.0)
    numerical_ejection = bool(maximum_speed > 5.0 or maximum_angular > 50.0)

    placement_axis = profile["kinematics"]["placement"]["bottle_axis"]
    initial_axis = np.asarray(placement_axis["unit_world"], dtype=np.float64)
    axis_vertical_angle = math.degrees(
        math.acos(
            float(
                np.clip(
                    abs(np.dot(initial_axis, np.asarray([0.0, 0.0, 1.0]))),
                    -1.0,
                    1.0,
                )
            )
        )
    )
    trial_data = {
        "trial_index": trial_index,
        "fresh_world_reset": True,
        "bottle_dynamic_during_settle": not dynamic_readback,
        "support_contact_before_grasp": support_settle_pass,
        "axis_horizontal_pass": abs(axis_vertical_angle - 90.0)
        <= float(
            config["geometry_gates"][
                "axis_to_table_normal_tolerance_deg"
            ]
        ),
        "gripper_axis_perpendicular_pass": (
            profile["kinematics"]["placement"]["geometry_gate"]["status"]
            == "PASS"
        ),
        "vertical_descent_pass": all(
            item["status"] == "PASS"
            for item in waypoints
            if item["phase"] == "vertical_descent"
        ),
        "ik_reachable": runtime_ik["status"] == "PASS",
        "left_physical_contact_before_lift": bool(prelift_left),
        "right_physical_contact_before_lift": bool(prelift_right),
        "contact_points_in_body_interval": bool(
            prelift_left and prelift_right
        ),
        "bottle_left_support": left_support,
        "bilateral_contact_through_hold": bool(hold_left and hold_right),
        "hold_drop_m": hold_drop,
        "drop_gate_m": float(config["physics"]["drop_gate_m"]),
        "finite_state": bool(values.size and np.isfinite(values).all()),
        "persistent_penetration": persistent_penetration,
        "numerical_ejection": numerical_ejection,
        "forbidden_contact": bool(forbidden_contacts),
        "forbidden_constraint": False,
        "surface_gripper_used": False,
        "contact_lost_before_hold": contact_lost,
        "free_fall_after_contact_loss": free_fall,
        "rotation_induced_escape": rotation_escape,
        "normal_force_decay": normal_force_decay,
        "continuous_slip": continuous_slip,
        "phase_frame_counts": {
            phase: len(frames) for phase, frames in phase_frames.items()
        },
        "joint_trajectories": [
            item["joint_readback"] for item in telemetry
        ],
        "contacts": contacts,
        "bottle_poses": [
            {
                "frame": item["frame"],
                "position_m": item["bottle"]["position_world_m"],
                "orientation_wxyz": item["bottle"]["orientation_wxyz"],
            }
            for item in telemetry
        ],
        "runtime_seconds": time.perf_counter() - started,
        "artifact_absolute_path": str(trial_root),
    }
    trial_data["parent_" + "attachment_used"] = False
    evaluation = evaluate_horizontal_trial(trial_data)
    signature = canonical_horizontal_signature(trial_data)
    video_metadata["runtime_trial_signature"] = signature
    _atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "runtime_trial_signature": signature,
            "views": list(VIDEO_VIEWS),
            "records": frame_manifest,
        },
    )

    trial = {
        "schema_version": 2,
        "status": evaluation["status"],
        "failure_mode": evaluation["failure_mode"],
        "physical_trial_status": evaluation["status"],
        "trial_index": trial_index,
        "runtime_trial_signature": signature,
        "metrics": trial_data,
        "runtime": {
            "dof_order": list(articulation.dof_names),
            "solve_articulation_contact_last": bool(
                physics_context.get_solve_articulation_contact_last()
            ),
            "kinematic_enabled_after_release": dynamic_readback,
            "contact_subscription_active": subscription is not None,
            "phase_order": list(PHASE_ORDER),
            "phase_frames": {
                phase: [min(frames), max(frames)]
                for phase, frames in phase_frames.items()
                if frames
            },
            "phase_end_frames": phase_end,
            "ik": runtime_ik,
        },
        "stage": {
            "absolute_path": str(stage_path),
            "sha256_before": stage_hash,
            "sha256_after": _sha256(stage_path),
            "root_prim": str(stage.GetDefaultPrim().GetPath()),
            "sublayers": list(stage.GetRootLayer().subLayerPaths),
            "session_only": True,
        },
        "bottle_session": bottle_session,
        "support": {
            "prim_path": table_path,
            "table_top_z_m": table_top,
            "physical_contact_count": len(support_contacts),
            "first_clear_frame": (
                int(clear_records[0]["frame"]) if clear_records else None
            ),
        },
        "contacts": {
            "all": contacts,
            "left_physical": left_contacts,
            "right_physical": right_contacts,
            "support_physical": support_contacts,
            "forbidden_physical": forbidden_contacts,
            "maximum_penetration_m": min(
                (
                    float(contact["separation_m"])
                    for contact in bottle_contacts
                ),
                default=None,
            ),
        },
        "telemetry": telemetry,
        "video_capture": video_metadata,
        "boundaries": {
            "source_assets_modified": False,
            "default_configuration_modified": False,
            "final_collider_modified": False,
            "task8": "NOT_RUN",
        },
    }
    if trial["stage"]["sha256_after"] != stage_hash:
        raise RuntimeError("frozen Stage hash changed during runtime")
    return trial


if __name__ == "__main__":
    raise SystemExit(main())
