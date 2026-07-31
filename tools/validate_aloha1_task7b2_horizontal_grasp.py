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
from importlib.metadata import version
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
from tools.probe_aloha1_task7b2_horizontal_kinematics import solve_adaptive_linear_ik
from tools.run_aloha1_grasp_editor_variant_b_gui import build_external_close_targets
from tools.validate_aloha1_gripper_coupling_ab import author_coupling_variant

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/aloha1_task7b2_horizontal_grasp.yaml"
DEFAULT_OUTPUT = ROOT / "reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp_v2.json"
DEFAULT_TRIALS = ROOT / "reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp_v2_trials.jsonl"
DEFAULT_MARKDOWN = ROOT / "reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp_v2.md"
DEFAULT_ARTIFACT_ROOT = ROOT / ".codex/artifacts/20260730-aloha1-official-gripper-unattended/stage6/runtime"
KINEMATICS_REPORT = ROOT / "reports/aloha1_mapping/aloha1_task7b2_horizontal_kinematics_v2.json"
IK_CORRESPONDENCE_REPORT = ROOT / "reports/aloha1_mapping/aloha1_ik_correspondence_v2.json"
LULA_DESCRIPTOR = ROOT / "configs/aloha1_lula_follower_left.yaml"

VIDEO_VIEWS = ("overview", "gripper_closeup")
SCREENSHOT_VIEWS = ("true_top", "side")
FULL_ARM_LINK_PRIMS = {
    "base": ("/World/follower_left/vx300s_left/follower_left_base_link",),
    "shoulder": (
        "/World/follower_left/vx300s_left/follower_left_shoulder_link",
        "/World/follower_left/vx300s_left/follower_left_upper_arm_link",
    ),
    "elbow": ("/World/follower_left/vx300s_left/follower_left_upper_forearm_link",),
    "forearm": ("/World/follower_left/vx300s_left/follower_left_lower_forearm_link",),
    "wrist": ("/World/follower_left/vx300s_left/follower_left_wrist_link",),
    "gripper": (
        "/World/follower_left/vx300s_left/follower_left_gripper_link",
        "/World/follower_left/vx300s_left/follower_left_left_finger_link",
        "/World/follower_left/vx300s_left/follower_left_right_finger_link",
    ),
}
FULL_ARM_NUMERIC_EVIDENCE_SCOPE = "WORLD_AABB_CAMERA_FRUSTUM_AND_IMAGE_BOUNDS_ONLY"
FULL_ARM_OCCLUSION_EVALUATION_STATUS = "NOT_EVALUATED_REQUIRES_VISUAL_REVIEW"
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
DIAGNOSTIC_COUPLING_CLASSIFICATION = "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
DIAGNOSTIC_FORCE_DRIVE_CLASSIFICATION = "DIAGNOSTIC_ONLY_FORCE_DRIVE_UNCALIBRATED"
FINGER_JOINT_PATHS = (
    "/World/follower_left/vx300s_left/joints/left_finger",
    "/World/follower_left/vx300s_left/joints/right_finger",
)
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
    if not 0.0 <= args.preload_delta_m <= 0.002:
        raise ValueError("--preload-delta-m must be within the frozen 0.0 to 0.002 m diagnostic range")
    profile = _load_profile(args.config)
    profile["diagnostic_preload_delta_m"] = float(args.preload_delta_m)
    profile["diagnostic_finger_drive_type"] = str(args.finger_drive_type)
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
        capture_views = VIDEO_VIEWS if args.capture_profile == "video" else SCREENSHOT_VIEWS
        for trial_index in range(args.repeats):
            trials.append(
                _run_trial(
                    app,
                    profile,
                    trial_index=trial_index,
                    artifact_root=artifact_root,
                    capture_video_frames=bool(args.capture_video_frames),
                    capture_collider_evidence=bool(
                        args.capture_collider_evidence
                    ),
                    capture_profile=str(args.capture_profile),
                    capture_views=capture_views,
                    resolution=(int(args.width), int(args.height)),
                )
            )
        summary = summarize_horizontal_trials([trial["metrics"] for trial in trials])
        physical_status = trials[0]["status"] if len(trials) == 1 else summary["status"]
        report = {
            "schema_version": 2,
            "status": (physical_status if len(trials) == 1 else summary["status"]),
            "trial_kind": ("SINGLE_DYNAMIC_SMOKE" if len(trials) == 1 else "REPEATED_DYNAMIC_DIAGNOSTIC"),
            "acceptance_random_trials": "NOT_RUN",
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
    parser.add_argument(
        "--capture-collider-evidence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "capture paired normal/physics-collider-overlay stills at the "
            "release, open, bilateral-contact, support-clear, and hold-end phases"
        ),
    )
    parser.add_argument(
        "--capture-profile",
        choices=("video", "screenshots"),
        default="video",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument(
        "--preload-delta-m",
        type=float,
        default=0.0,
        help=("diagnostic-only symmetric extra closure after the supplier-CAD first-contact target; range 0..0.002 m"),
    )
    parser.add_argument(
        "--finger-drive-type",
        choices=("acceleration", "force"),
        default="acceleration",
        help=(
            "session-only diagnostic finger drive type; stiffness, damping, "
            "maxForce, collider, friction, mimic adapter, bottle, and timestep stay frozen"
        ),
    )
    args, kit_args = parser.parse_known_args()
    invalid = [value for value in kit_args if not value.startswith("--/")]
    if invalid:
        parser.error(f"unrecognized arguments: {' '.join(invalid)}")
    args.kit_args = kit_args
    return args


def _resolve_source(root: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve(strict=True)


def validate_session_drive_type_readback(
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    requested_type: str,
) -> dict[str, Any]:
    """Fail closed unless a session drive edit changes only the requested type."""
    if requested_type not in {"acceleration", "force"}:
        raise ValueError(f"unsupported finger drive type: {requested_type}")
    if str(after["type"]) != requested_type:
        raise RuntimeError(f"drive type readback {after['type']!r} != requested {requested_type!r}")

    invariant_fields = ("stiffness", "damping", "max_force")
    drifted = [
        field
        for field in invariant_fields
        if not math.isclose(
            float(before[field]),
            float(after[field]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if drifted:
        raise RuntimeError(f"session drive edit changed frozen fields: {', '.join(drifted)}")

    return {
        "status": "PASS",
        "requested_type": requested_type,
        "type_changed": str(before["type"]) != str(after["type"]),
        "only_drive_type_changed": True,
        "frozen_fields_unchanged": list(invariant_fields),
        "classification": (
            DIAGNOSTIC_FORCE_DRIVE_CLASSIFICATION
            if requested_type == "force"
            else "BASELINE_ACCELERATION_DRIVE"
        ),
    }


def _read_usd_drive_parameters(drive: Any) -> dict[str, Any]:
    values = {
        "type": drive.GetTypeAttr().Get(),
        "stiffness": drive.GetStiffnessAttr().Get(),
        "damping": drive.GetDampingAttr().Get(),
        "max_force": drive.GetMaxForceAttr().Get(),
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise RuntimeError(f"finger drive missing authored/readable attributes: {', '.join(missing)}")
    return {
        "type": str(values["type"]),
        "stiffness": float(values["stiffness"]),
        "damping": float(values["damping"]),
        "max_force": float(values["max_force"]),
    }


def _author_session_finger_drive_type(
    *,
    stage: Any,
    usd_physics: Any,
    requested_type: str,
) -> dict[str, Any]:
    joints: dict[str, Any] = {}
    for joint_path in FINGER_JOINT_PATHS:
        joint = stage.GetPrimAtPath(joint_path)
        if not joint.IsValid():
            raise RuntimeError(f"missing finger joint for drive diagnostic: {joint_path}")
        drive = usd_physics.DriveAPI.Get(joint, "linear")
        if not drive:
            raise RuntimeError(f"missing linear DriveAPI for finger joint: {joint_path}")
        before = _read_usd_drive_parameters(drive)
        if not drive.GetTypeAttr().Set(requested_type):
            raise RuntimeError(f"failed to author session drive type {requested_type!r}: {joint_path}")
        after = _read_usd_drive_parameters(drive)
        validation = validate_session_drive_type_readback(
            before=before,
            after=after,
            requested_type=requested_type,
        )
        joints[joint_path] = {
            "before": before,
            "after": after,
            "validation": validation,
        }
    return {
        "status": "PASS",
        "requested_type": requested_type,
        "classification": (
            DIAGNOSTIC_FORCE_DRIVE_CLASSIFICATION
            if requested_type == "force"
            else "BASELINE_ACCELERATION_DRIVE"
        ),
        "session_layer_identifier": stage.GetSessionLayer().identifier,
        "source_stage_modified": False,
        "joints": joints,
    }


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
        "grasp_editor_v2_semantics": frozen["grasp_editor_v2_semantics"],
        "grasp_editor_v2_native_raw_yaml": frozen["grasp_editor_v2_native_raw_yaml"],
        "supplier_cad_grasp_candidate": frozen["supplier_cad_grasp_candidate"],
        "grasp_editor_v2_derived_yaml": frozen["grasp_editor_v2_derived_yaml"],
        "gripper_coupling_ab": frozen["gripper_coupling_ab"],
    }
    inputs = {name: _resolve_source(ROOT, str(spec["path"])) for name, spec in input_specs.items()}
    inputs["kinematics_report"] = KINEMATICS_REPORT.resolve(strict=True)
    inputs["ik_correspondence_report"] = IK_CORRESPONDENCE_REPORT.resolve(strict=True)
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
        raise RuntimeError("frozen input hash mismatch: " + json.dumps(mismatches, sort_keys=True))

    kinematics = json.loads(inputs["kinematics_report"].read_text(encoding="utf-8"))
    if kinematics.get("status") != "PASS":
        raise RuntimeError("horizontal kinematics report is not PASS")
    stage_record = kinematics.get("stage", {})
    if (
        Path(stage_record.get("path", "")).resolve() != inputs["task7a_stage"]
        or stage_record.get("sha256_after") != hashes["task7a_stage"]
        or not stage_record.get("immutable")
    ):
        raise RuntimeError("kinematics report does not bind frozen Stage")
    if kinematics.get("ik", {}).get("status") != "PASS":
        raise RuntimeError("kinematics report IK gate is not PASS")
    clearance_record = kinematics["placement"]["supplier_cad_finger_geometry"]["clearance_report"]
    clearance_path = Path(clearance_record["absolute_path"]).resolve(strict=True)
    clearance_hash = _sha256(clearance_path)
    if clearance_hash != clearance_record["sha256"]:
        raise RuntimeError("supplier-CAD clearance report hash mismatch")
    inputs["supplier_cad_clearance_report"] = clearance_path
    hashes["supplier_cad_clearance_report"] = clearance_hash
    ik_correspondence = json.loads(inputs["ik_correspondence_report"].read_text(encoding="utf-8"))
    if not (
        ik_correspondence.get("status") == "PASS"
        and ik_correspondence.get("aloha_6dof_correspondence") == "PASS"
        and ik_correspondence.get("ik") == "PASS"
        and ik_correspondence.get("diagnostic_coupling", {}).get("promotion_authorized") is False
    ):
        raise RuntimeError("ALOHA-specific FK/IK correspondence gate failed")
    return {
        "path": path,
        "sha256": _sha256(path),
        "config": config,
        "inputs": inputs,
        "hashes": hashes,
        "kinematics": kinematics,
        "ik_correspondence": ik_correspondence,
    }


def derive_interpolation_steps(
    start: Sequence[float],
    end: Sequence[float],
    episode_delta_limits: Sequence[float],
) -> int:
    """Return the minimum steps that stay inside episode command deltas."""
    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    limits = np.asarray(episode_delta_limits, dtype=np.float64)
    if (
        start_array.shape != end_array.shape
        or limits.shape != start_array.shape
        or not np.isfinite(start_array).all()
        or not np.isfinite(end_array).all()
        or not np.isfinite(limits).all()
    ):
        raise ValueError("interpolation vectors must be finite and aligned")
    delta = np.abs(end_array - start_array)
    blocked = (delta > 0.0) & (limits <= 0.0)
    if np.any(blocked):
        raise ValueError("episode command delta is zero for a moving joint")
    ratios = np.divide(
        delta,
        limits,
        out=np.zeros_like(delta),
        where=limits > 0.0,
    )
    return max(1, math.ceil(float(np.max(ratios, initial=0.0))))


def episode_gripper_targets(
    records: Sequence[Mapping[str, Any]],
    *,
    start_frame: int,
    end_frame: int,
    lower_m: float,
    scale_m: float,
) -> list[float]:
    """Map the proven episode action interval to URDF left-finger targets."""
    selected = [record for record in records if start_frame <= int(record["frame"]) <= end_frame]
    if not selected:
        raise ValueError("episode gripper interval is empty")
    targets = []
    for record in selected:
        action = float(record["gripper_action"])
        if not math.isfinite(action):
            raise ValueError("episode gripper action is non-finite")
        targets.append(lower_m + scale_m * min(max(action, 0.0), 1.0))
    return targets


def _command_positions(articulation: Any, target: np.ndarray) -> None:
    from isaacsim.core.utils.types import ArticulationAction

    target_array = np.asarray(target, dtype=np.float32)
    if target_array.shape != (len(EXPECTED_DOF_ORDER),):
        raise ValueError("ALOHA target must contain the explicit nine-DOF order")
    articulation.apply_action(
        ArticulationAction(
            joint_positions=target_array[:6],
            joint_indices=np.arange(6, dtype=np.int32),
        )
    )
    articulation.apply_action(
        ArticulationAction(
            joint_positions=target_array[[7, 8]],
            joint_indices=np.asarray([7, 8], dtype=np.int32),
        )
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


def _collision_world_bounds(
    stage: Any,
    rigid_body_path: str,
) -> dict[str, list[float]]:
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    root = stage.GetPrimAtPath(rigid_body_path)
    if not root.IsValid():
        raise RuntimeError(f"missing rigid body for collision bounds: {rigid_body_path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
    )
    minima: list[np.ndarray] = []
    maxima: list[np.ndarray] = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        minima.append(np.asarray(aligned.GetMin(), dtype=np.float64))
        maxima.append(np.asarray(aligned.GetMax(), dtype=np.float64))
    if not minima:
        raise RuntimeError(f"no collision prims below {rigid_body_path}")
    return {
        "minimum": np.min(np.vstack(minima), axis=0).tolist(),
        "maximum": np.max(np.vstack(maxima), axis=0).tolist(),
    }


def _collect_rigid_local_collision_points(
    stage: Any,
    rigid_body_path: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    from pxr import Gf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    root = stage.GetPrimAtPath(rigid_body_path)
    if not root.IsValid():
        raise RuntimeError(f"missing rigid body for local collision points: {rigid_body_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    points: list[list[float]] = []
    manifest: list[dict[str, Any]] = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"unsupported Bottle500 collider type {prim.GetTypeName()}: {prim.GetPath()}")
        authored = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not authored:
            raise RuntimeError(f"Bottle500 collider has no points: {prim.GetPath()}")
        relative, _ = cache.ComputeRelativeTransform(prim, root)
        local = [
            [float(value) for value in relative.Transform(Gf.Vec3d(*point))]
            for point in authored
        ]
        points.extend(local)
        manifest.append(
            {
                "prim_path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "point_count": len(local),
            }
        )
    if not points:
        raise RuntimeError(f"no Bottle500 collision points below {rigid_body_path}")
    return np.asarray(points, dtype=np.float64), manifest


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
                    "position_world_m": [float(value) for value in item.position],
                    "normal_world": [float(value) for value in item.normal],
                    "impulse_ns": float(np.linalg.norm(impulse)),
                    "impulse_vector_ns": [float(value) for value in impulse],
                    "estimated_normal_force_n": float(np.linalg.norm(impulse) / dt),
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
        if all(token in _pair_text(contact) for token in tokens) and float(contact["separation_m"]) <= 0.0
    ]


def read_physx_bottle_state(bottle: Any) -> dict[str, Any]:
    if int(bottle.count) != 1:
        raise RuntimeError(f"expected one Bottle500 rigid body, got {bottle.count}")
    transform_xyzw = np.asarray(bottle.get_transforms()[0], dtype=np.float64)
    velocity = np.asarray(bottle.get_velocities()[0], dtype=np.float64)
    if transform_xyzw.shape != (7,) or velocity.shape != (6,):
        raise RuntimeError("unexpected PhysX rigid-body tensor shape")
    position = transform_xyzw[:3]
    orientation = transform_xyzw[[6, 3, 4, 5]]
    linear = velocity[:3]
    angular = velocity[3:]
    return {
        "state_source": "OMNI_PHYSICS_TENSORS_RIGID_BODY_VIEW",
        "position_world_m": position.tolist(),
        "orientation_wxyz": orientation.tolist(),
        "linear_velocity_world_m_s": linear.tolist(),
        "angular_velocity_world_rad_s": angular.tolist(),
        "vertical_velocity_m_s": float(linear[2]),
        "angular_speed_rad_s": float(np.linalg.norm(angular)),
    }


def derive_pose_finite_difference_velocity(
    *,
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    dt_s: float,
) -> dict[str, Any]:
    """Derive world-frame velocity from consecutive rigid-body poses."""
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    previous_position = np.asarray(
        previous["position_world_m"],
        dtype=np.float64,
    )
    current_position = np.asarray(
        current["position_world_m"],
        dtype=np.float64,
    )
    previous_orientation = np.asarray(
        previous["orientation_wxyz"],
        dtype=np.float64,
    )
    current_orientation = np.asarray(
        current["orientation_wxyz"],
        dtype=np.float64,
    )
    if previous_position.shape != (3,) or current_position.shape != (3,):
        raise ValueError("positions must be 3-vectors")
    if previous_orientation.shape != (4,) or current_orientation.shape != (4,):
        raise ValueError("orientations must be wxyz quaternions")
    previous_norm = float(np.linalg.norm(previous_orientation))
    current_norm = float(np.linalg.norm(current_orientation))
    if previous_norm <= 0.0 or current_norm <= 0.0:
        raise ValueError("orientation quaternions must be nonzero")
    q0 = previous_orientation / previous_norm
    q1 = current_orientation / current_norm
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    w0, xyz0 = float(q0[0]), q0[1:]
    w1, xyz1 = float(q1[0]), q1[1:]
    delta_w = float(np.clip(w1 * w0 + np.dot(xyz1, xyz0), -1.0, 1.0))
    delta_xyz = -w1 * xyz0 + w0 * xyz1 - np.cross(xyz1, xyz0)
    delta_xyz_norm = float(np.linalg.norm(delta_xyz))
    if delta_xyz_norm <= 1.0e-12:
        angular_velocity = np.zeros(3, dtype=np.float64)
    else:
        angle = 2.0 * math.atan2(delta_xyz_norm, delta_w)
        if angle > math.pi:
            angle -= 2.0 * math.pi
        angular_velocity = (
            delta_xyz / delta_xyz_norm * (angle / float(dt_s))
        )
    linear_velocity = (
        current_position - previous_position
    ) / float(dt_s)
    return {
        "state_source": "POSE_FINITE_DIFFERENCE",
        "linear_velocity_world_m_s": linear_velocity.tolist(),
        "angular_velocity_world_rad_s": angular_velocity.tolist(),
        "vertical_velocity_m_s": float(linear_velocity[2]),
        "angular_speed_rad_s": float(np.linalg.norm(angular_velocity)),
    }


def transform_local_points_to_world_bounds(
    *,
    local_points: np.ndarray,
    position_world: np.ndarray,
    orientation_world_wxyz: np.ndarray,
) -> dict[str, Any]:
    """Transform frozen rigid-local collider points by a PhysX dynamic pose."""
    points = np.asarray(local_points, dtype=np.float64)
    position = np.asarray(position_world, dtype=np.float64)
    quaternion = np.asarray(orientation_world_wxyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("local collider points must be a finite Nx3 array")
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("world position must be a finite 3-vector")
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("world orientation must be a finite wxyz quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise ValueError("world orientation quaternion must be nonzero")
    w, x, y, z = quaternion / norm
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    world_points = points @ rotation.T + position
    return {
        "minimum": np.min(world_points, axis=0).tolist(),
        "maximum": np.max(world_points, axis=0).tolist(),
        "point_count": int(points.shape[0]),
        "source": "PHYSX_POSE_TRANSFORMED_FROZEN_LOCAL_COLLIDER_POINTS",
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
        scale = (
            math.sqrt(1.0 + matrix[index, index] - matrix[next_index, next_index] - matrix[last_index, last_index])
            * 2.0
        )
        xyz = np.zeros(3, dtype=np.float64)
        xyz[index] = 0.25 * scale
        xyz[next_index] = (matrix[next_index, index] + matrix[index, next_index]) / scale
        xyz[last_index] = (matrix[last_index, index] + matrix[index, last_index]) / scale
        w = (matrix[last_index, next_index] - matrix[next_index, last_index]) / scale
        quaternion = np.asarray([w, *xyz], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def _look_at_quaternion(
    camera_position: np.ndarray,
    target_position: np.ndarray,
    *,
    up_world: np.ndarray | None = None,
) -> np.ndarray:
    from tools.aloha1_mapping.isaac_screenshot import look_at_orientation_wxyz

    return np.asarray(
        look_at_orientation_wxyz(
            camera_position,
            target_position,
            up_world=up_world,
        ),
        dtype=np.float64,
    )


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
        bindingStrength=(UsdShade.Tokens.strongerThanDescendants if strong else UsdShade.Tokens.weakerThanDescendants),
        materialPurpose="physics",
    )


def _create_session_bottle(
    stage: Any,
    profile: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], np.ndarray]:
    from pxr import Gf
    from pxr import PhysxSchema
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    config = profile["config"]
    bottle_path = str(config["bottle"]["session_path"])
    product_prim = str(config["frozen_inputs"]["project_bottle_usd"]["reference_prim"])
    session_root = str(Path(bottle_path).parent).replace("\\", "/")
    stage.DefinePrim(session_root, "Scope")
    bottle = UsdGeom.Xform.Define(stage, bottle_path)
    if (
        not bottle.GetPrim()
        .GetReferences()
        .AddReference(
            str(profile["inputs"]["project_bottle_usd"]),
            product_prim,
        )
    ):
        raise RuntimeError("failed to reference /Bottle500 product")

    placement = np.asarray(
        profile["kinematics"]["placement"]["placement_matrix"],
        dtype=np.float64,
    )
    quaternion = _rotation_matrix_to_quaternion_wxyz(placement[:3, :3])
    bottle.AddTranslateOp().Set(Gf.Vec3d(*placement[:3, 3]))
    bottle.AddOrientOp().Set(
        Gf.Quatf(
            float(quaternion[0]),
            Gf.Vec3f(*[float(value) for value in quaternion[1:]]),
        )
    )
    bottle_prim = bottle.GetPrim()
    collision_prims = [prim for prim in Usd.PrimRange(bottle_prim) if prim.HasAPI(UsdPhysics.CollisionAPI)]
    expected_count = int(config["frozen_inputs"]["project_bottle_usd"]["collision_prim_count"])
    if len(collision_prims) != expected_count:
        raise RuntimeError(f"Bottle500 collision count {len(collision_prims)} != {expected_count}")
    collision_points_local, collision_point_manifest = _collect_rigid_local_collision_points(
        stage,
        bottle_path,
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
    PhysxSchema.PhysxContactReportAPI.Apply(bottle_prim).CreateThresholdAttr().Set(0.0)

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
        link = stage.GetPrimAtPath(f"/World/follower_left/vx300s_left/follower_left_{side}_finger_link")
        if not link.IsValid():
            raise RuntimeError(f"missing {side} finger rigid body")
        PhysxSchema.PhysxContactReportAPI.Apply(link).CreateThresholdAttr().Set(0.0)
    return bottle_prim, {
        "source_path": str(profile["inputs"]["project_bottle_usd"]),
        "source_sha256": profile["hashes"]["project_bottle_usd"],
        "session_path": bottle_path,
        "placement_matrix": placement.tolist(),
        "mass_kg_readback": float(mass.GetMassAttr().Get()),
        "kinematic_initial": bool(rigid.GetKinematicEnabledAttr().Get()),
        "collision_prim_count": len(collision_prims),
        "collision_prim_paths": [str(prim.GetPath()) for prim in collision_prims],
        "collision_local_geometry": {
            "source": "FROZEN_USD_COLLIDER_POINTS_IN_BOTTLE_RIGID_LOCAL_FRAME",
            "point_count": int(collision_points_local.shape[0]),
            "minimum": np.min(collision_points_local, axis=0).tolist(),
            "maximum": np.max(collision_points_local, axis=0).tolist(),
            "colliders": collision_point_manifest,
        },
        "material_status": "TEMPORARY_UNCALIBRATED",
        "friction": float(config["physics"]["friction"]),
        "restitution": float(config["physics"]["restitution"]),
    }, collision_points_local


def _save_rgb_array(pixels: Any, path: Path) -> tuple[int, int]:
    rgba = np.asarray(pixels)
    if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise RuntimeError(f"Replicator RGB invalid shape for {path.name}: {rgba.shape}")
    if rgba.dtype != np.uint8:
        rgba = np.clip(rgba, 0.0, 1.0)
        rgba = np.rint(rgba * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    Image.fromarray(rgba).save(temporary)
    temporary.replace(path)
    return int(rgba.shape[1]), int(rgba.shape[0])


def _capture_viewport_png(
    app: Any,
    viewport: Any,
    *,
    camera_path: str,
    destination: Path,
) -> tuple[int, int]:
    from omni.kit.viewport.utility import capture_viewport_to_file
    from pxr import Sdf

    destination.parent.mkdir(parents=True, exist_ok=True)
    viewport.camera_path = Sdf.Path(camera_path)
    for _ in range(20):
        app.update()
    helper = capture_viewport_to_file(
        viewport,
        file_path=str(destination),
    )
    previous_size = -1
    stable_updates = 0
    for _ in range(300):
        app.update()
        if not destination.exists():
            continue
        size = destination.stat().st_size
        if size > 0 and size == previous_size:
            stable_updates += 1
        else:
            stable_updates = 0
        previous_size = size
        if stable_updates >= 2:
            break
    del helper
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"viewport capture failed: {destination}")
    with Image.open(destination) as image:
        image.load()
        return int(image.width), int(image.height)


def _camera_world_matrix(position: np.ndarray, quaternion: np.ndarray) -> list[list[float]]:
    from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quats_to_rot_matrices(quaternion)
    matrix[:3, 3] = position
    return matrix.tolist()


def _required_full_arm_contract(
    *,
    bottle_path: str,
    table_path: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    required_links = tuple(FULL_ARM_LINK_PRIMS)
    required_prims = tuple(
        dict.fromkeys(
            [prim_path for prim_paths in FULL_ARM_LINK_PRIMS.values() for prim_path in prim_paths]
            + [bottle_path, table_path]
        )
    )
    return required_prims, required_links


def _full_arm_framing_evidence(
    *,
    stage: Any,
    camera: Any,
    camera_world_matrix: Sequence[Sequence[float]],
    resolution: tuple[int, int],
    required_link_prims: Mapping[str, Sequence[str]],
    required_scene_prims: Sequence[str],
) -> dict[str, Any]:
    width, height = (int(value) for value in resolution)
    projection_by_prim: dict[str, dict[str, Any]] = {}
    projected_in_frame_prims: list[str] = []
    candidate_prims = list(
        dict.fromkeys(
            [prim_path for prim_paths in required_link_prims.values() for prim_path in prim_paths]
            + list(required_scene_prims)
        )
    )
    try:
        matrix = np.asarray(camera_world_matrix, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("camera world matrix must be finite 4x4")
        world_to_camera = np.linalg.inv(matrix)
    except (ValueError, np.linalg.LinAlgError):
        world_to_camera = None
    try:
        clipping_range = np.asarray(
            camera.get_clipping_range(),
            dtype=np.float64,
        )
        if (
            clipping_range.shape != (2,)
            or not np.isfinite(clipping_range).all()
            or clipping_range[0] < 0.0
            or clipping_range[1] <= clipping_range[0]
        ):
            raise ValueError("invalid camera clipping range")
        near_distance, far_distance = clipping_range
    except (AttributeError, RuntimeError, TypeError, ValueError):
        near_distance = None
        far_distance = None

    for prim_path in candidate_prims:
        try:
            prim = stage.GetPrimAtPath(prim_path)
            prim_valid = bool(prim.IsValid())
        except Exception:
            prim_valid = False
        if not prim_valid:
            projection_by_prim[prim_path] = {
                "status": "MISSING_STAGE_PRIM",
                "sample_count": 0,
                "front_facing_sample_count": 0,
                "in_frame_sample_count": 0,
            }
            continue
        if world_to_camera is None:
            projection_by_prim[prim_path] = {
                "status": "INVALID_CAMERA_TRANSFORM",
                "sample_count": 0,
                "front_facing_sample_count": 0,
                "in_frame_sample_count": 0,
            }
            continue
        if near_distance is None or far_distance is None:
            projection_by_prim[prim_path] = {
                "status": "INVALID_CAMERA_CLIPPING_RANGE",
                "sample_count": 0,
                "front_facing_sample_count": 0,
                "in_frame_sample_count": 0,
            }
            continue
        try:
            bounds = _world_bounds(stage, prim_path)
            minimum = np.asarray(bounds["minimum"], dtype=np.float64)
            maximum = np.asarray(bounds["maximum"], dtype=np.float64)
            if (
                minimum.shape != (3,)
                or maximum.shape != (3,)
                or not np.isfinite(minimum).all()
                or not np.isfinite(maximum).all()
                or np.any(maximum < minimum)
            ):
                raise ValueError("invalid world bounds")
            grid = np.meshgrid(
                *[np.linspace(low, high, num=3) for low, high in zip(minimum, maximum, strict=True)],
                indexing="ij",
            )
            world_points = np.column_stack([coordinates.reshape(-1) for coordinates in grid])
        except (KeyError, RuntimeError, TypeError, ValueError):
            projection_by_prim[prim_path] = {
                "status": "WORLD_BOUNDS_UNAVAILABLE",
                "sample_count": 0,
                "front_facing_sample_count": 0,
                "in_frame_sample_count": 0,
            }
            continue

        homogeneous = np.column_stack([world_points, np.ones(len(world_points), dtype=np.float64)])
        camera_points = (world_to_camera @ homogeneous.T).T[:, :3]
        front_mask = camera_points[:, 2] < -1.0e-9
        front_count = int(np.count_nonzero(front_mask))
        if not front_count:
            projection_by_prim[prim_path] = {
                "status": "BEHIND_CAMERA",
                "sample_count": len(world_points),
                "front_facing_sample_count": 0,
                "in_frame_sample_count": 0,
            }
            continue
        depths = -camera_points[:, 2]
        clipping_mask = front_mask & (depths >= near_distance) & (depths <= far_distance)
        projection_points = world_points[clipping_mask]
        if not len(projection_points):
            projection_by_prim[prim_path] = {
                "status": "OUTSIDE_CLIPPING_RANGE",
                "sample_count": len(world_points),
                "front_facing_sample_count": front_count,
                "within_clipping_range_sample_count": 0,
                "in_frame_sample_count": 0,
                "clipping_range_m": clipping_range.tolist(),
            }
            continue
        try:
            pixels = np.asarray(
                camera.get_image_coords_from_world_points(projection_points),
                dtype=np.float64,
            )
            if pixels.shape != (len(projection_points), 2):
                raise ValueError("unexpected camera projection shape")
        except (RuntimeError, TypeError, ValueError):
            projection_by_prim[prim_path] = {
                "status": "PROJECTION_UNAVAILABLE",
                "sample_count": len(world_points),
                "front_facing_sample_count": front_count,
                "within_clipping_range_sample_count": len(projection_points),
                "in_frame_sample_count": 0,
            }
            continue
        finite_mask = np.isfinite(pixels).all(axis=1)
        in_frame_mask = (
            finite_mask
            & (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] < height)
        )
        in_frame_count = int(np.count_nonzero(in_frame_mask))
        finite_pixels = pixels[finite_mask]
        projection_by_prim[prim_path] = {
            "status": ("PROJECTED_IN_FRAME" if in_frame_count else "OUTSIDE_IMAGE"),
            "sample_count": len(world_points),
            "front_facing_sample_count": front_count,
            "within_clipping_range_sample_count": len(projection_points),
            "in_frame_sample_count": in_frame_count,
            "clipping_range_m": clipping_range.tolist(),
            "projected_pixel_min_xy": (np.min(finite_pixels, axis=0).tolist() if len(finite_pixels) else None),
            "projected_pixel_max_xy": (np.max(finite_pixels, axis=0).tolist() if len(finite_pixels) else None),
        }
        if in_frame_count:
            projected_in_frame_prims.append(prim_path)

    projected_prim_set = set(projected_in_frame_prims)
    projected_in_frame_links = [
        link_name
        for link_name, prim_paths in required_link_prims.items()
        if prim_paths and all(path in projected_prim_set for path in prim_paths)
    ]
    return {
        "method": ("WORLD_AABB_27_POINT_USD_CAMERA_CLIPPED_PROJECTION_IN_FRAME"),
        "numeric_evidence_scope": FULL_ARM_NUMERIC_EVIDENCE_SCOPE,
        "occlusion_evaluation_status": (FULL_ARM_OCCLUSION_EVALUATION_STATUS),
        "projected_in_frame_prims": projected_in_frame_prims,
        "projected_in_frame_links": projected_in_frame_links,
        "projection_by_prim": projection_by_prim,
    }


def _finalize_frame_manifest(
    *,
    frame_records: Sequence[dict[str, Any]],
    capture_views: Sequence[str],
    runtime_trial_signature: str,
    required_full_arm_prims: Sequence[str],
    required_full_arm_links: Sequence[str],
) -> dict[str, Any]:
    video_manifest = tuple(capture_views) == VIDEO_VIEWS
    for record in frame_records:
        physics_frame = int(record["physics_frame"])
        time_s = float(record["time_s"])
        missing_views = [view for view in capture_views if view not in record["views"]]
        if missing_views:
            raise ValueError(f"physics frame {physics_frame} missing views {missing_views}")
        if video_manifest:
            framing = record["views"]["overview"].get("framing_evidence")
            if not isinstance(framing, Mapping):
                raise ValueError(f"physics frame {physics_frame} missing overview framing_evidence")
            if (
                framing.get("numeric_evidence_scope") != FULL_ARM_NUMERIC_EVIDENCE_SCOPE
                or framing.get("occlusion_evaluation_status") != FULL_ARM_OCCLUSION_EVALUATION_STATUS
            ):
                raise ValueError(f"physics frame {physics_frame} invalid overview framing evidence scope")
            projected_prims = set(framing.get("projected_in_frame_prims", ()))
            projected_links = set(framing.get("projected_in_frame_links", ()))
            missing_prims = sorted(set(required_full_arm_prims) - projected_prims)
            missing_links = sorted(set(required_full_arm_links) - projected_links)
            if missing_prims or missing_links:
                raise ValueError(
                    f"physics frame {physics_frame} full-arm framing missing "
                    f"prims {missing_prims}; links {missing_links}"
                )
        for view in capture_views:
            record["views"][view].update(
                {
                    "physics_frame": physics_frame,
                    "time_s": time_s,
                    "runtime_trial_signature": runtime_trial_signature,
                }
            )
    return {
        "schema_version": 1,
        "runtime_trial_signature": runtime_trial_signature,
        "views": list(capture_views),
        "required_full_arm_prims": list(required_full_arm_prims),
        "required_full_arm_links": list(required_full_arm_links),
        "records": list(frame_records),
    }


def _create_cameras(
    *,
    config: Mapping[str, Any],
    kinematics: Mapping[str, Any],
    capture_profile: str,
    capture_views: Sequence[str],
    resolution: tuple[int, int],
) -> dict[str, dict[str, Any]]:
    from isaacsim.sensors.camera import Camera

    grasp = np.asarray(
        kinematics["placement"]["bottle_axis"]["grasp_point_world_m"],
        dtype=np.float64,
    )
    base = np.asarray(
        kinematics["fk_correspondence"]["cases"][0]["base_position_world_m"],
        dtype=np.float64,
    )
    overview_target = (base + grasp) / 2.0
    overview_target[2] += 0.08
    if capture_profile == "video":
        camera_specs = {
            "overview": {
                "position": overview_target + np.asarray([1.85, -1.75, 1.45]),
                "target": overview_target,
            },
            "gripper_closeup": {
                # Move along the already verified overview viewing ray.
                "position": grasp + np.asarray([0.925, -0.875, 0.725]),
                "target": grasp,
                "reuse_initial_orientation": True,
            },
        }
    elif capture_profile == "screenshots":
        camera_specs = {
            "true_top": {
                "position": grasp + np.asarray([0.0, 0.0, 1.65]),
                "target": grasp,
                "up_world": np.asarray([0.0, 1.0, 0.0]),
            },
            "side": {
                # Reuse the already visually accepted close-up ray.  This is
                # recorded as an oblique side view by its actual camera
                # forward vector; it is not a calibrated orthographic side.
                "position": grasp + np.asarray([0.925, -0.875, 0.725]),
                "target": grasp,
                "orientation_position": grasp + np.asarray([1.85, -1.75, 1.45]),
                "orientation_target": grasp,
            },
        }
    else:
        raise ValueError(f"unknown capture profile: {capture_profile}")
    initial_spec = camera_specs[capture_views[0]]
    initial_quaternion = _look_at_quaternion(
        initial_spec["position"],
        initial_spec["target"],
        up_world=initial_spec.get("up_world"),
    )
    capture_camera = Camera(
        prim_path="/World/Task7B2HorizontalCameras/capture_camera",
        position=initial_spec["position"],
        orientation=initial_quaternion,
        frequency=float(config["physics"]["frequency_hz"]),
        resolution=resolution,
    )
    capture_camera.initialize(attach_rgb_annotator=False)
    capture_camera.set_world_pose(
        position=initial_spec["position"],
        orientation=initial_quaternion,
        camera_axes="usd",
    )
    records: dict[str, dict[str, Any]] = {}
    for view in capture_views:
        spec = camera_specs[view]
        quaternion = (
            initial_quaternion
            if spec.get("reuse_initial_orientation", False)
            else _look_at_quaternion(
                spec.get("orientation_position", spec["position"]),
                spec.get("orientation_target", spec["target"]),
                up_world=spec.get("up_world"),
            )
        )
        records[view] = {
            "camera": capture_camera,
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


def _create_render_streams(
    cameras: Mapping[str, Mapping[str, Any]],
    *,
    resolution: tuple[int, int],
) -> tuple[Any, dict[str, dict[str, Any]]]:
    import omni.replicator.core as rep

    rep.orchestrator.set_capture_on_play(False)
    streams: dict[str, dict[str, Any]] = {}
    for view in VIDEO_VIEWS:
        camera = cameras[view]["camera"]
        render_product = rep.create.render_product(
            camera.prim_path,
            resolution,
            name=f"Task7B2Horizontal_{view}",
        )
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach(render_product)
        streams[view] = {
            "render_product": render_product,
            "render_product_path": str(render_product.path),
            "annotator": annotator,
        }
    rep.orchestrator.step(
        rt_subframes=2,
        pause_timeline=True,
        delta_time=0.0,
        wait_for_render=True,
    )
    return rep, streams


def _verify_runtime_versions(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    import carb
    import omni.kit.app

    app = omni.kit.app.get_app()
    extension_manager = app.get_extension_manager()
    motion_id = extension_manager.get_enabled_extension_id("isaacsim.robot_motion.motion_generation")
    physx_id = extension_manager.get_enabled_extension_id("omni.physx")
    motion_extension = extension_manager.get_extension_dict(motion_id) if motion_id else None
    physx_extension = extension_manager.get_extension_dict(physx_id) if physx_id else None
    motion_version = motion_extension.get("package", {}).get("version", None) if motion_extension else None
    physx_version = physx_extension.get("package", {}).get("version", None) if physx_extension else None
    kit_version = str(carb.tokens.get_tokens_interface().resolve("${kit_version}")).split("+", maxsplit=1)[0]
    isaac_version = version("isaacsim")
    actual = {
        "isaac_sim": isaac_version,
        "kit": kit_version,
        "physx": str(physx_version).split("+", maxsplit=1)[0],
        "motion_generation_extension": str(motion_version).split("+", maxsplit=1)[0],
        "python": platform.python_version(),
        "carbonite": str(carb.__file__),
        "use_fabric_scene_delegate": bool(carb.settings.get_settings().get_as_bool("/app/useFabricSceneDelegate")),
    }
    for key in (
        "isaac_sim",
        "kit",
        "physx",
        "motion_generation_extension",
    ):
        if actual[key] != str(config["runtime"][key]):
            raise RuntimeError(f"runtime mismatch {key}: {actual[key]} != {config['runtime'][key]}")
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
        profile["kinematics"]["episode_fk"]["lift_onset_requested_qpos_arm_6d"],
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
            position_tolerance=float(profile["config"]["motion"]["ik_position_tolerance_m"]),
            orientation_tolerance=float(profile["config"]["motion"]["ik_orientation_tolerance_rad"]),
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


def canonicalize_horizontal_cylindrical_grasp(
    *,
    world_from_object: np.ndarray,
    object_from_gripper_base: np.ndarray,
    object_axis_local: np.ndarray,
    table_normal_world: np.ndarray,
) -> dict[str, Any]:
    """Select the top-down equivalent of a cylindrical object-relative grasp.

    A horizontal cylindrical bottle may settle at an arbitrary roll about its
    longitudinal axis. Blindly composing that roll into ``T_W_G`` tilts the
    fingers relative to the table. This selection preserves the base Grasp
    Editor grasp's axial coordinate and radial distance while selecting the
    symmetry-equivalent roll with a vertical approach and horizontal fingers.
    """
    world_from_object = np.asarray(world_from_object, dtype=np.float64)
    object_from_gripper_base = np.asarray(
        object_from_gripper_base,
        dtype=np.float64,
    )
    object_axis_local = np.asarray(object_axis_local, dtype=np.float64)
    object_axis_local /= np.linalg.norm(object_axis_local)
    table_normal_world = np.asarray(
        table_normal_world,
        dtype=np.float64,
    )
    table_normal_world /= np.linalg.norm(table_normal_world)

    axis_world = world_from_object[:3, :3] @ object_axis_local
    axis_world /= np.linalg.norm(axis_world)
    axis_table_dot = float(np.dot(axis_world, table_normal_world))
    axis_horizontal = axis_world - axis_table_dot * table_normal_world
    axis_horizontal /= np.linalg.norm(axis_horizontal)

    translation_base = object_from_gripper_base[:3, 3]
    axial_coordinate = float(np.dot(translation_base, object_axis_local))
    radial_local = translation_base - axial_coordinate * object_axis_local
    radial_distance = float(np.linalg.norm(radial_local))

    approach_world = -table_normal_world
    gripper_bottle_axis_world = -axis_horizontal
    finger_line_world = np.cross(
        gripper_bottle_axis_world,
        approach_world,
    )
    finger_line_world /= np.linalg.norm(finger_line_world)
    world_from_gripper = np.eye(4, dtype=np.float64)
    world_from_gripper[:3, :3] = np.column_stack(
        (
            approach_world,
            finger_line_world,
            gripper_bottle_axis_world,
        )
    )
    world_from_gripper[:3, 3] = (
        world_from_object[:3, 3] + axial_coordinate * axis_horizontal + radial_distance * table_normal_world
    )
    object_from_gripper_selected = np.linalg.inv(world_from_object) @ world_from_gripper
    return {
        "world_from_gripper": world_from_gripper,
        "object_from_gripper_selected": (object_from_gripper_selected),
        "axis_world": axis_world,
        "axis_horizontal_world": axis_horizontal,
        "approach_world": approach_world,
        "finger_line_world": finger_line_world,
        "axial_coordinate_m": axial_coordinate,
        "radial_distance_m": radial_distance,
        "axis_to_table_normal_deg": math.degrees(
            math.acos(
                float(
                    np.clip(
                        abs(axis_table_dot),
                        -1.0,
                        1.0,
                    )
                )
            )
        ),
        "rotation_determinant": float(np.linalg.det(world_from_gripper[:3, :3])),
    }


def _solve_settled_bottle_runtime_ik(
    profile: Mapping[str, Any],
    *,
    base_position: np.ndarray,
    base_orientation: np.ndarray,
    bottle_state: Mapping[str, Any],
    current_ee_position: np.ndarray,
    current_ee_orientation: np.ndarray,
    current_arm_q: np.ndarray,
) -> dict[str, Any]:
    from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices
    from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver

    native_raw_document = yaml.safe_load(
        profile["inputs"]["grasp_editor_v2_native_raw_yaml"].read_text(encoding="utf-8")
    )
    native_raw_grasp = next(iter(native_raw_document["grasps"].values()))
    native_raw_orientation = native_raw_grasp["orientation"]
    object_from_gripper_native_raw = np.eye(4, dtype=np.float64)
    object_from_gripper_native_raw[:3, :3] = quats_to_rot_matrices(
        np.asarray(
            [
                native_raw_orientation["w"],
                *native_raw_orientation["xyz"],
            ],
            dtype=np.float64,
        )
    )
    object_from_gripper_native_raw[:3, 3] = np.asarray(
        native_raw_grasp["position"],
        dtype=np.float64,
    )
    candidate_document = yaml.safe_load(profile["inputs"]["supplier_cad_grasp_candidate"].read_text(encoding="utf-8"))
    candidate_grasp = next(iter(candidate_document["grasps"].values()))
    candidate_orientation = candidate_grasp["orientation"]
    object_from_gripper_base = np.eye(4, dtype=np.float64)
    object_from_gripper_base[:3, :3] = quats_to_rot_matrices(
        np.asarray(
            [
                candidate_orientation["w"],
                *candidate_orientation["xyz"],
            ],
            dtype=np.float64,
        )
    )
    object_from_gripper_base[:3, 3] = np.asarray(
        candidate_grasp["position"],
        dtype=np.float64,
    )
    world_from_object = np.eye(4, dtype=np.float64)
    world_from_object[:3, :3] = quats_to_rot_matrices(
        np.asarray(
            bottle_state["orientation_wxyz"],
            dtype=np.float64,
        )
    )
    world_from_object[:3, 3] = np.asarray(
        bottle_state["position_world_m"],
        dtype=np.float64,
    )
    object_axis_local = np.asarray(
        profile["config"]["bottle"]["axis"]["b_local_m"],
        dtype=np.float64,
    ) - np.asarray(
        profile["config"]["bottle"]["axis"]["a_local_m"],
        dtype=np.float64,
    )
    canonical = canonicalize_horizontal_cylindrical_grasp(
        world_from_object=world_from_object,
        object_from_gripper_base=object_from_gripper_base,
        object_axis_local=object_axis_local,
        table_normal_world=np.asarray(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        ),
    )
    world_from_gripper = canonical["world_from_gripper"]
    object_from_gripper_selected = canonical["object_from_gripper_selected"]
    target_orientation = _rotation_matrix_to_quaternion_wxyz(world_from_gripper[:3, :3])
    original_targets = profile["kinematics"]["placement"]["target_poses"]
    original_grasp = np.asarray(
        original_targets["grasp_ee_position_world_m"],
        dtype=np.float64,
    )
    pregrasp_clearance = float(
        np.asarray(
            original_targets["pregrasp_ee_position_world_m"],
            dtype=np.float64,
        )[2]
        - original_grasp[2]
    )
    lift_distance = float(
        np.asarray(
            original_targets["lift_ee_position_world_m"],
            dtype=np.float64,
        )[2]
        - original_grasp[2]
    )
    grasp_position = world_from_gripper[:3, 3].copy()
    pregrasp_position = grasp_position + np.asarray(
        [0.0, 0.0, pregrasp_clearance],
        dtype=np.float64,
    )
    lift_position = grasp_position + np.asarray(
        [0.0, 0.0, lift_distance],
        dtype=np.float64,
    )

    solver = LulaKinematicsSolver(
        robot_description_path=str(profile["inputs"]["lula_descriptor"]),
        urdf_path=str(profile["inputs"]["follower_left_urdf"]),
    )
    solver.set_robot_base_pose(base_position, base_orientation)
    ik_contract = profile["kinematics"]["ik"]
    lower_limits = np.asarray(
        ik_contract["joint_lower_limits_rad"],
        dtype=np.float64,
    )
    upper_limits = np.asarray(
        ik_contract["joint_upper_limits_rad"],
        dtype=np.float64,
    )
    velocity_limits = np.asarray(
        ik_contract["joint_velocity_limits_rad_s"],
        dtype=np.float64,
    )
    phases = (
        (
            "move_to_pregrasp",
            current_ee_position,
            pregrasp_position,
            current_ee_orientation,
        ),
        (
            "vertical_descent",
            pregrasp_position,
            grasp_position,
            target_orientation,
        ),
        (
            "vertical_lift",
            grasp_position,
            lift_position,
            target_orientation,
        ),
    )
    previous_q = np.asarray(current_arm_q, dtype=np.float64)
    waypoints: list[dict[str, Any]] = []
    phase_summaries: dict[str, Any] = {}
    for phase, start, end, start_orientation in phases:
        phase_waypoints, phase_summary = solve_adaptive_linear_ik(
            solver=solver,
            frame_name=profile["config"]["robot"]["end_effector_frame"],
            start_position=np.asarray(start, dtype=np.float64),
            end_position=np.asarray(end, dtype=np.float64),
            start_orientation_wxyz=np.asarray(
                start_orientation,
                dtype=np.float64,
            ),
            orientation_wxyz=target_orientation,
            start_q=previous_q,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            velocity_limits=velocity_limits,
            physics_dt=float(ik_contract["physics_dt_s"]),
            phase=phase,
            position_tolerance=float(ik_contract["position_tolerance_m"]),
            orientation_tolerance=float(ik_contract["orientation_tolerance_rad"]),
        )
        phase_summaries[phase] = phase_summary
        if phase_summary["status"] != "PASS":
            return {
                "status": "FAIL",
                "failure_phase": phase,
                "phase_summaries": phase_summaries,
                "waypoints": waypoints,
            }
        waypoints.extend(phase_waypoints)
        previous_q = np.asarray(
            phase_waypoints[-1]["joint_positions_rad"],
            dtype=np.float64,
        )
    closure = world_from_object @ object_from_gripper_selected
    return {
        "status": "PASS",
        "source": ("DYNAMIC_SETTLED_CYLINDRICAL_ROLL_EQUIVALENT_SELECTED_FROM_SUPPLIER_CAD_GRASP_CANDIDATE_T_O_G"),
        "classification": ("DIAGNOSTIC_CYLINDRICAL_SYMMETRY_VARIANT_PENDING_CORRECTED_NATIVE_GRASP_EDITOR_EXPORT"),
        "formula": (
            "T_W_G_SELECTED = canonical_top_down("
            "T_W_O_SETTLED, T_O_G_SUPPLIER_CAD_BASE); "
            "T_O_G_SELECTED = inv(T_W_O_SETTLED) @ T_W_G_SELECTED"
        ),
        "world_from_object": world_from_object.tolist(),
        "object_from_gripper_base": (object_from_gripper_base.tolist()),
        "object_from_gripper_native_raw_rejected": (object_from_gripper_native_raw.tolist()),
        "native_raw_vs_supplier_candidate": {
            "translation_residual_m": float(
                np.linalg.norm(object_from_gripper_native_raw[:3, 3] - object_from_gripper_base[:3, 3])
            ),
            "rotation_residual_rad": float(
                math.acos(
                    float(
                        np.clip(
                            (
                                np.trace(object_from_gripper_native_raw[:3, :3].T @ object_from_gripper_base[:3, :3])
                                - 1.0
                            )
                            / 2.0,
                            -1.0,
                            1.0,
                        )
                    )
                )
            ),
            "classification": ("NATIVE_RAW_POSE_REJECTED_FOR_DYNAMIC_TASK_PENDING_CORRECTED_GRASP_EDITOR_EXPORT"),
        },
        "object_from_gripper_selected": (object_from_gripper_selected.tolist()),
        "world_from_gripper": world_from_gripper.tolist(),
        "cylindrical_symmetry_selection": {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in canonical.items()
            if key
            not in {
                "world_from_gripper",
                "object_from_gripper_selected",
            }
        },
        "closure_max_abs": float(np.max(np.abs(closure - world_from_gripper))),
        "target_orientation_world_wxyz": target_orientation.tolist(),
        "pregrasp_position_world_m": pregrasp_position.tolist(),
        "grasp_position_world_m": grasp_position.tolist(),
        "lift_position_world_m": lift_position.tolist(),
        "pregrasp_clearance_m": pregrasp_clearance,
        "lift_distance_m": lift_distance,
        "phase_summaries": phase_summaries,
        "waypoints": waypoints,
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
    capture_collider_evidence: bool,
    capture_profile: str,
    capture_views: Sequence[str],
    resolution: tuple[int, int],
) -> dict[str, Any]:
    import carb
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.core.utils.xforms import get_world_pose
    from omni.kit.viewport.utility import get_active_viewport
    from omni.physx import get_physx_interface
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysxSchema
    from pxr import Usd
    from pxr import UsdPhysics

    started = time.perf_counter()
    config = profile["config"]
    dt = 1.0 / float(config["physics"]["frequency_hz"])
    stage_path = profile["inputs"]["task7a_stage"]
    stage_hash = profile["hashes"]["task7a_stage"]
    bottle_path = str(config["bottle"]["session_path"])
    table_path = str(config["frozen_inputs"]["task7a_stage"]["support_path"])
    required_full_arm_prims, required_full_arm_links = _required_full_arm_contract(
        bottle_path=bottle_path,
        table_path=table_path,
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
        coupling_readback = author_coupling_variant(
            stage=stage,
            variant="official_symmetric_adapter",
            physx_schema=PhysxSchema,
            usd_physics=UsdPhysics,
        )
        if coupling_readback["classification"] != DIAGNOSTIC_COUPLING_CLASSIFICATION:
            raise RuntimeError("unexpected diagnostic coupling classification")
        finger_drive_type_readback = _author_session_finger_drive_type(
            stage=stage,
            usd_physics=UsdPhysics,
            requested_type=str(profile["diagnostic_finger_drive_type"]),
        )
        bottle_prim, bottle_session, bottle_collision_points_local = _create_session_bottle(stage, profile)

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
        raise RuntimeError(f"unexpected DOF order: {list(articulation.dof_names)}")
    articulation_controller = articulation.get_articulation_controller()
    finger_indices = np.asarray([7, 8], dtype=np.int32)
    max_efforts_before = np.asarray(
        articulation_controller.get_max_efforts(),
        dtype=np.float64,
    )
    copied_max_force = float(coupling_readback["copied_left_drive_parameters"]["max_force"])
    articulation_controller.set_max_efforts(
        np.asarray(
            [copied_max_force, copied_max_force],
            dtype=np.float32,
        ),
        finger_indices,
    )
    max_efforts_after = np.asarray(
        articulation_controller.get_max_efforts(),
        dtype=np.float64,
    )
    if not np.allclose(
        max_efforts_after[finger_indices],
        copied_max_force,
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError("diagnostic finger max-effort readback mismatch")

    waypoints = profile["kinematics"]["ik"]["waypoints"]
    pregrasp_waypoints = [item for item in waypoints if item["phase"] == "move_to_pregrasp"]
    if not pregrasp_waypoints:
        raise RuntimeError("verified pregrasp waypoint is missing")
    initial_arm = np.asarray(
        pregrasp_waypoints[-1]["joint_positions_rad"],
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

    base_position, base_orientation = get_world_pose("/World/follower_left/vx300s_left/follower_left_base_link")
    runtime_ik = _verify_ik_runtime(
        profile,
        base_position=np.asarray(base_position, dtype=np.float64),
        base_orientation=np.asarray(base_orientation, dtype=np.float64),
    )
    simulation_view = SimulationManager.get_physics_sim_view()
    if simulation_view is None or not simulation_view.is_valid:
        raise RuntimeError("Isaac Sim physics tensor SimulationView is unavailable")
    bottle = simulation_view.create_rigid_body_view(bottle_path)
    if bottle is None or int(bottle.count) != 1:
        raise RuntimeError(f"expected one PhysX Bottle500 rigid body at {bottle_path}")
    table_bounds = _world_bounds(stage, table_path)
    table_top = float(table_bounds["maximum"][2])

    cameras = (
        _create_cameras(
            config=config,
            kinematics=profile["kinematics"],
            capture_profile=capture_profile,
            capture_views=capture_views,
            resolution=resolution,
        )
        if capture_video_frames
        else {}
    )
    if capture_video_frames:
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("no active viewport for two-view recording")
    else:
        viewport = None
    if capture_collider_evidence and not capture_video_frames:
        raise RuntimeError(
            "--capture-collider-evidence requires --capture-video-frames"
        )
    settings = carb.settings.get_settings()
    collider_setting = (
        "/persistent/physics/visualizationDisplayColliders"
    )
    collider_setting_before = int(settings.get(collider_setting) or 0)
    collider_overlay_captures: list[dict[str, Any]] = []

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
    clearance_report = json.loads(profile["inputs"]["supplier_cad_clearance_report"].read_text(encoding="utf-8"))
    finger_line_axis_reference = np.asarray(
        clearance_report["grasp_frame"]["finger_line_axis_reference"],
        dtype=np.float64,
    )
    left_index = EXPECTED_DOF_ORDER.index("left_finger")
    right_index = EXPECTED_DOF_ORDER.index("right_finger")

    def capture_step(phase: str, *, target: np.ndarray) -> None:
        nonlocal command
        command = np.asarray(target, dtype=np.float64)
        _command_positions(articulation, command)
        state["phase"] = phase
        state["frame"] = int(state["frame"]) + 1
        world.play()
        world.step(render=False)
        physx.update_transformations(True, True, False, False)
        bottle_state = read_physx_bottle_state(bottle)
        pose_derived_velocity = (
            derive_pose_finite_difference_velocity(
                previous=telemetry[-1]["bottle"],
                current=bottle_state,
                dt_s=dt,
            )
            if telemetry
            else None
        )
        visual_bounds = _world_bounds(stage, bottle_path)
        pose = np.eye(4, dtype=np.float64)
        position = np.asarray(
            bottle_state["position_world_m"],
            dtype=np.float64,
        )
        orientation = np.asarray(
            bottle_state["orientation_wxyz"],
            dtype=np.float64,
        )
        collision_bounds = transform_local_points_to_world_bounds(
            local_points=bottle_collision_points_local,
            position_world=position,
            orientation_world_wxyz=orientation,
        )
        from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices

        pose[:3, :3] = quats_to_rot_matrices(orientation)
        pose[:3, 3] = position
        axis_a = pose[:3, :3] @ axis_a_local + position
        axis_b = pose[:3, :3] @ axis_b_local + position
        ee_position, ee_orientation = get_world_pose("/World/follower_left/vx300s_left/follower_left_ee_gripper_link")
        ee_rotation = quats_to_rot_matrices(np.asarray(ee_orientation, dtype=np.float64))
        effective_finger_line = ee_rotation @ finger_line_axis_reference
        left_center, _ = get_world_pose(
            config["robot"]["left_finger_collider"],
        )
        right_center, _ = get_world_pose(
            config["robot"]["right_finger_collider"],
        )
        left_center = np.asarray(left_center, dtype=np.float64)
        right_center = np.asarray(right_center, dtype=np.float64)
        projection_world_points = {
            "bottle_a": axis_a.tolist(),
            "bottle_b": axis_b.tolist(),
            "left_finger_collider_origin": left_center.tolist(),
            "right_finger_collider_origin": right_center.tolist(),
        }
        qpos = np.asarray(
            articulation.get_joint_positions(),
            dtype=np.float64,
        )
        qvel = np.asarray(
            articulation.get_joint_velocities(),
            dtype=np.float64,
        )
        applied_action = articulation.get_applied_action()
        applied_positions = (
            np.asarray(
                applied_action.joint_positions,
                dtype=np.float64,
            ).tolist()
            if applied_action.joint_positions is not None
            else None
        )
        right_drive = UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath("/World/follower_left/vx300s_left/joints/right_finger"),
            "linear",
        )
        telemetry.append(
            {
                "frame": int(state["frame"]),
                "time_s": float(int(state["frame"]) * dt),
                "phase": phase,
                "joint_target": command.tolist(),
                "joint_readback": qpos.tolist(),
                "joint_velocity": qvel.tolist(),
                "controller_applied_joint_positions": applied_positions,
                "right_drive_target_position_authored": (
                    float(right_drive.GetTargetPositionAttr().Get())
                    if right_drive and right_drive.GetTargetPositionAttr().Get() is not None
                    else None
                ),
                "gripper": {
                    "ee_position_world_m": [float(value) for value in ee_position],
                    "ee_orientation_world_wxyz": [float(value) for value in ee_orientation],
                    "effective_finger_line_axis_world": (effective_finger_line.tolist()),
                    "effective_finger_line_source": ("SUPPLIER_CAD_EFFECTIVE_PAD_CONTACT_FRAME"),
                },
                "bottle": {
                    **bottle_state,
                    "pose_finite_difference_velocity": (
                        pose_derived_velocity
                    ),
                    "a_world_m": axis_a.tolist(),
                    "b_world_m": axis_b.tolist(),
                    "axis_world": ((axis_b - axis_a) / np.linalg.norm(axis_b - axis_a)).tolist(),
                    "bottom_clearance_m": float(collision_bounds["minimum"][2] - table_top),
                    "collision_bounds": collision_bounds,
                    "visual_bounds": visual_bounds,
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
            world.pause()
            for view in capture_views:
                capture_camera = cameras[view]["camera"]
                capture_camera.set_world_pose(
                    position=np.asarray(
                        cameras[view]["position_world_m"],
                        dtype=np.float64,
                    ),
                    orientation=np.asarray(
                        cameras[view]["orientation_wxyz"],
                        dtype=np.float64,
                    ),
                    camera_axes="usd",
                )
                output = trial_root / "frames" / view / f"{int(state['frame']):06d}.png"
                width, height = _capture_viewport_png(
                    app,
                    viewport,
                    camera_path=capture_camera.prim_path,
                    destination=output,
                )
                view_record = {
                    "absolute_path": str(output),
                    "sha256": _sha256(output),
                    "resolution": [width, height],
                    "projection_world_points": projection_world_points,
                    "projection_pixels_xy": {
                        label: pixel.tolist()
                        for label, pixel in zip(
                            projection_world_points,
                            np.asarray(
                                capture_camera.get_image_coords_from_world_points(
                                    np.asarray(
                                        list(projection_world_points.values()),
                                        dtype=np.float64,
                                    )
                                ),
                                dtype=np.float64,
                            ),
                            strict=True,
                        )
                        if np.isfinite(pixel).all()
                    },
                    "finger_center_method": ("COLLIDER_PRIM_WORLD_XFORM_ORIGIN_NOT_EFFECTIVE_CONTACT_REGION"),
                }
                if capture_profile == "video" and view == "overview":
                    view_record["framing_evidence"] = _full_arm_framing_evidence(
                        stage=stage,
                        camera=capture_camera,
                        camera_world_matrix=cameras[view]["camera_world_matrix"],
                        resolution=(width, height),
                        required_link_prims=FULL_ARM_LINK_PRIMS,
                        required_scene_prims=(bottle_path, table_path),
                    )
                frame_record["views"][view] = view_record
        frame_manifest.append(frame_record)

    def capture_collider_overlay_pair(phase: str) -> None:
        if not capture_collider_evidence:
            return
        if viewport is None or not frame_manifest:
            raise RuntimeError(
                "collider overlay capture requires an existing viewport frame"
            )
        world.pause()
        physics_frame = int(state["frame"])
        try:
            settings.set_int(collider_setting, 2)
            for view in capture_views:
                capture_camera = cameras[view]["camera"]
                capture_camera.set_world_pose(
                    position=np.asarray(
                        cameras[view]["position_world_m"],
                        dtype=np.float64,
                    ),
                    orientation=np.asarray(
                        cameras[view]["orientation_wxyz"],
                        dtype=np.float64,
                    ),
                    camera_axes="usd",
                )
                normal_path = (
                    trial_root
                    / "frames"
                    / view
                    / f"{physics_frame:06d}.png"
                ).resolve()
                overlay_path = (
                    trial_root
                    / "collider_overlay"
                    / phase
                    / f"{view}_physics_collider_overlay_raw.png"
                ).resolve()
                width, height = _capture_viewport_png(
                    app,
                    viewport,
                    camera_path=capture_camera.prim_path,
                    destination=overlay_path,
                )
                collider_overlay_captures.append(
                    {
                        "phase": phase,
                        "view": view,
                        "mode": "physics_collider_overlay",
                        "physics_frame": physics_frame,
                        "time_s": physics_frame * dt,
                        "normal_absolute_path": str(normal_path),
                        "normal_sha256": _sha256(normal_path),
                        "physics_collider_overlay_absolute_path": str(
                            overlay_path
                        ),
                        "physics_collider_overlay_sha256": _sha256(
                            overlay_path
                        ),
                        "resolution": [width, height],
                        "display_colliders_readback": int(
                            settings.get(collider_setting) or 0
                        ),
                        "camera_world_matrix": cameras[view][
                            "camera_world_matrix"
                        ],
                    }
                )
        finally:
            settings.set_int(
                collider_setting,
                collider_setting_before,
            )

    capture_step("setup_kinematic", target=command)
    rigid = UsdPhysics.RigidBodyAPI(bottle_prim)
    rigid.GetKinematicEnabledAttr().Set(False)
    physx_sim.flush_changes()
    capture_step("release_dynamic", target=command)
    capture_collider_overlay_pair("release_dynamic")
    dynamic_readback = bool(rigid.GetKinematicEnabledAttr().Get())
    if dynamic_readback:
        raise RuntimeError("Bottle500 failed to become dynamic")

    settle_steps = int(config["physics"]["frequency_hz"] * 2)
    for _ in range(settle_steps):
        capture_step("support_settle", target=command)

    settled_bottle_state = read_physx_bottle_state(bottle)
    settled_ee_position, settled_ee_orientation = get_world_pose(
        "/World/follower_left/vx300s_left/follower_left_ee_gripper_link"
    )
    settled_runtime_ik = _solve_settled_bottle_runtime_ik(
        profile,
        base_position=np.asarray(base_position, dtype=np.float64),
        base_orientation=np.asarray(base_orientation, dtype=np.float64),
        bottle_state=settled_bottle_state,
        current_ee_position=np.asarray(
            settled_ee_position,
            dtype=np.float64,
        ),
        current_ee_orientation=np.asarray(
            settled_ee_orientation,
            dtype=np.float64,
        ),
        current_arm_q=np.asarray(
            articulation.get_joint_positions(),
            dtype=np.float64,
        )[:6],
    )
    if settled_runtime_ik["status"] != "PASS":
        raise RuntimeError(f"settled-bottle runtime IK failed: {settled_runtime_ik.get('failure_phase')}")
    waypoints = settled_runtime_ik["waypoints"]

    episode_records = profile["kinematics"]["episode_fk"]["records"]
    episode_arm_commands = np.asarray(
        [record["action_arm_6d"] for record in episode_records],
        dtype=np.float64,
    )
    episode_arm_delta_limits = np.max(
        np.abs(np.diff(episode_arm_commands, axis=0)),
        axis=0,
    )

    def execute_arm_waypoint(
        phase: str,
        waypoint: Mapping[str, Any],
        *,
        minimum_steps: int = 1,
    ) -> None:
        start = command.copy()
        goal = command.copy()
        goal[:6] = waypoint["joint_positions_rad"]
        steps = derive_interpolation_steps(
            start[:6],
            goal[:6],
            episode_arm_delta_limits,
        )
        steps = max(steps, int(minimum_steps))
        for step in range(1, steps + 1):
            target = start.copy()
            target[:6] = start[:6] + (step / steps) * (goal[:6] - start[:6])
            capture_step(phase, target=target)

    for waypoint in [item for item in waypoints if item["phase"] == "move_to_pregrasp"]:
        execute_arm_waypoint("open_pregrasp", waypoint)
    capture_collider_overlay_pair("open_pregrasp")

    for waypoint in [item for item in waypoints if item["phase"] == "vertical_descent"]:
        execute_arm_waypoint("vertical_descent", waypoint)

    capture_step("vertical_descent", target=command)
    cad_contact_target_m = float(clearance_report["contact_solution"]["left_finger_q_m"])
    preload_delta_m = float(profile["diagnostic_preload_delta_m"])
    commanded_close_target_m = cad_contact_target_m - preload_delta_m
    left_close_targets = build_external_close_targets(
        open_position_m=float(config["robot"]["open_targets_m"][0]),
        contact_target_m=commanded_close_target_m,
        speed_m_s=0.02,
        physics_dt_s=dt,
    )
    for left_target in left_close_targets:
        target = command.copy()
        target[left_index] = left_target
        target[right_index] = -left_target
        capture_step("closing_preload", target=target)

    bottle_token = bottle_path
    left_token = "diagnostic_supplier_cad_left_finger"
    right_token = "diagnostic_supplier_cad_right_finger"
    preload_settle_limit = int(config["physics"]["frequency_hz"] * 2)
    preload_stable_required = 5
    preload_stable_count = 0
    preload_settle_steps = 0
    for _preload_settle_steps in range(1, preload_settle_limit + 1):
        capture_step("closing_preload", target=command)
        readback = np.asarray(
            articulation.get_joint_positions(),
            dtype=np.float64,
        )
        residual = abs(float(readback[left_index] + readback[right_index]))
        current_left = _physical_contacts(
            contacts,
            tokens=(bottle_token, left_token),
        )
        current_right = _physical_contacts(
            contacts,
            tokens=(bottle_token, right_token),
        )
        if current_left and current_right and residual <= 0.001:
            preload_stable_count += 1
        else:
            preload_stable_count = 0
        if preload_stable_count >= preload_stable_required:
            break
    preload_settle_steps = _preload_settle_steps
    capture_step("bilateral_contact", target=command)
    capture_collider_overlay_pair("bilateral_contact")

    lift_waypoints = [item for item in waypoints if item["phase"] == "vertical_lift"]
    episode_lift_onset_frame = int(profile["kinematics"]["lift_detection"]["lift_onset_frame"])
    episode_lift_end_frame = max(int(record["frame"]) for record in episode_records)
    episode_lift_transition_count = episode_lift_end_frame - episode_lift_onset_frame
    minimum_steps_per_lift_waypoint = max(
        1,
        math.ceil(episode_lift_transition_count / max(len(lift_waypoints), 1)),
    )
    for waypoint in lift_waypoints:
        execute_arm_waypoint(
            "vertical_lift",
            waypoint,
            minimum_steps=minimum_steps_per_lift_waypoint,
        )

    capture_step("support_clear", target=command)
    capture_collider_overlay_pair("support_clear")
    for _ in range(int(config["physics"]["hold_steps"])):
        capture_step("hold_end", target=command)
    capture_collider_overlay_pair("hold_end")

    manifest_path = trial_root / "frame_manifest.json"
    video_metadata: dict[str, Any] = {
        "capture_enabled": capture_video_frames,
        "capture_method": (
            (f"LOCAL_OMNIHYDRA_ACTIVE_VIEWPORT_PAUSED_TWO_VIEW_{capture_profile.upper()}")
            if capture_video_frames
            else "DISABLED"
        ),
        "frame_manifest": str(manifest_path),
        "runtime_trial_signature": "PENDING_TRACE_FINALIZATION",
        "first_physics_frame": (int(frame_manifest[0]["physics_frame"]) if frame_manifest else None),
        "last_physics_frame": (int(frame_manifest[-1]["physics_frame"]) if frame_manifest else None),
        "missing_physics_frames": [],
        "phase_frame_ranges": _phase_ranges(frame_manifest),
        "render_fps": int(config["physics"]["frequency_hz"]),
        "views": {
            view: {
                **{key: value for key, value in cameras[view].items() if key != "camera"},
            }
            for view in cameras
        },
        "collider_overlay_captures": collider_overlay_captures,
        "collider_display_setting": {
            "path": collider_setting,
            "before": collider_setting_before,
            "overlay": 2,
            "restored": int(settings.get(collider_setting) or 0),
        },
    }

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
    phase_end = {phase: max(frames) for phase, frames in phase_frames.items() if frames}
    lift_start_frame = min(phase_frames.get("vertical_lift", [10**9]))
    hold_frames = set(phase_frames.get("hold_end", []))
    prelift_left = [contact for contact in left_contacts if int(contact["frame"]) < lift_start_frame]
    prelift_right = [contact for contact in right_contacts if int(contact["frame"]) < lift_start_frame]
    hold_left = [contact for contact in left_contacts if int(contact["frame"]) in hold_frames]
    hold_right = [contact for contact in right_contacts if int(contact["frame"]) in hold_frames]
    settle_samples = [item for item in telemetry if item["phase"] == "support_settle"][-30:]
    support_settle_pass = bool(
        support_contacts
        and settle_samples
        and max(abs(float(item["bottle"]["vertical_velocity_m_s"])) for item in settle_samples) < 0.02
        and max(float(item["bottle"]["angular_speed_rad_s"]) for item in settle_samples) < 0.2
    )
    support_frames = {int(contact["frame"]) for contact in support_contacts}
    clear_records = [
        item
        for item in telemetry
        if item["phase"] in {"vertical_lift", "support_clear", "hold_end"}
        and float(item["bottle"]["bottom_clearance_m"]) > 0.0
        and int(item["frame"]) not in support_frames
    ]
    left_support = bool(clear_records)
    lift_records = [item for item in telemetry if item["phase"] == "vertical_lift"]
    hold_records = [item for item in telemetry if item["phase"] == "hold_end"]
    lift_end_z = float(lift_records[-1]["bottle"]["position_world_m"][2]) if lift_records else float("nan")
    hold_min_z = min(
        (float(item["bottle"]["position_world_m"][2]) for item in hold_records),
        default=float("nan"),
    )
    hold_drop = lift_end_z - hold_min_z if math.isfinite(lift_end_z) and math.isfinite(hold_min_z) else float("nan")
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
    bottle_contacts = [contact for contact in contacts if bottle_token in _pair_text(contact)]
    allowed = (table_token, left_token, right_token)
    forbidden_contacts = [
        contact
        for contact in bottle_contacts
        if not any(token in _pair_text(contact) for token in allowed) and float(contact["separation_m"]) <= 0.0
    ]
    deep_frames = {int(contact["frame"]) for contact in bottle_contacts if float(contact["separation_m"]) < -0.005}
    persistent_penetration = any(frame + 1 in deep_frames and frame + 2 in deep_frames for frame in deep_frames)
    maximum_speed = max(
        (abs(float(item["bottle"]["vertical_velocity_m_s"])) for item in telemetry),
        default=0.0,
    )
    maximum_angular = max(
        (float(item["bottle"]["angular_speed_rad_s"]) for item in telemetry),
        default=0.0,
    )
    normal_force_decay = bool(prelift_left and prelift_right and (not hold_left or not hold_right))
    continuous_slip = bool(
        hold_left and hold_right and math.isfinite(hold_drop) and hold_drop > float(config["physics"]["drop_gate_m"])
    )
    contact_lost = bool(prelift_left and prelift_right and (not hold_left or not hold_right))
    free_fall = bool(
        contact_lost and any(float(item["bottle"]["vertical_velocity_m_s"]) < -0.2 for item in hold_records)
    )
    rotation_escape = bool(contact_lost and maximum_angular > 3.0)
    numerical_ejection = bool(maximum_speed > 5.0 or maximum_angular > 50.0)

    settled_axis = np.asarray(
        settle_samples[-1]["bottle"]["axis_world"],
        dtype=np.float64,
    )
    axis_vertical_angle = math.degrees(
        math.acos(
            float(
                np.clip(
                    abs(
                        np.dot(
                            settled_axis,
                            np.asarray([0.0, 0.0, 1.0]),
                        )
                    ),
                    -1.0,
                    1.0,
                )
            )
        )
    )
    contact_reference = max(
        (item for item in telemetry if int(item["frame"]) < lift_start_frame),
        key=lambda item: int(item["frame"]),
    )
    contact_axis = np.asarray(
        contact_reference["bottle"]["axis_world"],
        dtype=np.float64,
    )
    contact_a = np.asarray(
        contact_reference["bottle"]["a_world_m"],
        dtype=np.float64,
    )

    def weighted_contact_center(
        samples: Sequence[Mapping[str, Any]],
    ) -> np.ndarray | None:
        if not samples:
            return None
        positions = np.asarray(
            [sample["position_world_m"] for sample in samples],
            dtype=np.float64,
        )
        weights = np.asarray(
            [max(float(sample["impulse_ns"]), 0.0) for sample in samples],
            dtype=np.float64,
        )
        if float(np.sum(weights)) <= 0.0:
            return np.mean(positions, axis=0)
        return np.average(positions, axis=0, weights=weights)

    left_center = weighted_contact_center(prelift_left)
    right_center = weighted_contact_center(prelift_right)
    body_interval = [float(value) for value in config["bottle"]["body_interval_m"]]
    contact_coordinates = (
        [
            float(np.dot(left_center - contact_a, contact_axis)),
            float(np.dot(right_center - contact_a, contact_axis)),
        ]
        if left_center is not None and right_center is not None
        else []
    )
    contact_points_in_body_interval = bool(
        contact_coordinates and all(body_interval[0] <= value <= body_interval[1] for value in contact_coordinates)
    )
    contact_point_pair_angle = None
    if left_center is not None and right_center is not None:
        gripper_line = right_center - left_center
        gripper_line[2] = 0.0
        axis_xy = contact_axis.copy()
        axis_xy[2] = 0.0
        if np.linalg.norm(gripper_line) > 0.0 and np.linalg.norm(axis_xy) > 0.0:
            cosine = abs(
                float(np.dot(gripper_line, axis_xy) / (np.linalg.norm(gripper_line) * np.linalg.norm(axis_xy)))
            )
            contact_point_pair_angle = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
    effective_line = np.asarray(
        contact_reference["gripper"]["effective_finger_line_axis_world"],
        dtype=np.float64,
    )
    effective_line[2] = 0.0
    axis_xy = contact_axis.copy()
    axis_xy[2] = 0.0
    if np.linalg.norm(effective_line) <= 0.0 or np.linalg.norm(axis_xy) <= 0.0:
        gripper_contact_angle = None
        gripper_axis_perpendicular_pass = False
    else:
        effective_cosine = abs(
            float(np.dot(effective_line, axis_xy) / (np.linalg.norm(effective_line) * np.linalg.norm(axis_xy)))
        )
        gripper_contact_angle = math.degrees(math.acos(float(np.clip(effective_cosine, -1.0, 1.0))))
        gripper_axis_perpendicular_pass = abs(
            gripper_contact_angle - float(config["geometry_gates"]["gripper_line_to_axis_target_deg"])
        ) <= float(config["geometry_gates"]["gripper_line_to_axis_tolerance_deg"])
    final_close_frame = max(phase_frames.get("closing_preload", [-1]))
    coupled_hold_records = [item for item in telemetry if int(item["frame"]) > final_close_frame]
    coupled_hold_residuals = [
        abs(float(item["joint_readback"][left_index]) + float(item["joint_readback"][right_index]))
        for item in coupled_hold_records
    ]
    maximum_coupling_residual_m = max(
        coupled_hold_residuals,
        default=math.inf,
    )
    final_coupling_residual_m = coupled_hold_residuals[-1] if coupled_hold_residuals else math.inf
    coupling_tolerance_m = 0.001
    prelift_reference = max(
        (item for item in telemetry if item["phase"] == "bilateral_contact"),
        key=lambda item: int(item["frame"]),
    )
    prelift_coupling_residual_m = abs(
        float(prelift_reference["joint_readback"][left_index]) + float(prelift_reference["joint_readback"][right_index])
    )
    coupling_accuracy_pass = bool(
        math.isfinite(prelift_coupling_residual_m)
        and prelift_coupling_residual_m <= coupling_tolerance_m
        and math.isfinite(final_coupling_residual_m)
        and final_coupling_residual_m <= coupling_tolerance_m
    )
    trial_data = {
        "trial_index": trial_index,
        "fresh_world_reset": True,
        "bottle_dynamic_during_settle": not dynamic_readback,
        "support_contact_before_grasp": support_settle_pass,
        "axis_horizontal_pass": abs(axis_vertical_angle - 90.0)
        <= float(config["geometry_gates"]["axis_to_table_normal_tolerance_deg"]),
        "gripper_axis_perpendicular_pass": (gripper_axis_perpendicular_pass),
        "coupling_accuracy_pass": coupling_accuracy_pass,
        "vertical_descent_pass": all(
            item["status"] == "PASS" for item in waypoints if item["phase"] == "vertical_descent"
        ),
        "ik_reachable": (runtime_ik["status"] == "PASS" and settled_runtime_ik["status"] == "PASS"),
        "left_physical_contact_before_lift": bool(prelift_left),
        "right_physical_contact_before_lift": bool(prelift_right),
        "contact_points_in_body_interval": contact_points_in_body_interval,
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
        "phase_frame_counts": {phase: len(frames) for phase, frames in phase_frames.items()},
        "joint_trajectories": [item["joint_readback"] for item in telemetry],
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
        "contact_geometry": {
            "left_impulse_weighted_center_world_m": (left_center.tolist() if left_center is not None else None),
            "right_impulse_weighted_center_world_m": (right_center.tolist() if right_center is not None else None),
            "bottle_body_coordinates_m": contact_coordinates,
            "body_interval_m": body_interval,
            "gripper_line_to_axis_deg": gripper_contact_angle,
            "gripper_line_definition": ("SUPPLIER_CAD_EFFECTIVE_PAD_REGION_CENTER_LINE"),
            "contact_point_pair_line_to_axis_deg": (contact_point_pair_angle),
            "settled_axis_to_table_normal_deg": axis_vertical_angle,
        },
        "coupling_tracking": {
            "status": "PASS" if coupling_accuracy_pass else "FAIL",
            "tolerance_m": coupling_tolerance_m,
            "maximum_post_close_transient_residual_m": (maximum_coupling_residual_m),
            "prelift_residual_m": prelift_coupling_residual_m,
            "final_residual_m": final_coupling_residual_m,
            "sample_count": len(coupled_hold_residuals),
            "preload_settle_steps": preload_settle_steps,
            "preload_settle_limit": preload_settle_limit,
            "stable_consecutive_frames": preload_stable_count,
            "stable_required_frames": preload_stable_required,
            "close_trajectory": {
                "source": ("STAGE3_PASSING_GRASP_EDITOR_SUPPLIER_CAD_CONTACT_TRAJECTORY"),
                "contact_target_m": cad_contact_target_m,
                "preload_delta_m": preload_delta_m,
                "commanded_close_target_m": (commanded_close_target_m),
                "step_count": len(left_close_targets),
                "speed_m_s": 0.02,
                "speed_status": ("DIAGNOSTIC_ONLY_NOT_HARDWARE_CALIBRATION"),
                "episode18_gripper_action_used_as_physical_meters": False,
            },
        },
    }
    trial_data["parent_" + "attachment_used"] = False
    evaluation = evaluate_horizontal_trial(trial_data)
    signature = canonical_horizontal_signature(trial_data)
    video_metadata["runtime_trial_signature"] = signature
    _atomic_json(
        manifest_path,
        _finalize_frame_manifest(
            frame_records=frame_manifest,
            capture_views=(capture_views if capture_video_frames else ()),
            runtime_trial_signature=signature,
            required_full_arm_prims=required_full_arm_prims,
            required_full_arm_links=required_full_arm_links,
        ),
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
            "solve_articulation_contact_last": bool(physics_context.get_solve_articulation_contact_last()),
            "kinematic_enabled_after_release": dynamic_readback,
            "contact_subscription_active": subscription is not None,
            "phase_order": list(PHASE_ORDER),
            "phase_frames": {phase: [min(frames), max(frames)] for phase, frames in phase_frames.items() if frames},
            "phase_end_frames": phase_end,
            "frozen_ik_reverification": runtime_ik,
            "settled_bottle_runtime_ik": settled_runtime_ik,
            "lift_time_resolution": {
                "source": "EPISODE18_LIFT_ONSET_TO_FRAME_244",
                "lift_onset_frame": episode_lift_onset_frame,
                "lift_end_frame": episode_lift_end_frame,
                "transition_count": episode_lift_transition_count,
                "waypoint_count": len(lift_waypoints),
                "minimum_steps_per_waypoint": (minimum_steps_per_lift_waypoint),
                "physics_frequency_hz": float(config["physics"]["frequency_hz"]),
            },
            "diagnostic_coupling": {
                **coupling_readback,
                "source_stage_modified": False,
                "tracking": trial_data["coupling_tracking"],
                "controller_max_efforts_before": (max_efforts_before[finger_indices].tolist()),
                "controller_max_efforts_after": (max_efforts_after[finger_indices].tolist()),
                "controller_max_effort_source": ("COPIED_UNCHANGED_LEFT_DRIVE_MAX_FORCE"),
            },
            "diagnostic_finger_drive_type": finger_drive_type_readback,
            "initial_arm_policy": ("ARM_INITIALIZED_AT_VERIFIED_PREGRASP_BEFORE_BOTTLE_DYNAMIC_RELEASE"),
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
            "first_clear_frame": (int(clear_records[0]["frame"]) if clear_records else None),
        },
        "contacts": {
            "all": contacts,
            "left_physical": left_contacts,
            "right_physical": right_contacts,
            "support_physical": support_contacts,
            "forbidden_physical": forbidden_contacts,
            "maximum_penetration_m": min(
                (float(contact["separation_m"]) for contact in bottle_contacts),
                default=None,
            ),
        },
        "telemetry": telemetry,
        "video_capture": video_metadata,
        "boundaries": {
            "source_assets_modified": False,
            "default_configuration_modified": False,
            "final_collider_modified": False,
            "finger_drive_type_session_only": True,
            "finger_drive_type_classification": finger_drive_type_readback["classification"],
            "task8": "NOT_RUN",
        },
    }
    if trial["stage"]["sha256_after"] != stage_hash:
        raise RuntimeError("frozen Stage hash changed during runtime")
    return trial


if __name__ == "__main__":
    raise SystemExit(main())
