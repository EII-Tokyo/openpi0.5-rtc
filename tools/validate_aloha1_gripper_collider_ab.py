#!/usr/bin/env python3
"""Fresh-reset Isaac Sim 5.1 A/B trials for ALOHA follower finger colliders."""

# Reuse the already validated baseline contact/material/aperture helpers so the
# A/B changes only the requested experimental variables.
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.gripper_collider_ab import assert_profile_pair_is_frozen
from tools.aloha1_mapping.gripper_collider_ab import canonical_signature
from tools.aloha1_mapping.gripper_collider_ab import classify_decomposition_status
from tools.aloha1_mapping.gripper_collider_ab import classify_root_cause
from tools.aloha1_mapping.gripper_collider_ab import load_collision_profiles
from tools.aloha1_mapping.gripper_collider_ab import sha256_file
from tools.aloha1_mapping.gripper_collider_ab import summarize_ab_trials
from tools.aloha1_mapping.gripper_collider_ab import trial_passes_hold_gate
from tools.aloha1_mapping.gripper_validation import build_gripper_validation_plan
from tools.aloha1_mapping.gripper_validation import summarize_contact_events
from tools.aloha1_mapping.isaac_screenshot import look_at_orientation_wxyz
from tools.aloha1_mapping.isaac_screenshot import save_camera_rgba_png
from tools.aloha1_mapping.screenshot_manifest import validate_screenshot
import tools.validate_aloha1_gripper as baseline


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=_json_default,
                    allow_nan=False,
                )
                + "\n"
            )
    temporary.replace(path)


def _verify_baseline(root: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    results = []
    for item in manifest["protected_baseline"]:
        path = root / item["path"]
        actual = sha256_file(path) if path.is_file() else None
        results.append(
            {
                "path": item["path"],
                "expected_sha256": item["sha256"],
                "actual_sha256": actual,
                "match": actual == item["sha256"],
            }
        )
    failures = [item["path"] for item in results if not item["match"]]
    if failures:
        raise RuntimeError(f"protected baseline changed: {failures}")
    return results


def _diagnostic_asset(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    profile_name: str,
    robot: str,
) -> Path:
    path = (
        root
        / manifest["diagnostic_directories"][profile_name]
        / robot
        / f"{robot}_{profile_name}.usd"
    )
    return path.resolve(strict=True)


def _collider_token_readback(
    stage: Any,
    *,
    robot: str,
) -> dict[str, str]:
    from pxr import Usd
    from pxr import UsdPhysics

    tokens = {}
    root = stage.GetPrimAtPath("/World/Robot")
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not (
            "_left_finger_link/" in path or "_right_finger_link/" in path
        ):
            continue
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            continue
        tokens[path] = UsdPhysics.MeshCollisionAPI(
            prim
        ).GetApproximationAttr().Get()
    if len(tokens) != 2:
        raise RuntimeError(f"expected two finger approximation tokens for {robot}: {tokens}")
    return dict(sorted(tokens.items()))


def _drive_and_mimic_snapshot(stage: Any, robot: str) -> dict[str, Any]:
    result = {}
    for joint_name in ("left_finger", "right_finger"):
        prim_path = f"/World/Robot/{robot}_{joint_name}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            # Importer joint prims use their source joint names below root_joint.
            candidates = [
                candidate
                for candidate in stage.Traverse()
                if str(candidate.GetPath()).endswith(f"/{robot}_{joint_name}")
                or str(candidate.GetPath()).endswith(f"/{joint_name}")
            ]
            if len(candidates) != 1:
                raise RuntimeError(f"finger joint prim not uniquely found: {joint_name}")
            prim = candidates[0]
        attributes = {}
        for attribute in prim.GetAttributes():
            name = attribute.GetName()
            if "drive:" in name or "mimic" in name or "limit" in name:
                value = attribute.Get()
                attributes[name] = value
        result[joint_name] = {
            "path": str(prim.GetPath()),
            "attributes": attributes,
            "applied_schemas": list(prim.GetAppliedSchemas()),
        }
    return result


def _apply_contact_reports(stage: Any, robot: str) -> list[str]:
    from pxr import PhysxSchema

    paths = []
    for side in ("left", "right"):
        path = f"/World/Robot/{robot}_{side}_finger_link"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"finger rigid body missing: {path}")
        report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        report_api.CreateThresholdAttr().Set(0.0)
        paths.append(path)
    return paths


def _command_fingers(
    articulation: Any,
    *,
    left_index: int,
    right_index: int,
    left_target: float,
    control_mode: str,
) -> None:
    from isaacsim.core.utils.types import ArticulationAction

    if control_mode == "current_mimic":
        baseline._command_left_finger(
            articulation,
            left_index=left_index,
            target=left_target,
        )
        return
    if control_mode != "explicit_symmetric":
        raise ValueError(f"unsupported control mode: {control_mode}")
    right_target = -left_target
    articulation.get_articulation_controller().apply_action(
        ArticulationAction(
            joint_positions=np.asarray(
                [left_target, right_target],
                dtype=np.float32,
            ),
            joint_indices=np.asarray(
                [left_index, right_index],
                dtype=np.int32,
            ),
        )
    )


def _bottle_state(bottle: Any) -> dict[str, Any]:
    position, orientation = bottle.get_world_pose()
    linear = bottle.get_linear_velocity()
    angular = bottle.get_angular_velocity()
    position = np.asarray(position, dtype=np.float64)
    linear = np.asarray(linear, dtype=np.float64)
    angular = np.asarray(angular, dtype=np.float64)
    return {
        "position_world_m": position,
        "orientation_wxyz": np.asarray(orientation, dtype=np.float64),
        "z_m": float(position[2]),
        "linear_velocity_world_m_s": linear,
        "vertical_velocity_m_s": float(linear[2]),
        "angular_velocity_world_rad_s": angular,
        "angular_speed_rad_s": float(np.linalg.norm(angular)),
    }


def _step(
    world: Any,
    *,
    steps: int,
    phase: str,
    frame_state: dict[str, int],
    articulation: Any,
    bottle: Any,
    left_index: int,
    right_index: int,
    telemetry: list[dict[str, Any]],
) -> None:
    for phase_step in range(steps):
        frame_state["frame"] += 1
        world.step(render=False)
        telemetry.append(
            {
                "frame": frame_state["frame"],
                "phase": phase,
                "phase_step": phase_step,
                "finger": baseline._finger_state(
                    articulation,
                    left_index,
                    right_index,
                ),
                "bottle": _bottle_state(bottle),
            }
        )


def _side_contact_summary(
    events: Sequence[Mapping[str, Any]],
    *,
    side: str,
    placement_frame: int,
    dt: float,
) -> dict[str, Any]:
    finger_token = f"_{side}_finger_link/"
    relevant = [
        event
        for event in events
        if int(event["frame"]) >= placement_frame
        and "/BottleProxy" in (str(event["collider0"]) + str(event["collider1"]))
        and finger_token in (str(event["collider0"]) + str(event["collider1"]))
    ]
    with_contacts = [event for event in relevant if event.get("contacts")]
    if not with_contacts:
        return {
            "contact": False,
            "first_contact_frame": None,
            "first_contact_time_s": None,
            "first_contact_after_placement_s": None,
            "contact_duration_s": 0.0,
            "contact_loss_frame": None,
            "contact_loss_time_s": None,
            "contact_samples": 0,
        }
    first = min(with_contacts, key=lambda event: int(event["frame"]))
    first_contact = first["contacts"][0]
    contact_frames = sorted({int(event["frame"]) for event in with_contacts})
    lost = [
        event
        for event in relevant
        if "LOST" in str(event["type"]).upper()
        and int(event["frame"]) >= int(first["frame"])
    ]
    loss_frame = min((int(event["frame"]) for event in lost), default=None)
    samples = [
        contact
        for event in with_contacts
        for contact in event.get("contacts", [])
    ]
    normal_impulses = []
    separations = []
    for contact in samples:
        normal = np.asarray(contact["normal"], dtype=np.float64)
        impulse = np.asarray(contact["impulse"], dtype=np.float64)
        normal_impulses.append(float(abs(np.dot(impulse, normal))))
        separations.append(float(contact["separation"]))
    first_normal = np.asarray(first_contact["normal"], dtype=np.float64)
    first_impulse = np.asarray(first_contact["impulse"], dtype=np.float64)
    first_normal_impulse = float(abs(np.dot(first_impulse, first_normal)))
    return {
        "contact": True,
        "first_contact_frame": int(first["frame"]),
        "first_contact_time_s": float(int(first["frame"]) * dt),
        "first_contact_after_placement_s": float(
            (int(first["frame"]) - placement_frame) * dt
        ),
        "first_contact_paths": {
            "collider0": first["collider0"],
            "collider1": first["collider1"],
        },
        "first_contact": {
            "position_world_m": first_contact["position"],
            "normal": first_contact["normal"],
            "impulse": first_contact["impulse"],
            "normal_impulse_n_s": first_normal_impulse,
            "estimated_normal_force_n": first_normal_impulse / dt,
            "separation_m": float(first_contact["separation"]),
            "material0": first_contact["material0"],
            "material1": first_contact["material1"],
        },
        "contact_duration_s": float(len(contact_frames) * dt),
        "contact_frame_count": len(contact_frames),
        "contact_loss_frame": loss_frame,
        "contact_loss_time_s": (
            float(loss_frame * dt) if loss_frame is not None else None
        ),
        "contact_samples": len(samples),
        "normal_impulse_n_s": {
            "maximum": max(normal_impulses),
            "mean": float(np.mean(normal_impulses)),
            "all_finite": bool(np.all(np.isfinite(normal_impulses))),
        },
        "estimated_normal_force_n": {
            "maximum": max(normal_impulses) / dt,
            "mean": float(np.mean(normal_impulses)) / dt,
        },
        "separation_m": {
            "minimum": min(separations),
            "maximum_penetration_depth": max(0.0, -min(separations)),
        },
    }


def _normal_quality(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    closing_axis: Sequence[float],
) -> dict[str, Any]:
    if not left.get("contact") or not right.get("contact"):
        return {
            "status": "FAIL",
            "reason": "bilateral_contact_missing",
        }
    axis = np.asarray(closing_axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    left_normal = np.asarray(left["first_contact"]["normal"], dtype=np.float64)
    right_normal = np.asarray(right["first_contact"]["normal"], dtype=np.float64)
    return {
        "status": "MEASURED_NO_CALIBRATED_THRESHOLD",
        "left_abs_alignment_with_closing_axis": float(abs(np.dot(left_normal, axis))),
        "right_abs_alignment_with_closing_axis": float(abs(np.dot(right_normal, axis))),
        "left_right_normal_dot": float(np.dot(left_normal, right_normal)),
    }


def _signature_payload(trial: Mapping[str, Any]) -> dict[str, Any]:
    # Exclude wall-clock runtime, trial index, asset path, and stage IDs.
    telemetry = [
        {
            "frame": item["frame"],
            "phase": item["phase"],
            "finger": item["finger"],
            "bottle": item["bottle"],
        }
        for item in trial["telemetry"]
    ]
    return {
        "profile": trial["profile"],
        "control_mode": trial["control_mode"],
        "robot": trial["robot"],
        "approximation_readback": trial["approximation_readback"],
        "metrics": trial["metrics"],
        "states": trial["states"],
        "aperture": trial["aperture"],
        "contacts": trial["contacts"],
        "released_hold": trial["released_hold"],
        "telemetry": telemetry,
    }


def _augment_trial_derived_fields(trial: dict[str, Any]) -> dict[str, Any]:
    residuals = {
        name: abs(
            float(state["right_finger_m"])
            + float(state["left_finger_m"])
        )
        for name, state in (
            ("start", trial["states"]["start_fingers"]),
            ("open", trial["states"]["open_fingers"]),
            (
                "closed_against_fixed_bottle",
                trial["states"]["closed_against_fixed_bottle"],
            ),
        )
    }
    commanded = [
        abs(
            float(item["finger"]["right_finger_m"])
            + float(item["finger"]["left_finger_m"])
        )
        for item in trial["telemetry"]
        if item["phase"] != "settle"
    ]
    trial["states"]["symmetric_residual_m"] = residuals
    trial["states"]["maximum_sampled_control_residual_m"] = max(
        residuals.values()
    )
    trial["states"]["maximum_post_command_dynamic_residual_m"] = max(commanded)
    trial["states"]["control_residual_scope"] = (
        "sampled control residual uses start/open/closed checkpoints; dynamic "
        "maximum also includes bottle-driven released hold"
    )
    trial["deterministic_signature"] = canonical_signature(
        json.loads(
            json.dumps(
                _signature_payload(trial),
                default=_json_default,
                allow_nan=False,
            )
        )
    )
    return trial


def _diagnostic_group_metrics(
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    drops = [float(trial["released_hold"]["drop_m"]) for trial in trials]
    contact_points = [
        int(trial["contacts"]["all_contact_summary"]["contact_point_count"])
        for trial in trials
    ]
    maximum_penetrations = [
        float(
            trial["contacts"]["all_contact_summary"][
                "maximum_penetration_depth_m"
            ]
        )
        for trial in trials
    ]
    sampled_residuals = [
        float(trial["states"]["maximum_sampled_control_residual_m"])
        for trial in trials
    ]
    dynamic_residuals = [
        float(trial["states"]["maximum_post_command_dynamic_residual_m"])
        for trial in trials
    ]
    left_forces = [
        float(trial["contacts"]["left"]["estimated_normal_force_n"]["maximum"])
        for trial in trials
    ]
    right_forces = [
        float(trial["contacts"]["right"]["estimated_normal_force_n"]["maximum"])
        for trial in trials
    ]
    return {
        "drop_m": {
            "minimum": min(drops),
            "maximum": max(drops),
            "mean": float(np.mean(drops)),
            "unique_values": sorted(set(drops)),
        },
        "bilateral_contact_trial_count": sum(
            bool(trial["metrics"]["bilateral_contact_before_release"])
            for trial in trials
        ),
        "persistent_penetration_trial_count": sum(
            bool(trial["metrics"]["persistent_penetration"])
            for trial in trials
        ),
        "unexpected_internal_collision_trial_count": sum(
            bool(trial["metrics"]["unexpected_gripper_collision"])
            for trial in trials
        ),
        "contact_point_count": {
            "minimum": min(contact_points),
            "maximum": max(contact_points),
            "mean": float(np.mean(contact_points)),
        },
        "maximum_transient_penetration_depth_m": max(maximum_penetrations),
        "maximum_sampled_control_residual_m": max(sampled_residuals),
        "maximum_post_command_dynamic_residual_m": max(dynamic_residuals),
        "maximum_estimated_normal_force_n": {
            "left": max(left_forces),
            "right": max(right_forces),
        },
    }


def _runtime_invariant_audit(
    trials_by_group: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    locked_paths = (
        ("friction",),
        ("restitution",),
        ("bottle_mass_kg",),
        ("bottle_diameter_m",),
        ("physics_frequency_hz",),
        ("physics_dt_s",),
        ("solve_articulation_contact_last_requested",),
        ("solve_articulation_contact_last_readback",),
        ("self_collision",),
        ("hold_interval_s",),
        ("drop_gate_m",),
        ("contact_rest_offsets_authored",),
    )
    unique_locked = {}
    all_trials = [
        trial
        for trials in trials_by_group.values()
        for trial in trials
    ]
    for (name,) in locked_paths:
        values = {
            json.dumps(
                trial["frozen"][name],
                sort_keys=True,
                separators=(",", ":"),
            )
            for trial in all_trials
        }
        unique_locked[name] = [
            json.loads(value) for value in sorted(values)
        ]
    initial_qpos_signatures = {}
    drive_mimic_signatures = {}
    phase_step_signatures = {}
    for trial in all_trials:
        robot = trial["robot"]
        initial_qpos_signatures.setdefault(robot, set()).add(
            canonical_signature(trial["states"]["initial_qpos"])
        )
        drive_mimic_signatures.setdefault(robot, set()).add(
            canonical_signature(trial["frozen"]["drive_and_mimic"])
        )
        phase_counts = {}
        for item in trial["telemetry"]:
            phase_counts[item["phase"]] = phase_counts.get(item["phase"], 0) + 1
        phase_step_signatures.setdefault(robot, set()).add(
            canonical_signature(phase_counts)
        )
    first_round = {
        name: trials_by_group[name]
        for name in ("hull_current", "decomposition_current")
        if name in trials_by_group
    }
    first_round_controls = {
        trial["control_mode"]
        for trials in first_round.values()
        for trial in trials
    }
    approximation_tokens = {
        name: sorted(
            {
                token
                for trial in trials
                for token in trial["approximation_readback"].values()
            }
        )
        for name, trials in trials_by_group.items()
    }
    status = (
        all(len(values) == 1 for values in unique_locked.values())
        and all(len(values) == 1 for values in initial_qpos_signatures.values())
        and all(len(values) == 1 for values in drive_mimic_signatures.values())
        and all(len(values) == 1 for values in phase_step_signatures.values())
        and first_round_controls == {"current_mimic"}
        and approximation_tokens.get("hull_current") == ["convexHull"]
        and approximation_tokens.get("decomposition_current")
        == ["convexDecomposition"]
    )
    return {
        "status": "PASS" if status else "FAIL",
        "trial_count": len(all_trials),
        "unique_locked_values": unique_locked,
        "initial_qpos_signature_count_per_robot": {
            robot: len(values)
            for robot, values in initial_qpos_signatures.items()
        },
        "drive_mimic_signature_count_per_robot": {
            robot: len(values)
            for robot, values in drive_mimic_signatures.items()
        },
        "trajectory_phase_signature_count_per_robot": {
            robot: len(values)
            for robot, values in phase_step_signatures.items()
        },
        "first_round_control_modes": sorted(first_round_controls),
        "approximation_tokens": approximation_tokens,
        "allowed_first_round_difference": "finger approximation token only",
    }


def _capture_trial_screenshot(
    *,
    world: Any,
    camera: Any,
    articulation: Any,
    bottle: Any,
    open_aperture: Mapping[str, Any],
    screenshot_context: dict[str, Any],
    phase: str,
    capture_name: str,
    view: str,
    frame: int,
    physical_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Render without a physics step and append a verified capture record."""

    current_target = np.asarray(
        _bottle_state(bottle)["position_world_m"],
        dtype=np.float64,
    )
    if "fixed_camera_target_world_m" not in screenshot_context:
        screenshot_context["fixed_camera_target_world_m"] = (
            current_target.tolist()
        )
    target = np.asarray(
        screenshot_context["fixed_camera_target_world_m"],
        dtype=np.float64,
    )
    closing = np.asarray(
        open_aperture["closing_axis_world"],
        dtype=np.float64,
    )
    closing /= np.linalg.norm(closing)
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    lateral = np.cross(up, closing)
    if np.linalg.norm(lateral) < 1.0e-6:
        lateral = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    lateral /= np.linalg.norm(lateral)
    if view == "closing_axis":
        # Elevated contact-focused view. Visual self-review rejected the old
        # near-horizontal view because the bottle and gripper bar hid both
        # inner finger-to-bottle interfaces.
        position = target - 0.27 * lateral - 0.08 * closing + 0.25 * up
    elif view == "isometric":
        position = target - 0.32 * lateral - 0.18 * closing + 0.18 * up
    else:
        raise ValueError(f"unsupported trial screenshot view: {view}")
    orientation = look_at_orientation_wxyz(position, target)
    camera.set_world_pose(
        position=position,
        orientation=orientation,
        camera_axes="usd",
    )
    # SimulationContext.render() explicitly disables playSimulations during
    # the Kit update in local Isaac Sim 5.1; screenshots therefore do not add
    # hidden physics steps to the frozen trajectory.
    pixels = None
    for _ in range(8):
        world.render()
    for _ in range(22):
        candidate = camera.get_rgba()
        if candidate is not None:
            candidate_array = np.asarray(candidate)
            if (
                candidate_array.ndim == 3
                and candidate_array.shape[2] == 4
                and candidate_array.size > 0
            ):
                pixels = candidate
                break
        world.render()
    if pixels is None:
        raise RuntimeError(
            f"Isaac camera produced no RGBA frame for {capture_name}"
        )
    output = (
        Path(screenshot_context["artifact_root"])
        / phase
        / f"{capture_name}.png"
    )
    render_readback = save_camera_rgba_png(camera, output, rgba=pixels)
    camera_position, camera_orientation = camera.get_world_pose(
        camera_axes="usd"
    )
    record = validate_screenshot(
        output.resolve(strict=True),
        artifact_root=Path(screenshot_context["artifact_root"]),
        phase=phase,
        capture_name=capture_name,
        gate_status="PASS",
        camera={
            "runtime": "isaacsim.sensors.camera.Camera",
            "view": view,
            "position_world_m": np.asarray(camera_position).tolist(),
            "orientation_wxyz": np.asarray(camera_orientation).tolist(),
            "target_world_m": target.tolist(),
            "current_bottle_target_world_m": current_target.tolist(),
            "camera_anchor_frozen_across_runtime_phases": True,
            "resolution": [1280, 900],
            "render_readback": render_readback,
        },
        simulation={
            "stage_asset": str(screenshot_context["asset"]),
            "robot": screenshot_context["robot"],
            "profile": screenshot_context["profile"],
            "trial_index": int(screenshot_context["trial_index"]),
            "frame": int(frame),
            "physics_frequency_hz": 60,
            "physics_steps_added_for_capture": 0,
            "joint_positions": np.asarray(
                articulation.get_joint_positions(),
                dtype=np.float64,
            ).tolist(),
            "bottle": _bottle_state(bottle),
            "physical_state": dict(physical_state),
        },
    )
    screenshot_context["captures"].append(record)
    return record


def _run_trial(
    *,
    robot_plan: Mapping[str, Any],
    base_plan: Mapping[str, Any],
    asset: Path,
    profile_name: str,
    approximation: str,
    control_mode: str,
    trial_index: int,
    screenshot_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.stage import create_new_stage
    from isaacsim.core.utils.stage import get_current_stage
    from omni.physx import get_physx_simulation_interface
    from pxr import Gf
    from pxr import UsdGeom
    from pxr import UsdLux
    from pxr import UsdPhysics

    start_time = time.perf_counter()
    World.clear_instance()
    create_new_stage()
    stage = get_current_stage()
    world_prim = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world_prim)
    stage.DefinePrim("/World/Materials", "Scope")
    add_reference_to_stage(str(asset), "/World/Robot")
    approximation_readback = _collider_token_readback(
        stage,
        robot=robot_plan["name"],
    )
    if set(approximation_readback.values()) != {approximation}:
        raise RuntimeError(
            f"trial approximation mismatch: {approximation_readback}"
        )
    frozen_drive_mimic = _drive_and_mimic_snapshot(stage, robot_plan["name"])
    material = baseline._apply_fingertip_material(
        stage,
        robot_name=robot_plan["name"],
        friction=0.7,
    )
    report_bodies = _apply_contact_reports(stage, robot_plan["name"])
    bottle_prim, bottle_description = baseline._create_bottle(
        stage,
        base_plan,
        friction=0.7,
    )

    dt = 1.0 / 60.0
    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=dt,
        rendering_dt=dt,
    )
    physics_context = world.get_physics_context()
    physics_context.set_solve_articulation_contact_last(True)
    solve_contact_last = physics_context.get_solve_articulation_contact_last()
    articulation = SingleArticulation(
        prim_path="/World/Robot/root_joint",
        name=f"{profile_name}_{control_mode}_{robot_plan['name']}_{trial_index}",
        reset_xform_properties=False,
    )
    world.scene.add(articulation)
    camera = None
    if screenshot_context is not None:
        from isaacsim.sensors.camera import Camera

        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
        dome.CreateIntensityAttr(650.0)
        dome.CreateColorAttr(Gf.Vec3f(0.85, 0.88, 1.0))
        key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
        key.CreateIntensityAttr(1100.0)
        key.CreateAngleAttr(1.0)
        camera = Camera(
            prim_path="/World/DiagnosticCamera",
            name=(
                f"{profile_name}_{robot_plan['name']}_"
                f"{trial_index}_diagnostic_camera"
            ),
            resolution=(1280, 900),
            frequency=60,
        )
        world.scene.add(camera)

    frame_state = {"frame": -1}
    events: list[dict[str, Any]] = []

    def on_contact(headers: Sequence[Any], data: Sequence[Any]) -> None:
        events.extend(
            baseline._serialize_contacts(
                headers,
                data,
                frame=frame_state["frame"],
            )
        )

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(
        on_contact
    )
    world.reset()
    if camera is not None:
        camera.initialize()
        camera.set_clipping_range(0.01, 10.0)
    # Initialize the rigid-body readback wrapper after reset. Adding a
    # kinematic bottle to Scene before reset makes the Scene reset path attempt
    # to author linear/angular velocity on a kinematic PhysX body.
    bottle = SingleRigidPrim(
        "/World/BottleProxy",
        name=f"bottle_{profile_name}_{control_mode}_{robot_plan['name']}_{trial_index}",
        reset_xform_properties=False,
    )
    bottle.initialize()
    reset_serial = hashlib.sha256(
        f"{profile_name}:{control_mode}:{robot_plan['name']}:{trial_index}:world.reset".encode()
    ).hexdigest()
    order = list(articulation.dof_names)
    if order != robot_plan["dof_order"]:
        raise RuntimeError(f"DOF order mismatch: {order}")
    left_index = order.index("left_finger")
    right_index = order.index("right_finger")
    telemetry: list[dict[str, Any]] = []
    motion = base_plan["motion"]

    _step(
        world,
        steps=motion["settle_steps"],
        phase="settle",
        frame_state=frame_state,
        articulation=articulation,
        bottle=bottle,
        left_index=left_index,
        right_index=right_index,
        telemetry=telemetry,
    )
    initial_qpos = articulation.get_joint_positions().tolist()
    start_fingers = baseline._finger_state(articulation, left_index, right_index)
    _command_fingers(
        articulation,
        left_index=left_index,
        right_index=right_index,
        left_target=robot_plan["open_left_finger_m"],
        control_mode=control_mode,
    )
    _step(
        world,
        steps=motion["open_steps"],
        phase="open",
        frame_state=frame_state,
        articulation=articulation,
        bottle=bottle,
        left_index=left_index,
        right_index=right_index,
        telemetry=telemetry,
    )
    opened = baseline._finger_state(articulation, left_index, right_index)
    left_root = f"/World/Robot/{robot_plan['name']}_left_finger_link/collisions"
    right_root = f"/World/Robot/{robot_plan['name']}_right_finger_link/collisions"
    left_open_bounds = baseline._collision_bounds(stage, left_root)
    right_open_bounds = baseline._collision_bounds(stage, right_root)
    open_aperture = baseline._aperture(left_open_bounds, right_open_bounds)

    bottle_position = np.asarray(
        open_aperture["midpoint_world_m"],
        dtype=np.float64,
    )
    bottle_xform = UsdGeom.Xformable(bottle_prim)
    bottle_ops = bottle_xform.GetOrderedXformOps()
    if len(bottle_ops) != 1:
        raise RuntimeError(f"unexpected bottle xform op count: {len(bottle_ops)}")
    bottle_ops[0].Set(Gf.Vec3d(*bottle_position.tolist()))
    get_physx_simulation_interface().flush_changes()
    _step(
        world,
        steps=motion["settle_steps"],
        phase="fixed_bottle_settle",
        frame_state=frame_state,
        articulation=articulation,
        bottle=bottle,
        left_index=left_index,
        right_index=right_index,
        telemetry=telemetry,
    )
    placement_frame = frame_state["frame"]
    trial_screenshots = []
    if screenshot_context is not None:
        screenshot_context["fixed_camera_target_world_m"] = np.asarray(
            _bottle_state(bottle)["position_world_m"],
            dtype=np.float64,
        ).tolist()
        profile_token = (
            "hull"
            if profile_name == "convex_hull"
            else "decomposition"
        )
        trial_screenshots.append(
            _capture_trial_screenshot(
                world=world,
                camera=camera,
                articulation=articulation,
                bottle=bottle,
                open_aperture=open_aperture,
                screenshot_context=screenshot_context,
                phase="runtime_open",
                capture_name=(
                    f"{robot_plan['name']}_{profile_token}_"
                    "open_with_bottle_isometric"
                ),
                view="isometric",
                frame=frame_state["frame"],
                physical_state={
                    "bottle_kinematic": True,
                    "left_finger_contact": False,
                    "right_finger_contact": False,
                    "surface_gap_m": open_aperture["surface_gap_m"],
                },
            )
        )
    _command_fingers(
        articulation,
        left_index=left_index,
        right_index=right_index,
        left_target=robot_plan["closed_left_finger_m"],
        control_mode=control_mode,
    )
    _step(
        world,
        steps=motion["close_steps"],
        phase="close_fixed_bottle",
        frame_state=frame_state,
        articulation=articulation,
        bottle=bottle,
        left_index=left_index,
        right_index=right_index,
        telemetry=telemetry,
    )
    closed = baseline._finger_state(articulation, left_index, right_index)
    _step(
        world,
        steps=motion["fixed_contact_steps"],
        phase="fixed_contact",
        frame_state=frame_state,
        articulation=articulation,
        bottle=bottle,
        left_index=left_index,
        right_index=right_index,
        telemetry=telemetry,
    )
    fixed_contact_end_frame = frame_state["frame"]
    left_closed_bounds = baseline._collision_bounds(stage, left_root)
    right_closed_bounds = baseline._collision_bounds(stage, right_root)
    closed_aperture = baseline._aperture(left_closed_bounds, right_closed_bounds)
    fixed_events = [
        event
        for event in events
        if placement_frame <= int(event["frame"]) <= fixed_contact_end_frame
    ]
    fixed_contact = summarize_contact_events(
        fixed_events,
        bottle_path_token="/BottleProxy",
        penetration_limit_m=base_plan["penetration"]["maximum_persistent_depth_m"],
        persistence_steps=base_plan["penetration"]["persistence_steps"],
    )
    if screenshot_context is not None:
        for view in ("closing_axis", "isometric"):
            trial_screenshots.append(
                _capture_trial_screenshot(
                    world=world,
                    camera=camera,
                    articulation=articulation,
                    bottle=bottle,
                    open_aperture=open_aperture,
                    screenshot_context=screenshot_context,
                    phase="bilateral_contact",
                    capture_name=(
                        f"{robot_plan['name']}_{profile_token}_"
                        f"bilateral_contact_established_{view}"
                    ),
                    view=view,
                    frame=frame_state["frame"],
                    physical_state={
                        "left_finger_contact": fixed_contact[
                            "left_finger_contact"
                        ],
                        "right_finger_contact": fixed_contact[
                            "right_finger_contact"
                        ],
                    },
                )
            )
    constraint_found, constraint_paths = baseline._has_bottle_constraint(stage)
    release_state = _bottle_state(bottle)
    UsdPhysics.RigidBodyAPI(bottle_prim).GetKinematicEnabledAttr().Set(
        False  # noqa: FBT003 - USD attribute API is positional-only
    )
    get_physx_simulation_interface().flush_changes()
    release_frame = frame_state["frame"] + 1
    if screenshot_context is not None:
        trial_screenshots.append(
            _capture_trial_screenshot(
                world=world,
                camera=camera,
                articulation=articulation,
                bottle=bottle,
                open_aperture=open_aperture,
                screenshot_context=screenshot_context,
                phase="release_hold",
                capture_name=(
                    f"{robot_plan['name']}_{profile_token}_release_isometric"
                ),
                view="isometric",
                frame=frame_state["frame"],
                physical_state={
                    "bottle_kinematic": False,
                    "constraint_found": constraint_found,
                },
            )
        )
    _step(
        world,
        steps=base_plan["released_hold"]["hold_steps"],
        phase="released_hold",
        frame_state=frame_state,
        articulation=articulation,
        bottle=bottle,
        left_index=left_index,
        right_index=right_index,
        telemetry=telemetry,
    )
    final_state = _bottle_state(bottle)
    all_contact = summarize_contact_events(
        events,
        bottle_path_token="/BottleProxy",
        penetration_limit_m=base_plan["penetration"]["maximum_persistent_depth_m"],
        persistence_steps=base_plan["penetration"]["persistence_steps"],
    )
    left_contact = _side_contact_summary(
        events,
        side="left",
        placement_frame=placement_frame,
        dt=dt,
    )
    right_contact = _side_contact_summary(
        events,
        side="right",
        placement_frame=placement_frame,
        dt=dt,
    )
    drop_m = float(release_state["z_m"] - final_state["z_m"])
    if screenshot_context is not None:
        trial_screenshots.append(
            _capture_trial_screenshot(
                world=world,
                camera=camera,
                articulation=articulation,
                bottle=bottle,
                open_aperture=open_aperture,
                screenshot_context=screenshot_context,
                phase="release_hold",
                capture_name=(
                    f"{robot_plan['name']}_{profile_token}_hold_end_isometric"
                ),
                view="isometric",
                frame=frame_state["frame"],
                physical_state={
                    "drop_m": drop_m,
                    "drop_gate_m": base_plan["released_hold"]["max_drop_m"],
                },
            )
        )
    metrics = {
        "actual_approximation_token_ok": set(approximation_readback.values())
        == {approximation},
        "fresh_world_reset": True,
        "solve_articulation_contact_last_ok": bool(solve_contact_last),
        "left_finger_contact": fixed_contact["left_finger_contact"],
        "right_finger_contact": fixed_contact["right_finger_contact"],
        "bilateral_contact_before_release": (
            fixed_contact["left_finger_contact"]
            and fixed_contact["right_finger_contact"]
        ),
        "impulses_finite": all_contact["impulses_finite"],
        "persistent_penetration": all_contact["persistent_penetration"],
        "unexpected_gripper_collision": all_contact["unexpected_gripper_collision"],
        "released_without_constraint": not constraint_found,
        "held_for_required_steps": (
            math.isfinite(drop_m)
            and drop_m <= base_plan["released_hold"]["max_drop_m"]
        ),
        "finite_state": bool(
            math.isfinite(drop_m)
            and all(
                math.isfinite(float(item["bottle"]["z_m"]))
                and math.isfinite(float(item["bottle"]["vertical_velocity_m_s"]))
                and math.isfinite(float(item["bottle"]["angular_speed_rad_s"]))
                for item in telemetry
            )
        ),
    }
    contact_quality = _normal_quality(
        left_contact,
        right_contact,
        open_aperture["closing_axis_world"],
    )
    trial = {
        "schema_version": 1,
        "status": "PASS" if trial_passes_hold_gate({"metrics": metrics}) else "FAIL",
        "profile": profile_name,
        "approximation_requested": approximation,
        "approximation_readback": approximation_readback,
        "control_mode": control_mode,
        "control_status": (
            "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
            if control_mode == "explicit_symmetric"
            else "CURRENT_IMPORTED_CONTROL_MAPPING"
        ),
        "robot": robot_plan["name"],
        "trial_index": trial_index,
        "fresh_reset": {
            "world_clear_instance": True,
            "new_stage": True,
            "world_reset": True,
            "reset_serial": reset_serial,
            "resumed_from_contact_state": False,
        },
        "frozen": {
            "friction": material["static_friction"],
            "restitution": material["restitution"],
            "bottle_mass_kg": bottle_description["mass_kg"],
            "bottle_diameter_m": bottle_description["diameter_m"],
            "physics_frequency_hz": 60,
            "physics_dt_s": dt,
            "solve_articulation_contact_last_requested": True,
            "solve_articulation_contact_last_readback": bool(solve_contact_last),
            "self_collision": False,
            "hold_interval_s": base_plan["released_hold"]["hold_time_s"],
            "drop_gate_m": base_plan["released_hold"]["max_drop_m"],
            "contact_rest_offsets_authored": False,
            "drive_and_mimic": frozen_drive_mimic,
        },
        "metrics": metrics,
        "states": {
            "initial_qpos": initial_qpos,
            "dof_order": order,
            "start_fingers": start_fingers,
            "open_fingers": opened,
            "closed_against_fixed_bottle": closed,
        },
        "aperture": {
            "open": open_aperture,
            "closed_against_fixed_bottle": closed_aperture,
        },
        "contacts": {
            "left": left_contact,
            "right": right_contact,
            "normal_quality": contact_quality,
            "fixed_bilateral_summary": fixed_contact,
            "all_contact_summary": all_contact,
            "raw_event_count": len(events),
            "contact_report_bodies": report_bodies,
        },
        "released_hold": {
            "release_frame": release_frame,
            "required_steps": base_plan["released_hold"]["hold_steps"],
            "required_time_s": base_plan["released_hold"]["hold_time_s"],
            "drop_gate_m": base_plan["released_hold"]["max_drop_m"],
            "release_state": release_state,
            "final_state": final_state,
            "drop_m": drop_m,
            "constraint_found": constraint_found,
            "constraint_paths": constraint_paths,
            "support_surface": "NONE_BASELINE_SUSPENDED_HOLD",
            "bottle_left_support_surface": "NOT_APPLICABLE",
        },
        "telemetry": telemetry,
        "screenshots": trial_screenshots,
        "runtime_s": time.perf_counter() - start_time,
        "contact_subscription_active": subscription is not None,
    }
    _augment_trial_derived_fields(trial)
    if camera is not None:
        camera.destroy()
    del subscription
    return trial


def _group_name(profile: str, control_mode: str) -> str:
    prefix = "hull" if profile == "convex_hull" else "decomposition"
    suffix = "current" if control_mode == "current_mimic" else "explicit"
    return f"{prefix}_{suffix}"


def normalize_report_status_and_determinism(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Separate experiment completeness from the physical hold gate."""

    deterministic_per_robot = True
    for group in report["groups"].values():
        group_deterministic = all(
            robot["summary"]["exact_signature_repeat"]
            for robot in group["robots"].values()
        )
        group["combined"]["deterministic_per_robot"] = group_deterministic
        group["combined"]["combined_signature_note"] = (
            "combined signatures include robot identity; determinism is "
            "evaluated within each robot/profile/control group"
        )
        deterministic_per_robot = deterministic_per_robot and group_deterministic
    report["determinism"] = {
        "status": "PASS" if deterministic_per_robot else "FAIL",
        "deterministic_within_every_robot_group": deterministic_per_robot,
    }
    experiment_complete = all(
        group["combined"]["complete"] for group in report["groups"].values()
    )
    report["experiment_execution_status"] = (
        "PASS" if experiment_complete else "FAIL"
    )
    current_control_groups = [
        group
        for group in report["groups"].values()
        if group["control_mode"] == "current_mimic"
    ]
    any_current_profile_holds = any(
        group["combined"]["all_trials_pass_hold_gate"]
        for group in current_control_groups
    )
    report["status"] = (
        "PARTIAL"
        if report["run_mode"] == "NON_ACCEPTANCE_SMOKE"
        else ("PARTIAL" if any_current_profile_holds else "FAIL")
    )
    report["status_semantics"] = {
        "experiment_execution_status": (
            "whether all requested fresh-reset trials and report writes completed"
        ),
        "status": (
            "physical hold gate; FAIL means no current-control collider profile "
            "passed every unchanged hold trial"
        ),
        "PARTIAL": (
            "a diagnostic candidate passed the interface gate but calibrated "
            "material/bottle dynamics remain blocked"
        ),
    }
    return report


def run(
    *,
    project_root: Path,
    repeats: int,
    include_explicit: bool,
    smoke: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile_path = project_root / "configs/aloha1_gripper_collision_profiles.yaml"
    manifest = load_collision_profiles(profile_path, project_root)
    required_repeats = int(manifest["experiment"]["repeats_per_robot"])
    if repeats < required_repeats and not smoke:
        raise ValueError(
            f"acceptance run requires at least {required_repeats} repeats; "
            "use --smoke only for non-acceptance diagnostics"
        )
    assert_profile_pair_is_frozen(
        manifest["profiles"]["convex_hull"],
        manifest["profiles"]["convex_decomposition"],
        allowed_differences={"approximation"},
    )
    baseline_before = _verify_baseline(project_root, manifest)
    base_plan = build_gripper_validation_plan(project_root)
    if (
        base_plan["physics"]["physics_dt_s"] != 1.0 / 60.0
        or base_plan["bottle_proxy"]["mass_kg"] != 0.020
        or base_plan["bottle_proxy"]["diameter_m"] != 0.065
        or base_plan["released_hold"]["hold_time_s"] != 2.0
        or base_plan["released_hold"]["max_drop_m"] != 0.010
    ):
        raise RuntimeError("existing gripper baseline no longer matches frozen A/B values")
    robots = {item["name"]: item for item in base_plan["robots"]}
    control_modes = ["current_mimic"]
    if include_explicit:
        control_modes.append("explicit_symmetric")
    output_dir = project_root / "reports/aloha1_mapping/gripper_collider_ab_trials"
    groups = {}
    for profile_name, profile in manifest["profiles"].items():
        approximation = profile["approximation"]
        for control_mode in control_modes:
            group_name = _group_name(profile_name, control_mode)
            robot_results = {}
            combined_trials = []
            for robot_name in manifest["experiment"]["robots"]:
                asset = _diagnostic_asset(
                    project_root,
                    manifest,
                    profile_name=profile_name,
                    robot=robot_name,
                )
                trials = []
                for trial_index in range(repeats):
                    print(
                        json.dumps(
                            {
                                "ab_event": "trial_start",
                                "group": group_name,
                                "robot": robot_name,
                                "trial_index": trial_index,
                                "repeat_count": repeats,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    trial = _run_trial(
                        robot_plan=robots[robot_name],
                        base_plan=base_plan,
                        asset=asset,
                        profile_name=profile_name,
                        approximation=approximation,
                        control_mode=control_mode,
                        trial_index=trial_index,
                    )
                    trials.append(trial)
                    print(
                        json.dumps(
                            {
                                "ab_event": "trial_complete",
                                "group": group_name,
                                "robot": robot_name,
                                "trial_index": trial_index,
                                "status": trial["status"],
                                "drop_m": trial["released_hold"]["drop_m"],
                                "signature": trial["deterministic_signature"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                trial_path = output_dir / group_name / f"{robot_name}.jsonl"
                _write_jsonl(trial_path, trials)
                summary = summarize_ab_trials(
                    trials,
                    minimum_repeats=(1 if smoke else required_repeats),
                )
                robot_results[robot_name] = {
                    "summary": summary,
                    "trial_file": str(trial_path),
                    "trial_file_sha256": sha256_file(trial_path),
                }
                combined_trials.extend(trials)
            groups[group_name] = {
                "profile": profile_name,
                "approximation": approximation,
                "control_mode": control_mode,
                "control_status": manifest["control_modes"][control_mode]["status"],
                "robots": robot_results,
                "combined": summarize_ab_trials(
                    combined_trials,
                    minimum_repeats=(
                        len(manifest["experiment"]["robots"])
                        if smoke
                        else required_repeats
                        * len(manifest["experiment"]["robots"])
                    ),
                ),
                "diagnostic_metrics": _diagnostic_group_metrics(combined_trials),
            }

    trials_by_group = {
        name: [
            line
            for robot in manifest["experiment"]["robots"]
            for line in _read_jsonl(
                Path(groups[name]["robots"][robot]["trial_file"])
            )
        ]
        for name in groups
    }
    classification_inputs = {
        name: [
            line["metrics"]["held_for_required_steps"]
            and line["metrics"]["bilateral_contact_before_release"]
            and line["metrics"]["impulses_finite"]
            and not line["metrics"]["persistent_penetration"]
            and not line["metrics"]["unexpected_gripper_collision"]
            for line in trials_by_group[name]
        ]
        for name in groups
    }
    per_group_required = (
        len(manifest["experiment"]["robots"])
        if smoke
        else required_repeats * len(manifest["experiment"]["robots"])
    )
    first_round_complete = {
        "hull_current",
        "decomposition_current",
    }.issubset(groups)
    decomposition_status = (
        classify_decomposition_status(
            classification_inputs["hull_current"],
            classification_inputs["decomposition_current"],
            minimum_repeats=per_group_required,
        )
        if first_round_complete
        else {"status": "INCONCLUSIVE", "reason": "first_round_missing"}
    )
    root_cause = (
        classify_root_cause(
            classification_inputs,
            minimum_repeats=per_group_required,
        )
        if include_explicit
        else {
            "status": "PARTIAL",
            "classification": "inconclusive",
            "reason": "second_round_not_run",
        }
    )
    baseline_after = _verify_baseline(project_root, manifest)
    report = {
        "schema_version": 1,
        "status": "PARTIAL",
        "scope": "ALOHA follower gripper collider A/B only",
        "run_mode": "NON_ACCEPTANCE_SMOKE" if smoke else "ACCEPTANCE",
        "repeats_per_robot": repeats,
        "fresh_reset_per_trial": True,
        "frozen_manifest": str(profile_path),
        "frozen_values": manifest["frozen"],
        "groups": groups,
        "first_round_complete": first_round_complete,
        "second_round_complete": include_explicit,
        "CONVEX_DECOMPOSITION_STATUS": decomposition_status["status"],
        "decomposition_evidence": decomposition_status,
        "root_cause_classification": root_cause,
        "baseline_protection": {
            "before": baseline_before,
            "after": baseline_after,
        },
        "runtime_invariant_audit": _runtime_invariant_audit(
            trials_by_group
        ),
        "parameter_scan_run": False,
        "task8": "NOT_RUN",
        "default_asset_collider_modified": False,
    }
    report = normalize_report_status_and_determinism(report)
    classification_report = {
        "schema_version": 1,
        "status": root_cause["status"],
        "classification": root_cause["classification"],
        "classification_detail": root_cause,
        "CONVEX_DECOMPOSITION_STATUS": decomposition_status["status"],
        "decomposition_evidence": decomposition_status,
        "control_warning": "explicit_symmetric is DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING",
        "task8": "NOT_RUN",
        "default_asset_collider_modified": False,
    }
    return report, classification_report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def finalize_existing_report(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = (
        project_root / "reports/aloha1_mapping/gripper_collider_ab_results.json"
    )
    classification_path = (
        project_root
        / "reports/aloha1_mapping/gripper_root_cause_classification.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    repeats = int(report["repeats_per_robot"])
    trials_by_group = {}
    for group_name, group in report["groups"].items():
        combined = []
        for robot in group["robots"].values():
            path = Path(robot["trial_file"])
            trials = [
                _augment_trial_derived_fields(trial)
                for trial in _read_jsonl(path)
            ]
            _write_jsonl(path, trials)
            robot["trial_file_sha256"] = sha256_file(path)
            robot["summary"] = summarize_ab_trials(
                trials,
                minimum_repeats=repeats,
            )
            combined.extend(trials)
        trials_by_group[group_name] = combined
        group["combined"] = summarize_ab_trials(
            combined,
            minimum_repeats=repeats * len(group["robots"]),
        )
        group["diagnostic_metrics"] = _diagnostic_group_metrics(combined)
    report["runtime_invariant_audit"] = _runtime_invariant_audit(
        trials_by_group
    )
    normalize_report_status_and_determinism(report)
    classification["experiment_execution_status"] = report[
        "experiment_execution_status"
    ]
    classification["physical_hold_status"] = report["status"]
    classification["determinism"] = report["determinism"]
    classification["group_evidence"] = {
        name: {
            "trial_count": group["combined"]["trial_count"],
            "hold_success_count": group["combined"]["hold_success_count"],
            "hold_success_rate": group["combined"]["hold_success_rate"],
            "bilateral_contact_trial_count": group["diagnostic_metrics"][
                "bilateral_contact_trial_count"
            ],
            "mean_drop_m": group["diagnostic_metrics"]["drop_m"]["mean"],
            "deterministic_per_robot": group["combined"][
                "deterministic_per_robot"
            ],
        }
        for name, group in report["groups"].items()
    }
    classification["status_semantics"] = {
        "status": "classification evidence completeness",
        "physical_hold_status": "unchanged two-second bottle drop gate",
    }
    return report, classification


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--first-round-only",
        action="store_true",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="allow fewer than 20 repeats and label all outputs non-acceptance",
    )
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="recompute derived fields/hashes from existing trial JSONL without Isaac",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.project_root.resolve(strict=True)
    if args.finalize_existing:
        report, classification = finalize_existing_report(root)
        _write_json(
            root / "reports/aloha1_mapping/gripper_collider_ab_results.json",
            report,
        )
        _write_json(
            root
            / "reports/aloha1_mapping/gripper_root_cause_classification.json",
            classification,
        )
        return 0
    manifest = load_collision_profiles(
        root / "configs/aloha1_gripper_collision_profiles.yaml",
        root,
    )
    repeats = (
        args.repeats
        if args.repeats is not None
        else int(manifest["experiment"]["repeats_per_robot"])
    )
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    failure_path = root / "reports/aloha1_mapping/gripper_collider_ab_failure.json"
    try:
        report, classification = run(
            project_root=root,
            repeats=repeats,
            include_explicit=not args.first_round_only,
            smoke=args.smoke,
        )
        _write_json(
            root / "reports/aloha1_mapping/gripper_collider_ab_results.json",
            report,
        )
        _write_json(
            root
            / "reports/aloha1_mapping/gripper_root_cause_classification.json",
            classification,
        )
        failure_path.unlink(missing_ok=True)
    except BaseException as error:
        _write_json(
            failure_path,
            {
                "schema_version": 1,
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        traceback.print_exc()
        raise
    finally:
        app.close()
    return 0 if report["status"] in {"PASS", "PARTIAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
