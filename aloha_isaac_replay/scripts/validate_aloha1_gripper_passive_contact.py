from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import yaml

from aloha_isaac_replay.adapters.gripper_mapping import DEFAULT_GRIPPER_QPOS_CALIBRATION
from aloha_isaac_replay.adapters.gripper_mapping import GripperQposCalibration
from aloha_isaac_replay.adapters.gripper_mapping import gripper_qpos_calibration_from_loaded_contact
from aloha_isaac_replay.adapters.gripper_mapping import standard_gripper_qpos_to_isaac_fingers
from aloha_isaac_replay.adapters.isaac_dof_adapter import load_mapping
from aloha_isaac_replay.replay.arm_only_mapping import arm_only_targets_from_standard_qpos
from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG
from aloha_isaac_replay.scripts.audit_table_frame_candidate import audit_table_frame
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _apply_arm_gains
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _apply_gravity
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _apply_named_dof_gains
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _get_gains
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _get_limits
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _get_max_efforts
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _get_max_velocities
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _json_safe
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _set_full_state
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _set_full_target
from aloha_isaac_replay.scripts.validate_aloha1_gripper_proxy_gap import _bbox_row
from aloha_isaac_replay.scripts.validate_aloha1_gripper_proxy_gap import _gap_metrics
from aloha_isaac_replay.scripts.validate_aloha1_native_single_joint_response import _safe_target
from aloha_isaac_replay.validation.contact_proxy_profiles import contact_proxy_namespace_roots
from aloha_isaac_replay.validation.contact_proxy_profiles import contact_proxy_profile_names
from aloha_isaac_replay.validation.contact_proxy_profiles import finger_qpos_limits_for_side
from aloha_isaac_replay.validation.contact_proxy_profiles import finger_dof_names_for_side
from aloha_isaac_replay.validation.contact_proxy_profiles import resolve_contact_proxy_paths
from aloha_isaac_replay.validation.contact_proxy_profiles import resolve_contact_target_paths
from aloha_isaac_replay.validation.contact_proxy_profiles import robot_root_for_side
from aloha_isaac_replay.validation.bottle_grasp_semantics import BOTTLE_LENGTH_M
from aloha_isaac_replay.validation.bottle_grasp_semantics import BOTTLE_RADIUS_M
from aloha_isaac_replay.validation.bottle_grasp_semantics import evaluate_axis_aligned_finger_rear_quarter
from aloha_isaac_replay.validation.bottle_grasp_semantics import evaluate_grasp_file

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE = REPO_ROOT / "local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_bbox_proxy_runtime.usda"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase43_gripper_passive_contact_20260718"
DEFAULT_BOTTLE_USD = REPO_ROOT / "assets/bottle_500ml/isaac/bottle_500ml_sim.usd"
DEFAULT_MAPPING = REPO_ROOT / "configs/aloha/original_stationary_aloha_mapping.yaml"
DEFAULT_STAGE_UNITS_IN_METERS = 0.01
DEFAULT_SUPPORT_PLANE_SIZE = 2.0
DEFAULT_SUPPORT_PLANE_THICKNESS = 0.02
DEFAULT_MAX_FINGER_SURFACE_GAP_METERS = 0.12
DEFAULT_MAX_GENERATED_OBJECT_SIDE_METERS = 0.08
HDF5_ARM_START_THEN_GRIPPER_ONLY_MODE = "hdf5_arm_start_then_gripper_only"


def _replay_mode_controls_arm(replay_mode: str) -> bool:
    return replay_mode in {"left_arm_and_gripper", HDF5_ARM_START_THEN_GRIPPER_ONLY_MODE}


def _rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n")


def _open_stage_and_wait(stage_utils: Any, omni_usd: Any, app: Any, stage_path: str, timeout_s: float = 10.0) -> Any:
    """Open a USD stage and wait until Kit exposes the composed stage instance."""

    stage_utils.open_stage(stage_path)
    deadline = time.monotonic() + float(timeout_s)
    last_stage = None
    while time.monotonic() < deadline:
        last_stage = omni_usd.get_context().get_stage()
        if last_stage is not None:
            return last_stage
        try:
            app.update()
        except Exception:
            pass
        time.sleep(0.05)
    raise AssertionError(
        f"Expected a USD stage instance after opening {stage_path!r}, got: {type(last_stage)}"
    )


def _passive_contact_geometry_sanity(
    *,
    finger_surface_gap_stage_units: float,
    object_side_length_stage_units: float,
    stage_units_in_meters: float,
    max_finger_surface_gap_meters: float,
    max_generated_object_side_meters: float,
) -> dict[str, Any]:
    """Reject obviously implausible setup geometry before creating a physics object."""

    gap_stage = float(finger_surface_gap_stage_units)
    side_stage = float(object_side_length_stage_units)
    unit_scale = float(stage_units_in_meters)
    gap_m = gap_stage * unit_scale
    side_m = side_stage * unit_scale
    row: dict[str, Any] = {
        "pass": True,
        "status": "PASS",
        "finger_surface_gap_open": gap_stage,
        "finger_surface_gap_open_meters": gap_m,
        "max_finger_surface_gap_meters": float(max_finger_surface_gap_meters),
        "object_side_length_stage_units": side_stage,
        "object_side_length_meters": side_m,
        "max_generated_object_side_meters": float(max_generated_object_side_meters),
        "stage_units_in_meters": unit_scale,
    }
    if not all(np.isfinite(v) for v in [gap_stage, side_stage, unit_scale, gap_m, side_m]):
        row.update({"pass": False, "status": "FAIL_NONFINITE_CONTACT_SETUP_GEOMETRY"})
    elif gap_stage < 0.0 or side_stage <= 0.0 or unit_scale <= 0.0:
        row.update({"pass": False, "status": "FAIL_INVALID_CONTACT_SETUP_GEOMETRY"})
    elif gap_m > max_finger_surface_gap_meters:
        row.update({"pass": False, "status": "FAIL_IMPLAUSIBLE_FINGER_GAP"})
    elif side_m > max_generated_object_side_meters:
        row.update({"pass": False, "status": "FAIL_IMPLAUSIBLE_OBJECT_SIZE"})
    return row


def _load_support_plane_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    support = cfg.get("support_plane")
    if not isinstance(support, dict):
        raise ValueError(f"{config_path} must contain a support_plane mapping")
    center = support.get("center")
    size = support.get("size")
    if not (isinstance(center, list) and len(center) == 3):
        raise ValueError(f"{config_path} support_plane.center must be a 3-value list")
    if not (isinstance(size, list) and len(size) == 3):
        raise ValueError(f"{config_path} support_plane.size must be a 3-value list")
    return {
        "path": str(config_path),
        "raw": cfg,
        "mode": str(support.get("mode", "fixed_box")),
        "center": [float(v) for v in center],
        "size": [float(v) for v in size],
        "provenance": support.get("provenance"),
        "table_frame": cfg.get("table_frame"),
    }


def _resolve_support_plane_options(args: argparse.Namespace) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "mode": args.support_plane_mode,
        "center": args.support_plane_center,
        "size_x": args.support_plane_size_x if args.support_plane_size_x is not None else args.support_plane_size,
        "size_y": args.support_plane_size_y if args.support_plane_size_y is not None else args.support_plane_size,
        "thickness": args.support_plane_thickness,
        "patch_margin": args.support_plane_patch_margin,
        "config": None,
        "config_provenance": None,
        "table_frame": None,
    }
    if args.support_plane_config is None:
        return resolved

    cfg = _load_support_plane_config(args.support_plane_config)
    if cfg["mode"] != "fixed_box":
        raise ValueError("support_plane_config currently supports only mode: fixed_box")
    if args.support_plane_mode not in {"none", "fixed_box"}:
        raise ValueError(
            "--support-plane-config cannot be combined with generated support-plane modes "
            f"such as {args.support_plane_mode}"
        )
    size = cfg["size"]
    resolved.update(
        {
            "mode": "fixed_box",
            "center": cfg["center"] if args.support_plane_center is None else args.support_plane_center,
            "size_x": args.support_plane_size_x if args.support_plane_size_x is not None else size[0],
            "size_y": args.support_plane_size_y if args.support_plane_size_y is not None else size[1],
            "thickness": args.support_plane_thickness if args.support_plane_thickness != 0.02 else size[2],
            "config": _rel(cfg["path"]),
            "config_provenance": cfg.get("provenance"),
            "table_frame": cfg.get("table_frame"),
        }
    )
    return resolved


def _audit_required_table_frame(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.require_calibrated_table_frame:
        return None
    if args.support_plane_config is None:
        raise ValueError("--require-calibrated-table-frame requires --support-plane-config")
    overrides = _support_plane_cli_overrides(args)
    if overrides:
        raise ValueError(
            "--require-calibrated-table-frame cannot combine --support-plane-config with support-plane CLI overrides: "
            + ", ".join(overrides)
        )
    if abs(float(args.stage_units_in_meters) - 1.0) > 1e-9:
        raise ValueError(
            "--require-calibrated-table-frame requires --stage-units-in-meters 1.0 "
            "because Phase68/69 calibration YAML is authored in Isaac meters, +Z up."
        )
    audit = audit_table_frame(Path(args.support_plane_config))
    if audit["status"] != "PASS_TABLE_TO_BASE_CALIBRATION_READY":
        raise ValueError(
            "--require-calibrated-table-frame failed: "
            f"{audit['status']} ({'; '.join(audit.get('blocking_reasons', []))})"
        )
    return audit


def _guard_support_plane_calibration_mode(args: argparse.Namespace) -> None:
    if args.support_plane_config is None:
        return
    if args.require_calibrated_table_frame:
        return
    if getattr(args, "allow_diagnostic_support_plane_config", False):
        return
    raise ValueError(
        "--support-plane-config can load diagnostic table candidates. Final replay/contact validation must use "
        "--require-calibrated-table-frame, or explicitly pass --allow-diagnostic-support-plane-config for a "
        "non-final diagnostic run."
    )


def _finger_proxy_paths_for_args(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    profile = getattr(args, "contact_proxy_profile", "legacy_puppet")
    return resolve_contact_proxy_paths(profile)


def _stage_namespace_hints(stage_usd: str | Path) -> dict[str, Any]:
    path = Path(stage_usd)
    if not path.exists():
        return {
            "stage_usd": _rel(path),
            "exists": False,
            "uses_scene_namespace": False,
            "uses_legacy_puppet_namespace": False,
            "mentions_bbox_collision_proxy": False,
        }
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {
        "stage_usd": _rel(path),
        "exists": True,
        "uses_scene_namespace": any(marker in text for marker in ('over "scene"', 'def Xform "scene"', "/scene/")),
        "uses_legacy_puppet_namespace": any(marker in text for marker in ("puppet_left_vx300s", "puppet_right_vx300s")),
        "mentions_bbox_collision_proxy": "bbox_collision_proxy" in text,
    }


def _guard_final_contact_stage_namespace(args: argparse.Namespace) -> dict[str, Any] | None:
    if not getattr(args, "require_calibrated_table_frame", False):
        return None
    proxy_roots = contact_proxy_namespace_roots(_finger_proxy_paths_for_args(args))
    hints = _stage_namespace_hints(args.stage_usd)
    summary = {
        "stage_namespace_hints": hints,
        "finger_proxy_namespace_roots": proxy_roots,
        "contact_proxy_profile": getattr(args, "contact_proxy_profile", "legacy_puppet"),
    }
    uses_legacy_proxy_paths = any(root.startswith("puppet_") for root in proxy_roots)
    if hints["uses_scene_namespace"] and uses_legacy_proxy_paths:
        raise ValueError(
            "--require-calibrated-table-frame selected a /scene calibrated overlay, but this contact validator uses "
            "legacy /puppet_* FINGER_PROXY_PATHS. Final contact validation requires a contact-capable stage/profile "
            "whose articulation roots, fingertip proxies, and calibrated table collider share the same namespace."
        )
    return summary


def _support_plane_cli_overrides(args: argparse.Namespace) -> list[str]:
    overrides: list[str] = []
    if getattr(args, "support_plane_center", None) is not None:
        overrides.append("--support-plane-center")
    if getattr(args, "support_plane_size_x", None) is not None:
        overrides.append("--support-plane-size-x")
    if getattr(args, "support_plane_size_y", None) is not None:
        overrides.append("--support-plane-size-y")
    if abs(float(getattr(args, "support_plane_size", DEFAULT_SUPPORT_PLANE_SIZE)) - DEFAULT_SUPPORT_PLANE_SIZE) > 1e-12:
        overrides.append("--support-plane-size")
    if (
        abs(
            float(getattr(args, "support_plane_thickness", DEFAULT_SUPPORT_PLANE_THICKNESS))
            - DEFAULT_SUPPORT_PLANE_THICKNESS
        )
        > 1e-12
    ):
        overrides.append("--support-plane-thickness")
    return overrides


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    base_fieldnames = [
        "phase",
        "step",
        "object_center_x",
        "object_center_y",
        "object_center_z",
        "object_displacement",
        "left_finger_qpos",
        "right_finger_qpos",
        "finger_center_distance",
        "left_axis_min",
        "left_axis_max",
        "left_axis_center",
        "right_axis_min",
        "right_axis_max",
        "right_axis_center",
        "object_axis_min",
        "object_axis_max",
        "object_axis_center",
        "target_finger_object_surface_gap",
    ]
    fieldnames = list(base_fieldnames)
    seen = set(fieldnames)
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    unique_pairs = payload.get("unique_contact_pairs") or []
    pair_lines = [f"- `{pair}`" for pair in unique_pairs[:12]]
    if len(unique_pairs) > 12:
        pair_lines.append(f"- ... {len(unique_pairs) - 12} more unique pairs")
    finger_hits = payload.get("target_contact_finger_hits") or {}
    finger_hit_lines = [f"- `{path}`: `{hit}`" for path, hit in finger_hits.items()]
    cross_overlap = payload.get("cross_side_proxy_overlap") or {}
    support_plane = payload.get("support_plane") or {}
    debug_stage = payload.get("debug_stage_after_object_placement") or {}
    diagnostic_contacts = payload.get("diagnostic_contact_summaries") or {}
    support_size = support_plane.get("size")
    tracking_gate = payload.get("controller_tracking_gate") or {}
    physical_grasp_gate = payload.get("physical_grasp_gate") or {}
    tabletop_grasp_gate = payload.get("tabletop_grasp_contact_gate") or {}
    lift_transport_gate = payload.get("lift_transport_gate") or {}
    prelift_static_gate = payload.get("prelift_static_grasp_gate") or {}
    post_close_hold_gate = payload.get("post_close_hold_diagnostic_gate") or {}
    post_close_lift_gate = payload.get("post_close_lift_diagnostic_gate") or {}
    controller_replay_gate = payload.get("controller_replay_fidelity_gate") or {}
    command_smoothness = payload.get("command_smoothness_gate") or {}
    formal_replay_gate = payload.get("formal_replay_feasibility_gate") or {}
    top_command_spike = (command_smoothness.get("top_target_velocity_spikes") or [{}])[0]
    tracking_spike = payload.get("tracking_spike_packet") or {}
    drive_audit = payload.get("drive_authority_audit") or {}
    non_target_gate = payload.get("non_target_contact_gate") or {}
    active_target_gate = payload.get("active_target_contact_gate") or {}
    bilateral_gate = payload.get("bilateral_grasp_formation_gate") or {}
    landmark_alignment = payload.get("contact_landmark_alignment") or {}
    first_landmark_alignment = (landmark_alignment.get("samples") or [{}])[0]
    start_alignment = payload.get("start_finger_object_alignment") or {}
    final_alignment = payload.get("final_finger_object_alignment") or {}
    active_grasp_geometry = payload.get("active_grasp_geometry_precondition") or {}
    object_lift_gate = payload.get("object_lift_gate") or {}
    soft_contact_model = payload.get("soft_bottle_contact_model") or {}
    loaded_calibration = payload.get("loaded_gripper_soft_bottle_calibration_diagnostic") or {}
    bottle_gate = payload.get("bottle_runtime_composition_gate") or {}
    grasp_gate = payload.get("bottle_grasp_semantics_gate") or {}
    failure_reasons = payload.get("failure_reasons") or []
    lines = [
        "# Gripper Passive Contact Smoke",
        "",
        f"- status: `{payload['status']}`",
        f"- contact trace status: `{payload.get('contact_trace_status')}`",
        f"- failure reasons: `{failure_reasons}`",
        f"- tabletop grasp contact gate: `{tabletop_grasp_gate.get('status')}` pass=`{tabletop_grasp_gate.get('pass')}`",
        f"- lift transport gate: `{lift_transport_gate.get('status')}` pass=`{lift_transport_gate.get('pass')}`",
        f"- lift follow ratio: `{lift_transport_gate.get('object_follow_ratio')}`",
        f"- post-close hold diagnostic: `{post_close_hold_gate.get('status')}` pass=`{post_close_hold_gate.get('pass')}`",
        f"- pre-lift static grasp diagnostic: `{prelift_static_gate.get('status')}` pass=`{prelift_static_gate.get('pass')}`",
        f"- pre-lift tail bilateral contact steps: `{prelift_static_gate.get('tail_bilateral_contact_steps')}`",
        f"- pre-lift object XY sweep: `{prelift_static_gate.get('object_xy_sweep_m')}` m",
        f"- pre-lift object Z delta: `{prelift_static_gate.get('object_z_delta_m')}` m",
        f"- post-close lift diagnostic: `{post_close_lift_gate.get('status')}` pass=`{post_close_lift_gate.get('pass')}`",
        f"- stage: `{payload['inputs']['stage_usd']}`",
        f"- control mode: `{payload['inputs']['control_mode']}`",
        f"- moving fingers: `{payload['inputs'].get('moving_fingers')}`",
        f"- visible bottle runtime path: `{payload.get('visible_bottle_runtime_path')}`",
        f"- bottle runtime composition gate: `{bottle_gate.get('status')}`",
        f"- bottle visual mesh count: `{bottle_gate.get('visual_mesh_count')}`",
        f"- bottle collision prim count: `{bottle_gate.get('collision_prim_count')}`",
        f"- bottle grasp semantics gate: `{grasp_gate.get('status')}`",
        f"- selected grasp: `{grasp_gate.get('selected_grasp')}`",
        f"- finger rear-quarter fraction: `{grasp_gate.get('fraction_from_axis_min')}`",
        f"- finger rear-quarter target: `{grasp_gate.get('rear_fraction_target')}`",
        f"- closing axis dot bottle long axis: `{grasp_gate.get('closing_long_axis_dot_abs')}`",
        f"- debug stage after object placement: `{debug_stage.get('path')}` saved=`{debug_stage.get('saved')}`",
        f"- object side length: `{payload.get('object_side_length_stage_units')}` stage units",
        f"- object side length: `{payload.get('object_side_length_meters')}` m",
        f"- soft bottle contact model: `{soft_contact_model.get('enabled')}`",
        f"- soft effective contact width: `{soft_contact_model.get('effective_contact_width_m')}` m",
        f"- visible external diameter: `{soft_contact_model.get('visual_external_diameter_m')}` m",
        f"- loaded gripper calibration diagnostic: `{loaded_calibration.get('status')}`",
        f"- loaded calibration preserves formal gate: `{loaded_calibration.get('formal_gate_result_preserved')}`",
        f"- nearest qpos replay surface gap: `{loaded_calibration.get('nearest_surface_gap_m')}` m",
        f"- missing to contact distance: `{loaded_calibration.get('missing_to_contact_distance_m')}` m",
        f"- per-finger loaded closure residual: `{loaded_calibration.get('per_finger_loaded_closure_deficit_to_zero_gap_m')}` m",
        f"- finger surface gap open: `{payload.get('finger_surface_gap_open')}` stage units",
        f"- finger surface gap open: `{payload.get('finger_surface_gap_open_meters')}` m",
        f"- contact setup geometry sanity: `{payload.get('contact_setup_geometry_sanity_status')}`",
        f"- object settle displacement: `{payload.get('object_settle_displacement')}` stage units",
        f"- object close displacement: `{payload.get('object_displacement')}` stage units",
        f"- object total displacement: `{payload.get('total_object_displacement')}` stage units",
        f"- object lift: `{payload.get('object_lift')}` stage units",
        f"- object lift gate: `{object_lift_gate.get('status')}` required=`{object_lift_gate.get('required')}`",
        f"- max object displacement: `{payload.get('max_object_displacement')}` stage units",
        f"- finite object motion: `{payload.get('object_motion_finite')}`",
        f"- contact motion lower bound ok: `{payload.get('contact_motion_ok')}`",
        f"- no explosion upper bound ok: `{payload.get('no_explosion_ok')}`",
        f"- contact pair trace enabled: `{payload.get('contact_pair_trace_enabled')}`",
        f"- contact pair count: `{payload.get('contact_pair_count')}`",
        f"- target contact pair found: `{payload.get('target_contact_pair_found')}`",
        f"- all expected fingers contacted object: `{payload.get('all_expected_fingers_target_contact_pair_found')}`",
        f"- target contact found during settle: `{payload.get('target_contact_found_during_settle')}`",
        f"- target contact found during close: `{payload.get('target_contact_found_during_close')}`",
        f"- non-target object contact found: `{payload.get('non_target_object_contact_found')}`",
        f"- non-target object contact pair count: `{payload.get('non_target_object_contact_pair_count')}`",
        f"- non-target object contact categories: `{payload.get('non_target_object_contact_categories')}`",
        f"- allowed non-target object contact categories: `{payload.get('allowed_non_target_object_contact_categories')}`",
        f"- non-target contact gate: `{non_target_gate}`",
        f"- strict non-target object contact gate ok: `{payload.get('non_target_object_contact_ok')}`",
        f"- require active target contact: `{payload.get('require_active_target_contact')}`",
        f"- already-in-contact setup: `{payload.get('already_in_contact_setup')}`",
        f"- active grasp geometry precondition: `{active_grasp_geometry.get('status')}`",
        f"- active grasp open center gap: `{active_grasp_geometry.get('open_finger_center_gap_m')}` m",
        f"- active grasp open gap: `{active_grasp_geometry.get('open_finger_surface_gap_m')}` m",
        f"- active grasp surface gap diagnostic only: `{active_grasp_geometry.get('surface_gap_is_diagnostic_only')}`",
        f"- active grasp object width along gap: `{active_grasp_geometry.get('object_width_along_gap_axis_m')}` m",
        f"- active grasp free-space shortfall: `{active_grasp_geometry.get('shortfall_m')}` m",
        f"- active target contact gate: `{active_target_gate}`",
        f"- active target contact gate ok: `{payload.get('active_target_contact_ok')}`",
        f"- bilateral grasp formation gate: `{bilateral_gate.get('status')}` pass=`{bilateral_gate.get('pass')}`",
        f"- bilateral contact steps: `{bilateral_gate.get('bilateral_contact_step_count')}`",
        f"- pre-lift lateral sweep: `{bilateral_gate.get('lateral_sweep_for_gate_m')}` m",
        f"- start object cross-closing-axis offset: `{start_alignment.get('object_cross_closing_axis_offset_norm_m')}` m",
        f"- start reference grasp-band correction to midplane: `{((start_alignment.get('reference_contact_center') or {}).get('correction_to_midplane_norm_m'))}` m",
        f"- start closing-axis dot object long axis abs: `{((start_alignment.get('object_long_axis') or {}).get('closing_axis_dot_object_long_axis_abs'))}`",
        f"- start projected inner gap: `{(start_alignment.get('closing_axis_projected_inner_gap') or {}).get('finger_inner_gap_m')}` m",
        f"- start object inside projected finger gap: `{(start_alignment.get('closing_axis_projected_inner_gap') or {}).get('object_inside_inner_gap')}`",
        f"- contact landmark alignment: `{landmark_alignment.get('status')}`",
        f"- first landmark step: `{first_landmark_alignment.get('step')}`",
        f"- first landmark object cross-closing-axis offset: `{first_landmark_alignment.get('object_cross_closing_axis_offset_norm_m')}` m",
        f"- final object cross-closing-axis offset: `{final_alignment.get('object_cross_closing_axis_offset_norm_m')}` m",
        f"- cross-side proxy overlap detected: `{cross_overlap.get('overlap_detected')}`",
        f"- first contact pair: `{payload.get('first_contact_pair')}`",
        f"- first target contact pair: `{payload.get('first_target_contact_pair')}`",
        f"- first target contact phase: `{payload.get('first_target_contact_phase')}`",
        f"- first target contact step: `{payload.get('first_target_contact_step')}`",
        f"- first non-target object contact pair: `{payload.get('first_non_target_object_contact_pair')}`",
        f"- first non-target object contact phase: `{payload.get('first_non_target_object_contact_phase')}`",
        f"- target contact persistence steps: `{payload.get('target_contact_persistence_steps')}`",
        f"- target runtime-limit summary: `{payload.get('target_limit_summary')}`",
        f"- target runtime-limit gate ok: `{payload.get('target_limit_gate_ok')}`",
        f"- physical grasp semantics gate: `{physical_grasp_gate.get('status')}` pass=`{physical_grasp_gate.get('pass')}`",
        f"- formal replay feasibility gate: `{formal_replay_gate.get('status')}` pass=`{formal_replay_gate.get('pass')}`",
        f"- controller tracking gate: `{tracking_gate}`",
        f"- controller replay fidelity gate: `{controller_replay_gate.get('status')}` pass=`{controller_replay_gate.get('pass')}`",
        f"- command smoothness gate: `{command_smoothness.get('status')}` pass=`{command_smoothness.get('pass')}`",
        f"- formal replay targets modified: `{command_smoothness.get('formal_replay_targets_modified')}`",
        f"- top command velocity spike: dof=`{top_command_spike.get('dof_name')}` step=`{top_command_spike.get('step')}` velocity=`{top_command_spike.get('target_velocity')}`",
        f"- max tracking spike dof: `{tracking_spike.get('dof_name')}` phase=`{tracking_spike.get('phase')}` step=`{tracking_spike.get('step')}`",
        f"- max tracking spike target delta: `{tracking_spike.get('target_delta_from_previous')}`",
        f"- max tracking spike actual delta: `{tracking_spike.get('actual_delta_during_hold')}`",
        f"- max tracking spike contact categories: `{tracking_spike.get('contact_categories_at_step')}`",
        f"- drive authority audit: `{drive_audit.get('status')}` profile=`{drive_audit.get('profile_name')}`",
        f"- estimated net drive demand at spike: `{drive_audit.get('estimated_net_drive_demand')}` clipped=`{drive_audit.get('estimated_effort_clipped')}`",
        f"- pre-step tracking summary: `{payload.get('pre_step_tracking_summary')}`",
        f"- post-step tracking summary: `{payload.get('tracking_summary')}`",
        f"- support plane path: `{support_plane.get('path')}`",
        f"- support plane center: `{support_plane.get('center')}`",
        f"- support plane size: `{support_size}`",
        f"- support plane size xy: `{support_plane.get('size_xy')}`",
        f"- support plane thickness: `{support_plane.get('thickness')}`",
        f"- diagnostic contact summaries: `{diagnostic_contacts}`",
        "",
        "## Expected Finger Coverage",
        "",
        *(finger_hit_lines or ["- none"]),
        "",
        "## Unique Contact Pairs",
        "",
        *(pair_lines or ["- none"]),
        "",
        "## Interpretation",
        "",
        "This is a local contact smoke test for the configured target object collider. The target may be a simple primitive, a BottleUSD asset, or a BottleUSD visual asset paired with a simplified physics proxy.",
        "A non-zero contact count is not a success condition. The trace must show that the expected fingertip proxy contacts the target object collider in the required phase, and object motion must remain bounded.",
        "It does not by itself validate final grasp success, friction realism, lift stability, or full-arm task behavior.",
    ]
    path.write_text("\n".join(lines) + "\n")


def _finger_targets(
    art: Any,
    offset: float,
    limit_margin: float,
    finger_dof_names: dict[str, str],
    *,
    right_finger_sign: float = -1.0,
) -> tuple[np.ndarray, dict[str, float]]:
    dof_names = list(art.dof_names)
    limits = _get_limits(art)
    qpos = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
    target = qpos.copy()
    target_values: dict[str, float] = {}
    for logical_name, sign in [("left_finger", 1.0), ("right_finger", float(right_finger_sign))]:
        dof_name = finger_dof_names[logical_name]
        idx = dof_names.index(dof_name)
        lower, upper = [float(x) for x in limits[idx]]
        origin = (lower + upper) * 0.5
        target_value, _clipped = _safe_target(origin, offset * sign, lower, upper, limit_margin)
        target[idx] = target_value
        target_values[logical_name] = target_value
    return target, target_values


def _load_hdf5_qpos(path: str | Path, *, start: int | None, end: int | None, max_frames: int | None) -> np.ndarray:
    import h5py

    episode = Path(path)
    with h5py.File(episode, "r") as h5:
        qpos = np.asarray(h5["observations/qpos"][:], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] < 14:
        raise ValueError(f"Expected observations/qpos shape (T, >=14), got {qpos.shape} in {episode}")
    lo = 0 if start is None else int(start)
    hi = len(qpos) if end is None else int(end)
    seq = qpos[lo:hi]
    if max_frames is not None:
        seq = seq[: int(max_frames)]
    if seq.shape[0] < 2:
        raise ValueError(f"Need at least two HDF5 qpos samples, got {seq.shape[0]} from {episode}")
    if not np.isfinite(seq).all():
        raise ValueError(f"HDF5 qpos contains NaN/Inf: {episode}")
    return np.asarray(seq, dtype=np.float64)


def _load_hdf5_action(path: str | Path, *, start: int | None, end: int | None, max_frames: int | None) -> np.ndarray:
    import h5py

    episode = Path(path)
    with h5py.File(episode, "r") as h5:
        action = np.asarray(h5["action"][:], dtype=np.float64)
    if action.ndim != 2 or action.shape[1] < 14:
        raise ValueError(f"Expected action shape (T, >=14), got {action.shape} in {episode}")
    lo = 0 if start is None else int(start)
    hi = len(action) if end is None else int(end)
    seq = action[lo:hi]
    if max_frames is not None:
        seq = seq[: int(max_frames)]
    if seq.shape[0] < 2:
        raise ValueError(f"Need at least two HDF5 action samples, got {seq.shape[0]} from {episode}")
    if not np.isfinite(seq).all():
        raise ValueError(f"HDF5 action contains NaN/Inf: {episode}")
    return np.asarray(seq, dtype=np.float64)


def _target_from_standard_qpos(
    *,
    art: Any,
    side: str,
    qpos_frame: np.ndarray,
    mapping: dict[str, Any] | None,
    replay_mode: str,
    finger_dof_names: dict[str, str],
    finger_qpos_limits: Any,
    gripper_qpos_calibration: GripperQposCalibration = DEFAULT_GRIPPER_QPOS_CALIBRATION,
) -> np.ndarray:
    dof_names = list(art.dof_names)
    target = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1).copy()
    if replay_mode == "left_arm_and_gripper":
        if mapping is None:
            raise ValueError("left_arm_and_gripper replay requires a mapping")
        side_prefix = f"{side}/"
        for arm_target in arm_only_targets_from_standard_qpos(qpos_frame, mapping, side=side):
            if not arm_target.isaac_dof_name.startswith(side_prefix):
                continue
            dof_name = arm_target.isaac_dof_name[len(side_prefix) :]
            resolved_dof_name = _resolve_side_arm_dof_name(
                dof_name,
                dof_names=dof_names,
                side=side,
                source_name=arm_target.isaac_dof_name,
            )
            target[dof_names.index(resolved_dof_name)] = float(arm_target.value)
    channel = 6 if side == "left" else 13
    fingers = standard_gripper_qpos_to_isaac_fingers(
        float(qpos_frame[channel]),
        side=side,
        calibration=gripper_qpos_calibration,
        limits=finger_qpos_limits,
    )
    target[dof_names.index(finger_dof_names["left_finger"])] = float(fingers[f"{side}/left_finger"])
    target[dof_names.index(finger_dof_names["right_finger"])] = float(fingers[f"{side}/right_finger"])
    return target


def _resolve_side_arm_dof_name(source_suffix: str, *, dof_names: list[str], side: str, source_name: str) -> str:
    """Resolve mapping names like ``left/waist`` against stage DOF names.

    Different ALOHA/Trossen stages use either unprefixed arm names
    (``waist``) or side-prefixed names (``left_waist``).  Keep this strict:
    only the exact stripped name and the exact current-side-prefixed name are
    allowed, so a missing or cross-side DOF remains a hard error.
    """

    candidates = [source_suffix]
    if not source_suffix.startswith(f"{side}_"):
        candidates.append(f"{side}_{source_suffix}")
    for candidate in candidates:
        if candidate in dof_names:
            return candidate
    sample = ", ".join(dof_names[:12])
    raise ValueError(
        f"Could not resolve mapped DOF {source_name!r}; tried {candidates!r}; "
        f"available DOFs include: {sample}"
    )


def _targets_from_hdf5_qpos(
    *,
    art: Any,
    side: str,
    qpos: np.ndarray,
    gripper_sequence: np.ndarray | None = None,
    gripper_source: str = "observations/qpos",
    mapping: dict[str, Any] | None,
    replay_mode: str,
    finger_dof_names: dict[str, str],
    finger_qpos_limits: Any,
    gripper_qpos_calibration: GripperQposCalibration = DEFAULT_GRIPPER_QPOS_CALIBRATION,
    arm_hold_frame_offset: int = 0,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    dof_names = list(art.dof_names)
    left_idx = dof_names.index(finger_dof_names["left_finger"])
    right_idx = dof_names.index(finger_dof_names["right_finger"])
    channel = 6 if side == "left" else 13
    gripper_signal_array = qpos if gripper_sequence is None else np.asarray(gripper_sequence, dtype=np.float64)
    if gripper_signal_array.shape[0] != qpos.shape[0] or gripper_signal_array.shape[1] < 14:
        raise ValueError(
            "gripper_sequence must have the same frame count as qpos and at least 14 columns; "
            f"got qpos={qpos.shape}, gripper_sequence={gripper_signal_array.shape}"
        )
    gripper_signal = np.asarray(gripper_signal_array[:, channel], dtype=np.float64)
    targets: list[np.ndarray] = []
    if replay_mode == HDF5_ARM_START_THEN_GRIPPER_ONLY_MODE:
        if mapping is None:
            raise ValueError(f"{HDF5_ARM_START_THEN_GRIPPER_ONLY_MODE} replay requires a mapping")
        hold_offset = int(arm_hold_frame_offset)
        if hold_offset < 0 or hold_offset >= int(qpos.shape[0]):
            raise ValueError(
                f"arm_hold_frame_offset must be within the selected HDF5 qpos window [0, {qpos.shape[0] - 1}], "
                f"got {hold_offset}"
            )
        arm_hold_target = _target_from_standard_qpos(
            art=art,
            side=side,
            qpos_frame=qpos[hold_offset],
            mapping=mapping,
            replay_mode="left_arm_and_gripper",
            finger_dof_names=finger_dof_names,
            finger_qpos_limits=finger_qpos_limits,
            gripper_qpos_calibration=gripper_qpos_calibration,
        )
        for frame, gripper_frame in zip(qpos, gripper_signal_array, strict=True):
            target = np.asarray(arm_hold_target, dtype=np.float64).copy()
            fingers = standard_gripper_qpos_to_isaac_fingers(
                float(gripper_frame[channel]),
                side=side,
                calibration=gripper_qpos_calibration,
                limits=finger_qpos_limits,
            )
            target[left_idx] = float(fingers[f"{side}/left_finger"])
            target[right_idx] = float(fingers[f"{side}/right_finger"])
            targets.append(target)
    else:
        for frame, gripper_frame in zip(qpos, gripper_signal_array, strict=True):
            target = _target_from_standard_qpos(
                art=art,
                side=side,
                qpos_frame=frame,
                mapping=mapping,
                replay_mode=replay_mode,
                finger_dof_names=finger_dof_names,
                finger_qpos_limits=finger_qpos_limits,
                gripper_qpos_calibration=gripper_qpos_calibration,
            )
            if gripper_source != "observations/qpos":
                fingers = standard_gripper_qpos_to_isaac_fingers(
                    float(gripper_frame[channel]),
                    side=side,
                    calibration=gripper_qpos_calibration,
                    limits=finger_qpos_limits,
                )
                target[left_idx] = float(fingers[f"{side}/left_finger"])
                target[right_idx] = float(fingers[f"{side}/right_finger"])
            targets.append(target)
    arm_delta = None
    arm_target_behavior = "not_controlled"
    if _replay_mode_controls_arm(replay_mode):
        indices = slice(0, 6) if side == "left" else slice(7, 13)
        arm_qpos = np.asarray(qpos[:, indices], dtype=np.float64)
        arm_delta = {
            "max_abs_frame_delta": float(np.max(np.abs(np.diff(arm_qpos, axis=0)))) if len(arm_qpos) > 1 else 0.0,
            "max_abs_net_delta": float(np.max(np.abs(arm_qpos[-1] - arm_qpos[0]))),
        }
        if replay_mode == HDF5_ARM_START_THEN_GRIPPER_ONLY_MODE:
            arm_target_behavior = (
                "constant_hdf5_start_frame_hold"
                if int(arm_hold_frame_offset) == 0
                else "constant_hdf5_selected_frame_hold"
            )
        else:
            arm_target_behavior = "hdf5_frame_by_frame_targets"
    return targets, {
        "source": gripper_source,
        "arm_source": "observations/qpos",
        "side": side,
        "replay_mode": replay_mode,
        "formal_full_hdf5_replay": bool(replay_mode == "left_arm_and_gripper"),
        "arm_initialized_from_hdf5": bool(_replay_mode_controls_arm(replay_mode)),
        "hdf5_arm_targets_after_start_used": bool(replay_mode == "left_arm_and_gripper"),
        "arm_target_behavior": arm_target_behavior,
        "arm_hold_frame_offset": int(arm_hold_frame_offset) if replay_mode == HDF5_ARM_START_THEN_GRIPPER_ONLY_MODE else None,
        "sample_count": int(gripper_signal.size),
        "raw_start": float(gripper_signal[0]),
        "raw_end": float(gripper_signal[-1]),
        "raw_min": float(np.min(gripper_signal)),
        "raw_max": float(np.max(gripper_signal)),
        "raw_range": float(np.max(gripper_signal) - np.min(gripper_signal)),
        "raw_net": float(gripper_signal[-1] - gripper_signal[0]),
        "first_target_values": {
            "left_finger": float(targets[0][left_idx]),
            "right_finger": float(targets[0][right_idx]),
        },
        "last_target_values": {
            "left_finger": float(targets[-1][left_idx]),
            "right_finger": float(targets[-1][right_idx]),
        },
        "qpos_calibration": dataclasses.asdict(gripper_qpos_calibration),
        "qpos_source_is_loaded_gap_calibrated": bool(
            gripper_qpos_calibration != DEFAULT_GRIPPER_QPOS_CALIBRATION
        ),
        "arm_qpos_delta": arm_delta,
    }


def _apply_diagnostic_loaded_clamp_squeeze(
    *,
    enabled: bool,
    target: np.ndarray,
    dof_names: list[str],
    finger_dof_names: dict[str, str],
    runtime_limits: np.ndarray,
    limit_margin: float,
    squeeze_depth: float,
    phase: str,
    moving_fingers: str,
    reference_target: np.ndarray,
    reference_target_source: str = "reference_target",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a diagnostic-only post-close over-close target.

    The HDF5 close sequence remains untouched.  This only modifies appended
    post-close hold/lift diagnostics to test whether a small loaded soft-bottle
    compression can create enough normal force to carry the object.
    """

    row: dict[str, Any] = {
        "enabled": bool(enabled),
        "active": False,
        "status": "DISABLED",
        "phase": phase,
        "squeeze_depth_m": float(squeeze_depth),
        "per_finger_target_delta_m": 0.0,
        "reference_target_source": str(reference_target_source),
        "diagnostic_only": True,
        "formal_hdf5_close_sequence_modified": False,
        "notes": (
            "Diagnostic post-close target over-close for soft-bottle load-bearing tests. "
            "It must not be treated as original HDF5 replay success."
        ),
    }
    if not enabled:
        return target, row
    if phase not in {"post_close_hold", "post_close_lift", "post_close_lift_hold"}:
        row.update({"status": "SKIPPED_FORMAL_CLOSE_PHASE"})
        return target, row
    if moving_fingers != "both":
        row.update({"status": "SKIPPED_REQUIRES_BOTH_FINGERS"})
        return target, row
    depth = float(squeeze_depth)
    if not np.isfinite(depth) or depth <= 0.0:
        row.update({"status": "SKIPPED_NONPOSITIVE_SQUEEZE_DEPTH"})
        return target, row

    squeezed = np.asarray(target, dtype=np.float64).reshape(-1).copy()
    reference = np.asarray(reference_target, dtype=np.float64).reshape(-1)
    per_finger_delta = depth * 0.5
    updates: dict[str, Any] = {}
    any_clipped = False
    for logical_name in ("left_finger", "right_finger"):
        dof_name = finger_dof_names[logical_name]
        idx = dof_names.index(dof_name)
        lower, upper = [float(x) for x in runtime_limits[idx]]
        requested = float(reference[idx]) - per_finger_delta
        safe_value, clipped = _safe_target(float(reference[idx]), -per_finger_delta, lower, upper, limit_margin)
        squeezed[idx] = float(safe_value)
        any_clipped = bool(any_clipped or clipped)
        updates[logical_name] = {
            "dof_name": dof_name,
            "dof_index": int(idx),
            "reference_target": float(reference[idx]),
            "requested_target": requested,
            "applied_target": float(safe_value),
            "actual_delta_from_reference": float(safe_value - float(reference[idx])),
            "lower_limit": lower,
            "upper_limit": upper,
            "clipped": bool(clipped),
        }
    row.update(
        {
            "active": True,
            "status": "APPLIED_DIAGNOSTIC_LOADED_CLAMP_SQUEEZE",
            "per_finger_target_delta_m": -per_finger_delta,
            "target_inner_gap_reduction_m": float(2.0 * per_finger_delta),
            "clipped": bool(any_clipped),
            "finger_targets": updates,
        }
    )
    return squeezed, row


def _tracking_groups(
    dof_names: list[str], *, replay_mode: str, finger_dof_names: dict[str, str], side: str = "left"
) -> dict[str, list[int]]:
    finger_indices = [
        dof_names.index(finger_dof_names["left_finger"]),
        dof_names.index(finger_dof_names["right_finger"]),
    ]
    groups: dict[str, list[int]] = {"gripper": finger_indices}
    if _replay_mode_controls_arm(replay_mode):
        base_arm_names = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
        side_arm_names = tuple(f"{side}_{name}" for name in base_arm_names)
        arm_names = side_arm_names if all(name in dof_names for name in side_arm_names) else base_arm_names
        arm_indices = [dof_names.index(name) for name in arm_names if name in dof_names]
        groups["left_arm"] = arm_indices
        groups["controlled"] = arm_indices + finger_indices
    else:
        groups["controlled"] = finger_indices
    return groups


def _should_disable_workcell_environment_collision(path: str) -> bool:
    """Return True for uncalibrated workcell/table collision prims used only in diagnostic replay isolation."""

    normalized = str(path)
    return (
        normalized == "/World/Table"
        or normalized.startswith("/World/Table/")
        or normalized == "/scene/worldBody"
        or normalized.startswith("/scene/worldBody/")
    )


def _disable_workcell_environment_collisions_for_diagnostic_replay(stage: Any) -> list[str]:
    from pxr import UsdPhysics

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if _should_disable_workcell_environment_collision(path) and prim.IsInstanceable():
            prim.SetInstanceable(False)

    disabled: list[str] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not _should_disable_workcell_environment_collision(path):
            continue
        collision = UsdPhysics.CollisionAPI(prim)
        if not collision:
            continue
        attr = collision.GetCollisionEnabledAttr()
        if not attr:
            attr = collision.CreateCollisionEnabledAttr()
        attr.Set(False)
        disabled.append(path)
    return disabled


def _finger_qpos_values(qpos: np.ndarray, dof_names: list[str], finger_dof_names: dict[str, str]) -> dict[str, float]:
    return {
        "left_finger_qpos": float(qpos[dof_names.index(finger_dof_names["left_finger"])]),
        "right_finger_qpos": float(qpos[dof_names.index(finger_dof_names["right_finger"])]),
    }


def _tracking_step_errors(
    *,
    target: np.ndarray,
    actual: np.ndarray,
    groups: dict[str, list[int]],
) -> dict[str, dict[str, float]]:
    row: dict[str, dict[str, float]] = {}
    error = np.asarray(actual, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    for name, indices in groups.items():
        if not indices:
            row[name] = {"max_abs_error": float("nan"), "rms_error": float("nan")}
            continue
        group_error = error[np.asarray(indices, dtype=np.int64)]
        local_max_index = int(np.argmax(np.abs(group_error)))
        row[name] = {
            "max_abs_error": float(np.max(np.abs(group_error))),
            "max_abs_error_dof_index": int(indices[local_max_index]),
            "max_abs_error_signed": float(group_error[local_max_index]),
            "rms_error": float(np.sqrt(np.mean(np.square(group_error)))),
        }
    return row


def _target_limit_step_violations(
    *,
    target: np.ndarray,
    limits: np.ndarray,
    groups: dict[str, list[int]],
) -> dict[str, dict[str, float]]:
    row: dict[str, dict[str, float]] = {}
    target_arr = np.asarray(target, dtype=np.float64)
    limits_arr = np.asarray(limits, dtype=np.float64)
    lower = limits_arr[:, 0]
    upper = limits_arr[:, 1]
    lower_violation = np.maximum(lower - target_arr, 0.0)
    upper_violation = np.maximum(target_arr - upper, 0.0)
    max_violation_by_dof = np.maximum(lower_violation, upper_violation)
    signed_violation_by_dof = np.where(upper_violation > 0.0, upper_violation, -lower_violation)
    for name, indices in groups.items():
        if not indices:
            row[name] = {"max_violation": float("nan")}
            continue
        group_indices = np.asarray(indices, dtype=np.int64)
        group_violation = max_violation_by_dof[group_indices]
        local_max_index = int(np.argmax(group_violation))
        dof_index = int(indices[local_max_index])
        row[name] = {
            "max_violation": float(group_violation[local_max_index]),
            "max_violation_dof_index": dof_index,
            "max_violation_signed": float(signed_violation_by_dof[dof_index]),
            "target": float(target_arr[dof_index]),
            "lower": float(lower[dof_index]),
            "upper": float(upper[dof_index]),
        }
    return row


def _summarize_target_limit_violations(
    rows: list[dict[str, Any]],
    groups: dict[str, list[int]],
    dof_names: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sample_count": len(rows),
        "groups": {},
        "controller_ready": True,
        "status": "PASS_TARGETS_WITHIN_RUNTIME_LIMITS",
    }
    for name, indices in groups.items():
        group_rows = [row["groups"][name] for row in rows if name in row["groups"]]
        dof_group_names = [dof_names[i] for i in indices]
        if not group_rows:
            summary["groups"][name] = {"dof_names": dof_group_names, "sample_count": 0}
            continue
        violations = np.asarray([row["max_violation"] for row in group_rows], dtype=np.float64)
        max_row_index = int(np.nanargmax(violations))
        max_row = group_rows[max_row_index]
        source_row = [row for row in rows if name in row["groups"]][max_row_index]
        max_dof_index = int(max_row["max_violation_dof_index"])
        group_summary = {
            "dof_names": dof_group_names,
            "sample_count": int(len(group_rows)),
            "max_violation": float(np.nanmax(violations)),
            "max_violation_dof_name": dof_names[max_dof_index],
            "max_violation_signed": float(max_row["max_violation_signed"]),
            "max_violation_phase": source_row.get("phase"),
            "max_violation_step": source_row.get("step"),
            "target_at_max_violation": float(max_row["target"]),
            "lower_at_max_violation": float(max_row["lower"]),
            "upper_at_max_violation": float(max_row["upper"]),
        }
        if group_summary["max_violation"] > 0.0:
            summary["controller_ready"] = False
            summary["status"] = "FAIL_TARGET_OUTSIDE_RUNTIME_LIMITS"
        summary["groups"][name] = group_summary
    return summary


def _summarize_tracking_errors(
    tracking_rows: list[dict[str, Any]],
    groups: dict[str, list[int]],
    dof_names: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"sample_count": len(tracking_rows), "groups": {}}
    for name, indices in groups.items():
        group_rows = [row["groups"][name] for row in tracking_rows if name in row["groups"]]
        dof_group_names = [dof_names[i] for i in indices]
        if not group_rows:
            summary["groups"][name] = {"dof_names": dof_group_names, "sample_count": 0}
            continue
        max_abs = np.asarray([row["max_abs_error"] for row in group_rows], dtype=np.float64)
        rms = np.asarray([row["rms_error"] for row in group_rows], dtype=np.float64)
        max_row_index = int(np.nanargmax(max_abs))
        max_row = group_rows[max_row_index]
        source_row = [row for row in tracking_rows if name in row["groups"]][max_row_index]
        max_dof_index = int(max_row["max_abs_error_dof_index"])
        summary["groups"][name] = {
            "dof_names": dof_group_names,
            "sample_count": int(len(group_rows)),
            "max_abs_error": float(np.nanmax(max_abs)),
            "max_abs_error_dof_name": dof_names[max_dof_index],
            "max_abs_error_signed": float(max_row["max_abs_error_signed"]),
            "max_abs_error_phase": source_row.get("phase"),
            "max_abs_error_step": source_row.get("step"),
            "mean_max_abs_error": float(np.nanmean(max_abs)),
            "final_max_abs_error": float(max_abs[-1]),
            "max_rms_error": float(np.nanmax(rms)),
            "mean_rms_error": float(np.nanmean(rms)),
            "final_rms_error": float(rms[-1]),
        }
    return summary


def _controller_tracking_gate(
    *,
    tracking_summary: dict[str, Any],
    max_controlled_error: float | None,
) -> dict[str, Any]:
    controlled = (tracking_summary.get("groups") or {}).get("controlled") or {}
    max_error = controlled.get("max_abs_error")
    row: dict[str, Any] = {
        "threshold": max_controlled_error,
        "max_controlled_error": max_error,
        "max_controlled_error_dof_name": controlled.get("max_abs_error_dof_name"),
        "max_controlled_error_phase": controlled.get("max_abs_error_phase"),
        "max_controlled_error_step": controlled.get("max_abs_error_step"),
    }
    if max_controlled_error is None:
        row.update({"pass": True, "status": "SKIPPED_NO_TRACKING_THRESHOLD"})
        return row
    if max_error is None or not np.isfinite(float(max_error)):
        row.update({"pass": False, "status": "FAIL_TRACKING_ERROR_NOT_FINITE"})
        return row
    if float(max_error) <= float(max_controlled_error):
        row.update({"pass": True, "status": "PASS_POST_STEP_TRACKING_WITHIN_THRESHOLD"})
    else:
        row.update({"pass": False, "status": "FAIL_POST_STEP_TRACKING_EXCEEDS_THRESHOLD"})
    return row


def _safe_joint_velocities(art: Any) -> list[float] | None:
    try:
        values = art.get_joint_velocities()
    except Exception:
        return None
    if values is None:
        return None
    return np.asarray(values, dtype=np.float64).reshape(-1).tolist()


def _tracking_spike_packet(
    *,
    tracking_rows: list[dict[str, Any]],
    tracking_summary: dict[str, Any],
    contact_pair_rows: list[dict[str, Any]],
    dof_names: list[str],
    runtime_limits: np.ndarray,
    physics_dt: float,
    target_hold_steps: int,
    arm_gain_override: dict[str, float | None],
    finger_gain_override: dict[str, float | None],
    finger_dof_names: dict[str, str],
) -> dict[str, Any]:
    controlled = (tracking_summary.get("groups") or {}).get("controlled") or {}
    phase = controlled.get("max_abs_error_phase")
    step = controlled.get("max_abs_error_step")
    dof_name = controlled.get("max_abs_error_dof_name")
    row = next((r for r in tracking_rows if r.get("phase") == phase and r.get("step") == step), None)
    packet: dict[str, Any] = {
        "status": "FOUND_TRACKING_SPIKE" if row is not None and dof_name in dof_names else "MISSING_TRACKING_SPIKE_ROW",
        "phase": phase,
        "step": step,
        "dof_name": dof_name,
        "max_abs_error": controlled.get("max_abs_error"),
        "max_abs_error_signed": controlled.get("max_abs_error_signed"),
        "physics_dt": float(physics_dt),
        "target_hold_steps": int(target_hold_steps),
        "effective_target_dt": float(physics_dt) * float(target_hold_steps),
    }
    if row is None or dof_name not in dof_names:
        return packet

    dof_index = int(dof_names.index(dof_name))

    def at(name: str) -> float | None:
        values = row.get(name)
        if values is None:
            return None
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if dof_index >= arr.size:
            return None
        return float(arr[dof_index])

    target = at("target")
    previous_target = at("previous_target")
    next_target = at("next_target")
    pre_qpos = at("pre_qpos")
    post_qpos = at("post_qpos")
    qvel = at("qvel")
    target_dt = float(packet["effective_target_dt"])
    target_delta = None if target is None or previous_target is None else float(target - previous_target)
    next_target_delta = None if target is None or next_target is None else float(next_target - target)
    actual_delta = None if pre_qpos is None or post_qpos is None else float(post_qpos - pre_qpos)
    tracking_ratio = (
        None
        if target_delta is None or actual_delta is None or abs(target_delta) <= 1e-12
        else float(actual_delta / target_delta)
    )
    packet.update(
        {
            "dof_index": dof_index,
            "target": target,
            "previous_target": previous_target,
            "next_target": next_target,
            "target_delta_from_previous": target_delta,
            "next_target_delta": next_target_delta,
            "estimated_target_velocity": None if target_delta is None else float(target_delta / target_dt),
            "pre_step_qpos": pre_qpos,
            "post_step_qpos": post_qpos,
            "actual_delta_during_hold": actual_delta,
            "estimated_actual_velocity_during_hold": None if actual_delta is None else float(actual_delta / target_dt),
            "tracking_ratio": tracking_ratio,
            "qvel_after_hold": qvel,
            "runtime_limit": (
                {
                    "lower": float(runtime_limits[dof_index, 0]),
                    "upper": float(runtime_limits[dof_index, 1]),
                }
                if runtime_limits is not None and dof_index < runtime_limits.shape[0]
                else None
            ),
            "gain_override": (
                finger_gain_override
                if dof_name in {finger_dof_names["left_finger"], finger_dof_names["right_finger"]}
                else arm_gain_override
            ),
        }
    )
    step_contacts = [r for r in contact_pair_rows if r.get("phase") == phase and r.get("step") == step]
    packet["contact_pair_count_at_step"] = len(step_contacts)
    packet["contact_categories_at_step"] = sorted({str(r.get("category")) for r in step_contacts if r.get("category")})
    packet["contact_pairs_sample_at_step"] = [
        {
            "type_name": r.get("type_name"),
            "category": r.get("category"),
            "collider0": r.get("collider0"),
            "collider1": r.get("collider1"),
        }
        for r in step_contacts[:8]
    ]
    return packet


def _percentile_summary(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"p50": None, "p90": None, "p95": None, "p99": None, "max": None, "mean": None}
    return {
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _cluster_command_velocity_spikes(
    spike_rows: list[dict[str, Any]],
    *,
    cluster_gap_steps: int = 3,
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    by_dof: dict[str, list[dict[str, Any]]] = {}
    for row in spike_rows:
        by_dof.setdefault(str(row.get("dof_name")), []).append(row)
    for dof_name, rows in by_dof.items():
        ordered = sorted(rows, key=lambda row: int(row.get("step") or 0))
        current: list[dict[str, Any]] = []
        previous_step: int | None = None
        for row in ordered:
            step = int(row.get("step") or 0)
            if previous_step is None or step - previous_step <= int(cluster_gap_steps):
                current.append(row)
            else:
                clusters.append(_command_velocity_spike_cluster_row(dof_name, current))
                current = [row]
            previous_step = step
        if current:
            clusters.append(_command_velocity_spike_cluster_row(dof_name, current))
    return sorted(
        clusters,
        key=lambda row: float(row.get("largest_abs_target_velocity") or 0.0),
        reverse=True,
    )


def _command_velocity_spike_cluster_row(dof_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    steps = [int(row.get("step") or 0) for row in rows]
    peak = max(rows, key=lambda row: float(row.get("abs_target_velocity") or 0.0))
    return {
        "dof_name": dof_name,
        "cluster_start_step": min(steps),
        "cluster_end_step": max(steps),
        "length_steps": len(rows),
        "spike_steps": steps,
        "largest_abs_target_velocity": float(peak.get("abs_target_velocity") or 0.0),
        "largest_target_velocity": float(peak.get("target_velocity") or 0.0),
        "peak_step": int(peak.get("step") or 0),
    }


def _command_delta_distribution(
    *,
    tracking_rows: list[dict[str, Any]],
    groups: dict[str, list[int]],
    dof_names: list[str],
    effective_target_dt: float,
    max_abs_target_velocity: float | None,
    top_n: int = 20,
) -> dict[str, Any]:
    controlled_indices = groups.get("controlled") or []
    rows = [row for row in tracking_rows if row.get("phase") == "close" and row.get("target") is not None]
    report: dict[str, Any] = {
        "status": "SKIPPED_NO_CLOSE_TRACKING_ROWS" if not rows else "DIAGNOSTIC_ONLY_NO_THRESHOLD",
        "pass": True,
        "classification": "INSUFFICIENT_DATA" if not rows else "DIAGNOSTIC_ONLY_NO_THRESHOLD",
        "recommendation": "REVIEW_HDF5_TARGET_SOURCE" if not rows else "REPORT_ONLY_NO_BLOCKING_THRESHOLD",
        "mode": "reporting_only" if max_abs_target_velocity is None else "blocking_tuning",
        "formal_replay_targets_modified": False,
        "deleted_frames": 0,
        "smoothed_frames": 0,
        "interpolated_frames": 0,
        "diagnostic_only_transforms": [],
        "threshold_abs_target_velocity": max_abs_target_velocity,
        "spike_threshold_rad_s": max_abs_target_velocity,
        "spike_cluster_gap_steps": 3,
        "spike_count": 0,
        "spike_steps": [],
        "spike_clusters": [],
        "effective_target_dt": float(effective_target_dt),
        "effective_target_rate_hz": float(1.0 / effective_target_dt) if effective_target_dt > 0 else None,
        "sample_count": len(rows),
        "dofs": {},
        "top_target_velocity_spikes": [],
        "top_tracking_error_spikes": [],
        "notes": (
            "This diagnostic describes the formal 50 Hz replay command sequence. It does not change replay "
            "targets and is not a smoothing/interpolation mode."
        ),
    }
    if not rows or effective_target_dt <= 0:
        return report

    velocity_spikes: list[dict[str, Any]] = []
    tracking_spikes: list[dict[str, Any]] = []
    any_threshold_failure = False
    for dof_index in controlled_indices:
        dof_name = dof_names[int(dof_index)]
        target_delta_values: list[float] = []
        target_velocity_values: list[float] = []
        actual_velocity_values: list[float] = []
        qvel_values: list[float] = []
        tracking_ratio_values: list[float] = []
        tracking_error_values: list[float] = []
        for row in rows:
            target_arr = np.asarray(row["target"], dtype=np.float64).reshape(-1)
            previous_arr = np.asarray(row["previous_target"], dtype=np.float64).reshape(-1)
            pre_arr = np.asarray(row["pre_qpos"], dtype=np.float64).reshape(-1)
            post_arr = np.asarray(row["post_qpos"], dtype=np.float64).reshape(-1)
            target_delta = float(target_arr[dof_index] - previous_arr[dof_index])
            target_velocity = float(target_delta / effective_target_dt)
            actual_delta = float(post_arr[dof_index] - pre_arr[dof_index])
            actual_velocity = float(actual_delta / effective_target_dt)
            tracking_error = float(post_arr[dof_index] - target_arr[dof_index])
            tracking_ratio = float(actual_delta / target_delta) if abs(target_delta) > 1e-12 else float("nan")
            qvel_row = row.get("qvel")
            qvel_value = (
                float(np.asarray(qvel_row, dtype=np.float64).reshape(-1)[dof_index])
                if qvel_row is not None
                else float("nan")
            )
            target_delta_values.append(abs(target_delta))
            target_velocity_values.append(abs(target_velocity))
            actual_velocity_values.append(abs(actual_velocity))
            qvel_values.append(abs(qvel_value))
            tracking_ratio_values.append(tracking_ratio)
            tracking_error_values.append(abs(tracking_error))
            velocity_spikes.append(
                {
                    "phase": row.get("phase"),
                    "step": row.get("step"),
                    "dof_name": dof_name,
                    "target_delta": target_delta,
                    "abs_target_delta": abs(target_delta),
                    "target_velocity": target_velocity,
                    "abs_target_velocity": abs(target_velocity),
                    "actual_velocity": actual_velocity,
                    "tracking_ratio": tracking_ratio if np.isfinite(tracking_ratio) else None,
                    "tracking_error": tracking_error,
                }
            )
            tracking_spikes.append(
                {
                    "phase": row.get("phase"),
                    "step": row.get("step"),
                    "dof_name": dof_name,
                    "tracking_error": tracking_error,
                    "abs_tracking_error": abs(tracking_error),
                    "target_delta": target_delta,
                    "target_velocity": target_velocity,
                    "actual_velocity": actual_velocity,
                    "tracking_ratio": tracking_ratio if np.isfinite(tracking_ratio) else None,
                }
            )

        target_velocity_arr = np.asarray(target_velocity_values, dtype=np.float64)
        if max_abs_target_velocity is not None and np.nanmax(target_velocity_arr) > float(max_abs_target_velocity):
            any_threshold_failure = True
        report["dofs"][dof_name] = {
            "abs_target_delta": _percentile_summary(np.asarray(target_delta_values, dtype=np.float64)),
            "abs_target_velocity": _percentile_summary(target_velocity_arr),
            "abs_actual_velocity": _percentile_summary(np.asarray(actual_velocity_values, dtype=np.float64)),
            "abs_qvel_after_hold": _percentile_summary(np.asarray(qvel_values, dtype=np.float64)),
            "tracking_ratio": _percentile_summary(np.asarray(tracking_ratio_values, dtype=np.float64)),
            "abs_tracking_error": _percentile_summary(np.asarray(tracking_error_values, dtype=np.float64)),
            "tracking_error_sign_flip_count": int(
                np.sum(
                    np.diff(
                        np.sign(
                            np.asarray(
                                [row["post_qpos"][dof_index] - row["target"][dof_index] for row in rows],
                                dtype=np.float64,
                            )
                        )
                    )
                    != 0
                )
            ),
        }

    report["top_target_velocity_spikes"] = sorted(
        velocity_spikes, key=lambda row: float(row["abs_target_velocity"]), reverse=True
    )[:top_n]
    report["top_tracking_error_spikes"] = sorted(
        tracking_spikes, key=lambda row: float(row["abs_tracking_error"]), reverse=True
    )[:top_n]
    if max_abs_target_velocity is not None:
        spike_rows = [
            row
            for row in velocity_spikes
            if float(row["abs_target_velocity"]) > float(max_abs_target_velocity)
        ]
        spike_clusters = _cluster_command_velocity_spikes(spike_rows, cluster_gap_steps=3)
        largest_cluster = spike_clusters[0] if spike_clusters else {}
        report["spike_count"] = len(spike_rows)
        report["spike_steps"] = sorted({int(row.get("step") or 0) for row in spike_rows})
        report["spike_clusters"] = spike_clusters
        report["largest_cluster_start"] = largest_cluster.get("cluster_start_step")
        report["largest_cluster_end"] = largest_cluster.get("cluster_end_step")
        report["largest_cluster_length_steps"] = largest_cluster.get("length_steps")
        report["largest_cluster_length_seconds"] = (
            float(largest_cluster["length_steps"]) * float(effective_target_dt)
            if largest_cluster.get("length_steps") is not None
            else None
        )
        report["largest_cluster_peak_velocity"] = largest_cluster.get("largest_target_velocity")
        report["pass"] = not any_threshold_failure
        if report["pass"]:
            report["status"] = "PASS_COMMAND_TARGET_VELOCITY_WITHIN_THRESHOLD"
            report["classification"] = "COMMAND_SMOOTHNESS_PASS"
            report["recommendation"] = "ALLOW_DRIVE_PROFILE_TUNING"
        else:
            report["status"] = "FAIL_COMMAND_TARGET_VELOCITY_EXCEEDS_THRESHOLD"
            report["classification"] = (
                "REPEATED_SPIKE_CLUSTER"
                if len(spike_rows) > 1 or any(int(row.get("length_steps") or 0) > 1 for row in spike_clusters)
                else "SINGLE_SPIKE_RESIDUAL"
            )
            report["recommendation"] = "BLOCK_CCD_FIX_COMMAND_CONTINUITY_FIRST"
    return report


def _runtime_drive_profile(
    *,
    dof_names: list[str],
    groups: dict[str, list[int]],
    runtime_limits: np.ndarray,
    stiffness: list[float | None],
    damping: list[float | None],
    max_efforts: list[float | None],
    max_velocities: list[float | None],
    profile_name: str,
    profile_provenance: str,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for dof_index in groups.get("controlled") or []:
        idx = int(dof_index)
        rows[dof_names[idx]] = {
            "dof_index": idx,
            "runtime_limit": {
                "lower": float(runtime_limits[idx, 0]),
                "upper": float(runtime_limits[idx, 1]),
            },
            "stiffness": stiffness[idx] if idx < len(stiffness) else None,
            "damping": damping[idx] if idx < len(damping) else None,
            "max_effort": max_efforts[idx] if idx < len(max_efforts) else None,
            "max_velocity": max_velocities[idx] if idx < len(max_velocities) else None,
        }
    return {
        "profile_name": profile_name,
        "profile_provenance": profile_provenance,
        "controlled_dofs": rows,
    }


def _drive_authority_audit(
    *,
    tracking_spike: dict[str, Any],
    runtime_drive_profile: dict[str, Any],
) -> dict[str, Any]:
    dof_name = tracking_spike.get("dof_name")
    drive_row = (runtime_drive_profile.get("controlled_dofs") or {}).get(str(dof_name), {})
    kp = drive_row.get("stiffness")
    kd = drive_row.get("damping")
    max_effort = drive_row.get("max_effort")
    target = tracking_spike.get("target")
    pre_qpos = tracking_spike.get("pre_step_qpos")
    post_qpos = tracking_spike.get("post_step_qpos")
    qvel = tracking_spike.get("qvel_after_hold")
    pre_error = None if target is None or pre_qpos is None else float(target - pre_qpos)
    post_error = None if target is None or post_qpos is None else float(target - post_qpos)
    spring_term = None if kp is None or pre_error is None else float(float(kp) * pre_error)
    damping_term = None if kd is None or qvel is None else float(float(kd) * float(qvel))
    net_demand = None if spring_term is None or damping_term is None else float(spring_term - damping_term)
    clipped = None
    if net_demand is not None and max_effort is not None and np.isfinite(float(max_effort)):
        clipped = bool(abs(net_demand) > abs(float(max_effort)))
    return {
        "enabled": True,
        "status": "DRIVE_AUTHORITY_AUDIT_REPORTED",
        "profile_name": runtime_drive_profile.get("profile_name"),
        "profile_provenance": runtime_drive_profile.get("profile_provenance"),
        "spike_dof": dof_name,
        "spike_phase": tracking_spike.get("phase"),
        "spike_step": tracking_spike.get("step"),
        "drive": drive_row,
        "target": target,
        "pre_step_qpos": pre_qpos,
        "post_step_qpos": post_qpos,
        "qvel_after_hold": qvel,
        "pre_step_position_error": pre_error,
        "post_step_position_error": post_error,
        "target_delta_from_previous": tracking_spike.get("target_delta_from_previous"),
        "actual_delta_during_hold": tracking_spike.get("actual_delta_during_hold"),
        "estimated_target_velocity": tracking_spike.get("estimated_target_velocity"),
        "estimated_actual_velocity_during_hold": tracking_spike.get("estimated_actual_velocity_during_hold"),
        "tracking_ratio": tracking_spike.get("tracking_ratio"),
        "estimated_spring_term": spring_term,
        "estimated_damping_term": damping_term,
        "estimated_net_drive_demand": net_demand,
        "estimated_effort_clipped": clipped,
        "notes": (
            "This is a first-order position-drive demand estimate using the reported runtime gains and the "
            "spike state. It is not a measured motor torque."
        ),
    }


def _non_target_contact_gate(
    *,
    contact_summary: dict[str, Any],
    fail_on_non_target: bool,
    allowed_categories: list[str],
) -> dict[str, Any]:
    categories = list(contact_summary.get("non_target_object_contact_categories") or [])
    allowed = set(allowed_categories)
    blocking = sorted(category for category in categories if category not in allowed)
    if not fail_on_non_target:
        return {
            "pass": True,
            "status": "SKIPPED_NON_TARGET_CONTACT_GATE",
            "allowed_categories": sorted(allowed),
            "observed_categories": categories,
            "blocking_categories": blocking,
        }
    if blocking:
        return {
            "pass": False,
            "status": "FAIL_NON_TARGET_OBJECT_CONTACT",
            "allowed_categories": sorted(allowed),
            "observed_categories": categories,
            "blocking_categories": blocking,
        }
    return {
        "pass": True,
        "status": "PASS_NON_TARGET_CONTACTS_ALLOWED",
        "allowed_categories": sorted(allowed),
        "observed_categories": categories,
        "blocking_categories": [],
    }


def _load_workcell_contact_policy(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    policy_path = Path(path)
    data = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    rules = data.get("rules") or []
    if not isinstance(rules, list):
        raise ValueError("workcell contact policy must contain a list field named rules")
    normalized_rules: list[dict[str, Any]] = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"workcell contact policy rule {index} must be a mapping")
        prefix = str(rule.get("path_prefix") or "")
        if not prefix.startswith("/"):
            raise ValueError(f"workcell contact policy rule {index} path_prefix must be an absolute prim path")
        decision = str(rule.get("decision") or "deny")
        if decision not in {"allow", "deny"}:
            raise ValueError(f"workcell contact policy rule {index} decision must be allow or deny")
        normalized_rules.append(
            {
                "path_prefix": prefix.rstrip("/") or "/",
                "semantic_class": str(rule.get("semantic_class") or "unknown_workcell_collision"),
                "decision": decision,
                "notes": str(rule.get("notes") or ""),
            }
        )
    return {
        "path": _rel(policy_path),
        "default_decision": str(data.get("default_decision") or "deny"),
        "rules": sorted(normalized_rules, key=lambda item: len(item["path_prefix"]), reverse=True),
    }


def _other_path_from_unique_object_pair(pair: list[str], object_path: str) -> str | None:
    if len(pair) != 2:
        return None
    left, right = str(pair[0]), str(pair[1])
    left_is_object = _path_matches(left, object_path)
    right_is_object = _path_matches(right, object_path)
    if left_is_object and not right_is_object:
        return right
    if right_is_object and not left_is_object:
        return left
    return None


def _pair_summary_for_unique_object_pair(
    payload: dict[str, Any],
    pair: list[str],
) -> dict[str, Any] | None:
    pair_list = list(pair)
    for summary in payload.get("unique_contact_pair_summaries") or []:
        if list(summary.get("pair") or []) == pair_list:
            return dict(summary)
    return None


def _match_workcell_contact_rule(other_path: str, policy: dict[str, Any]) -> dict[str, Any]:
    for rule in policy.get("rules") or []:
        prefix = str(rule["path_prefix"])
        if other_path == prefix or other_path.startswith(prefix + "/"):
            return dict(rule)
    return {
        "path_prefix": None,
        "semantic_class": "unknown_workcell_collision",
        "decision": str(policy.get("default_decision") or "deny"),
        "notes": "No explicit workcell contact policy rule matched this path.",
    }


def _workcell_contact_policy_gate(
    *,
    contact_summary: dict[str, Any],
    object_path: str,
    policy: dict[str, Any] | None,
) -> dict[str, Any]:
    if policy is None:
        return {"pass": True, "status": "SKIPPED_NO_WORKCELL_CONTACT_POLICY", "policy": None, "rows": []}
    rows: list[dict[str, Any]] = []
    categories = contact_summary.get("object_contact_categories") or {}
    for category, payload in sorted(categories.items()):
        if category == "target_finger":
            continue
        for pair in payload.get("unique_contact_pairs") or []:
            other_path = _other_path_from_unique_object_pair(pair, object_path)
            if other_path is None:
                continue
            rule = _match_workcell_contact_rule(other_path, policy)
            pair_summary = _pair_summary_for_unique_object_pair(payload, pair) or {}
            rows.append(
                {
                    "category": category,
                    "other_path": other_path,
                    "semantic_class": rule["semantic_class"],
                    "decision": rule["decision"],
                    "matched_path_prefix": rule["path_prefix"],
                    "notes": rule.get("notes", ""),
                    "category_contact_pair_count": payload.get("contact_pair_count"),
                    "category_phase_counts": payload.get("phase_counts"),
                    "category_first_contact_pair": payload.get("first_contact_pair"),
                    "pair_contact_pair_count": pair_summary.get("contact_pair_count"),
                    "pair_phase_counts": pair_summary.get("phase_counts"),
                    "pair_first_contact_pair": pair_summary.get("first_contact_pair"),
                }
            )
    denied_rows = [row for row in rows if row["decision"] == "deny"]
    return {
        "pass": not denied_rows,
        "status": "PASS_WORKCELL_CONTACT_POLICY" if not denied_rows else "FAIL_WORKCELL_CONTACT_POLICY",
        "policy": {"path": policy.get("path"), "default_decision": policy.get("default_decision")},
        "rows": rows,
        "denied_rows": denied_rows,
        "denied_semantic_classes": sorted({row["semantic_class"] for row in denied_rows}),
    }


def _active_target_contact_gate(
    *,
    contact_summary: dict[str, Any],
    require_active_target_contact: bool,
    already_in_contact_setup: bool,
) -> dict[str, Any]:
    found_phases = list(contact_summary.get("target_contact_found_phases") or [])
    row: dict[str, Any] = {
        "required": bool(require_active_target_contact),
        "already_in_contact_setup": bool(already_in_contact_setup),
        "active_phases": ["close", "post_close_hold"],
        "observed_target_contact_found_phases": found_phases,
        "first_target_contact_phase": contact_summary.get("first_target_contact_phase"),
        "first_target_contact_found_phase": contact_summary.get("first_target_contact_found_phase"),
    }
    if already_in_contact_setup:
        row.update({"pass": True, "status": "SKIPPED_ALREADY_IN_CONTACT_SETUP"})
        return row
    if not require_active_target_contact:
        row.update({"pass": True, "status": "SKIPPED_ACTIVE_TARGET_CONTACT_GATE"})
        return row
    first_found_phase = contact_summary.get("first_target_contact_found_phase")
    found_during_close = bool(contact_summary.get("target_contact_found_during_close"))
    found_during_post_close_hold = bool(contact_summary.get("target_contact_found_during_post_close_hold"))
    if first_found_phase in {"close", "post_close_hold"}:
        row.update(
            {
                "pass": True,
                "status": "PASS_ACTIVE_TARGET_CONTACT_FOUND_DURING_CLOSE_OR_POST_CLOSE_HOLD",
            }
        )
    elif found_during_close:
        row.update({"pass": False, "status": "FAIL_TARGET_ALREADY_CONTACTING_BEFORE_CLOSE"})
    elif found_during_post_close_hold:
        row.update({"pass": False, "status": "FAIL_TARGET_ALREADY_CONTACTING_BEFORE_POST_CLOSE_HOLD"})
    else:
        row.update({"pass": False, "status": "FAIL_NO_ACTIVE_TARGET_CONTACT_DURING_CLOSE_OR_POST_CLOSE_HOLD"})
    return row


def _box_projection_interval(box: dict[str, Any], unit: np.ndarray) -> tuple[float, float] | None:
    if not (box.get("bbox_valid") and box.get("center") is not None and box.get("size") is not None):
        return None
    center = np.asarray(box["center"], dtype=np.float64).reshape(3)
    half_size = np.asarray(box["size"], dtype=np.float64).reshape(3) * 0.5
    unit = np.asarray(unit, dtype=np.float64).reshape(3)
    projected_center = float(np.dot(center, unit))
    projected_radius = float(np.dot(np.abs(unit), half_size))
    return projected_center - projected_radius, projected_center + projected_radius


def _box_oriented_projection_interval(box: dict[str, Any], unit: np.ndarray) -> tuple[float, float] | None:
    """Project a bbox row with its local oriented axes when that metadata exists.

    `_bbox_row()` historically returns world AABBs.  That is conservative for
    rendering diagnostics, but it can be too pessimistic for finger-pad gap
    checks when the proxy is rotated relative to the world axes.  Some rows also
    carry an authored world transform; in that case use the local box support
    function instead of the world AABB support.
    """

    if not (box.get("bbox_valid") and box.get("center") is not None and box.get("size") is not None):
        return None
    transform = (
        box.get("oriented_world_matrix")
        or box.get("world_transform")
        or box.get("transform")
        or box.get("matrix_world")
    )
    if transform is None:
        return _box_projection_interval(box, unit)
    try:
        matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    except Exception:
        return _box_projection_interval(box, unit)
    unit = np.asarray(unit, dtype=np.float64).reshape(3)
    unit_norm = float(np.linalg.norm(unit))
    if unit_norm <= 1e-12 or not np.isfinite(unit_norm):
        return None
    unit = unit / unit_norm
    center = np.asarray(box["center"], dtype=np.float64).reshape(3)
    size_key = "oriented_size" if box.get("oriented_size") is not None else "size"
    half_size = np.asarray(box[size_key], dtype=np.float64).reshape(3) * 0.5
    axes = [np.asarray(matrix[:3, i], dtype=np.float64).reshape(3) for i in range(3)]
    axis_norms = [float(np.linalg.norm(axis)) for axis in axes]
    if any(norm <= 1e-12 or not np.isfinite(norm) for norm in axis_norms):
        return _box_projection_interval(box, unit)
    projected_center = float(np.dot(center, unit))
    # Preserve the transform column scale. USD BBox3d stores an oriented local
    # box plus a matrix; for scaled cube proxies the local box can be unit-sized
    # while the scale lives in the matrix columns. Normalizing the axes would
    # inflate a 12-35 mm finger pad into a 1 m support interval.
    projected_radius = float(sum(abs(float(np.dot(axis, unit))) * half for axis, half in zip(axes, half_size)))
    return projected_center - projected_radius, projected_center + projected_radius


def _ordered_inner_gap(
    *,
    lower_box: dict[str, Any],
    upper_box: dict[str, Any],
    object_box: dict[str, Any],
    axis: int,
) -> dict[str, Any]:
    lower_inner = float(lower_box["max"][axis])
    upper_inner = float(upper_box["min"][axis])
    object_min = float(object_box["min"][axis])
    object_max = float(object_box["max"][axis])
    return {
        "finger_inner_gap_m": float(upper_inner - lower_inner),
        "object_inside_inner_gap": bool(object_min >= lower_inner and object_max <= upper_inner),
        "object_gap_to_lower_finger_m": float(object_min - lower_inner),
        "object_gap_to_upper_finger_m": float(upper_inner - object_max),
        "lower_inner_surface_m": lower_inner,
        "upper_inner_surface_m": upper_inner,
    }


def _projected_inner_gap(
    *,
    lower_box: dict[str, Any],
    upper_box: dict[str, Any],
    object_box: dict[str, Any],
    unit: np.ndarray,
    use_oriented_finger_boxes: bool = False,
) -> dict[str, Any]:
    interval_fn = _box_oriented_projection_interval if use_oriented_finger_boxes else _box_projection_interval
    lower_interval = interval_fn(lower_box, unit)
    upper_interval = interval_fn(upper_box, unit)
    object_interval = _box_projection_interval(object_box, unit)
    if lower_interval is None or upper_interval is None or object_interval is None:
        return {"valid": False}
    lower_inner = float(lower_interval[1])
    upper_inner = float(upper_interval[0])
    object_min = float(object_interval[0])
    object_max = float(object_interval[1])
    return {
        "valid": True,
        "finger_inner_gap_m": float(upper_inner - lower_inner),
        "object_inside_inner_gap": bool(object_min >= lower_inner and object_max <= upper_inner),
        "object_gap_to_lower_finger_m": float(object_min - lower_inner),
        "object_gap_to_upper_finger_m": float(upper_inner - object_max),
        "lower_inner_surface_m": lower_inner,
        "upper_inner_surface_m": upper_inner,
        "object_interval_m": [object_min, object_max],
    }


def _projected_inner_gap_for_interval(
    *,
    lower_box: dict[str, Any],
    upper_box: dict[str, Any],
    object_interval: tuple[float, float],
    unit: np.ndarray,
    use_oriented_finger_boxes: bool = False,
) -> dict[str, Any]:
    """Evaluate a finger inner gap against an oriented/proxy-aware object interval."""

    interval_fn = _box_oriented_projection_interval if use_oriented_finger_boxes else _box_projection_interval
    lower_interval = interval_fn(lower_box, unit)
    upper_interval = interval_fn(upper_box, unit)
    if lower_interval is None or upper_interval is None:
        return {"valid": False}
    lower_inner = float(lower_interval[1])
    upper_inner = float(upper_interval[0])
    object_min = float(object_interval[0])
    object_max = float(object_interval[1])
    return {
        "valid": True,
        "finger_interval_source": "oriented_box_support" if use_oriented_finger_boxes else "world_aabb",
        "finger_inner_gap_m": float(upper_inner - lower_inner),
        "object_inside_inner_gap": bool(object_min >= lower_inner and object_max <= upper_inner),
        "object_gap_to_lower_finger_m": float(object_min - lower_inner),
        "object_gap_to_upper_finger_m": float(upper_inner - object_max),
        "lower_inner_surface_m": lower_inner,
        "upper_inner_surface_m": upper_inner,
        "object_interval_m": [object_min, object_max],
    }


def _oriented_cylinder_projection_model(
    *,
    object_box: dict[str, Any],
    object_axis_unit_world: list[float] | tuple[float, float, float] | np.ndarray,
    projection_unit_world: list[float] | tuple[float, float, float] | np.ndarray,
    radius_m: float,
    half_length_m: float,
    source: str,
) -> dict[str, Any]:
    """Project a cylinder/capsule-like proxy without using its rotated world AABB.

    A world AABB overestimates rotated bottles because it projects part of the
    long axis into the closing direction.  PhysX still collides with the authored
    cylinder proxy, so the fixed-pose Gate2 geometry check should use the
    oriented cylinder support radius along the true closing axis.
    """

    row: dict[str, Any] = {
        "valid": False,
        "source": source,
        "shape_model": "oriented_cylinder_support",
    }
    if not object_box.get("bbox_valid") or object_box.get("center") is None:
        row["status"] = "FAIL_OBJECT_BBOX_INVALID"
        return row
    axis = np.asarray(object_axis_unit_world, dtype=np.float64).reshape(3)
    unit = np.asarray(projection_unit_world, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    unit_norm = float(np.linalg.norm(unit))
    radius = float(radius_m)
    half_length = float(half_length_m)
    if (
        axis_norm <= 1e-12
        or unit_norm <= 1e-12
        or not np.isfinite(axis_norm)
        or not np.isfinite(unit_norm)
        or radius <= 0.0
        or half_length <= 0.0
        or not np.isfinite(radius)
        or not np.isfinite(half_length)
    ):
        row["status"] = "FAIL_INVALID_PROJECTION_MODEL_INPUT"
        return row
    axis = axis / axis_norm
    unit = unit / unit_norm
    center = np.asarray(object_box["center"], dtype=np.float64).reshape(3)
    axis_dot = float(np.dot(axis, unit))
    radial_component = float(np.sqrt(max(0.0, 1.0 - axis_dot * axis_dot)))
    projected_radius = abs(axis_dot) * half_length + radius * radial_component
    center_projection = float(np.dot(center, unit))
    row.update(
        {
            "valid": True,
            "status": "PASS_ORIENTED_CYLINDER_PROJECTION_MODEL",
            "center_world_m": center.tolist(),
            "axis_unit_world": axis.tolist(),
            "projection_unit_world": unit.tolist(),
            "axis_dot_projection_abs": abs(axis_dot),
            "radius_m": radius,
            "half_length_m": half_length,
            "radial_component": radial_component,
            "projected_radius_m": float(projected_radius),
            "projected_width_m": float(projected_radius * 2.0),
            "center_projection_m": center_projection,
            "object_interval_m": [center_projection - projected_radius, center_projection + projected_radius],
            "notes": (
                "This interval is for the authored cylinder-like contact proxy. It avoids treating the rotated "
                "world AABB as the physical bottle width."
            ),
        }
    )
    return row


def _oriented_box_projection_model(
    *,
    object_box: dict[str, Any],
    projection_unit_world: list[float] | tuple[float, float, float] | np.ndarray,
    source: str,
) -> dict[str, Any]:
    """Project a cuboid contact coupon using its authored oriented box support."""

    row: dict[str, Any] = {
        "valid": False,
        "source": source,
        "shape_model": "oriented_box_support",
    }
    unit = np.asarray(projection_unit_world, dtype=np.float64).reshape(3)
    unit_norm = float(np.linalg.norm(unit))
    if unit_norm <= 1e-12 or not np.isfinite(unit_norm):
        row["status"] = "FAIL_INVALID_PROJECTION_UNIT"
        return row
    unit = unit / unit_norm
    interval = _box_oriented_projection_interval(object_box, unit)
    if interval is None:
        row["status"] = "FAIL_ORIENTED_BOX_INTERVAL_INVALID"
        return row
    center = np.asarray(object_box["center"], dtype=np.float64).reshape(3)
    projected_center = (float(interval[0]) + float(interval[1])) * 0.5
    projected_radius = (float(interval[1]) - float(interval[0])) * 0.5
    row.update(
        {
            "valid": True,
            "status": "PASS_ORIENTED_BOX_PROJECTION_MODEL",
            "center_world_m": center.tolist(),
            "projection_unit_world": unit.tolist(),
            "projected_radius_m": float(projected_radius),
            "projected_width_m": float(projected_radius * 2.0),
            "center_projection_m": float(projected_center),
            "object_interval_m": [float(interval[0]), float(interval[1])],
            "notes": (
                "This interval is for a local cuboid grasp coupon. It uses the authored oriented box support "
                "instead of the rotated world AABB."
            ),
        }
    )
    return row


def _closing_axis_gap_centering_solver(
    *,
    lower_box: dict[str, Any],
    upper_box: dict[str, Any],
    object_projection_model: dict[str, Any],
    projection_unit_world: list[float] | tuple[float, float, float] | np.ndarray,
    clearance: float,
    use_oriented_finger_boxes: bool = False,
) -> dict[str, Any]:
    """Compute a reset-time horizontal shift that centers the object inside the true finger gap."""

    row: dict[str, Any] = {
        "enabled": True,
        "provenance": "FORMAL_FIXED_POSE_CLOSING_AXIS_PLACEMENT_SOLVER",
        "pass": False,
        "applied": False,
    }
    unit = np.asarray(projection_unit_world, dtype=np.float64).reshape(3)
    unit_norm = float(np.linalg.norm(unit))
    if unit_norm <= 1e-12 or not np.isfinite(unit_norm):
        row["status"] = "FAIL_INVALID_CLOSING_AXIS"
        return row
    unit = unit / unit_norm
    if not object_projection_model.get("valid"):
        row["status"] = "FAIL_OBJECT_PROJECTION_MODEL_INVALID"
        row["object_projection_model"] = object_projection_model
        return row
    object_interval = tuple(float(v) for v in object_projection_model["object_interval_m"])
    gap = _projected_inner_gap_for_interval(
        lower_box=lower_box,
        upper_box=upper_box,
        object_interval=object_interval,
        unit=unit,
        use_oriented_finger_boxes=use_oriented_finger_boxes,
    )
    row["gap_before"] = gap
    if not gap.get("valid"):
        row["status"] = "FAIL_FINGER_INNER_GAP_INVALID"
        return row
    lower_inner = float(gap["lower_inner_surface_m"])
    upper_inner = float(gap["upper_inner_surface_m"])
    projected_radius = float(object_projection_model["projected_radius_m"])
    available_gap = upper_inner - lower_inner
    required_gap = projected_radius * 2.0 + 2.0 * float(clearance)
    row.update(
        {
            "available_inner_gap_m": float(available_gap),
            "required_inner_gap_m": float(required_gap),
            "clearance_m": float(clearance),
            "projected_object_width_m": float(projected_radius * 2.0),
        }
    )
    if required_gap > available_gap:
        row["status"] = "FAIL_CLOSING_AXIS_INNER_GAP_INFEASIBLE"
        row["shortfall_m"] = float(required_gap - available_gap)
        return row
    target_center_projection = (lower_inner + upper_inner) * 0.5
    current_center_projection = float(object_projection_model["center_projection_m"])
    delta_projection = target_center_projection - current_center_projection
    horizontal = np.asarray([unit[0], unit[1], 0.0], dtype=np.float64)
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm <= 1e-12 or not np.isfinite(horizontal_norm):
        row["status"] = "FAIL_CLOSING_AXIS_HAS_NO_HORIZONTAL_COMPONENT"
        return row
    horizontal_unit = horizontal / horizontal_norm
    projection_per_meter = float(np.dot(horizontal_unit, unit))
    if abs(projection_per_meter) <= 1e-12 or not np.isfinite(projection_per_meter):
        row["status"] = "FAIL_HORIZONTAL_SHIFT_CANNOT_CHANGE_CLOSING_PROJECTION"
        return row
    delta_world = horizontal_unit * (delta_projection / projection_per_meter)
    centered_interval = (
        target_center_projection - projected_radius,
        target_center_projection + projected_radius,
    )
    gap_after = _projected_inner_gap_for_interval(
        lower_box=lower_box,
        upper_box=upper_box,
        object_interval=centered_interval,
        unit=unit,
        use_oriented_finger_boxes=use_oriented_finger_boxes,
    )
    row.update(
        {
            "pass": True,
            "status": "PASS_CLOSING_AXIS_GAP_CENTERING_SHIFT_COMPUTED",
            "target_center_projection_m": float(target_center_projection),
            "current_center_projection_m": float(current_center_projection),
            "delta_projection_m": float(delta_projection),
            "horizontal_shift_unit_world": horizontal_unit.tolist(),
            "projection_per_meter": projection_per_meter,
            "delta_world_m": delta_world.tolist(),
            "gap_after_expected": gap_after,
        }
    )
    return row


def _diagnostic_force_target_overlap_shift(
    *,
    stage: Any,
    object_path: str,
    mode: str,
    lower_box: dict[str, Any],
    upper_box: dict[str, Any],
    object_projection_model: dict[str, Any],
    projection_unit_world: list[float] | tuple[float, float, float] | np.ndarray,
    overlap_m: float,
    use_oriented_finger_boxes: bool = False,
    named_finger_boxes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Diagnostic-only positive control for the contact-report pipeline.

    This intentionally perturbs reset geometry and must never be counted as a
    formal Gate2 pass.  It answers only whether the selected finger collider and
    selected object contact proxy can produce a PhysX contact report.
    """

    row: dict[str, Any] = {
        "enabled": bool(mode != "none"),
        "mode": str(mode),
        "formal_gate_allowed": False,
        "applied": False,
        "pass": False,
        "overlap_m": float(overlap_m),
    }
    if mode == "none":
        row["status"] = "DISABLED"
        return row
    unit = np.asarray(projection_unit_world, dtype=np.float64).reshape(3)
    unit_norm = float(np.linalg.norm(unit))
    if unit_norm <= 1e-12 or not np.isfinite(unit_norm):
        row["status"] = "FAIL_INVALID_CLOSING_AXIS"
        return row
    unit = unit / unit_norm
    if not object_projection_model.get("valid"):
        row["status"] = "FAIL_OBJECT_PROJECTION_MODEL_INVALID"
        row["object_projection_model"] = object_projection_model
        return row
    interval = tuple(float(v) for v in object_projection_model["object_interval_m"])
    gap = _projected_inner_gap_for_interval(
        lower_box=lower_box,
        upper_box=upper_box,
        object_interval=interval,
        unit=unit,
        use_oriented_finger_boxes=use_oriented_finger_boxes,
    )
    row["gap_before"] = gap
    if not gap.get("valid"):
        row["status"] = "FAIL_FINGER_INNER_GAP_INVALID"
        return row
    lower_gap = float(gap["object_gap_to_lower_finger_m"])
    upper_gap = float(gap["object_gap_to_upper_finger_m"])
    interval_fn = _box_oriented_projection_interval if use_oriented_finger_boxes else _box_projection_interval
    forced_finger = None
    if mode in {"left_finger", "right_finger"}:
        if not named_finger_boxes or mode not in named_finger_boxes:
            row["status"] = "FAIL_NAMED_FINGER_BOX_UNAVAILABLE"
            return row
        other_mode = "right_finger" if mode == "left_finger" else "left_finger"
        selected_interval = interval_fn(named_finger_boxes[mode], unit)
        other_interval = interval_fn(named_finger_boxes.get(other_mode, {}), unit)
        if selected_interval is None or other_interval is None:
            row["status"] = "FAIL_NAMED_FINGER_PROJECTION_INVALID"
            return row
        selected_center = (float(selected_interval[0]) + float(selected_interval[1])) * 0.5
        other_center = (float(other_interval[0]) + float(other_interval[1])) * 0.5
        side = "lower" if selected_center <= other_center else "upper"
        forced_finger = mode
        row["selected_finger_interval_m"] = [float(selected_interval[0]), float(selected_interval[1])]
        row["other_finger_interval_m"] = [float(other_interval[0]), float(other_interval[1])]
    elif mode == "nearest":
        side = "lower" if lower_gap <= upper_gap else "upper"
    else:
        side = mode
    projected_radius = float(object_projection_model["projected_radius_m"])
    if side == "lower":
        target_center_projection = float(gap["lower_inner_surface_m"]) - float(overlap_m) + projected_radius
    elif side == "upper":
        target_center_projection = float(gap["upper_inner_surface_m"]) + float(overlap_m) - projected_radius
    else:
        row["status"] = "FAIL_UNSUPPORTED_FORCE_OVERLAP_SIDE"
        return row
    current_center_projection = float(object_projection_model["center_projection_m"])
    delta_projection = target_center_projection - current_center_projection
    horizontal = np.asarray([unit[0], unit[1], 0.0], dtype=np.float64)
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm <= 1e-12 or not np.isfinite(horizontal_norm):
        row["status"] = "FAIL_CLOSING_AXIS_HAS_NO_HORIZONTAL_COMPONENT"
        return row
    horizontal_unit = horizontal / horizontal_norm
    projection_per_meter = float(np.dot(horizontal_unit, unit))
    if abs(projection_per_meter) <= 1e-12 or not np.isfinite(projection_per_meter):
        row["status"] = "FAIL_HORIZONTAL_SHIFT_CANNOT_CHANGE_CLOSING_PROJECTION"
        return row
    delta_world = horizontal_unit * (delta_projection / projection_per_meter)
    _shift_prim_world_translation(stage, object_path, delta_world)
    row.update(
        {
            "status": "DIAGNOSTIC_FORCED_TARGET_OVERLAP_APPLIED",
            "pass": True,
            "applied": True,
            "forced_side": side,
            "forced_finger": forced_finger,
            "target_center_projection_m": float(target_center_projection),
            "current_center_projection_m": float(current_center_projection),
            "delta_projection_m": float(delta_projection),
            "delta_world_m": delta_world.tolist(),
            "notes": (
                "Positive-control perturbation only. If contact is reported after this shift, the selected "
                "colliders and contact-report path can work. This run must not be used as a formal grasp pass."
            ),
        }
    )
    return row


def _finger_object_alignment_diagnostic(
    *,
    label: str,
    left_box: dict[str, Any],
    right_box: dict[str, Any],
    object_box: dict[str, Any],
    gap_axis: int,
    gap_axis_name: str,
    reference_contact_center_world: list[float] | tuple[float, float, float] | np.ndarray | None = None,
    object_long_axis_world: list[float] | tuple[float, float, float] | np.ndarray | None = None,
    object_projected_interval: tuple[float, float] | None = None,
    object_projection_model: dict[str, Any] | None = None,
    use_oriented_finger_boxes: bool = False,
) -> dict[str, Any]:
    """Summarize whether the object lies on the real two-finger closing line.

    The gap-axis AABB check can look correct while the object is laterally offset
    from the actual 3-D line between the two fingertip proxies.  This diagnostic
    keeps those two facts separate and does not modify the simulation.
    """

    row: dict[str, Any] = {
        "label": label,
        "gap_axis_index": int(gap_axis),
        "gap_axis_name": str(gap_axis_name),
        "valid": False,
    }
    if not (
        left_box.get("bbox_valid")
        and right_box.get("bbox_valid")
        and object_box.get("bbox_valid")
        and left_box.get("center") is not None
        and right_box.get("center") is not None
        and object_box.get("center") is not None
    ):
        row["status"] = "INVALID_BBOX"
        return row

    left_center = np.asarray(left_box["center"], dtype=np.float64).reshape(3)
    right_center = np.asarray(right_box["center"], dtype=np.float64).reshape(3)
    object_center = np.asarray(object_box["center"], dtype=np.float64).reshape(3)
    left_size = np.asarray(left_box.get("size", [np.nan, np.nan, np.nan]), dtype=np.float64).reshape(3)
    right_size = np.asarray(right_box.get("size", [np.nan, np.nan, np.nan]), dtype=np.float64).reshape(3)
    object_size = np.asarray(object_box.get("size", [np.nan, np.nan, np.nan]), dtype=np.float64).reshape(3)
    center_delta = left_center - right_center
    center_distance = float(np.linalg.norm(center_delta))
    if center_distance <= 1e-12 or not np.isfinite(center_distance):
        row["status"] = "INVALID_FINGER_CENTER_DISTANCE"
        return row
    closing_unit = center_delta / center_distance
    midpoint = (left_center + right_center) * 0.5
    object_offset = object_center - midpoint
    offset_along_closing = float(np.dot(object_offset, closing_unit))
    offset_cross = object_offset - offset_along_closing * closing_unit
    reference_row: dict[str, Any] = {"provided": False}
    if reference_contact_center_world is not None:
        reference_center = np.asarray(reference_contact_center_world, dtype=np.float64).reshape(3)
        reference_offset = reference_center - midpoint
        reference_offset_along_closing = float(np.dot(reference_offset, closing_unit))
        reference_cross = reference_offset - reference_offset_along_closing * closing_unit
        reference_row = {
            "provided": True,
            "center_world_m": reference_center.tolist(),
            "offset_from_finger_midpoint_world_m": reference_offset.tolist(),
            "offset_along_closing_axis_m": reference_offset_along_closing,
            "cross_closing_axis_offset_world_m": reference_cross.tolist(),
            "cross_closing_axis_offset_norm_m": float(np.linalg.norm(reference_cross)),
            "offset_along_gap_axis_m": float(reference_offset[gap_axis]),
            "correction_to_midplane_world_m": (-reference_offset_along_closing * closing_unit).tolist(),
            "correction_to_midplane_norm_m": abs(reference_offset_along_closing),
        }
    long_axis_row: dict[str, Any] = {"provided": False}
    if object_long_axis_world is not None:
        long_axis = np.asarray(object_long_axis_world, dtype=np.float64).reshape(3)
        long_axis_norm = float(np.linalg.norm(long_axis))
        if long_axis_norm > 1e-12 and np.isfinite(long_axis_norm):
            long_axis_unit = long_axis / long_axis_norm
            long_axis_row = {
                "provided": True,
                "unit_world": long_axis_unit.tolist(),
                "closing_axis_dot_object_long_axis": float(np.dot(closing_unit, long_axis_unit)),
                "closing_axis_dot_object_long_axis_abs": abs(float(np.dot(closing_unit, long_axis_unit))),
            }
    object_projection = _box_projection_interval(object_box, closing_unit)
    finger_interval_fn = _box_oriented_projection_interval if use_oriented_finger_boxes else _box_projection_interval
    left_projection = finger_interval_fn(left_box, closing_unit)
    right_projection = finger_interval_fn(right_box, closing_unit)
    if float(np.dot(right_center, closing_unit)) <= float(np.dot(left_center, closing_unit)):
        lower_box, upper_box = right_box, left_box
    else:
        lower_box, upper_box = left_box, right_box
    projected_inner_gap = (
        _projected_inner_gap_for_interval(
            lower_box=lower_box,
            upper_box=upper_box,
            object_interval=object_projected_interval,
            unit=closing_unit,
            use_oriented_finger_boxes=use_oriented_finger_boxes,
        )
        if object_projected_interval is not None
        else _projected_inner_gap(
            lower_box=lower_box,
            upper_box=upper_box,
            object_box=object_box,
            unit=closing_unit,
            use_oriented_finger_boxes=use_oriented_finger_boxes,
        )
    )

    if float(left_center[gap_axis]) <= float(right_center[gap_axis]):
        axis_inner_gap = _ordered_inner_gap(
            lower_box=left_box, upper_box=right_box, object_box=object_box, axis=gap_axis
        )
    else:
        axis_inner_gap = _ordered_inner_gap(
            lower_box=right_box, upper_box=left_box, object_box=object_box, axis=gap_axis
        )

    row.update(
        {
            "valid": True,
            "status": "PASS_DIAGNOSTIC_COMPUTED",
            "left_center_world_m": left_center.tolist(),
            "right_center_world_m": right_center.tolist(),
            "object_center_world_m": object_center.tolist(),
            "finger_midpoint_world_m": midpoint.tolist(),
            "left_size_m": left_size.tolist(),
            "right_size_m": right_size.tolist(),
            "object_size_m": object_size.tolist(),
            "finger_center_delta_world_m": center_delta.tolist(),
            "finger_center_distance_m": center_distance,
            "closing_axis_unit_world": closing_unit.tolist(),
            "object_offset_from_finger_midpoint_world_m": object_offset.tolist(),
            "object_offset_along_closing_axis_m": offset_along_closing,
            "object_cross_closing_axis_offset_world_m": offset_cross.tolist(),
            "object_cross_closing_axis_offset_norm_m": float(np.linalg.norm(offset_cross)),
            "object_offset_along_gap_axis_m": float(object_offset[gap_axis]),
            "reference_contact_center": reference_row,
            "object_long_axis": long_axis_row,
            "object_width_along_gap_axis_m": float(object_size[gap_axis]),
            "object_width_projected_on_closing_axis_m": None
            if object_projection is None
            else float(object_projection[1] - object_projection[0]),
            "finger_projection_model": "oriented_box_support" if use_oriented_finger_boxes else "world_aabb",
            "left_interval_projected_on_closing_axis_m": None
            if left_projection is None
            else [float(left_projection[0]), float(left_projection[1])],
            "right_interval_projected_on_closing_axis_m": None
            if right_projection is None
            else [float(right_projection[0]), float(right_projection[1])],
            "object_interval_projected_on_closing_axis_m": None
            if object_projection is None
            else [float(object_projection[0]), float(object_projection[1])],
            "axis_aligned_inner_gap": axis_inner_gap,
            "closing_axis_projected_inner_gap": projected_inner_gap,
            "object_projection_model": object_projection_model or {"source": "world_aabb"},
            "left_object_center_distance_m": float(np.linalg.norm(object_center - left_center)),
            "right_object_center_distance_m": float(np.linalg.norm(object_center - right_center)),
            "notes": (
                "Use this to distinguish an axis-aligned gap placement from true 3-D gripper-line alignment. "
                "A large cross-closing-axis offset means friction or CCD cannot create a symmetric clamp."
            ),
        }
    )
    return row


def _translated_reference_contact_center(
    *,
    reference_contact_center_world: list[float] | tuple[float, float, float] | np.ndarray | None,
    reset_object_box: dict[str, Any],
    final_object_box: dict[str, Any],
) -> dict[str, Any]:
    """Move the reference grasp point with the contact proxy center.

    The fixed-reference lift gate compares a bottle-local grasp point against the
    two-finger midplane after the bottle is lifted.  Reusing the initial world
    point would falsely report a midplane error as soon as the object moves.
    This first correction is translation-only; if future runs show large object
    rotation, the gate should be upgraded to a full rigid transform.
    """

    if reference_contact_center_world is None:
        return {
            "valid": False,
            "status": "SKIPPED_NO_REFERENCE_CONTACT_CENTER",
            "reference_contact_center_world_m": None,
        }
    reference = np.asarray(reference_contact_center_world, dtype=np.float64).reshape(3)
    if not (
        reset_object_box.get("bbox_valid")
        and final_object_box.get("bbox_valid")
        and reset_object_box.get("center") is not None
        and final_object_box.get("center") is not None
    ):
        return {
            "valid": False,
            "status": "INVALID_OBJECT_BOX_FOR_TRANSLATED_REFERENCE",
            "reference_contact_center_world_m": reference.astype(float).tolist(),
        }
    reset_center = np.asarray(reset_object_box["center"], dtype=np.float64).reshape(3)
    final_center = np.asarray(final_object_box["center"], dtype=np.float64).reshape(3)
    local_offset_world = reference - reset_center
    translated = final_center + local_offset_world
    return {
        "valid": True,
        "status": "PASS_TRANSLATED_REFERENCE_CONTACT_CENTER",
        "method": "contact_proxy_center_translation_only",
        "reference_contact_center_world_m": translated.astype(float).tolist(),
        "initial_reference_contact_center_world_m": reference.astype(float).tolist(),
        "reset_object_contact_center_world_m": reset_center.astype(float).tolist(),
        "final_object_contact_center_world_m": final_center.astype(float).tolist(),
        "object_contact_center_delta_world_m": (final_center - reset_center).astype(float).tolist(),
        "reference_local_offset_world_m": local_offset_world.astype(float).tolist(),
        "notes": (
            "The reference grasp point is moved with the contact proxy center for lift-phase validation. "
            "This avoids comparing a final gripper pose against a stale initial world-space point."
        ),
    }


def _closing_unit_from_finger_boxes(left_box: dict[str, Any], right_box: dict[str, Any]) -> np.ndarray | None:
    if not (
        left_box.get("bbox_valid")
        and right_box.get("bbox_valid")
        and left_box.get("center") is not None
        and right_box.get("center") is not None
    ):
        return None
    left_center = np.asarray(left_box["center"], dtype=np.float64).reshape(3)
    right_center = np.asarray(right_box["center"], dtype=np.float64).reshape(3)
    delta = left_center - right_center
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-12 or not np.isfinite(norm):
        return None
    return delta / norm


def _finger_object_center_offset_row(
    *,
    left_center: np.ndarray,
    right_center: np.ndarray,
    object_center: np.ndarray | None,
    closing_unit: np.ndarray,
) -> dict[str, Any]:
    """Decompose object/finger midpoint offset into closing and cross-axis parts."""

    midpoint = (left_center + right_center) * 0.5
    row: dict[str, Any] = {
        "finger_midpoint_world_m": midpoint.tolist(),
        "object_center_world_m": None,
        "object_offset_from_finger_midpoint_world_m": None,
        "object_offset_along_closing_axis_m": None,
        "object_cross_closing_axis_offset_world_m": None,
        "object_cross_closing_axis_offset_norm_m": None,
        "object_cross_closing_axis_offset_x_m": None,
        "object_cross_closing_axis_offset_y_m": None,
        "object_cross_closing_axis_offset_z_m": None,
    }
    if object_center is None:
        return row
    offset = object_center - midpoint
    along = float(np.dot(offset, closing_unit))
    cross = offset - along * closing_unit
    row.update(
        {
            "object_center_world_m": object_center.tolist(),
            "object_offset_from_finger_midpoint_world_m": offset.tolist(),
            "object_offset_along_closing_axis_m": along,
            "object_cross_closing_axis_offset_world_m": cross.tolist(),
            "object_cross_closing_axis_offset_norm_m": float(np.linalg.norm(cross)),
            "object_cross_closing_axis_offset_x_m": float(cross[0]),
            "object_cross_closing_axis_offset_y_m": float(cross[1]),
            "object_cross_closing_axis_offset_z_m": float(cross[2]),
        }
    )
    return row


def _finger_object_aabb_overlap_diagnostic(
    *,
    left_box: dict[str, Any],
    right_box: dict[str, Any],
    object_box: dict[str, Any],
    contact_distance: float,
) -> dict[str, Any]:
    """Report world-AABB overlap/separation for each finger proxy vs target.

    This is a conservative diagnostic, not a true signed-distance query.  If the
    enclosing AABBs are separated beyond the allowed contact distance on any
    world axis, a 1-D closing-axis projection is not sufficient evidence that
    PhysX should have produced a contact pair.
    """

    result: dict[str, Any] = {
        "valid": False,
        "contact_distance_m": float(contact_distance),
        "finger_rows": {},
        "all_fingers_within_aabb_contact_distance": False,
        "all_fingers_world_aabb_overlap_xyz": False,
    }
    if not (
        object_box.get("bbox_valid")
        and object_box.get("min") is not None
        and object_box.get("max") is not None
    ):
        result["status"] = "INVALID_OBJECT_BBOX"
        return result

    object_min = np.asarray(object_box["min"], dtype=np.float64).reshape(3)
    object_max = np.asarray(object_box["max"], dtype=np.float64).reshape(3)
    object_center = np.asarray(object_box.get("center", (object_min + object_max) * 0.5), dtype=np.float64).reshape(3)
    contact_distance_f = float(contact_distance)
    finger_rows: dict[str, Any] = {}
    for name, box in (("left_finger", left_box), ("right_finger", right_box)):
        row: dict[str, Any] = {"valid": False, "finger": name}
        if not (box.get("bbox_valid") and box.get("min") is not None and box.get("max") is not None):
            row["status"] = "INVALID_FINGER_BBOX"
            finger_rows[name] = row
            continue
        finger_min = np.asarray(box["min"], dtype=np.float64).reshape(3)
        finger_max = np.asarray(box["max"], dtype=np.float64).reshape(3)
        finger_center = np.asarray(box.get("center", (finger_min + finger_max) * 0.5), dtype=np.float64).reshape(3)
        overlap = np.minimum(finger_max, object_max) - np.maximum(finger_min, object_min)
        axis_gap = np.maximum(-overlap, 0.0)
        axis_overlap = np.maximum(overlap, 0.0)
        overlap_xyz = bool(np.all(overlap >= 0.0))
        within_contact_distance = bool(np.all(axis_gap <= contact_distance_f))
        row.update(
            {
                "valid": True,
                "status": (
                    "PASS_WORLD_AABB_OVERLAPS_OBJECT"
                    if overlap_xyz
                    else (
                        "PASS_WORLD_AABB_WITHIN_CONTACT_DISTANCE"
                        if within_contact_distance
                        else "FAIL_WORLD_AABB_SEPARATED_BEYOND_CONTACT_DISTANCE"
                    )
                ),
                "finger_center_world_m": finger_center.tolist(),
                "object_center_world_m": object_center.tolist(),
                "center_delta_world_m": (object_center - finger_center).tolist(),
                "axis_overlap_m": axis_overlap.tolist(),
                "axis_gap_m": axis_gap.tolist(),
                "max_axis_gap_m": float(np.max(axis_gap)),
                "world_aabb_overlap_xyz": overlap_xyz,
                "world_aabb_overlap_volume_m3": float(np.prod(axis_overlap)) if overlap_xyz else 0.0,
                "within_contact_distance_all_axes": within_contact_distance,
            }
        )
        finger_rows[name] = row

    valid_rows = [row for row in finger_rows.values() if row.get("valid")]
    result.update(
        {
            "valid": bool(len(valid_rows) == 2),
            "status": (
                "PASS_ALL_FINGER_AABBS_WITHIN_CONTACT_DISTANCE"
                if valid_rows and all(row["within_contact_distance_all_axes"] for row in valid_rows)
                else "FAIL_AT_LEAST_ONE_FINGER_AABB_SEPARATED_BEYOND_CONTACT_DISTANCE"
            ),
            "finger_rows": finger_rows,
            "all_fingers_within_aabb_contact_distance": bool(
                valid_rows and all(row["within_contact_distance_all_axes"] for row in valid_rows)
            ),
            "all_fingers_world_aabb_overlap_xyz": bool(valid_rows and all(row["world_aabb_overlap_xyz"] for row in valid_rows)),
        }
    )
    return result


def _live_target_reachability_row(
    *,
    phase: str,
    step: int,
    left_box: dict[str, Any],
    right_box: dict[str, Any],
    object_contact_box: dict[str, Any],
    object_projection_model: dict[str, Any],
    contact_rows: list[dict[str, Any]],
    object_path: str,
    expected_finger_paths: list[str],
    table_path: str | None,
    contact_distance: float,
    use_oriented_finger_boxes: bool,
) -> dict[str, Any]:
    """Per-step audit of whether the live target collider is physically reachable.

    This is intentionally diagnostic.  It does not move the object, change
    contact offsets, or relax the contact gate.  The goal is to avoid treating a
    stale or world-AABB projection as proof that the actual PhysX proxy touched
    the gripper.
    """

    row: dict[str, Any] = {
        "phase": str(phase),
        "step": int(step),
        "valid": False,
        "contact_distance_m": float(contact_distance),
        "object_projection_model": object_projection_model,
    }
    closing_unit = _closing_unit_from_finger_boxes(left_box, right_box)
    if closing_unit is None:
        row["status"] = "FAIL_INVALID_FINGER_CLOSING_AXIS"
        return row
    if not object_projection_model.get("valid"):
        row["status"] = "FAIL_OBJECT_PROJECTION_MODEL_INVALID"
        return row
    left_center = np.asarray(left_box["center"], dtype=np.float64).reshape(3)
    right_center = np.asarray(right_box["center"], dtype=np.float64).reshape(3)
    if float(np.dot(right_center, closing_unit)) <= float(np.dot(left_center, closing_unit)):
        lower_box, upper_box = right_box, left_box
    else:
        lower_box, upper_box = left_box, right_box
    gap = _projected_inner_gap_for_interval(
        lower_box=lower_box,
        upper_box=upper_box,
        object_interval=tuple(float(v) for v in object_projection_model["object_interval_m"]),
        unit=closing_unit,
        use_oriented_finger_boxes=use_oriented_finger_boxes,
    )
    object_center = (
        np.asarray(object_contact_box["center"], dtype=np.float64).reshape(3)
        if object_contact_box.get("bbox_valid") and object_contact_box.get("center") is not None
        else None
    )
    offset_row = _finger_object_center_offset_row(
        left_center=left_center,
        right_center=right_center,
        object_center=object_center,
        closing_unit=closing_unit,
    )
    aabb_overlap = _finger_object_aabb_overlap_diagnostic(
        left_box=left_box,
        right_box=right_box,
        object_box=object_contact_box,
        contact_distance=float(contact_distance),
    )
    target_rows = [contact for contact in contact_rows if _pair_touches_targets(contact, object_path, expected_finger_paths)]
    table_finger_rows: list[dict[str, Any]] = []
    if table_path:
        for contact in contact_rows:
            if not _pair_touches_path(contact, table_path):
                continue
            if any(_pair_touches_path(contact, finger_path) for finger_path in expected_finger_paths):
                table_finger_rows.append(contact)
    min_projected_surface_gap = None
    object_inside = bool(gap.get("object_inside_inner_gap")) if gap.get("valid") else False
    if gap.get("valid") and object_inside:
        min_projected_surface_gap = min(
            float(gap["object_gap_to_lower_finger_m"]),
            float(gap["object_gap_to_upper_finger_m"]),
        )
    projected_reaches_contact_distance = bool(
        min_projected_surface_gap is not None and min_projected_surface_gap <= float(contact_distance)
    )
    status = "OBSERVED_NO_TARGET_CONTACT"
    if target_rows:
        status = "PASS_GEOMETRIC_CONTACT_REPORTED"
    elif table_finger_rows:
        status = "FAIL_FINGER_TABLE_CONTACT_BLOCKS_TARGET_REACH"
    elif projected_reaches_contact_distance and not aabb_overlap.get("all_fingers_within_aabb_contact_distance"):
        status = "FAIL_1D_PROJECTION_OVERLAP_BUT_3D_AABB_SEPARATION"
    elif projected_reaches_contact_distance:
        status = "FAIL_PROXIMITY_WITHOUT_CONTACT_REPORT"
    elif gap.get("valid"):
        status = "FAIL_NO_GEOMETRIC_REACH_TO_TARGET_COLLIDER"
    row.update(
        {
            "valid": bool(gap.get("valid")),
            "status": status,
            "closing_axis_unit_world": closing_unit.tolist(),
            "projected_inner_gap": gap,
            "object_contact_box": object_contact_box,
            **offset_row,
            "finger_object_world_aabb_overlap_diagnostic": aabb_overlap,
            "min_projected_surface_gap_m": min_projected_surface_gap,
            "projected_reaches_contact_distance": projected_reaches_contact_distance,
            "target_contact_rows_at_step": len(target_rows),
            "table_finger_contact_rows_at_step": len(table_finger_rows),
            "target_contact_pairs_at_step": _unique_pairs(target_rows),
            "table_finger_contact_pairs_at_step": _unique_pairs(table_finger_rows),
        }
    )
    return row


def _summarize_target_reachability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"pass": False, "status": "NOT_COMPUTED", "row_count": 0}
    target_rows = [row for row in rows if row.get("target_contact_rows_at_step", 0) > 0]
    table_block_rows = [row for row in rows if row.get("table_finger_contact_rows_at_step", 0) > 0]
    proximity_rows = [row for row in rows if row.get("projected_reaches_contact_distance")]
    projection_aabb_separation_rows = [
        row for row in rows if row.get("status") == "FAIL_1D_PROJECTION_OVERLAP_BUT_3D_AABB_SEPARATION"
    ]
    valid_rows = [row for row in rows if row.get("valid")]
    min_gap_values = [
        float(row["min_projected_surface_gap_m"])
        for row in rows
        if row.get("min_projected_surface_gap_m") is not None
    ]
    cross_values = [
        float(row["object_cross_closing_axis_offset_norm_m"])
        for row in rows
        if row.get("object_cross_closing_axis_offset_norm_m") is not None
    ]
    min_gap_row = None
    rows_with_min_gap = [row for row in rows if row.get("min_projected_surface_gap_m") is not None]
    if rows_with_min_gap:
        min_gap_row = min(rows_with_min_gap, key=lambda row: float(row["min_projected_surface_gap_m"]))
    proximity_rows_sample = proximity_rows[:5]
    if target_rows:
        status = "PASS_GEOMETRIC_CONTACT_REPORTED"
    elif table_block_rows:
        status = "FAIL_FINGER_TABLE_CONTACT_BLOCKS_TARGET_REACH"
    elif projection_aabb_separation_rows:
        status = "FAIL_1D_PROJECTION_OVERLAP_BUT_3D_AABB_SEPARATION"
    elif proximity_rows:
        status = "FAIL_PROXIMITY_WITHOUT_CONTACT_REPORT"
    elif valid_rows:
        status = "FAIL_NO_GEOMETRIC_REACH_TO_TARGET_COLLIDER"
    else:
        status = "FAIL_REACHABILITY_INVALID"
    return {
        "pass": bool(target_rows),
        "status": status,
        "row_count": len(rows),
        "valid_row_count": len(valid_rows),
        "target_contact_step_count": len(target_rows),
        "first_target_contact_step": target_rows[0]["step"] if target_rows else None,
        "table_finger_contact_step_count": len(table_block_rows),
        "first_table_finger_contact_step": table_block_rows[0]["step"] if table_block_rows else None,
        "projection_reach_step_count": len(proximity_rows),
        "first_projection_reach_step": proximity_rows[0]["step"] if proximity_rows else None,
        "projection_aabb_separation_step_count": len(projection_aabb_separation_rows),
        "first_projection_aabb_separation_step": (
            projection_aabb_separation_rows[0]["step"] if projection_aabb_separation_rows else None
        ),
        "min_projected_surface_gap_m": min(min_gap_values) if min_gap_values else None,
        "min_projected_surface_gap_row": min_gap_row,
        "max_object_cross_closing_axis_offset_norm_m": max(cross_values) if cross_values else None,
        "projection_aabb_separation_rows_sample": projection_aabb_separation_rows[:5],
        "proximity_without_contact_rows_sample": proximity_rows_sample,
        "rows_sample": rows[:5],
        "last_row": rows[-1],
        "notes": (
            "Uses the live contact proxy projection for each close step. A world-AABB-only final gap is not "
            "accepted as evidence of PhysX contact. A small closing-axis gap with a large cross-axis object "
            "offset means the gripper and object are not aligned in the full 3-D contact volume."
        ),
    }


def _loaded_gripper_soft_bottle_calibration_diagnostic(
    *,
    final_alignment: dict[str, Any],
    hdf5_gripper_summary: dict[str, Any] | None,
    reachability_audit: dict[str, Any],
    contact_distance_m: float,
    object_effective_contact_width_m: float | None,
    visual_bottle_outer_diameter_m: float | None,
    moving_fingers: str,
    controller_tracking_gate: dict[str, Any] | None = None,
    positive_control_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Quantify qpos replay residual without relaxing the formal contact gate.

    Real mineral-water bottles deform under ALOHA finger pressure, and ALOHA
    gripper qpos is a normalized observed gripper-joint signal rather than a
    direct measurement of loaded finger-pad surface separation.  This diagnostic
    reports the residual between the formal qpos replay geometry and the target
    contact proxy.  It must not change `overall_pass` or contact gates.
    """

    result: dict[str, Any] = {
        "enabled": True,
        "formal_gate_result_preserved": True,
        "may_set_overall_pass": False,
        "may_set_physical_grasp_gate": False,
        "status": "NOT_COMPUTED",
        "source": (hdf5_gripper_summary or {}).get("source"),
        "moving_fingers": str(moving_fingers),
        "contact_distance_m": float(contact_distance_m),
        "object_effective_contact_width_m": object_effective_contact_width_m,
        "visual_bottle_outer_diameter_m": visual_bottle_outer_diameter_m,
        "reachability_status": reachability_audit.get("status"),
        "controller_tracking_pass": None if controller_tracking_gate is None else controller_tracking_gate.get("pass"),
        "positive_control_status": None if positive_control_gate is None else positive_control_gate.get("status"),
        "notes": (
            "Diagnostic only: this quantifies unmodeled loaded gripper compliance and soft-bottle deformation. "
            "It must not convert a missing PhysX target contact into a formal pass."
        ),
    }

    gap = final_alignment.get("closing_axis_projected_inner_gap") or {}
    object_projection_model = final_alignment.get("object_projection_model") or {}
    if not gap.get("valid"):
        result["status"] = "NOT_COMPUTED_INVALID_FINAL_PROJECTED_GAP"
        return result

    lower_gap = gap.get("object_gap_to_lower_finger_m")
    upper_gap = gap.get("object_gap_to_upper_finger_m")
    if lower_gap is None or upper_gap is None:
        result["status"] = "NOT_COMPUTED_MISSING_FINGER_OBJECT_GAPS"
        return result
    try:
        lower_gap_f = float(lower_gap)
        upper_gap_f = float(upper_gap)
    except Exception:
        result["status"] = "NOT_COMPUTED_NON_NUMERIC_FINGER_OBJECT_GAPS"
        return result
    if not (np.isfinite(lower_gap_f) and np.isfinite(upper_gap_f)):
        result["status"] = "NOT_COMPUTED_NON_FINITE_FINGER_OBJECT_GAPS"
        return result

    min_gap = min(lower_gap_f, upper_gap_f)
    max_gap = max(lower_gap_f, upper_gap_f)
    nearest_side = "lower" if lower_gap_f <= upper_gap_f else "upper"
    contact_distance = max(0.0, float(contact_distance_m))
    missing_nearest_to_zero = max(0.0, min_gap)
    missing_nearest_to_contact = max(0.0, min_gap - contact_distance)
    missing_bilateral_to_zero = max(0.0, max_gap)
    missing_bilateral_to_contact = max(0.0, max_gap - contact_distance)
    symmetric_divisor = 2.0 if str(moving_fingers) == "both" else 1.0

    object_width = None
    if object_projection_model.get("projected_width_m") is not None:
        object_width = float(object_projection_model["projected_width_m"])
    elif gap.get("object_interval_m") is not None:
        interval = [float(v) for v in gap["object_interval_m"]]
        object_width = interval[1] - interval[0]
    elif object_effective_contact_width_m is not None:
        object_width = float(object_effective_contact_width_m)

    implied_widths: dict[str, float | None] = {
        "nearest_touch_m": None,
        "nearest_contact_distance_m": None,
        "bilateral_touch_m": None,
        "bilateral_contact_distance_m": None,
    }
    if object_width is not None and np.isfinite(object_width):
        implied_widths = {
            "nearest_touch_m": float(object_width + 2.0 * missing_nearest_to_zero),
            "nearest_contact_distance_m": float(object_width + 2.0 * missing_nearest_to_contact),
            "bilateral_touch_m": float(object_width + 2.0 * missing_bilateral_to_zero),
            "bilateral_contact_distance_m": float(object_width + 2.0 * missing_bilateral_to_contact),
        }

    qpos_source_is_loaded_gap_calibrated = bool(
        (hdf5_gripper_summary or {}).get("qpos_source_is_loaded_gap_calibrated")
    )
    result.update(
        {
            "status": (
                "COMPUTED_LOADED_QPOS_CALIBRATED_CONTACT_RESIDUAL"
                if qpos_source_is_loaded_gap_calibrated
                else "COMPUTED_FORMAL_QPOS_LOADED_CONTACT_RESIDUAL"
            ),
            "qpos_source_is_loaded_gap_calibrated": qpos_source_is_loaded_gap_calibrated,
            "qpos_calibration": (hdf5_gripper_summary or {}).get("qpos_calibration"),
            "requires_raw_finger_or_spacer_calibration": not qpos_source_is_loaded_gap_calibrated,
            "nearest_side": nearest_side,
            "lower_surface_gap_m": lower_gap_f,
            "upper_surface_gap_m": upper_gap_f,
            "nearest_surface_gap_m": float(min_gap),
            "bilateral_surface_gap_m": float(max_gap),
            "missing_to_zero_gap_m": float(missing_nearest_to_zero),
            "missing_to_contact_distance_m": float(missing_nearest_to_contact),
            "bilateral_missing_to_zero_gap_m": float(missing_bilateral_to_zero),
            "bilateral_missing_to_contact_distance_m": float(missing_bilateral_to_contact),
            "per_finger_loaded_closure_deficit_to_zero_gap_m": float(
                missing_nearest_to_zero / symmetric_divisor
            ),
            "per_finger_loaded_closure_deficit_to_contact_distance_m": float(
                missing_nearest_to_contact / symmetric_divisor
            ),
            "per_finger_bilateral_loaded_closure_deficit_to_zero_gap_m": float(
                missing_bilateral_to_zero / symmetric_divisor
            ),
            "per_finger_bilateral_loaded_closure_deficit_to_contact_distance_m": float(
                missing_bilateral_to_contact / symmetric_divisor
            ),
            "projected_object_contact_width_m": None if object_width is None else float(object_width),
            "implied_effective_contact_widths_if_explained_as_soft_deformation": implied_widths,
            "hdf5_gripper_raw_start": (hdf5_gripper_summary or {}).get("raw_start"),
            "hdf5_gripper_raw_end": (hdf5_gripper_summary or {}).get("raw_end"),
            "hdf5_gripper_raw_range": (hdf5_gripper_summary or {}).get("raw_range"),
            "hdf5_gripper_sample_count": (hdf5_gripper_summary or {}).get("sample_count"),
        }
    )
    if visual_bottle_outer_diameter_m is not None and object_width is not None:
        result["visual_minus_projected_contact_width_m"] = float(visual_bottle_outer_diameter_m - object_width)
        result["nearest_touch_width_exceeds_visual_outer_diameter"] = bool(
            implied_widths["nearest_touch_m"] is not None
            and implied_widths["nearest_touch_m"] > float(visual_bottle_outer_diameter_m)
        )
        result["bilateral_touch_width_exceeds_visual_outer_diameter"] = bool(
            implied_widths["bilateral_touch_m"] is not None
            and implied_widths["bilateral_touch_m"] > float(visual_bottle_outer_diameter_m)
        )
    return result


def _axis_suffix(axis_name: str) -> str:
    normalized = str(axis_name).lower()
    if normalized not in {"x", "y", "z"}:
        raise ValueError(f"unsupported axis name: {axis_name!r}")
    return normalized


def _bilateral_grasp_formation_gate(
    *,
    rows: list[dict[str, Any]],
    contact_summary: dict[str, Any],
    moving_fingers: str,
    gap_axis_name: str,
    min_contact_steps: int,
    min_nonzero_impulse_steps: int,
    max_impulse_ratio: float,
    max_prelift_lateral_sweep: float,
    prelift_gripper_z_delta: float,
) -> dict[str, Any]:
    """Require real two-finger contact before lift validation.

    The gate is diagnostic-only.  It does not move the object, attach the object,
    alter contact offsets, or relax any existing contact policy.
    """

    result: dict[str, Any] = {
        "required": bool(moving_fingers == "both"),
        "pass": True,
        "status": "SKIPPED_NOT_BILATERAL_GRASP",
        "min_contact_steps": int(min_contact_steps),
        "min_nonzero_impulse_steps": int(min_nonzero_impulse_steps),
        "max_impulse_ratio": float(max_impulse_ratio),
        "max_prelift_lateral_sweep_m": float(max_prelift_lateral_sweep),
        "prelift_gripper_z_delta_m": float(prelift_gripper_z_delta),
    }
    if moving_fingers != "both":
        return result

    quality_by_finger = contact_summary.get("target_contact_quality_by_finger") or {}
    finger_rows: list[dict[str, Any]] = []
    contact_step_sets: list[set[int]] = []
    impulse_values: list[float] = []
    for finger_path, quality in sorted(quality_by_finger.items()):
        contact_steps = {int(step) for step in (quality.get("contact_steps") or [])}
        contact_step_sets.append(contact_steps)
        max_impulse = quality.get("max_impulse_norm")
        if max_impulse is not None:
            try:
                impulse_values.append(float(max_impulse))
            except Exception:
                pass
        finger_rows.append(
            {
                "finger_path": finger_path,
                "contact_step_count": int(quality.get("contact_step_count") or 0),
                "nonzero_impulse_step_count": int(quality.get("nonzero_impulse_step_count") or 0),
                "max_impulse_norm": quality.get("max_impulse_norm"),
                "first_step": quality.get("first_step"),
                "last_step": quality.get("last_step"),
            }
        )

    bilateral_steps = sorted(set.intersection(*contact_step_sets)) if contact_step_sets else []
    contact_ok = bool(
        finger_rows
        and all(item["contact_step_count"] >= int(min_contact_steps) for item in finger_rows)
        and len(bilateral_steps) >= int(min_contact_steps)
    )
    impulse_ok = bool(
        finger_rows
        and all(item["nonzero_impulse_step_count"] >= int(min_nonzero_impulse_steps) for item in finger_rows)
    )
    positive_impulses = [value for value in impulse_values if np.isfinite(value) and value > 1e-8]
    impulse_ratio = None
    impulse_balance_ok = False
    if len(positive_impulses) == len(finger_rows) and positive_impulses:
        impulse_ratio = float(max(positive_impulses) / max(min(positive_impulses), 1e-12))
        impulse_balance_ok = bool(impulse_ratio <= float(max_impulse_ratio))

    close_rows = [item for item in rows if item.get("phase") == "close"] or list(rows)
    axis = _axis_suffix(gap_axis_name)
    prelift_rows: list[dict[str, Any]] = []
    object_axis_values: list[float] = []
    relative_axis_values: list[float] = []
    if close_rows:
        initial_mid_z = close_rows[0].get("finger_mid_center_z")
        for item in close_rows:
            mid_z = item.get("finger_mid_center_z")
            if initial_mid_z is not None and mid_z is not None:
                try:
                    if float(mid_z) - float(initial_mid_z) > float(prelift_gripper_z_delta):
                        break
                except Exception:
                    pass
            prelift_rows.append(item)
        for item in prelift_rows:
            object_value = item.get(f"object_center_{axis}")
            finger_mid_value = item.get(f"finger_mid_center_{axis}")
            try:
                object_axis_values.append(float(object_value))
            except Exception:
                pass
            try:
                relative_axis_values.append(float(object_value) - float(finger_mid_value))
            except Exception:
                pass

    object_lateral_sweep = float(max(object_axis_values) - min(object_axis_values)) if object_axis_values else None
    relative_lateral_sweep = float(max(relative_axis_values) - min(relative_axis_values)) if relative_axis_values else None
    lateral_sweep_for_gate = relative_lateral_sweep if relative_lateral_sweep is not None else object_lateral_sweep
    lateral_sweep_ok = bool(
        lateral_sweep_for_gate is not None and lateral_sweep_for_gate <= float(max_prelift_lateral_sweep)
    )

    pass_gate = bool(contact_ok and impulse_ok and impulse_balance_ok and lateral_sweep_ok)
    if not contact_ok:
        status = "FAIL_BILATERAL_TARGET_CONTACT_NOT_FORMED"
    elif not impulse_ok:
        status = "FAIL_BILATERAL_NONZERO_IMPULSE_NOT_FORMED"
    elif not impulse_balance_ok:
        status = "FAIL_BILATERAL_IMPULSE_IMBALANCED"
    elif not lateral_sweep_ok:
        status = "FAIL_OBJECT_SWEPT_BEFORE_BILATERAL_GRASP"
    else:
        status = "PASS_BILATERAL_GRASP_FORMATION"
    result.update(
        {
            "pass": pass_gate,
            "status": status,
            "finger_rows": finger_rows,
            "bilateral_contact_steps": bilateral_steps,
            "bilateral_contact_step_count": len(bilateral_steps),
            "max_impulse_ratio_observed": impulse_ratio,
            "prelift_row_count": len(prelift_rows),
            "object_lateral_sweep_m": object_lateral_sweep,
            "object_relative_to_gripper_lateral_sweep_m": relative_lateral_sweep,
            "lateral_sweep_for_gate_m": lateral_sweep_for_gate,
            "notes": (
                "A dynamic lift is not meaningful until both finger proxies contact the target with nonzero "
                "impulse and the object is not swept sideways before lift."
            ),
        }
    )
    return result


def _consecutive_tail_count(steps: list[int], *, final_step: int) -> int:
    """Count consecutive contact steps ending at final_step."""

    step_set = {int(step) for step in steps}
    count = 0
    cursor = int(final_step)
    while cursor in step_set:
        count += 1
        cursor -= 1
    return count


def _prelift_static_grasp_gate(
    *,
    hold_rows: list[dict[str, Any]],
    hold_bilateral_gate: dict[str, Any] | None,
    hold_tracking_gate: dict[str, Any] | None,
    min_hold_steps: int,
    min_tail_bilateral_steps: int,
    min_each_finger_nonzero_impulse_steps: int,
    max_object_xy_sweep: float,
    max_object_z_delta: float,
) -> dict[str, Any]:
    """Diagnose whether the object is already stably grasped before lift.

    This gate is intentionally diagnostic-only. It evaluates an appended
    post-close hold phase and must not convert the raw HDF5 close window into a
    formal replay pass.
    """

    hold_bilateral_gate = hold_bilateral_gate or {}
    hold_tracking_gate = hold_tracking_gate or {}
    result: dict[str, Any] = {
        "required": bool(min_hold_steps > 0),
        "diagnostic_only": True,
        "formal_replay_targets_modified": False,
        "formal_close_replay_success": False,
        "pass": True,
        "status": "SKIPPED_NO_PRELIFT_HOLD",
        "hold_step_count": int(len(hold_rows)),
        "min_hold_steps": int(min_hold_steps),
        "min_tail_bilateral_steps": int(min_tail_bilateral_steps),
        "min_each_finger_nonzero_impulse_steps": int(min_each_finger_nonzero_impulse_steps),
        "max_object_xy_sweep_m": float(max_object_xy_sweep),
        "max_object_z_delta_m": float(max_object_z_delta),
    }
    if min_hold_steps <= 0:
        return result

    bilateral_steps = [int(step) for step in (hold_bilateral_gate.get("bilateral_contact_steps") or [])]
    final_step = max((int(row.get("step", -1)) for row in hold_rows), default=-1)
    tail_bilateral_steps = _consecutive_tail_count(bilateral_steps, final_step=final_step) if final_step >= 0 else 0
    finger_rows = hold_bilateral_gate.get("finger_rows") or []
    each_finger_nonzero_ok = bool(
        finger_rows
        and all(
            int(row.get("nonzero_impulse_step_count") or 0)
            >= int(min_each_finger_nonzero_impulse_steps)
            for row in finger_rows
        )
    )

    object_x_values: list[float] = []
    object_y_values: list[float] = []
    object_z_values: list[float] = []
    for row in hold_rows:
        for key, target in (
            ("object_center_x", object_x_values),
            ("object_center_y", object_y_values),
            ("object_center_z", object_z_values),
        ):
            value = row.get(key)
            try:
                target.append(float(value))
            except Exception:
                pass

    object_x_sweep = float(max(object_x_values) - min(object_x_values)) if object_x_values else None
    object_y_sweep = float(max(object_y_values) - min(object_y_values)) if object_y_values else None
    object_xy_sweep = (
        None
        if object_x_sweep is None or object_y_sweep is None
        else float(np.hypot(object_x_sweep, object_y_sweep))
    )
    object_z_delta = (
        None
        if len(object_z_values) < 2
        else float(abs(object_z_values[-1] - object_z_values[0]))
    )
    object_z_range = float(max(object_z_values) - min(object_z_values)) if object_z_values else None

    hold_length_ok = len(hold_rows) >= int(min_hold_steps)
    bilateral_ok = bool(hold_bilateral_gate.get("pass"))
    tracking_ok = bool(hold_tracking_gate.get("pass"))
    tail_ok = tail_bilateral_steps >= int(min_tail_bilateral_steps)
    object_xy_ok = bool(object_xy_sweep is not None and object_xy_sweep <= float(max_object_xy_sweep))
    object_z_ok = bool(object_z_delta is not None and object_z_delta <= float(max_object_z_delta))
    pass_gate = bool(
        hold_length_ok
        and bilateral_ok
        and tracking_ok
        and tail_ok
        and each_finger_nonzero_ok
        and object_xy_ok
        and object_z_ok
    )
    failed_checks = []
    if not hold_length_ok:
        failed_checks.append("hold_step_count_too_short")
    if not bilateral_ok:
        failed_checks.append("hold_bilateral_gate_failed")
    if not tracking_ok:
        failed_checks.append("hold_tracking_gate_failed")
    if not tail_ok:
        failed_checks.append("tail_bilateral_contact_not_sustained")
    if not each_finger_nonzero_ok:
        failed_checks.append("each_finger_nonzero_impulse_not_sustained")
    if not object_xy_ok:
        failed_checks.append("object_xy_sweep_too_large")
    if not object_z_ok:
        failed_checks.append("object_z_delta_too_large")

    result.update(
        {
            "pass": pass_gate,
            "status": "PASS_PRELIFT_STATIC_GRASP_DIAGNOSTIC"
            if pass_gate
            else "FAIL_PRELIFT_STATIC_GRASP_DIAGNOSTIC",
            "failed_checks": failed_checks,
            "hold_bilateral_status": hold_bilateral_gate.get("status"),
            "hold_tracking_status": hold_tracking_gate.get("status"),
            "tail_bilateral_contact_steps": int(tail_bilateral_steps),
            "bilateral_contact_step_count": int(hold_bilateral_gate.get("bilateral_contact_step_count") or 0),
            "finger_rows": finger_rows,
            "object_x_sweep_m": object_x_sweep,
            "object_y_sweep_m": object_y_sweep,
            "object_xy_sweep_m": object_xy_sweep,
            "object_z_delta_m": object_z_delta,
            "object_z_range_m": object_z_range,
            "notes": (
                "Diagnostic only. A pass means the appended pre-lift hold has a stable two-finger clamp "
                "before transport begins. It does not make the raw HDF5 close window a formal replay pass."
            ),
        }
    )
    return result


def _early_geometric_escape_gate(
    *,
    object_width_stop_rows: list[dict[str, Any]],
    bilateral_grasp_formation_gate: dict[str, Any],
    tolerance_m: float = 1e-4,
) -> dict[str, Any]:
    """Classify whether the object leaves the two-finger capture region early.

    This gate is diagnostic.  It is intentionally based on live geometric
    reachability rows, not contact reports.  A negative gap on exactly one side
    means the bottle interval has crossed one finger's inner surface while the
    opposite side still has clearance.  That is different from centered
    over-closure, where both sides can be negative and PhysX contact reporting is
    the primary failure signal.
    """

    result: dict[str, Any] = {
        "required": bool(bilateral_grasp_formation_gate.get("required")),
        "pass": True,
        "status": "SKIPPED_NOT_BILATERAL_GRASP",
        "tolerance_m": float(tolerance_m),
        "first_escape_step": None,
        "first_unilateral_escape_row": None,
        "first_left_contact_step": None,
        "first_right_contact_step": None,
        "first_bilateral_contact_step": None,
        "notes": (
            "This is a geometry classifier only. It reports when the bottle leaves the live "
            "closing-axis capture region before both fingers form target contact; it does not "
            "turn geometric overlap into contact success."
        ),
    }
    if not result["required"]:
        return result

    finger_rows = bilateral_grasp_formation_gate.get("finger_rows") or []
    bilateral_steps = [int(step) for step in (bilateral_grasp_formation_gate.get("bilateral_contact_steps") or [])]
    first_bilateral_step = min(bilateral_steps) if bilateral_steps else None
    result["first_bilateral_contact_step"] = first_bilateral_step

    for item in finger_rows:
        path = str(item.get("finger_path") or "")
        first_step = item.get("first_step")
        if first_step is None:
            continue
        try:
            first_step_int = int(first_step)
        except Exception:
            continue
        if "left_left_finger" in path:
            result["first_left_contact_step"] = first_step_int
        elif "left_right_finger" in path:
            result["first_right_contact_step"] = first_step_int

    for row in object_width_stop_rows:
        projected_gap = row.get("projected_inner_gap") or {}
        if not projected_gap.get("valid"):
            continue
        step_value = row.get("step")
        try:
            step = int(step_value)
        except Exception:
            continue
        if first_bilateral_step is not None and step >= first_bilateral_step:
            continue
        lower_gap = projected_gap.get("object_gap_to_lower_finger_m")
        upper_gap = projected_gap.get("object_gap_to_upper_finger_m")
        try:
            lower_gap_f = float(lower_gap)
            upper_gap_f = float(upper_gap)
        except Exception:
            continue
        if not (np.isfinite(lower_gap_f) and np.isfinite(upper_gap_f)):
            continue
        lower_crossed = lower_gap_f < -float(tolerance_m)
        upper_crossed = upper_gap_f < -float(tolerance_m)
        unilateral_escape = bool(lower_crossed ^ upper_crossed)
        if not unilateral_escape:
            continue
        result.update(
            {
                "pass": False,
                "status": "FAIL_OBJECT_ESCAPES_CAPTURE_REGION_BEFORE_BILATERAL_CONTACT",
                "first_escape_step": step,
                "first_unilateral_escape_row": {
                    "phase": row.get("phase"),
                    "step": step,
                    "active": bool(row.get("active")),
                    "status": row.get("status"),
                    "current_center_gap_m": row.get("current_center_gap_m"),
                    "projected_inner_gap_m": projected_gap.get("finger_inner_gap_m"),
                    "object_gap_to_lower_finger_m": lower_gap_f,
                    "object_gap_to_upper_finger_m": upper_gap_f,
                    "lower_crossed": lower_crossed,
                    "upper_crossed": upper_crossed,
                    "object_interval_m": projected_gap.get("object_interval_m"),
                    "lower_inner_surface_m": projected_gap.get("lower_inner_surface_m"),
                    "upper_inner_surface_m": projected_gap.get("upper_inner_surface_m"),
                },
            }
        )
        return result

    result["status"] = "PASS_OBJECT_STAYS_IN_CAPTURE_REGION_UNTIL_BILATERAL_CONTACT"
    return result


def _bilateral_grasp_failure_diagnostic(
    *,
    bilateral_grasp_formation_gate: dict[str, Any],
    contact_summary: dict[str, Any],
    start_alignment: dict[str, Any],
    final_alignment: dict[str, Any],
) -> dict[str, Any]:
    """Explain one-sided grasp failures without changing the formal gate."""

    if bool(bilateral_grasp_formation_gate.get("pass")):
        return {
            "required": True,
            "status": "PASS_BILATERAL_GRASP_FORMED",
            "notes": "Both expected finger proxies formed target contact under the existing bilateral gate.",
        }

    quality_by_finger = contact_summary.get("target_contact_quality_by_finger") or {}
    finger_rows: list[dict[str, Any]] = []
    contacted_fingers: list[str] = []
    missing_fingers: list[str] = []
    for finger_path, quality in sorted(quality_by_finger.items()):
        row = {
            "finger_path": finger_path,
            "contact_step_count": int(quality.get("contact_step_count") or 0),
            "nonzero_impulse_step_count": int(quality.get("nonzero_impulse_step_count") or 0),
            "first_step": quality.get("first_step"),
            "last_step": quality.get("last_step"),
            "max_impulse_norm": quality.get("max_impulse_norm"),
        }
        finger_rows.append(row)
        if row["contact_step_count"] > 0:
            contacted_fingers.append(finger_path)
        else:
            missing_fingers.append(finger_path)

    start_gap = start_alignment.get("closing_axis_projected_inner_gap") or {}
    final_gap = final_alignment.get("closing_axis_projected_inner_gap") or {}
    final_along = final_alignment.get("object_offset_along_closing_axis_m")
    final_cross = final_alignment.get("object_cross_closing_axis_offset_norm_m")
    suggested_shift = None
    try:
        suggested_shift = -float(final_along)
    except Exception:
        pass

    gap_values = []
    for value in (final_gap.get("object_gap_to_lower_finger_m"), final_gap.get("object_gap_to_upper_finger_m")):
        try:
            gap_values.append(float(value))
        except Exception:
            pass

    if len(contacted_fingers) == 1 and len(missing_fingers) == 1:
        if gap_values and max(gap_values) > 0.005 and min(gap_values) < 0.0:
            status = "FAIL_SINGLE_SIDE_GEOMETRIC_MISS"
            notes = (
                "One target finger contacted the bottle while the final closing-axis interval remained biased "
                "toward one side. Diagnose bottle reset pose, yaw, and contact-band alignment before changing "
                "friction, CCD, or contact reporting."
            )
        else:
            status = "FAIL_SINGLE_SIDE_CONTACT_MISSING_UNCLASSIFIED"
            notes = (
                "One target finger contacted but the final projected gap does not conclusively classify the miss. "
                "Inspect per-side closest gap/contact rows before modifying colliders."
            )
    elif not contacted_fingers:
        status = "FAIL_NO_TARGET_FINGER_CONTACT"
        notes = "No expected target finger contacted the object; this is not yet a bilateral grasp formation issue."
    else:
        status = str(bilateral_grasp_formation_gate.get("status") or "FAIL_BILATERAL_GRASP_FORMATION")
        notes = "Bilateral contact did not satisfy contact count, nonzero impulse, impulse balance, or prelift sweep."

    return {
        "required": True,
        "status": status,
        "contacted_fingers": contacted_fingers,
        "missing_fingers": missing_fingers,
        "finger_rows": finger_rows,
        "start_closing_axis_projected_inner_gap": start_gap,
        "final_closing_axis_projected_inner_gap": final_gap,
        "final_object_offset_along_closing_axis_m": final_along,
        "final_object_cross_closing_axis_offset_norm_m": final_cross,
        "suggested_centering_shift_along_closing_axis_m": suggested_shift,
        "notes": notes,
    }


def _object_lift_gate(*, object_lift: float, min_object_lift: float) -> dict[str, Any]:
    """Gate lift only when the caller explicitly requests a positive lift.

    Contact-only tabletop grasp gates can legitimately roll or settle by a few
    millimeters while closing.  A positive ``--min-object-lift`` upgrades the
    validation to a dynamic grasp/lift gate.
    """

    required = bool(float(min_object_lift) > 0.0)
    lift = float(object_lift)
    threshold = float(min_object_lift)
    if not required:
        return {
            "required": False,
            "pass": True,
            "status": "SKIPPED_CONTACT_ONLY_GATE",
            "object_lift_m": lift,
            "min_object_lift_m": threshold,
            "notes": (
                "Lift is not required because --min-object-lift is <= 0. "
                "Use a positive threshold for dynamic lift validation."
            ),
        }
    return {
        "required": True,
        "pass": bool(lift >= threshold),
        "status": "PASS_OBJECT_LIFT" if lift >= threshold else "FAIL_OBJECT_LIFT_BELOW_THRESHOLD",
        "object_lift_m": lift,
        "min_object_lift_m": threshold,
        "notes": "Positive --min-object-lift requests a dynamic lift/transport gate.",
    }


def _lift_transport_gate(
    *,
    rows: list[dict[str, Any]],
    object_lift_gate: dict[str, Any],
    contact_summary: dict[str, Any],
    min_object_lift: float,
    diagnostic_held_object_mode: str,
    min_follow_ratio: float = 0.5,
    min_contact_steps: int = 10,
) -> dict[str, Any]:
    required = bool(float(min_object_lift) > 0.0)
    lift_rows = [row for row in rows if row.get("phase") == "post_close_lift"]
    close_rows = [row for row in rows if row.get("phase") == "close"]
    eval_rows = lift_rows or close_rows
    eval_phase = "post_close_lift" if lift_rows else "close"
    if not required:
        return {
            "required": False,
            "pass": True,
            "status": "SKIPPED_NO_LIFT_THRESHOLD",
            "lift_mode": "contact_only",
            "object_attachment": "none" if diagnostic_held_object_mode == "none" else diagnostic_held_object_mode,
            "notes": "Lift/transport is skipped because --min-object-lift is <= 0.",
        }
    if not eval_rows:
        return {
            "required": True,
            "pass": False,
            "status": "FAIL_NO_CLOSE_ROWS_FOR_LIFT",
            "lift_mode": "recorded_hdf5_zero_order_hold",
        }

    first = eval_rows[0]
    last = eval_rows[-1]
    object_height_delta = float(last["object_center_z"] - first["object_center_z"])
    gripper_start = first.get("finger_mid_center_z")
    gripper_end = last.get("finger_mid_center_z")
    gripper_height_delta = (
        None if gripper_start is None or gripper_end is None else float(gripper_end) - float(gripper_start)
    )
    object_follow_ratio = (
        None
        if gripper_height_delta is None or abs(gripper_height_delta) < 1e-9
        else float(object_height_delta / gripper_height_delta)
    )
    target_contact_steps = list(contact_summary.get("target_contact_steps") or [])
    phase_target_contact_steps = [
        step
        for step in target_contact_steps
        if isinstance(step, dict) and step.get("phase") == eval_phase
    ]
    if not phase_target_contact_steps:
        phase_target_contact_steps = [
            row.get("step")
            for row in rows
            if row.get("phase") == eval_phase and row.get("target_contact_pair_found")
        ]
    categories = contact_summary.get("object_contact_categories") or {}
    table_like = categories.get("workcell_or_environment") or {}
    table_phase_counts = table_like.get("phase_counts") or {}
    object_attachment = "none" if diagnostic_held_object_mode == "none" else diagnostic_held_object_mode
    no_attachment = object_attachment == "none"
    contact_persist_ok = len(phase_target_contact_steps or target_contact_steps) >= int(min_contact_steps)
    gripper_lift_ok = bool(gripper_height_delta is not None and gripper_height_delta >= float(min_object_lift))
    follow_ok = bool(object_follow_ratio is not None and object_follow_ratio >= float(min_follow_ratio))
    strict_lift_clearance_gate = {
        "pass": bool(object_lift_gate["pass"]),
        "status": object_lift_gate["status"],
        "min_object_lift_m": float(min_object_lift),
        "object_lift_gate": object_lift_gate,
    }
    transport_follow_gate = {
        "pass": bool(object_height_delta > 0.0 and follow_ok and contact_persist_ok),
        "status": "PASS_TRANSPORT_FOLLOW_RATIO"
        if bool(object_height_delta > 0.0 and follow_ok and contact_persist_ok)
        else "FAIL_TRANSPORT_FOLLOW_RATIO",
        "object_height_delta_m": object_height_delta,
        "gripper_height_delta_m": gripper_height_delta,
        "object_follow_ratio": object_follow_ratio,
        "min_follow_ratio": float(min_follow_ratio),
        "contact_persist_ok": contact_persist_ok,
        "target_contact_persistence_steps": len(phase_target_contact_steps or target_contact_steps),
        "min_contact_steps": int(min_contact_steps),
    }
    table_load_bearing_contact_gate = _table_load_bearing_contact_gate(
        contact_summary=contact_summary,
        eval_phase=eval_phase,
    )
    pass_gate = bool(
        no_attachment
        and object_lift_gate["pass"]
        and gripper_lift_ok
        and follow_ok
        and contact_persist_ok
        and table_load_bearing_contact_gate["pass"]
    )
    if not no_attachment:
        status = "FAIL_OBJECT_ATTACHMENT_ENABLED"
    elif not gripper_lift_ok:
        status = "FAIL_GRIPPER_DID_NOT_LIFT"
    elif not follow_ok:
        status = "FAIL_OBJECT_DID_NOT_FOLLOW_GRIPPER"
    elif not contact_persist_ok:
        status = "FAIL_FINGER_CONTACT_NOT_PERSISTENT_DURING_LIFT"
    elif not table_load_bearing_contact_gate["pass"]:
        status = "FAIL_TABLE_STILL_LOAD_BEARING"
    elif not object_lift_gate["pass"]:
        status = "FAIL_STRICT_OBJECT_LIFT_CLEARANCE"
    else:
        status = "PASS_LIFT_TRANSPORT"
    return {
        "required": True,
        "pass": pass_gate,
        "status": status,
        "lift_mode": "post_close_lift_diagnostic" if eval_phase == "post_close_lift" else "recorded_hdf5_zero_order_hold",
        "eval_phase": eval_phase,
        "formal_replay": bool(eval_phase == "close"),
        "diagnostic_only": bool(eval_phase == "post_close_lift"),
        "object_attachment": object_attachment,
        "object_height_initial_m": float(first["object_center_z"]),
        "object_height_final_m": float(last["object_center_z"]),
        "object_height_delta_m": object_height_delta,
        "gripper_height_initial_m": None if gripper_start is None else float(gripper_start),
        "gripper_height_final_m": None if gripper_end is None else float(gripper_end),
        "gripper_height_delta_m": gripper_height_delta,
        "object_follow_ratio": object_follow_ratio,
        "min_object_lift_m": float(min_object_lift),
        "min_follow_ratio": float(min_follow_ratio),
        "target_contact_persistence_steps": len(phase_target_contact_steps or target_contact_steps),
        "target_contact_persistence_steps_all_phases": len(target_contact_steps),
        "min_contact_steps": int(min_contact_steps),
        "table_contact_phase_counts": table_phase_counts,
        "strict_lift_clearance_gate": strict_lift_clearance_gate,
        "transport_follow_gate": transport_follow_gate,
        "table_load_bearing_contact_gate": table_load_bearing_contact_gate,
        "object_lift_gate_status": object_lift_gate["status"],
        "notes": (
            "This gate validates whether the recorded post-grasp HDF5 motion naturally lifts the dynamic object. "
            "It does not allow object pose following, attachment, frame deletion, or target smoothing."
        ),
    }


def _active_grasp_geometry_precondition(
    *,
    require_active_target_contact: bool,
    already_in_contact_setup: bool,
    loaded_contact_placement: bool,
    open_left_box: dict[str, Any],
    open_right_box: dict[str, Any],
    object_box: dict[str, Any],
    gap_axis: int,
    clearance: float,
    object_projected_interval: tuple[float, float] | None = None,
    object_projection_model: dict[str, Any] | None = None,
    use_oriented_finger_boxes: bool = False,
) -> dict[str, Any]:
    """Check whether a no-contact-at-start active grasp is geometrically possible.

    This is intentionally a narrow precondition for the free-space active-contact
    gate. It applies when the object is initially placed inside the future finger
    gap. The scene_base_link ALOHA proxies are small fingertip reference proxies,
    but their axis-aligned bboxes can include enough geometry to make an AABB
    surface-gap test overly conservative.  Use the two proxy centers as the
    active grasp line, and keep the AABB surface gap as a diagnostic only.
    """

    row: dict[str, Any] = {
        "required": bool(require_active_target_contact),
        "already_in_contact_setup": bool(already_in_contact_setup),
        "loaded_contact_placement": bool(loaded_contact_placement),
        "applies": bool(require_active_target_contact and not already_in_contact_setup),
        "mode": "in_gap_free_space_first_contact",
        "pass": True,
        "status": "SKIPPED_ACTIVE_GRASP_GEOMETRY_PRECONDITION",
    }
    if already_in_contact_setup or not require_active_target_contact:
        row["applies"] = False
        row["pass"] = None
        return row
    if loaded_contact_placement:
        row.update(
            {
                "applies": False,
                "pass": True,
                "status": "SKIPPED_LOADED_CONTACT_PLACEMENT",
                "mode": "loaded_contact_placement",
                "notes": (
                    "The free-space closing-axis gap precondition applies to open-frame object placement. "
                    "For hdf5_close_* loaded-contact placement, the close-frame gap may intentionally be "
                    "smaller than the soft-bottle effective contact width, so contact formation must be "
                    "judged by the phase-specific contact, tracking, and fixed-reference geometry gates."
                ),
            }
        )
        return row
    if not (
        open_left_box.get("bbox_valid")
        and open_right_box.get("bbox_valid")
        and object_box.get("bbox_valid")
        and object_box.get("size")
    ):
        row.update({"pass": False, "status": "FAIL_ACTIVE_GRASP_GEOMETRY_BBOX_INVALID"})
        return row

    left_center = np.asarray(open_left_box["center"], dtype=np.float64)
    right_center = np.asarray(open_right_box["center"], dtype=np.float64)
    center_delta = left_center - right_center
    open_center_gap = float(np.linalg.norm(center_delta))
    open_surface_gap = _surface_gap(open_left_box, open_right_box, gap_axis)
    object_size = np.asarray(object_box["size"], dtype=np.float64).reshape(-1)
    object_width_along_gap_axis = float(object_size[gap_axis])
    object_width_centerline = (
        float(object_projected_interval[1] - object_projected_interval[0])
        if object_projected_interval is not None
        else float(np.median(object_size))
    )
    required_open_center_gap = object_width_centerline + float(clearance)
    centerline_pass = bool(open_center_gap >= required_open_center_gap)

    projected_inner_gap: dict[str, Any] = {"valid": False}
    if open_center_gap > 1e-12 and np.isfinite(open_center_gap):
        closing_unit = center_delta / open_center_gap
        left_projection = float(np.dot(left_center, closing_unit))
        right_projection = float(np.dot(right_center, closing_unit))
        lower_box, upper_box = (
            (open_right_box, open_left_box) if right_projection <= left_projection else (open_left_box, open_right_box)
        )
        if object_projected_interval is not None:
            projected_inner_gap = _projected_inner_gap_for_interval(
                lower_box=lower_box,
                upper_box=upper_box,
                object_interval=object_projected_interval,
                unit=closing_unit,
                use_oriented_finger_boxes=use_oriented_finger_boxes,
            )
        else:
            projected_inner_gap = _projected_inner_gap(
                lower_box=lower_box,
                upper_box=upper_box,
                object_box=object_box,
                unit=closing_unit,
                use_oriented_finger_boxes=use_oriented_finger_boxes,
            )
    projected_gap_pass = bool(
        projected_inner_gap.get("valid")
        and projected_inner_gap.get("object_gap_to_lower_finger_m") is not None
        and projected_inner_gap.get("object_gap_to_upper_finger_m") is not None
        and float(projected_inner_gap["object_gap_to_lower_finger_m"]) >= float(clearance)
        and float(projected_inner_gap["object_gap_to_upper_finger_m"]) >= float(clearance)
    )
    pass_gate = bool(centerline_pass and projected_gap_pass)
    if pass_gate:
        status = "PASS_ACTIVE_GRASP_GEOMETRY_PRECONDITION"
    elif not centerline_pass:
        status = "FAIL_ACTIVE_FREE_SPACE_CENTERLINE_GEOMETRY_PRECONDITION"
    else:
        status = "FAIL_ACTIVE_FREE_SPACE_TRUE_CLOSING_AXIS_GEOMETRY_PRECONDITION"
    row.update(
        {
            "pass": pass_gate,
            "status": status,
            "mode": "proxy_centerline_free_space_first_contact",
            "gap_axis_index": int(gap_axis),
            "open_finger_center_gap_m": float(open_center_gap),
            "open_finger_surface_gap_m": float(open_surface_gap),
            "surface_gap_is_diagnostic_only": True,
            "finger_center_delta_world_m": center_delta.tolist(),
            "object_width_along_gap_axis_m": object_width_along_gap_axis,
            "object_width_centerline_m": object_width_centerline,
            "required_open_center_gap_m": float(required_open_center_gap),
            "centerline_gap_pass": centerline_pass,
            "true_closing_axis_gap_pass": projected_gap_pass,
            "closing_axis_projected_inner_gap": projected_inner_gap,
            "finger_projection_model": "oriented_box_support" if use_oriented_finger_boxes else "world_aabb",
            "object_projection_model": object_projection_model or {"source": "world_aabb"},
            "clearance_m": float(clearance),
            "shortfall_m": float(max(required_open_center_gap - open_center_gap, 0.0)),
            "notes": (
                "The centerline gap is only a coarse feasibility check. The hard condition is the true "
                "closing-axis projected inner gap, because AABB/world-axis gaps can report inside while the "
                "bottle body still penetrates one or both moving finger pads."
            ),
        }
    )
    return row


def _open_finger_object_height_alignment(
    *,
    require_active_target_contact: bool,
    already_in_contact_setup: bool,
    open_left_box: dict[str, Any],
    open_right_box: dict[str, Any],
    object_box: dict[str, Any],
    max_error: float,
) -> dict[str, Any]:
    """Check that the open gripper is at the table bottle body's height.

    A free-space tabletop grasp test is invalid if the bottle is placed on the
    table but the first replay fingertip midpoint is far above the bottle body.
    Keep this as a gate instead of silently shifting or deleting frames.
    """

    row: dict[str, Any] = {
        "required": bool(require_active_target_contact),
        "already_in_contact_setup": bool(already_in_contact_setup),
        "pass": True,
        "status": "SKIPPED_OPEN_FINGER_OBJECT_HEIGHT_ALIGNMENT",
        "max_error_m": float(max_error),
    }
    if already_in_contact_setup or not require_active_target_contact:
        return row
    if not (
        open_left_box.get("bbox_valid")
        and open_right_box.get("bbox_valid")
        and object_box.get("bbox_valid")
        and object_box.get("center")
    ):
        row.update({"pass": False, "status": "FAIL_HEIGHT_ALIGNMENT_BBOX_INVALID"})
        return row

    left_center = np.asarray(open_left_box["center"], dtype=np.float64)
    right_center = np.asarray(open_right_box["center"], dtype=np.float64)
    object_center = np.asarray(object_box["center"], dtype=np.float64)
    finger_midpoint = (left_center + right_center) / 2.0
    height_error = float(abs(finger_midpoint[2] - object_center[2]))
    pass_gate = bool(height_error <= float(max_error))
    row.update(
        {
            "pass": pass_gate,
            "status": "PASS_OPEN_FINGER_OBJECT_HEIGHT_ALIGNMENT"
            if pass_gate
            else "FAIL_OPEN_FINGER_OBJECT_HEIGHT_MISMATCH",
            "finger_midpoint_world_m": finger_midpoint.tolist(),
            "object_center_world_m": object_center.tolist(),
            "finger_midpoint_z_m": float(finger_midpoint[2]),
            "object_center_z_m": float(object_center[2]),
            "height_error_m": height_error,
        }
    )
    return row


def _tabletop_collision_audit(stage: Any, table_path: str | None, max_rows: int = 24) -> dict[str, Any]:
    """Summarize whether a tabletop prim has enabled collider descendants.

    Isaac/PhysX contact depends on collision schemas, not visible mesh names.
    Keep this audit bounded so a large composed stage does not flood reports.
    """

    if not table_path:
        return {
            "table_path": table_path,
            "table_exists": False,
            "enabled_collision_prim_count": 0,
            "collision_prim_count": 0,
            "rows": [],
            "status": "FAIL_TABLETOP_REFERENCE_PATH_MISSING",
        }
    from pxr import Usd

    root = stage.GetPrimAtPath(table_path)
    if not root or not root.IsValid():
        return {
            "table_path": table_path,
            "table_exists": False,
            "enabled_collision_prim_count": 0,
            "collision_prim_count": 0,
            "rows": [],
            "status": "FAIL_TABLETOP_REFERENCE_PRIM_MISSING",
        }
    rows: list[dict[str, Any]] = []
    collision_count = 0
    enabled_count = 0
    for prim in Usd.PrimRange(root):
        schemas = [str(item) for item in prim.GetAppliedSchemas()]
        if "PhysicsCollisionAPI" not in schemas:
            continue
        collision_count += 1
        enabled_attr = prim.GetAttribute("physics:collisionEnabled")
        enabled_value = enabled_attr.Get() if enabled_attr and enabled_attr.HasAuthoredValueOpinion() else True
        enabled = bool(enabled_value)
        if enabled:
            enabled_count += 1
        if len(rows) < int(max_rows):
            rows.append(
                {
                    "path": str(prim.GetPath()),
                    "enabled": enabled,
                    "authored_collision_enabled": bool(
                        enabled_attr and enabled_attr.HasAuthoredValueOpinion()
                    ),
                }
            )
    status = "PASS_TABLETOP_REFERENCE_HAS_ENABLED_COLLIDER" if enabled_count else "FAIL_TABLETOP_REFERENCE_NO_ENABLED_COLLIDER"
    return {
        "table_path": table_path,
        "table_exists": True,
        "enabled_collision_prim_count": enabled_count,
        "collision_prim_count": collision_count,
        "rows": rows,
        "truncated": bool(collision_count > len(rows)),
        "status": status,
    }


def _tabletop_reference_contract(
    *,
    required: bool,
    require_open_height_alignment: bool,
    tabletop_adjustment: dict[str, Any] | None,
    table_collision_audit: dict[str, Any] | None,
    open_left_box: dict[str, Any],
    open_right_box: dict[str, Any],
    object_box: dict[str, Any],
    max_finger_object_center_height_error: float,
    max_tabletop_gap_error: float = 0.002,
) -> dict[str, Any]:
    """Formal tabletop/reference gate for fixed-pose tabletop grasp validation.

    A table is calibrated enough for Gate2 only if it is a valid collidable
    tabletop and the bottle is resting on that tabletop.  Open-frame height is
    a hard check only for open-frame placement.  For loaded/close-frame
    placement the open frame is an approach pose, so height is reported as a
    diagnostic and the contact-frame/bilateral gates carry the formal burden.
    """

    row: dict[str, Any] = {
        "required": bool(required),
        "open_height_alignment_required": bool(require_open_height_alignment),
        "pass": True,
        "status": "SKIPPED_TABLETOP_REFERENCE_CONTRACT",
    }
    if not required:
        return row
    if not tabletop_adjustment:
        row.update({"pass": False, "status": "FAIL_TABLETOP_ADJUSTMENT_MISSING"})
        return row
    if not tabletop_adjustment.get("pass"):
        row.update(
            {
                "pass": False,
                "status": tabletop_adjustment.get("status", "FAIL_TABLETOP_ADJUSTMENT_FAILED"),
                "tabletop_adjustment": tabletop_adjustment,
                "table_collision_audit": table_collision_audit,
            }
        )
        return row
    table_box = tabletop_adjustment.get("table_bbox") or {}
    if not table_box.get("bbox_valid"):
        row.update(
            {
                "pass": False,
                "status": "FAIL_TABLETOP_REFERENCE_BBOX_INVALID",
                "tabletop_adjustment": tabletop_adjustment,
                "table_collision_audit": table_collision_audit,
            }
        )
        return row
    enabled_collision_count = 0
    if table_collision_audit is not None:
        enabled_collision_count = int(table_collision_audit.get("enabled_collision_prim_count") or 0)
    if enabled_collision_count <= 0:
        row.update(
            {
                "pass": False,
                "status": "FAIL_TABLETOP_REFERENCE_NO_ENABLED_COLLIDER",
                "tabletop_adjustment": tabletop_adjustment,
                "table_collision_audit": table_collision_audit,
            }
        )
        return row
    tabletop_gap = tabletop_adjustment.get("tabletop_gap_after_m")
    if tabletop_gap is None or abs(float(tabletop_gap) - float(tabletop_adjustment["tabletop_clearance_m"])) > float(
        max_tabletop_gap_error
    ):
        row.update(
            {
                "pass": False,
                "status": "FAIL_TABLETOP_OBJECT_NOT_ON_TABLETOP",
                "tabletop_adjustment": tabletop_adjustment,
                "table_collision_audit": table_collision_audit,
                "max_tabletop_gap_error_m": float(max_tabletop_gap_error),
            }
        )
        return row
    if not (
        open_left_box.get("bbox_valid")
        and open_right_box.get("bbox_valid")
        and object_box.get("bbox_valid")
        and object_box.get("center")
    ):
        row.update(
            {
                "pass": False,
                "status": "FAIL_TABLE_ROBOT_FRAME_AUDIT_BBOX_INVALID",
                "tabletop_adjustment": tabletop_adjustment,
                "table_collision_audit": table_collision_audit,
            }
        )
        return row
    left_center = np.asarray(open_left_box["center"], dtype=np.float64)
    right_center = np.asarray(open_right_box["center"], dtype=np.float64)
    object_center = np.asarray(object_box["center"], dtype=np.float64)
    finger_midpoint = (left_center + right_center) / 2.0
    height_error = float(abs(finger_midpoint[2] - object_center[2]))
    height_ok = bool(height_error <= float(max_finger_object_center_height_error))
    pass_gate = bool(height_ok or not require_open_height_alignment)
    if height_ok:
        status = "PASS_CALIBRATED_TABLETOP_REFERENCE"
    elif require_open_height_alignment:
        status = "FAIL_TABLE_ROBOT_FRAME_MISMATCH"
    else:
        status = "PASS_TABLETOP_REFERENCE_OPEN_HEIGHT_DIAGNOSTIC_ONLY"
    row.update(
        {
            "pass": pass_gate,
            "status": status,
            "table_path": tabletop_adjustment.get("table_path"),
            "table_top_z_m": tabletop_adjustment.get("table_top_z_m"),
            "object_bottom_z_after_m": tabletop_adjustment.get("object_bottom_z_after_m"),
            "tabletop_gap_after_m": tabletop_gap,
            "tabletop_clearance_m": tabletop_adjustment.get("tabletop_clearance_m"),
            "finger_midpoint_world_m": finger_midpoint.tolist(),
            "object_center_world_m": object_center.tolist(),
            "finger_midpoint_z_m": float(finger_midpoint[2]),
            "object_center_z_m": float(object_center[2]),
            "finger_object_center_height_error_m": height_error,
            "max_finger_object_center_height_error_m": float(max_finger_object_center_height_error),
            "table_collision_audit": table_collision_audit,
            "notes": (
                "Gate2 requires a real collidable tabletop and a bottle resting on that table. For "
                "open-frame placement this also requires open finger height compatibility. For close-frame "
                "placement the open frame is an approach pose, so the height mismatch is diagnostic only."
            ),
        }
    )
    return row


def _object_width_stop_target(
    *,
    enabled: bool,
    current_qpos: np.ndarray,
    target: np.ndarray,
    dof_names: list[str],
    finger_dof_names: dict[str, str],
    left_box: dict[str, Any],
    right_box: dict[str, Any],
    object_box: dict[str, Any],
    clearance: float,
    predictive_margin: float = 0.0,
    object_projected_interval: tuple[float, float] | None = None,
    use_oriented_finger_boxes: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Prevent commanded finger targets from closing past the object body width.

    This is a control-target guard, not a replacement for PhysX contact.  If the
    current measured finger center distance is already at the bottle-width
    threshold, keep the finger targets at the current realized finger qpos while
    allowing arm targets to continue.  The goal is to avoid asking the simulated
    gripper to pass through the bottle body when contact/friction tuning is still
    under validation.
    """

    row: dict[str, Any] = {
        "enabled": bool(enabled),
        "active": False,
        "status": "DISABLED",
    }
    if not enabled:
        return target, row
    if not (
        left_box.get("bbox_valid")
        and right_box.get("bbox_valid")
        and object_box.get("bbox_valid")
        and object_box.get("size")
    ):
        row.update({"status": "SKIPPED_BBOX_INVALID"})
        return target, row

    left_center = np.asarray(left_box["center"], dtype=np.float64)
    right_center = np.asarray(right_box["center"], dtype=np.float64)
    center_delta = left_center - right_center
    current_center_gap = float(np.linalg.norm(center_delta))
    gap_axis = int(np.argmax(np.abs(center_delta))) if np.all(np.isfinite(center_delta)) else 0
    current_surface_gap = float(_surface_gap(left_box, right_box, gap_axis))
    object_size = np.asarray(object_box["size"], dtype=np.float64).reshape(-1)
    object_width_centerline = float(np.median(object_size))
    object_width_along_gap_axis = float(object_size[gap_axis])
    early_margin = max(0.0, float(predictive_margin))
    stop_center_gap = object_width_centerline + float(clearance)
    stop_surface_gap = object_width_along_gap_axis + float(clearance)
    projected_stop_gap: dict[str, Any] = {"valid": False}
    projected_stop_threshold = None
    if current_center_gap > 1e-12 and np.isfinite(current_center_gap) and object_projected_interval is not None:
        closing_unit = center_delta / current_center_gap
        left_projection = float(np.dot(left_center, closing_unit))
        right_projection = float(np.dot(right_center, closing_unit))
        lower_box, upper_box = (
            (right_box, left_box) if right_projection <= left_projection else (left_box, right_box)
        )
        projected_stop_gap = _projected_inner_gap_for_interval(
            lower_box=lower_box,
            upper_box=upper_box,
            object_interval=object_projected_interval,
            unit=closing_unit,
            use_oriented_finger_boxes=use_oriented_finger_boxes,
        )
        if projected_stop_gap.get("valid"):
            projected_width = float(object_projected_interval[1] - object_projected_interval[0])
            projected_stop_threshold = projected_width + float(clearance)
    row.update(
        {
            "status": "OBSERVED_FINGER_GAP_ABOVE_OBJECT_WIDTH",
            "mode": "closing_axis_projected_inner_gap"
            if projected_stop_gap.get("valid")
            else "axis_aligned_aabb_fallback",
            "finger_projection_model": "oriented_box_support" if use_oriented_finger_boxes else "world_aabb",
            "gap_axis_index": gap_axis,
            "current_center_gap_m": current_center_gap,
            "current_surface_gap_m": current_surface_gap,
            "object_width_centerline_m": object_width_centerline,
            "object_width_along_gap_axis_m": object_width_along_gap_axis,
            "clearance_m": float(clearance),
            "stop_center_gap_m": stop_center_gap,
            "stop_surface_gap_m": stop_surface_gap,
            "projected_inner_gap": projected_stop_gap,
            "predictive_margin_m": early_margin,
            "projected_stop_gap_m": None
            if projected_stop_threshold is None
            else float(projected_stop_threshold),
            "projected_effective_stop_gap_m": None
            if projected_stop_threshold is None
            else float(projected_stop_threshold + early_margin),
        }
    )
    if projected_stop_gap.get("valid"):
        if float(projected_stop_gap["finger_inner_gap_m"]) > float(projected_stop_threshold + early_margin):
            return target, row
    elif current_center_gap > stop_center_gap + early_margin and current_surface_gap > stop_surface_gap + early_margin:
        return target, row

    guarded = np.asarray(target, dtype=np.float64).copy()
    for logical_name in ("left_finger", "right_finger"):
        idx = dof_names.index(finger_dof_names[logical_name])
        guarded[idx] = float(current_qpos[idx])
    row.update(
        {
            "active": True,
            "status": "ACTIVE_HOLD_FINGER_TARGETS_AT_OBJECT_WIDTH",
            "held_left_finger_qpos": float(guarded[dof_names.index(finger_dof_names["left_finger"])]),
            "held_right_finger_qpos": float(guarded[dof_names.index(finger_dof_names["right_finger"])]),
        }
    )
    return guarded, row


def _contact_geometry_bbox_path(object_shape: str, object_path: str) -> str:
    """Return the prim whose bbox should define physical contact width.

    BottleUSD cylinder-proxy objects use the Bottle500 mesh for visual/semantic
    checks and a separate cylinder for contact.  Width guards and active-contact
    free-space checks must use the contact proxy, or the visual mesh bbox can
    stop the fingers before the physical proxy is actually reachable.
    """

    if object_shape == "bottle_usd_cylinder_proxy":
        return f"{object_path}/physics_proxy"
    if object_shape in {"bottle_usd_segmented_proxy", "bottle_usd_grasp_band_proxy", "bottle_usd_grasp_box_proxy"}:
        return f"{object_path}/physics_proxy/body"
    return object_path


def _bind_contact_physics_material(
    stage: Any,
    *,
    prim_path: str,
    material_path: str,
    static_friction: float | None,
    dynamic_friction: float | None,
    restitution: float | None,
) -> dict[str, Any]:
    if static_friction is None and dynamic_friction is None and restitution is None:
        return {
            "bound": False,
            "status": "SKIPPED_NO_CONTACT_MATERIAL_REQUESTED",
            "prim_path": prim_path,
            "material_path": material_path,
        }

    from pxr import UsdPhysics
    from pxr import UsdShade

    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return {
            "bound": False,
            "status": "FAIL_CONTACT_MATERIAL_TARGET_MISSING",
            "prim_path": prim_path,
            "material_path": material_path,
        }

    material = UsdShade.Material.Define(stage, material_path)
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    if static_friction is not None:
        material_api.CreateStaticFrictionAttr(float(static_friction))
    if dynamic_friction is not None:
        material_api.CreateDynamicFrictionAttr(float(dynamic_friction))
    if restitution is not None:
        material_api.CreateRestitutionAttr(float(restitution))
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    return {
        "bound": True,
        "status": "PASS_CONTACT_MATERIAL_BOUND",
        "prim_path": prim_path,
        "material_path": material_path,
        "static_friction": static_friction,
        "dynamic_friction": dynamic_friction,
        "restitution": restitution,
    }


def _set_finger_target_and_step(world: Any, art: Any, target: np.ndarray, steps: int) -> None:
    for _ in range(steps):
        _set_full_target(art, target)
        world.step(render=False)


def _apply_replay_target_and_step(
    world: Any,
    art: Any,
    target: np.ndarray,
    *,
    actuation_mode: str,
    substep_mode: str = "zero_order_hold",
    previous_target: np.ndarray | None = None,
    target_hold_steps: int = 1,
) -> np.ndarray:
    """Apply one replay target for one or more physics steps.

    HDF5 actions are recorded at 50 Hz, but Isaac articulation drives may need
    several physics steps to settle near a new target.  The returned qpos is the
    state immediately before the first physics step, which preserves the old
    pre-step tracking diagnostic when ``target_hold_steps == 1``.
    """

    if target_hold_steps <= 0:
        raise ValueError(f"target_hold_steps must be positive, got {target_hold_steps}")
    if substep_mode not in {"zero_order_hold", "linear_interpolation_diagnostic"}:
        raise ValueError(f"unknown HDF5 replay substep mode: {substep_mode!r}")
    target_arr = np.asarray(target, dtype=np.float64).reshape(-1)
    previous_arr = target_arr if previous_target is None else np.asarray(previous_target, dtype=np.float64).reshape(-1)

    pre_step_qpos: np.ndarray | None = None
    for substep in range(target_hold_steps):
        if substep_mode == "linear_interpolation_diagnostic":
            alpha = float(substep + 1) / float(target_hold_steps)
            step_target = previous_arr + alpha * (target_arr - previous_arr)
        else:
            step_target = target_arr
        if actuation_mode == "drive_target":
            _set_full_target(art, step_target)
        elif actuation_mode == "state_teleport":
            _set_full_state(art, step_target)
            _set_full_target(art, step_target)
        else:
            raise ValueError(f"unknown HDF5 replay actuation mode: {actuation_mode!r}")
        if pre_step_qpos is None:
            pre_step_qpos = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1).copy()
        world.step(render=False)
    assert pre_step_qpos is not None
    return pre_step_qpos


def _surface_gap(left_box: dict[str, Any], right_box: dict[str, Any], axis: int) -> float:
    left_min = float(left_box["min"][axis])
    left_max = float(left_box["max"][axis])
    right_min = float(right_box["min"][axis])
    right_max = float(right_box["max"][axis])
    if left_max <= right_min:
        return right_min - left_max
    if right_max <= left_min:
        return left_min - right_max
    return 0.0


def _axis_probe_row(
    *,
    axis: int,
    left_box: dict[str, Any],
    right_box: dict[str, Any],
    object_box: dict[str, Any],
    target_finger_box: dict[str, Any],
) -> dict[str, float | None]:
    def pick(box: dict[str, Any], key: str) -> float | None:
        values = box.get(key)
        if values is None:
            return None
        return float(values[axis])

    return {
        "left_axis_min": pick(left_box, "min"),
        "left_axis_max": pick(left_box, "max"),
        "left_axis_center": pick(left_box, "center"),
        "right_axis_min": pick(right_box, "min"),
        "right_axis_max": pick(right_box, "max"),
        "right_axis_center": pick(right_box, "center"),
        "object_axis_min": pick(object_box, "min"),
        "object_axis_max": pick(object_box, "max"),
        "object_axis_center": pick(object_box, "center"),
        "target_finger_object_surface_gap": _surface_gap(target_finger_box, object_box, axis)
        if target_finger_box.get("bbox_valid") and object_box.get("bbox_valid")
        else None,
    }


def _finger_center_row(left_box: dict[str, Any], right_box: dict[str, Any]) -> dict[str, float | None]:
    def center_value(box: dict[str, Any], axis: int) -> float | None:
        center = box.get("center")
        if center is None:
            return None
        return float(center[axis])

    row: dict[str, float | None] = {
        "left_finger_center_x": center_value(left_box, 0),
        "left_finger_center_y": center_value(left_box, 1),
        "left_finger_center_z": center_value(left_box, 2),
        "right_finger_center_x": center_value(right_box, 0),
        "right_finger_center_y": center_value(right_box, 1),
        "right_finger_center_z": center_value(right_box, 2),
    }
    if row["left_finger_center_z"] is None or row["right_finger_center_z"] is None:
        row.update(
            {
                "finger_mid_center_x": None,
                "finger_mid_center_y": None,
                "finger_mid_center_z": None,
            }
        )
    else:
        row.update(
            {
                "finger_mid_center_x": (float(row["left_finger_center_x"]) + float(row["right_finger_center_x"])) / 2.0,
                "finger_mid_center_y": (float(row["left_finger_center_y"]) + float(row["right_finger_center_y"])) / 2.0,
                "finger_mid_center_z": (float(row["left_finger_center_z"]) + float(row["right_finger_center_z"])) / 2.0,
            }
        )
    return row


def _finger_mid_center_from_boxes(left_box: dict[str, Any], right_box: dict[str, Any]) -> np.ndarray:
    row = _finger_center_row(left_box, right_box)
    values = [row.get("finger_mid_center_x"), row.get("finger_mid_center_y"), row.get("finger_mid_center_z")]
    if any(value is None for value in values):
        return np.asarray([np.nan, np.nan, np.nan], dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _arm_indices_for_tracking_group(tracking_groups: dict[str, list[int]], dof_names: list[str], side: str) -> list[int]:
    indices = list(tracking_groups.get("left_arm") or [])
    if indices:
        return [int(index) for index in indices]
    base_arm_names = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
    side_arm_names = tuple(f"{side}_{name}" for name in base_arm_names)
    candidate_names = side_arm_names if all(name in dof_names for name in side_arm_names) else base_arm_names
    return [dof_names.index(name) for name in candidate_names if name in dof_names]


def _finite_difference_finger_mid_z_jacobian(
    *,
    stage: Any,
    art: Any,
    paths: dict[str, str],
    base_qpos: np.ndarray,
    arm_indices: list[int],
    epsilon: float,
    body_name: str | None = None,
) -> dict[str, Any]:
    """Estimate local gripper height sensitivity without stepping physics."""

    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    def measure_position() -> tuple[np.ndarray, str]:
        if body_name:
            view = getattr(art, "_articulation_view", None)
            body_names = list(getattr(view, "body_names", []) or [])
            if body_name in body_names:
                body_idx = body_names.index(body_name)
                raw = np.asarray(view._physics_view.get_link_transforms(), dtype=np.float64)
                raw = raw.reshape((-1, 7))
                return raw[body_idx, :3].copy(), f"articulation_body:{body_name}"
        return (
            _finger_mid_center_from_boxes(
                _bbox_row(stage, paths["left_finger"]),
                _bbox_row(stage, paths["right_finger"]),
            ),
            "finger_proxy_bbox_midpoint",
        )

    base_qpos = np.asarray(base_qpos, dtype=np.float64).reshape(-1)
    _set_full_state(art, base_qpos)
    _set_full_target(art, base_qpos)
    base_mid, measurement_source = measure_position()
    jacobian: list[float] = []
    perturb_rows: list[dict[str, Any]] = []
    dof_names = list(art.dof_names)
    for index in arm_indices:
        probe = base_qpos.copy()
        probe[int(index)] += float(epsilon)
        _set_full_state(art, probe)
        _set_full_target(art, probe)
        mid, _source = measure_position()
        dz = float(mid[2] - base_mid[2]) if np.isfinite(mid[2]) and np.isfinite(base_mid[2]) else float("nan")
        value = float(dz / float(epsilon)) if np.isfinite(dz) else float("nan")
        jacobian.append(value)
        perturb_rows.append(
            {
                "dof_index": int(index),
                "dof_name": dof_names[int(index)],
                "epsilon_rad": float(epsilon),
                "finger_mid_z_delta_m": dz,
                "finger_mid_z_jacobian_m_per_rad": value,
            }
        )
    _set_full_state(art, base_qpos)
    _set_full_target(art, base_qpos)
    jac = np.asarray(jacobian, dtype=np.float64)
    finite = bool(np.all(np.isfinite(jac)) and np.isfinite(base_mid).all())
    norm = float(np.linalg.norm(jac)) if finite else float("nan")
    return {
        "status": "PASS_JACOBIAN_ESTIMATED" if finite and norm > 0 else "FAIL_INVALID_OR_ZERO_JACOBIAN",
        "pass": bool(finite and norm > 0),
        "measurement_source": measurement_source,
        "body_name": body_name,
        "base_gripper_measurement_m": base_mid.tolist(),
        "arm_indices": [int(index) for index in arm_indices],
        "jacobian_m_per_rad": jac.tolist(),
        "jacobian_norm_m_per_rad": norm,
        "perturbations": perturb_rows,
        "notes": (
            "Finite-difference diagnostic for a small vertical lift. It sets articulation state directly, "
            "does not step physics, and does not move or attach the bottle."
        ),
    }


def _build_jacobian_vertical_lift_sequence(
    *,
    stage: Any,
    art: Any,
    paths: dict[str, str],
    base_qpos: np.ndarray,
    close_target: np.ndarray,
    tracking_groups: dict[str, list[int]],
    side: str,
    lift_height: float,
    lift_steps: int,
    epsilon: float,
    damping: float,
    max_joint_delta: float,
    body_name: str | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if lift_height <= 0:
        raise ValueError(f"lift_height must be positive, got {lift_height}")
    if lift_steps <= 0:
        raise ValueError(f"lift_steps must be positive, got {lift_steps}")
    if damping < 0:
        raise ValueError(f"damping must be non-negative, got {damping}")
    if max_joint_delta <= 0:
        raise ValueError(f"max_joint_delta must be positive, got {max_joint_delta}")
    dof_names = list(art.dof_names)
    arm_indices = _arm_indices_for_tracking_group(tracking_groups, dof_names, side)
    base_qpos = np.asarray(base_qpos, dtype=np.float64).reshape(-1)
    close_target = np.asarray(close_target, dtype=np.float64).reshape(-1)
    jacobian_report = _finite_difference_finger_mid_z_jacobian(
        stage=stage,
        art=art,
        paths=paths,
        base_qpos=base_qpos,
        arm_indices=arm_indices,
        epsilon=epsilon,
        body_name=body_name,
    )
    jac = np.asarray(jacobian_report.get("jacobian_m_per_rad") or [], dtype=np.float64)
    denom = float(np.dot(jac, jac) + float(damping))
    if not jacobian_report.get("pass") or not np.isfinite(denom) or denom <= 0:
        sequence = [close_target.copy() for _ in range(int(lift_steps))]
        return sequence, {
            "source": "jacobian_vertical",
            "status": "FAIL_JACOBIAN_VERTICAL_SEQUENCE_NOT_GENERATED",
            "pass": False,
            "requested_lift_height_m": float(lift_height),
            "step_count": int(lift_steps),
            "jacobian": jacobian_report,
            "notes": "Jacobian was invalid; generated a hold sequence so the diagnostic fails safely.",
        }
    delta_arm = jac * (float(lift_height) / denom)
    max_abs = float(np.max(np.abs(delta_arm))) if delta_arm.size else 0.0
    scale = 1.0
    clipped = False
    if max_abs > float(max_joint_delta):
        scale = float(max_joint_delta) / max_abs
        delta_arm = delta_arm * scale
        clipped = True
    achieved_lift_estimate = float(np.dot(jac, delta_arm))
    sequence: list[np.ndarray] = []
    for step in range(int(lift_steps)):
        alpha = 0.5 - 0.5 * np.cos(np.pi * float(step + 1) / float(lift_steps))
        target = close_target.copy()
        for local_idx, dof_index in enumerate(arm_indices):
            target[int(dof_index)] = float(base_qpos[int(dof_index)] + alpha * delta_arm[local_idx])
        sequence.append(target)
    return sequence, {
        "source": "jacobian_vertical",
        "status": "PASS_JACOBIAN_VERTICAL_SEQUENCE_GENERATED",
        "pass": True,
        "requested_lift_height_m": float(lift_height),
        "estimated_lift_height_m": achieved_lift_estimate,
        "step_count": int(lift_steps),
        "epsilon_rad": float(epsilon),
        "damping": float(damping),
        "max_joint_delta_rad": float(max_joint_delta),
        "delta_clipped": bool(clipped),
        "delta_scale": float(scale),
        "arm_delta_by_dof": [
            {"dof_index": int(index), "dof_name": dof_names[int(index)], "delta_rad": float(delta)}
            for index, delta in zip(arm_indices, delta_arm, strict=True)
        ],
        "jacobian": jacobian_report,
        "notes": (
            "Diagnostic-only local Jacobian lift from the post-close hold state. It tests whether the current "
            "contact model can transport the bottle under a small smooth upward gripper motion."
        ),
    }


def _timeseries_gripper_object_alignment_samples(
    *,
    rows: list[dict[str, Any]],
    contact_summary: dict[str, Any],
    moving_fingers: str,
    max_samples: int = 12,
) -> dict[str, Any]:
    """Sample center-line alignment at the target-contact landmarks."""

    if moving_fingers == "both":
        expected_finger_paths = sorted((contact_summary.get("target_contact_quality_by_finger") or {}).keys())
    else:
        expected_finger_paths = sorted((contact_summary.get("target_contact_quality_by_finger") or {}).keys())
    landmark_steps: set[int] = set()
    for quality in (contact_summary.get("target_contact_quality_by_finger") or {}).values():
        for key in ("first_step", "last_step", "first_nonzero_impulse_step", "last_nonzero_impulse_step"):
            value = quality.get(key)
            if value is not None:
                try:
                    landmark_steps.add(int(value))
                except Exception:
                    pass
    if not landmark_steps:
        return {
            "status": "NO_TARGET_CONTACT_LANDMARKS",
            "samples": [],
            "expected_finger_paths": expected_finger_paths,
        }

    close_rows_by_step = {
        int(row["step"]): row
        for row in rows
        if row.get("phase") == "close" and row.get("step") is not None
    }
    samples: list[dict[str, Any]] = []
    for step in sorted(landmark_steps)[: int(max_samples)]:
        row = close_rows_by_step.get(step)
        if row is None:
            samples.append({"step": step, "status": "MISSING_TIMESERIES_ROW"})
            continue
        keys = [
            "left_finger_center_x",
            "left_finger_center_y",
            "left_finger_center_z",
            "right_finger_center_x",
            "right_finger_center_y",
            "right_finger_center_z",
            "object_center_x",
            "object_center_y",
            "object_center_z",
        ]
        try:
            values = {key: float(row[key]) for key in keys}
        except Exception:
            samples.append({"step": step, "status": "INVALID_TIMESERIES_CENTER_ROW"})
            continue
        left_center = np.asarray(
            [values["left_finger_center_x"], values["left_finger_center_y"], values["left_finger_center_z"]],
            dtype=np.float64,
        )
        right_center = np.asarray(
            [values["right_finger_center_x"], values["right_finger_center_y"], values["right_finger_center_z"]],
            dtype=np.float64,
        )
        object_center = np.asarray(
            [values["object_center_x"], values["object_center_y"], values["object_center_z"]],
            dtype=np.float64,
        )
        center_delta = left_center - right_center
        center_distance = float(np.linalg.norm(center_delta))
        if center_distance <= 1e-12 or not np.isfinite(center_distance):
            samples.append({"step": step, "status": "INVALID_FINGER_CENTER_DISTANCE"})
            continue
        closing_unit = center_delta / center_distance
        midpoint = (left_center + right_center) * 0.5
        object_offset = object_center - midpoint
        along = float(np.dot(object_offset, closing_unit))
        cross = object_offset - along * closing_unit
        samples.append(
            {
                "step": step,
                "status": "PASS_DIAGNOSTIC_COMPUTED",
                "finger_midpoint_world_m": midpoint.tolist(),
                "finger_center_distance_m": center_distance,
                "closing_axis_unit_world": closing_unit.tolist(),
                "object_center_world_m": object_center.tolist(),
                "object_offset_from_finger_midpoint_world_m": object_offset.tolist(),
                "object_offset_along_closing_axis_m": along,
                "object_cross_closing_axis_offset_norm_m": float(np.linalg.norm(cross)),
                "object_center_z_m": float(object_center[2]),
            }
        )
    return {
        "status": "PASS_SAMPLED_TARGET_CONTACT_LANDMARKS",
        "expected_finger_paths": expected_finger_paths,
        "landmark_steps": sorted(landmark_steps),
        "samples": samples,
        "notes": "Center-line samples at first/last target contact landmarks; bbox sizes are reported separately.",
    }


def _axis_rotation_xyz(axis: str) -> tuple[float, float, float]:
    """Rotate Bottle500 local +Z long axis onto the requested world axis."""
    normalized_axis = axis.upper()
    if normalized_axis == "X":
        return (0.0, 90.0, 0.0)
    if normalized_axis == "Y":
        return (-90.0, 0.0, 0.0)
    if normalized_axis == "Z":
        return (0.0, 0.0, 0.0)
    raise ValueError(f"Unsupported object axis: {axis}")


def _axis_unit_vector(axis: str) -> np.ndarray:
    normalized_axis = axis.upper()
    if normalized_axis == "X":
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if normalized_axis == "Y":
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    if normalized_axis == "Z":
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    raise ValueError(f"Unsupported object axis: {axis}")


def _derive_open_finger_horizontal_perpendicular_axis(
    *,
    left_box: dict[str, Any],
    right_box: dict[str, Any],
    preferred_axis: str,
) -> dict[str, Any]:
    """Derive a horizontal bottle axis perpendicular to the open finger closing line."""

    left_center = np.asarray(left_box["center"], dtype=np.float64).reshape(3)
    right_center = np.asarray(right_box["center"], dtype=np.float64).reshape(3)
    closing = left_center - right_center
    closing_xy = np.asarray([closing[0], closing[1], 0.0], dtype=np.float64)
    closing_norm = float(np.linalg.norm(closing_xy))
    if closing_norm <= 1e-12 or not np.isfinite(closing_norm):
        raise ValueError(f"Cannot derive object yaw from degenerate horizontal closing axis: {closing.tolist()}")
    closing_unit = closing_xy / closing_norm
    world_z = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    candidate = np.cross(world_z, closing_unit)
    candidate_norm = float(np.linalg.norm(candidate))
    if candidate_norm <= 1e-12 or not np.isfinite(candidate_norm):
        raise ValueError(f"Cannot derive object yaw from closing axis: {closing.tolist()}")
    candidate = candidate / candidate_norm
    preferred = _axis_unit_vector(preferred_axis)
    if float(np.dot(candidate, preferred)) < 0.0:
        candidate = -candidate
    return {
        "source": "open_finger_horizontal_perpendicular",
        "provenance": "DIAGNOSTIC_OPEN_FRAME_FINGER_DERIVED_BOTTLE_YAW",
        "left_center_world_m": left_center.tolist(),
        "right_center_world_m": right_center.tolist(),
        "closing_axis_world_m": closing.tolist(),
        "closing_axis_horizontal_unit_world_m": closing_unit.tolist(),
        "preferred_axis": preferred_axis.upper(),
        "object_axis_unit_world": candidate.tolist(),
        "abs_dot_closing_axis": abs(float(np.dot(closing_unit, candidate))),
        "horizontal_abs_z": abs(float(candidate[2])),
    }


def _transform_with_local_x_axis(center: np.ndarray, x_axis: np.ndarray) -> np.ndarray:
    x = np.asarray(x_axis, dtype=np.float64).reshape(3)
    x_norm = float(np.linalg.norm(x))
    if x_norm <= 1e-12 or not np.isfinite(x_norm):
        raise ValueError(f"Cannot build transform from invalid x axis: {x.tolist()}")
    x = x / x_norm
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    z = up - float(np.dot(up, x)) * x
    z_norm = float(np.linalg.norm(z))
    if z_norm <= 1e-12 or not np.isfinite(z_norm):
        up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        z = up - float(np.dot(up, x)) * x
        z_norm = float(np.linalg.norm(z))
    z = z / z_norm
    y = np.cross(z, x)
    y = y / float(np.linalg.norm(y))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 0] = x
    transform[:3, 1] = y
    transform[:3, 2] = z
    transform[:3, 3] = np.asarray(center, dtype=np.float64).reshape(3)
    return transform


def _nominal_object_axis_length_stage_units(args: argparse.Namespace, side_length: float) -> float:
    if args.object_shape in {
        "bottle_usd",
        "bottle_usd_cylinder_proxy",
        "bottle_usd_segmented_proxy",
        "bottle_usd_grasp_band_proxy",
        "bottle_usd_grasp_box_proxy",
    }:
        return float(BOTTLE_LENGTH_M) / float(args.stage_units_in_meters)
    if args.object_shape in {"cylinder", "capsule", "bottle_proxy"}:
        return float(side_length) * float(args.object_length_multiplier)
    return float(side_length)


def _grasp_band_proxy_axis_length_stage_units(side_length: float, object_length_multiplier: float) -> float:
    """Return the short local collision band length used for rear-quarter grasp proofs."""

    total_length = float(side_length) * float(object_length_multiplier)
    return min(max(float(side_length) * 0.90, total_length * 0.20), total_length * 0.34)


def _contact_projection_model_for_args(
    *,
    args: argparse.Namespace,
    object_box: dict[str, Any],
    object_axis_unit_world: list[float] | tuple[float, float, float] | np.ndarray,
    projection_unit_world: list[float] | tuple[float, float, float] | np.ndarray,
    side_length: float,
) -> dict[str, Any]:
    if args.object_shape == "bottle_usd_grasp_box_proxy":
        return _oriented_box_projection_model(
            object_box=object_box,
            projection_unit_world=projection_unit_world,
            source="bottle_usd_grasp_box_proxy_oriented_contact_coupon",
        )
    if args.object_shape in {"cylinder", "capsule", "bottle_usd_cylinder_proxy", "bottle_usd_grasp_band_proxy"}:
        radius = float(side_length) * 0.5
        if args.object_shape == "bottle_usd_grasp_band_proxy":
            axis_length = _grasp_band_proxy_axis_length_stage_units(side_length, args.object_length_multiplier)
        else:
            axis_length = _nominal_object_axis_length_stage_units(args, side_length)
        half_length = float(axis_length) * 0.5
        return _oriented_cylinder_projection_model(
            object_box=object_box,
            object_axis_unit_world=object_axis_unit_world,
            projection_unit_world=projection_unit_world,
            radius_m=radius,
            half_length_m=half_length,
            source=f"{args.object_shape}_oriented_contact_proxy",
        )
    return {
        "valid": False,
        "status": "SKIPPED_NO_ORIENTED_CONTACT_PROJECTION_MODEL",
        "source": "world_aabb",
        "object_shape": args.object_shape,
    }


def _bbox_center(stage: Any, path: str) -> np.ndarray:
    box = _bbox_row(stage, path)
    if not box.get("bbox_valid"):
        raise RuntimeError(f"Cannot compute bbox center for {path}")
    return np.asarray(box["center"], dtype=np.float64)


def _tabletop_z_shift_from_bboxes(
    *,
    table_box: dict[str, Any],
    object_box: dict[str, Any],
    clearance: float,
) -> dict[str, Any]:
    if not table_box.get("bbox_valid"):
        return {"pass": False, "status": "FAIL_TABLETOP_REFERENCE_BBOX_INVALID"}
    if not object_box.get("bbox_valid"):
        return {"pass": False, "status": "FAIL_TABLETOP_OBJECT_BBOX_INVALID"}
    table_top_z = float(table_box["max"][2])
    object_bottom_z = float(object_box["min"][2])
    target_bottom_z = table_top_z + float(clearance)
    z_shift = target_bottom_z - object_bottom_z
    return {
        "pass": True,
        "status": "PASS_TABLETOP_Z_SHIFT_COMPUTED",
        "table_top_z_m": table_top_z,
        "object_bottom_z_before_m": object_bottom_z,
        "target_object_bottom_z_m": target_bottom_z,
        "tabletop_clearance_m": float(clearance),
        "z_shift_m": float(z_shift),
    }


def _tabletop_z_shift_from_top_z(
    *,
    table_top_z: float,
    object_box: dict[str, Any],
    clearance: float,
) -> dict[str, Any]:
    if not object_box.get("bbox_valid"):
        return {"pass": False, "status": "FAIL_TABLETOP_OBJECT_BBOX_INVALID"}
    object_bottom_z = float(object_box["min"][2])
    target_bottom_z = float(table_top_z) + float(clearance)
    z_shift = target_bottom_z - object_bottom_z
    return {
        "pass": True,
        "status": "PASS_TABLETOP_Z_SHIFT_COMPUTED",
        "table_top_z_m": float(table_top_z),
        "object_bottom_z_before_m": object_bottom_z,
        "target_object_bottom_z_m": target_bottom_z,
        "tabletop_clearance_m": float(clearance),
        "z_shift_m": float(z_shift),
    }


def _derived_tabletop_top_z_from_open_finger(
    *,
    open_left_box: dict[str, Any],
    open_right_box: dict[str, Any],
    object_contact_radius: float,
    clearance: float,
) -> dict[str, Any]:
    if not (open_left_box.get("bbox_valid") and open_right_box.get("bbox_valid")):
        return {
            "pass": False,
            "status": "FAIL_DERIVED_TABLETOP_OPEN_FINGER_BBOX_INVALID",
        }
    left_center = np.asarray(open_left_box["center"], dtype=np.float64)
    right_center = np.asarray(open_right_box["center"], dtype=np.float64)
    finger_midpoint = (left_center + right_center) / 2.0
    contact_radius = max(float(object_contact_radius), 0.0)
    tabletop_clearance = max(float(clearance), 0.0)
    derived_top_z = float(finger_midpoint[2] - contact_radius - tabletop_clearance)
    return {
        "pass": True,
        "status": "PASS_DERIVED_TABLETOP_TOP_Z_FROM_OPEN_FINGER",
        "mode": "derived_tabletop_top_z_from_open_finger_and_contact_radius",
        "open_finger_contact_midpoint_world_m": finger_midpoint.tolist(),
        "open_finger_contact_midpoint_z_m": float(finger_midpoint[2]),
        "object_contact_vertical_radius_m": contact_radius,
        "tabletop_clearance_m": tabletop_clearance,
        "derived_table_top_z_m": derived_top_z,
        "notes": (
            "Gate2 fixed-reset calibration: move the table collider in the composed validation stage so "
            "a soft bottle resting on the table has its contact proxy center at the HDF5 open-finger "
            "contact midpoint. This is a reset contract, not a lift or RL proof."
        ),
    }


def _calibrate_tabletop_top_z(
    *,
    stage: Any,
    table_path: str,
    target_top_z: float,
) -> dict[str, Any]:
    table_box_before = _bbox_row(stage, table_path)
    if not table_box_before.get("bbox_valid"):
        return {
            "pass": False,
            "status": "FAIL_TABLETOP_CALIBRATION_TABLE_BBOX_INVALID",
            "table_path": table_path,
            "table_bbox_before": table_box_before,
        }
    current_top_z = float(table_box_before["max"][2])
    z_shift = float(target_top_z) - current_top_z
    _shift_prim_world_translation(stage, table_path, np.asarray([0.0, 0.0, z_shift], dtype=np.float64))
    table_box_after = _bbox_row(stage, table_path)
    if not table_box_after.get("bbox_valid"):
        return {
            "pass": False,
            "status": "FAIL_TABLETOP_CALIBRATION_TABLE_BBOX_AFTER_SHIFT_INVALID",
            "table_path": table_path,
            "target_table_top_z_m": float(target_top_z),
            "table_top_z_before_m": current_top_z,
            "z_shift_m": z_shift,
            "table_bbox_before": table_box_before,
            "table_bbox_after": table_box_after,
        }
    table_top_after = float(table_box_after["max"][2])
    return {
        "pass": abs(table_top_after - float(target_top_z)) <= 1e-6,
        "status": "PASS_TABLETOP_TOP_Z_CALIBRATED"
        if abs(table_top_after - float(target_top_z)) <= 1e-6
        else "FAIL_TABLETOP_TOP_Z_CALIBRATION_ERROR",
        "table_path": table_path,
        "target_table_top_z_m": float(target_top_z),
        "table_top_z_before_m": current_top_z,
        "table_top_z_after_m": table_top_after,
        "z_shift_m": z_shift,
        "table_bbox_before": table_box_before,
        "table_bbox_after": table_box_after,
    }


def _shift_prim_world_translation(stage: Any, prim_path: str, delta_world: np.ndarray) -> None:
    from pxr import Gf
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Cannot shift missing prim: {prim_path}")
    transform = _world_matrix(UsdGeom, stage, prim_path)
    transform[:3, 3] = transform[:3, 3] + np.asarray(delta_world, dtype=np.float64)
    _set_xform_matrix(UsdGeom, Gf, prim, transform)


def _place_object_on_tabletop(
    *,
    stage: Any,
    object_path: str,
    table_path: str,
    clearance: float,
    table_top_z: float | None = None,
) -> dict[str, Any]:
    object_box_before = _bbox_row(stage, object_path)
    if table_top_z is None:
        table_box = _bbox_row(stage, table_path)
        row = _tabletop_z_shift_from_bboxes(table_box=table_box, object_box=object_box_before, clearance=clearance)
    else:
        table_box = {
            "path": table_path,
            "bbox_valid": None,
            "status": "SKIPPED_EXPLICIT_TABLETOP_TOP_Z",
        }
        row = _tabletop_z_shift_from_top_z(
            table_top_z=float(table_top_z),
            object_box=object_box_before,
            clearance=clearance,
        )
    row.update(
        {
            "mode": "object_bottom_to_tabletop",
            "table_path": table_path,
            "table_top_z_source": "explicit_arg" if table_top_z is not None else "reference_prim_bbox",
            "object_path": object_path,
            "table_bbox": table_box,
            "object_bbox_before": object_box_before,
        }
    )
    if not row["pass"]:
        return row
    delta = np.asarray([0.0, 0.0, float(row["z_shift_m"])], dtype=np.float64)
    _shift_prim_world_translation(stage, object_path, delta)
    object_box_after = _bbox_row(stage, object_path)
    row["object_bbox_after"] = object_box_after
    if object_box_after.get("bbox_valid"):
        row["object_bottom_z_after_m"] = float(object_box_after["min"][2])
        row["tabletop_gap_after_m"] = float(object_box_after["min"][2]) - float(row["table_top_z_m"])
        row["status"] = "PASS_TABLETOP_OBJECT_PLACED"
    else:
        row["pass"] = False
        row["status"] = "FAIL_TABLETOP_OBJECT_BBOX_AFTER_SHIFT_INVALID"
    return row


def _matrix_to_numpy(matrix: Any) -> np.ndarray:
    raw = np.array([[float(matrix[i][j]) for j in range(4)] for i in range(4)], dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = raw[:3, :3].T
    result[:3, 3] = raw[3, :3]
    return result


def _world_matrix(UsdGeom: Any, stage: Any, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Cannot compute world transform for missing prim: {prim_path}")
    return _matrix_to_numpy(UsdGeom.XformCache().GetLocalToWorldTransform(prim))


def _geometry_audit_prim(UsdGeom: Any, stage: Any, prim_path: str) -> dict[str, Any]:
    prim = stage.GetPrimAtPath(prim_path)
    row: dict[str, Any] = {"path": prim_path, "exists": bool(prim and prim.IsValid())}
    if not row["exists"]:
        row["status"] = "MISSING_PRIM"
        return row
    try:
        row["bbox"] = _bbox_row(stage, prim_path)
    except Exception as exc:  # pragma: no cover - defensive Isaac runtime path
        row["bbox_error"] = f"{type(exc).__name__}: {exc}"
    try:
        matrix = _world_matrix(UsdGeom, stage, prim_path)
        row["world_translation"] = matrix[:3, 3].tolist()
        row["world_matrix"] = matrix.tolist()
    except Exception as exc:  # pragma: no cover - defensive Isaac runtime path
        row["world_transform_error"] = f"{type(exc).__name__}: {exc}"
    return row


def _geometry_audit_snapshot(
    *,
    UsdGeom: Any,
    stage: Any,
    phase: str,
    step: int,
    object_path: str,
    contact_geometry_path: str,
    object_gripper_frame: str,
    extra_paths: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    paths = [
        {"label": "object_root", "path": object_path},
        {"label": "object_contact_geometry", "path": contact_geometry_path},
        {"label": "object_gripper_frame", "path": object_gripper_frame},
    ]
    paths.extend(extra_paths or [])
    return {
        "phase": phase,
        "step": int(step),
        "prims": {item["label"]: _geometry_audit_prim(UsdGeom, stage, item["path"]) for item in paths},
    }


def _grasp_geometry_audit_extra_paths(
    *,
    side: str,
    contact_proxy_profile: str,
    finger_proxy_paths: dict[str, str],
    contact_target_paths: dict[str, str],
    support_plane_path: str | None,
) -> list[dict[str, str]]:
    rows = [
        {"label": "left_finger_proxy", "path": finger_proxy_paths["left_finger"]},
        {"label": "right_finger_proxy", "path": finger_proxy_paths["right_finger"]},
        {"label": "left_contact_target", "path": contact_target_paths["left_finger"]},
        {"label": "right_contact_target", "path": contact_target_paths["right_finger"]},
    ]
    if contact_proxy_profile in {"scene_base_link", "scene_base_link_finger_mesh", "scene_base_link_inner_pad"}:
        prefix = "left" if side == "left" else "right"
        rows.extend(
            [
                {
                    "label": "same_side_gripper_bar",
                    "path": (
                        f"/scene/{prefix}_base_link/{prefix}_gripper_base/collisions/"
                        "vx300s_7_gripper_bar/vx300s_7_gripper_bar"
                    ),
                },
                {"label": "same_side_gripper_base", "path": f"/scene/{prefix}_base_link/{prefix}_gripper_base"},
            ]
        )
    if support_plane_path:
        rows.append({"label": "support_plane", "path": support_plane_path})
    return rows


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    norm = float(np.linalg.norm([w, x, y, z]))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_from_pose(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quat_wxyz_to_matrix(quat_wxyz)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def _set_xform_matrix(UsdGeom: Any, Gf: Any, prim: Any, matrix: np.ndarray) -> None:
    # Gf.Matrix4d uses row-vector convention; translation lives in row 3.
    m = np.asarray(matrix, dtype=np.float64)
    gf_matrix = Gf.Matrix4d(
        float(m[0, 0]),
        float(m[1, 0]),
        float(m[2, 0]),
        0.0,
        float(m[0, 1]),
        float(m[1, 1]),
        float(m[2, 1]),
        0.0,
        float(m[0, 2]),
        float(m[1, 2]),
        float(m[2, 2]),
        0.0,
        float(m[0, 3]),
        float(m[1, 3]),
        float(m[2, 3]),
        1.0,
    )
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(gf_matrix)


def _set_object_from_gripper_relative_transform(
    *,
    stage: Any,
    UsdGeom: Any,
    Gf: Any,
    object_path: str,
    object_gripper_frame: str,
    t_gripper_object: np.ndarray,
) -> dict[str, Any]:
    object_prim = stage.GetPrimAtPath(object_path)
    if not object_prim or not object_prim.IsValid():
        raise RuntimeError(f"Cannot update held object; missing prim: {object_path}")
    t_world_gripper = _world_matrix(UsdGeom, stage, object_gripper_frame)
    t_world_object = t_world_gripper @ np.asarray(t_gripper_object, dtype=np.float64)
    _set_xform_matrix(UsdGeom, Gf, object_prim, t_world_object)
    return {
        "object_path": object_path,
        "object_gripper_frame": object_gripper_frame,
        "object_world_position": t_world_object[:3, 3].tolist(),
    }


def _target_contact_hits_for_phase(
    *,
    rows: list[dict[str, Any]],
    object_path: str,
    expected_finger_paths: list[str],
    phase: str,
) -> dict[str, Any]:
    """Summarize contact hits for expected object/finger pairs in one phase.

    This is used only for diagnostic held-object replay. It waits for actual
    PhysX contact pairs during close instead of geometry overlap. CONTACT_FOUND
    events are also reported, but CONTACT_PERSIST counts as contact because a
    finger can already be touching while the other finger closes.
    """

    phase_rows = [
        row
        for row in rows
        if row.get("phase") == phase
        and _pair_touches_targets(row, object_path, expected_finger_paths)
    ]
    found_rows = [row for row in phase_rows if row.get("type_name") == "CONTACT_FOUND"]
    finger_rows = {
        finger_path: [row for row in phase_rows if _pair_touches_finger(row, object_path, finger_path)]
        for finger_path in expected_finger_paths
    }
    finger_found_rows = {
        finger_path: [row for row in found_rows if _pair_touches_finger(row, object_path, finger_path)]
        for finger_path in expected_finger_paths
    }
    return {
        "phase": phase,
        "expected_finger_paths": expected_finger_paths,
        "triggered": bool(expected_finger_paths) and all(bool(rows_for_finger) for rows_for_finger in finger_rows.values()),
        "finger_hits": {finger_path: bool(rows_for_finger) for finger_path, rows_for_finger in finger_rows.items()},
        "finger_found_hits": {
            finger_path: bool(rows_for_finger) for finger_path, rows_for_finger in finger_found_rows.items()
        },
        "first_contact_pair": phase_rows[0] if phase_rows else None,
        "first_contact_found_pair": found_rows[0] if found_rows else None,
        "contact_pair_count": len(phase_rows),
        "contact_found_pair_count": len(found_rows),
    }


def _diagnostic_object_frame_features(UsdGeom: Any, stage: Any, object_path: str) -> dict[str, float | bool]:
    features: dict[str, float | bool] = {}
    frame_specs = [
        ("object_origin", object_path),
        ("object_mouth", f"{object_path}/Frames/MouthFrame"),
    ]
    for label, frame_path in frame_specs:
        prim = stage.GetPrimAtPath(frame_path)
        exists = bool(prim and prim.IsValid())
        features[f"{label}_frame_exists"] = exists
        if not exists:
            continue
        transform = _world_matrix(UsdGeom, stage, frame_path)
        features[f"{label}_x"] = float(transform[0, 3])
        features[f"{label}_y"] = float(transform[1, 3])
        features[f"{label}_z"] = float(transform[2, 3])
        if label == "object_mouth":
            axis = np.asarray(transform[:3, 2], dtype=np.float64)
            norm = float(np.linalg.norm(axis))
            if norm > 1e-12:
                axis = axis / norm
            features["object_mouth_axis_x"] = float(axis[0])
            features["object_mouth_axis_y"] = float(axis[1])
            features["object_mouth_axis_z"] = float(axis[2])
    return features


def _bottle_usd_runtime_composition_gate(stage: Any, object_path: str) -> dict[str, Any]:
    """Validate that the runtime object is a visible Bottle500 composition.

    Contact gates alone can pass while a GUI user cannot find a visible bottle,
    because the runtime path is not the source asset path. This gate records the
    real prim path created for the replay run and checks visual, collision, frame,
    and bbox evidence under that path.
    """

    from pxr import UsdGeom
    from pxr import UsdPhysics

    root = stage.GetPrimAtPath(object_path)
    row: dict[str, Any] = {
        "runtime_object_path": object_path,
        "pass": False,
        "status": "FAIL_MISSING_RUNTIME_OBJECT",
        "notes": (
            "The source Bottle500 asset may be referenced under a runtime path. "
            "Inspect runtime_object_path, not only /World/Bottle500."
        ),
    }
    if not root or not root.IsValid():
        return row

    visual_meshes: list[str] = []
    collision_prims: list[str] = []
    enabled_collision_prims: list[str] = []
    mouth_frame_path: str | None = None
    inner_bottom_frame_path: str | None = None
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not _path_matches(path, object_path):
            continue
        if prim.IsA(UsdGeom.Mesh) and "/Visuals/" in path:
            visual_meshes.append(path)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_prims.append(path)
            collision = UsdPhysics.CollisionAPI(prim)
            attr = collision.GetCollisionEnabledAttr()
            if attr is None or attr.Get() is not False:
                enabled_collision_prims.append(path)
        if path.endswith("/Frames/MouthFrame"):
            mouth_frame_path = path
        if path.endswith("/Frames/InnerBottomFrame"):
            inner_bottom_frame_path = path

    bbox = _bbox_row(stage, object_path)
    bbox_size = [float(v) for v in bbox.get("size") or []]
    bbox_valid = bool(bbox.get("bbox_valid") and len(bbox_size) == 3 and all(np.isfinite(v) for v in bbox_size))
    longest_axis = max(bbox_size) if bbox_valid else float("nan")
    positive_extents = sum(1 for value in bbox_size if value > 0.025) if bbox_valid else 0
    bottle_sized = bool(bbox_valid and 0.16 <= longest_axis <= 0.24 and positive_extents >= 3)
    visual_ok = bool(visual_meshes)
    collision_ok = bool(enabled_collision_prims)
    mouth_frame_ok = bool(mouth_frame_path and stage.GetPrimAtPath(mouth_frame_path))
    inner_bottom_frame_ok = bool(inner_bottom_frame_path and stage.GetPrimAtPath(inner_bottom_frame_path))
    pass_gate = bool(visual_ok and collision_ok and mouth_frame_ok and inner_bottom_frame_ok and bottle_sized)
    row.update(
        {
            "pass": pass_gate,
            "status": "PASS_BOTTLE_USD_RUNTIME_COMPOSITION" if pass_gate else "FAIL_BOTTLE_USD_RUNTIME_COMPOSITION",
            "visual_mesh_count": len(visual_meshes),
            "visual_mesh_sample": visual_meshes[:8],
            "collision_prim_count": len(collision_prims),
            "collision_prim_sample": collision_prims[:8],
            "enabled_collision_prim_count": len(enabled_collision_prims),
            "enabled_collision_prim_sample": enabled_collision_prims[:8],
            "mouth_frame_path": mouth_frame_path,
            "mouth_frame_exists": mouth_frame_ok,
            "inner_bottom_frame_path": inner_bottom_frame_path,
            "inner_bottom_frame_exists": inner_bottom_frame_ok,
            "bbox": bbox,
            "bbox_longest_axis_m": longest_axis,
            "bbox_bottle_sized": bottle_sized,
        }
    )
    return row


def _load_grasp_transform(grasp_yaml: str | Path, grasp_name: str) -> dict[str, Any]:
    grasp_path = Path(grasp_yaml).expanduser().resolve()
    data = yaml.safe_load(grasp_path.read_text())
    grasps = data.get("grasps") or {}
    grasp = grasps.get(grasp_name) if isinstance(grasps, dict) else None
    if grasp is None:
        raise ValueError(f"Cannot find grasp {grasp_name!r} in {grasp_path}")
    quat = np.asarray([grasp["orientation"]["w"], *grasp["orientation"]["xyz"]], dtype=np.float64)
    position = np.asarray(grasp["position"], dtype=np.float64)
    return {
        "path": str(grasp_path),
        "name": grasp_name,
        "object_frame": data.get("object_frame"),
        "gripper_frame": data.get("gripper_frame"),
        "t_object_gripper": _transform_from_pose(position, quat),
        "position": position.tolist(),
        "quat_wxyz": quat.tolist(),
    }


def _create_passive_cube(
    *,
    world: Any,
    stage: Any,
    path: str,
    center: np.ndarray,
    side_length: float,
    mass: float,
    creation_mode: str,
    shape: str = "cube",
    axis: str = "X",
    length_multiplier: float = 4.0,
    usd_path: str | Path | None = None,
    usd_prim_path: str = "/Bottle500",
    grasp_band_local_axis_offset: float = 0.0,
    rigid_body: bool = True,
) -> None:
    from pxr import Gf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    if shape != "cube" and creation_mode != "raw_usd":
        raise ValueError(f"{shape} object shape requires raw_usd creation; got {creation_mode}")

    def strip_visual_physics(prim: Any) -> None:
        for child in Usd.PrimRange(prim):
            if not child:
                continue
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                try:
                    child.RemoveAPI(UsdPhysics.RigidBodyAPI)
                except Exception:
                    pass
            if child.HasAPI(UsdPhysics.MassAPI):
                try:
                    child.RemoveAPI(UsdPhysics.MassAPI)
                except Exception:
                    pass
            if child.HasAPI(UsdPhysics.CollisionAPI):
                collision = UsdPhysics.CollisionAPI(child)
                attr = collision.GetCollisionEnabledAttr()
                if not attr:
                    attr = collision.CreateCollisionEnabledAttr()
                attr.Set(False)

    if creation_mode == "dynamic_cuboid":
        from isaacsim.core.api.objects import DynamicCuboid

        world.scene.add(
            DynamicCuboid(
                prim_path=path,
                name="phase43_passive_contact_cube",
                position=np.asarray(center, dtype=np.float64),
                scale=np.asarray([side_length, side_length, side_length], dtype=np.float64),
                size=1.0,
                mass=float(mass),
                color=np.asarray([0.9, 0.2, 0.1], dtype=np.float64),
            )
        )
        return
    if creation_mode != "raw_usd":
        raise ValueError(f"Unsupported object creation mode: {creation_mode}")
    normalized_axis = axis.upper()
    if normalized_axis not in {"X", "Y", "Z"}:
        raise ValueError(f"Unsupported object axis: {axis}")
    if shape == "cube":
        geom = UsdGeom.Cube.Define(stage, path)
        geom.CreateSizeAttr(1.0)
        scale = Gf.Vec3d(side_length, side_length, side_length)
    elif shape == "cylinder":
        geom = UsdGeom.Cylinder.Define(stage, path)
        geom.CreateAxisAttr(normalized_axis)
        geom.CreateRadiusAttr(side_length * 0.5)
        geom.CreateHeightAttr(side_length * length_multiplier)
        scale = Gf.Vec3d(1.0, 1.0, 1.0)
    elif shape == "capsule":
        geom = UsdGeom.Capsule.Define(stage, path)
        geom.CreateAxisAttr(normalized_axis)
        geom.CreateRadiusAttr(side_length * 0.5)
        geom.CreateHeightAttr(side_length * length_multiplier)
        scale = Gf.Vec3d(1.0, 1.0, 1.0)
    elif shape == "bottle_proxy":
        root = UsdGeom.Xform.Define(stage, path)
        root_xform = UsdGeom.Xformable(root.GetPrim())
        root_xform.ClearXformOpOrder()
        root_xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*[float(x) for x in center]))

        body_length = side_length * length_multiplier
        neck_length = side_length * max(length_multiplier * 0.35, 1.0)
        body_radius = side_length * 0.5
        neck_radius = side_length * 0.18
        mouth_radius = side_length * 0.22

        body = UsdGeom.Cylinder.Define(stage, f"{path}/body")
        body.CreateAxisAttr(normalized_axis)
        body.CreateRadiusAttr(body_radius)
        body.CreateHeightAttr(body_length)
        body.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.35, 0.95)])

        neck = UsdGeom.Cylinder.Define(stage, f"{path}/neck")
        neck.CreateAxisAttr(normalized_axis)
        neck.CreateRadiusAttr(neck_radius)
        neck.CreateHeightAttr(neck_length)
        neck.CreateDisplayColorAttr([Gf.Vec3f(0.75, 0.9, 1.0)])

        mouth = UsdGeom.Sphere.Define(stage, f"{path}/mouth")
        mouth.CreateRadiusAttr(mouth_radius)
        mouth.CreateDisplayColorAttr([Gf.Vec3f(0.02, 0.04, 0.1)])

        axis_index = {"X": 0, "Y": 1, "Z": 2}[normalized_axis]

        def offset_vec(distance: float) -> Gf.Vec3d:
            values = [0.0, 0.0, 0.0]
            values[axis_index] = distance
            return Gf.Vec3d(*values)

        neck_distance = body_length * 0.5 + neck_length * 0.5
        mouth_distance = body_length * 0.5 + neck_length + mouth_radius
        UsdGeom.Xformable(neck.GetPrim()).AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            offset_vec(neck_distance)
        )
        UsdGeom.Xformable(mouth.GetPrim()).AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            offset_vec(mouth_distance)
        )

        for child in (body.GetPrim(), neck.GetPrim(), mouth.GetPrim()):
            UsdPhysics.CollisionAPI.Apply(child).CreateCollisionEnabledAttr().Set(True)
        if rigid_body:
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
            UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(float(mass))
        return
    elif shape == "bottle_usd":
        if usd_path is None:
            raise ValueError("bottle_usd requires a USD asset path")
        asset_path = Path(usd_path).expanduser().resolve()
        if not asset_path.exists():
            raise FileNotFoundError(f"bottle_usd asset does not exist: {asset_path}")
        root = UsdGeom.Xform.Define(stage, path)
        root.GetPrim().GetReferences().AddReference(str(asset_path), usd_prim_path)
        root_xform = UsdGeom.Xformable(root.GetPrim())
        root_xform.ClearXformOpOrder()
        translate_op = root_xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        rotate_op = root_xform.AddRotateXYZOp(precision=UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(*[float(x) for x in center]))
        rotate_op.Set(Gf.Vec3d(*_axis_rotation_xyz(normalized_axis)))

        # The referenced asset origin is semantic, not necessarily its collision
        # bbox center. Move once more after composition so the actual object used
        # by PhysX is centered between the fingertips.
        composed_center = _bbox_center(stage, path)
        correction = np.asarray(center, dtype=np.float64) - composed_center
        translate_op.Set(Gf.Vec3d(*[float(x) for x in np.asarray(center, dtype=np.float64) + correction]))

        if rigid_body:
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
            UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(float(mass))
        return
    elif shape in {
        "bottle_usd_cylinder_proxy",
        "bottle_usd_segmented_proxy",
        "bottle_usd_grasp_band_proxy",
        "bottle_usd_grasp_box_proxy",
    }:
        if usd_path is None:
            raise ValueError(f"{shape} requires a USD asset path")
        asset_path = Path(usd_path).expanduser().resolve()
        if not asset_path.exists():
            raise FileNotFoundError(f"bottle_usd asset does not exist: {asset_path}")
        root = UsdGeom.Xform.Define(stage, path)
        root_xform = UsdGeom.Xformable(root.GetPrim())
        root_xform.ClearXformOpOrder()
        root_xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*[float(x) for x in center])
        )

        visual_path = f"{path}/visual_bottle"
        visual = UsdGeom.Xform.Define(stage, visual_path)
        visual.GetPrim().GetReferences().AddReference(str(asset_path), usd_prim_path)
        visual_xform = UsdGeom.Xformable(visual.GetPrim())
        visual_xform.ClearXformOpOrder()
        visual_translate = visual_xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        visual_rotate = visual_xform.AddRotateXYZOp(precision=UsdGeom.XformOp.PrecisionDouble)
        visual_translate.Set(Gf.Vec3d(0.0, 0.0, 0.0))
        visual_rotate.Set(Gf.Vec3d(*_axis_rotation_xyz(normalized_axis)))

        proxy_path = f"{path}/physics_proxy"
        if shape == "bottle_usd_cylinder_proxy":
            proxy = UsdGeom.Cylinder.Define(stage, proxy_path)
            proxy.CreateAxisAttr(normalized_axis)
            proxy.CreateRadiusAttr(side_length * 0.5)
            proxy.CreateHeightAttr(float(BOTTLE_LENGTH_M))
            proxy.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.2, 0.1)])
            UsdPhysics.CollisionAPI.Apply(proxy.GetPrim()).CreateCollisionEnabledAttr().Set(True)
        elif shape == "bottle_usd_grasp_box_proxy":
            from pxr import Sdf

            proxy_root = UsdGeom.Xform.Define(stage, proxy_path)
            band_length = _grasp_band_proxy_axis_length_stage_units(side_length, length_multiplier)
            axis_index = {"X": 0, "Y": 1, "Z": 2}[normalized_axis]
            scale_values = [float(side_length), float(side_length), float(side_length)]
            scale_values[axis_index] = float(band_length)
            offset_values = [0.0, 0.0, 0.0]
            offset_values[axis_index] = float(grasp_band_local_axis_offset)

            body = UsdGeom.Cube.Define(stage, f"{proxy_path}/body")
            body.CreateSizeAttr(1.0)
            body.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.2, 0.1)])
            body_xform = UsdGeom.Xformable(body.GetPrim())
            body_xform.ClearXformOpOrder()
            body_xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*offset_values))
            body_xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale_values))
            UsdPhysics.CollisionAPI.Apply(body.GetPrim()).CreateCollisionEnabledAttr().Set(True)
            proxy_root.GetPrim().CreateAttribute("aloha:proxyType", Sdf.ValueTypeNames.String).Set(
                "local_grasp_box_coupon"
            )
            proxy_root.GetPrim().CreateAttribute("aloha:graspBandLocalAxisOffset", Sdf.ValueTypeNames.Double).Set(
                float(grasp_band_local_axis_offset)
            )
        elif shape == "bottle_usd_grasp_band_proxy":
            from pxr import Sdf

            proxy_root = UsdGeom.Xform.Define(stage, proxy_path)
            total_length = float(side_length) * float(length_multiplier)
            band_length = _grasp_band_proxy_axis_length_stage_units(side_length, length_multiplier)
            axis_index = {"X": 0, "Y": 1, "Z": 2}[normalized_axis]
            offset_values = [0.0, 0.0, 0.0]
            offset_values[axis_index] = float(grasp_band_local_axis_offset)
            # Rear-quarter placement keeps the runtime root at the full bottle
            # center so the visual BottleUSD stays semantically correct.  The
            # local contact band must therefore be shifted back along the bottle
            # axis to the actual gripper contact section.
            body = UsdGeom.Cylinder.Define(stage, f"{proxy_path}/body")
            body.CreateAxisAttr(normalized_axis)
            body.CreateRadiusAttr(float(side_length) * 0.5)
            body.CreateHeightAttr(band_length)
            body.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.2, 0.1)])
            UsdGeom.Xformable(body.GetPrim()).AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Vec3d(*offset_values)
            )
            UsdPhysics.CollisionAPI.Apply(body.GetPrim()).CreateCollisionEnabledAttr().Set(True)
            proxy_root.GetPrim().CreateAttribute("aloha:proxyType", Sdf.ValueTypeNames.String).Set(
                "local_grasp_band_only"
            )
            proxy_root.GetPrim().CreateAttribute("aloha:graspBandLocalAxisOffset", Sdf.ValueTypeNames.Double).Set(
                float(grasp_band_local_axis_offset)
            )
        else:
            from pxr import Sdf

            proxy_root = UsdGeom.Xform.Define(stage, proxy_path)
            axis_index = {"X": 0, "Y": 1, "Z": 2}[normalized_axis]
            total_length = float(side_length) * float(length_multiplier)
            mouth_radius = min(float(side_length) * 0.12, total_length * 0.04)
            body_length = total_length * 0.76
            neck_length = max(total_length - body_length - 2.0 * mouth_radius, total_length * 0.08)
            body_radius = float(side_length) * 0.5
            neck_radius = float(side_length) * 0.25

            def segment_translate(distance: float) -> Gf.Vec3d:
                values = [0.0, 0.0, 0.0]
                values[axis_index] = distance
                return Gf.Vec3d(*values)

            body = UsdGeom.Cylinder.Define(stage, f"{proxy_path}/body")
            body.CreateAxisAttr(normalized_axis)
            body.CreateRadiusAttr(body_radius)
            body.CreateHeightAttr(body_length)
            body.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.2, 0.1)])
            UsdGeom.Xformable(body.GetPrim()).AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                segment_translate(-total_length * 0.5 + body_length * 0.5)
            )

            neck = UsdGeom.Cylinder.Define(stage, f"{proxy_path}/neck")
            neck.CreateAxisAttr(normalized_axis)
            neck.CreateRadiusAttr(neck_radius)
            neck.CreateHeightAttr(neck_length)
            neck.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.55, 1.0)])
            UsdGeom.Xformable(neck.GetPrim()).AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                segment_translate(-total_length * 0.5 + body_length + neck_length * 0.5)
            )

            mouth = UsdGeom.Sphere.Define(stage, f"{proxy_path}/mouth")
            mouth.CreateRadiusAttr(mouth_radius)
            mouth.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.08, 0.12)])
            UsdGeom.Xformable(mouth.GetPrim()).AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                segment_translate(-total_length * 0.5 + body_length + neck_length + mouth_radius)
            )

            for child in (body.GetPrim(), neck.GetPrim(), mouth.GetPrim()):
                UsdPhysics.CollisionAPI.Apply(child).CreateCollisionEnabledAttr().Set(True)
            proxy_root.GetPrim().CreateAttribute("aloha:proxyType", Sdf.ValueTypeNames.String).Set(
                "segmented_bottle_body_neck_mouth"
            )

        visual_center_target = np.asarray(center, dtype=np.float64).copy()
        axis_index = {"X": 0, "Y": 1, "Z": 2}[normalized_axis]
        if axis_index != 2:
            # Keep the visible bottle bottom aligned with the smaller physics
            # proxy bottom on the tabletop. The root/proxy center remains the
            # physical contact center used by the replay gate.
            visual_center_target[2] += max(float(BOTTLE_RADIUS_M) - float(side_length) * 0.5, 0.0)
        visual_center = _bbox_center(stage, visual_path)
        visual_translate.Set(Gf.Vec3d(*[float(x) for x in visual_center_target - visual_center]))

        strip_visual_physics(visual.GetPrim())
        if rigid_body:
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
            UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(float(mass))
        return
    else:
        raise ValueError(f"Unsupported object shape: {shape}")
    geom.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.2, 0.1)])
    xform = UsdGeom.Xformable(geom.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*[float(x) for x in center]))
    xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(scale)
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    if rigid_body:
        UsdPhysics.RigidBodyAPI.Apply(geom.GetPrim())
        UsdPhysics.MassAPI.Apply(geom.GetPrim()).CreateMassAttr(float(mass))


def _parse_vec3(values: list[float] | tuple[float, ...] | None, *, name: str) -> np.ndarray:
    if values is None:
        return np.zeros(3, dtype=np.float64)
    if len(values) != 3:
        raise ValueError(f"{name} requires exactly three values")
    result = np.asarray([float(v) for v in values], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN/Inf: {values}")
    return result


def _create_static_support_box(
    *,
    stage: Any,
    path: str,
    center: np.ndarray,
    size_x: float,
    size_y: float,
    thickness: float,
) -> dict[str, Any]:
    from pxr import Gf
    from pxr import UsdGeom
    from pxr import UsdPhysics

    geom = UsdGeom.Cube.Define(stage, path)
    geom.CreateSizeAttr(1.0)
    geom.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.45, 0.45)])
    xform = UsdGeom.Xformable(geom.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*[float(x) for x in center]))
    xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(size_x), float(size_y), float(thickness))
    )
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    return {
        "path": path,
        "center": [float(x) for x in center],
        "size": [float(size_x), float(size_y), float(thickness)],
        "size_x": float(size_x),
        "size_y": float(size_y),
        "size_xy": float(size_x) if float(size_x) == float(size_y) else None,
        "thickness": float(thickness),
    }


def _local_object_support_patch_size(
    object_box: dict[str, Any],
    *,
    margin: float,
    min_size: float = 0.05,
) -> tuple[float, float]:
    """Derive a diagnostic support patch from the object's XY footprint.

    The patch isolates bottle-on-table support without putting a broad collider
    under robot base links. It is diagnostic only and must not be reported as a
    final full-workcell table validation.
    """

    size = np.asarray(object_box.get("size", [np.nan, np.nan, np.nan]), dtype=np.float64).reshape(3)
    if not np.isfinite(size).all():
        raise ValueError(f"Cannot derive local support patch size from invalid object bbox size: {size}")
    pad = max(float(margin), 0.0) * 2.0
    return (
        max(float(size[0]) + pad, float(min_size)),
        max(float(size[1]) + pad, float(min_size)),
    )


def _set_collision_offsets(
    stage: Any, prim_path: str, contact_offset: float | None, rest_offset: float | None
) -> dict[str, Any]:
    from pxr import PhysxSchema

    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return {"path": prim_path, "exists": False, "applied": False}
    author_offsets = contact_offset is not None or rest_offset is not None
    api = PhysxSchema.PhysxCollisionAPI.Apply(prim) if author_offsets else PhysxSchema.PhysxCollisionAPI(prim)
    if contact_offset is not None:
        api.CreateContactOffsetAttr(float(contact_offset)).Set(float(contact_offset))
    if rest_offset is not None:
        api.CreateRestOffsetAttr(float(rest_offset)).Set(float(rest_offset))
    return {
        "path": prim_path,
        "exists": True,
        "applied": author_offsets,
        "contact_offset": api.GetContactOffsetAttr().Get() if api.GetContactOffsetAttr() else None,
        "rest_offset": api.GetRestOffsetAttr().Get() if api.GetRestOffsetAttr() else None,
    }


def _set_object_collision_offsets(
    stage: Any, prim_path: str, contact_offset: float | None, rest_offset: float | None
) -> dict[str, Any]:
    from pxr import Usd
    from pxr import UsdPhysics

    root = stage.GetPrimAtPath(prim_path)
    if not root:
        return {"path": prim_path, "exists": False, "targets": []}
    targets = [str(prim.GetPath()) for prim in Usd.PrimRange(root) if prim and prim.HasAPI(UsdPhysics.CollisionAPI)]
    if not targets:
        targets = [prim_path]
    return {
        "path": prim_path,
        "exists": True,
        "targets": [_set_collision_offsets(stage, target, contact_offset, rest_offset) for target in targets],
    }


def _export_debug_stage(stage: Any, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ok = bool(stage.Export(str(path)))
    except BaseException as exc:
        return {"path": _rel(path), "saved": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"path": _rel(path), "saved": ok, "error": None}


def _begin_contact_pair_trace(stage: Any, *, disable_usd_updates: bool) -> dict[str, Any]:
    import carb
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings._physx import SETTING_UPDATE_TO_USD
    from pxr import PhysxSchema
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdUtils
    import usdrt

    session_sub_layer = Sdf.Layer.CreateAnonymous()
    stage.GetSessionLayer().subLayerPaths.append(session_sub_layer.identifier)
    old_layer = stage.GetEditTarget().GetLayer()
    stage.SetEditTarget(Usd.EditTarget(session_sub_layer))

    stage_cache = UsdUtils.StageCache.Get()
    stage_cache.Insert(stage)
    stage_id = stage_cache.GetId(stage).ToLongInt()
    usdrt_stage = usdrt.Usd.Stage.Attach(stage_id)
    rigid_body_paths = [str(path) for path in usdrt_stage.GetPrimsWithAppliedAPIName("PhysicsRigidBodyAPI")]
    for prim_path in rigid_body_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if prim:
            contact_report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            contact_report_api.CreateThresholdAttr().Set(0)

    settings = carb.settings.get_settings()
    write_usd = settings.get_as_bool(SETTING_UPDATE_TO_USD)
    write_fabric = settings.get_as_bool("/physics/fabricEnabled")
    if disable_usd_updates:
        settings.set(SETTING_UPDATE_TO_USD, False)
        settings.set("/physics/fabricEnabled", False)
    return {
        "enabled": True,
        "stage_id": stage_id,
        "session_sub_layer": session_sub_layer,
        "old_layer": old_layer,
        "settings": settings,
        "write_usd": write_usd,
        "write_fabric": write_fabric,
        "disable_usd_updates": disable_usd_updates,
        "rigid_body_paths": rigid_body_paths,
        "physx_interface": get_physx_simulation_interface(),
    }


def _apply_contact_report_api(stage: Any, prim_paths: list[str]) -> list[dict[str, Any]]:
    from pxr import PhysxSchema
    from pxr import UsdPhysics

    rows: list[dict[str, Any]] = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        row: dict[str, Any] = {"path": prim_path, "exists": bool(prim)}
        if not prim:
            row["applied"] = False
            row["reason"] = "missing_prim"
            rows.append(row)
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            row["applied"] = False
            row["reason"] = "missing_rigid_body_api"
            rows.append(row)
            continue
        api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        api.CreateThresholdAttr().Set(0)
        row["applied"] = True
        row["threshold"] = api.GetThresholdAttr().Get()
        rows.append(row)
    return rows


def _finish_contact_pair_trace(stage: Any, trace_state: dict[str, Any] | None) -> None:
    if not trace_state:
        return
    from omni.physx.bindings._physx import SETTING_UPDATE_TO_USD

    settings = trace_state["settings"]
    if trace_state.get("disable_usd_updates"):
        settings.set(SETTING_UPDATE_TO_USD, trace_state["write_usd"])
        settings.set("/physics/fabricEnabled", trace_state["write_fabric"])
    stage.SetEditTarget(trace_state["old_layer"])
    layer_id = trace_state["session_sub_layer"].identifier
    if layer_id in stage.GetSessionLayer().subLayerPaths:
        stage.GetSessionLayer().subLayerPaths.remove(layer_id)


def _record_value(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    try:
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    try:
        if hasattr(value, "__len__") and not isinstance(value, (bytes, bytearray)):
            return [_record_value(item) for item in list(value)]
    except Exception:
        pass
    return str(value)


def _public_record_fields(record: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in dir(record):
        if name.startswith("_"):
            continue
        try:
            value = getattr(record, name)
        except Exception:
            continue
        if callable(value):
            continue
        fields[name] = _record_value(value)
    return fields


def _first_int_field(fields: dict[str, Any], names: list[str]) -> int | None:
    for name in names:
        if name not in fields:
            continue
        try:
            return int(fields[name])
        except Exception:
            continue
    return None


def _contact_data_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    field_names = sorted({name for sample in samples for name in sample})
    separations: list[float] = []
    impulse_norms: list[float] = []
    for sample in samples:
        for name, value in sample.items():
            lower_name = name.lower()
            if lower_name in {"separation", "distance"}:
                try:
                    separations.append(float(value))
                except Exception:
                    pass
            if "impulse" in lower_name or lower_name.endswith("force"):
                try:
                    arr = np.asarray(value, dtype=np.float64).reshape(-1)
                    if arr.size:
                        impulse_norms.append(float(np.linalg.norm(arr)))
                except Exception:
                    pass
    return {
        "field_names": field_names,
        "separation_min": min(separations) if separations else None,
        "separation_max": max(separations) if separations else None,
        "max_impulse_norm": max(impulse_norms) if impulse_norms else None,
    }


def _contact_report_rows_from_records(
    contact_headers: Any,
    contact_data: Any,
    *,
    path_from_id: Any,
    contact_found_type: int,
    max_contact_data_sample: int = 8,
) -> list[dict[str, Any]]:
    data_list = list(contact_data or [])
    rows: list[dict[str, Any]] = []
    running_offset = 0
    for contact_header in contact_headers:
        header_fields = _public_record_fields(contact_header)
        collider0 = str(path_from_id(contact_header.collider0))
        collider1 = str(path_from_id(contact_header.collider1))
        contact_count = _first_int_field(
            header_fields,
            ["numContactData", "num_contact_data", "numContacts", "contactCount"],
        )
        contact_offset = _first_int_field(
            header_fields,
            ["contactDataOffset", "contact_data_offset", "startIndex", "contactDataStartIndex"],
        )
        if contact_count is None:
            contact_count = 0
        if contact_offset is None:
            contact_offset = running_offset
        contact_slice = data_list[contact_offset : contact_offset + max(contact_count, 0)]
        contact_samples = [_public_record_fields(item) for item in contact_slice[:max_contact_data_sample]]
        rows.append(
            {
                "type": int(contact_header.type),
                "type_name": "CONTACT_FOUND" if int(contact_header.type) == int(contact_found_type) else str(contact_header.type),
                "collider0": collider0,
                "collider1": collider1,
                "sorted_pair": sorted([collider0, collider1]),
                "raw_header_fields": header_fields,
                "num_contact_data": int(contact_count),
                "contact_data_offset": int(contact_offset),
                "contact_data_sample": contact_samples,
                "contact_data_summary": _contact_data_summary(contact_samples),
                "contact_data_sample_truncated": bool(contact_count > max_contact_data_sample),
            }
        )
        running_offset = max(running_offset, int(contact_offset) + max(int(contact_count), 0))
    return rows


def _read_contact_pairs(trace_state: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not trace_state:
        return []
    from omni.physx.bindings._physx import ContactEventType
    from pxr import PhysicsSchemaTools

    contact_headers, _contact_data = trace_state["physx_interface"].get_contact_report()
    return _contact_report_rows_from_records(
        contact_headers,
        _contact_data,
        path_from_id=PhysicsSchemaTools.intToSdfPath,
        contact_found_type=int(ContactEventType.CONTACT_FOUND),
    )


def _path_matches(path: str, target: str) -> bool:
    return path == target or path.startswith(f"{target}/")


def _pair_touches_targets(pair: dict[str, Any], object_path: str, finger_paths: list[str]) -> bool:
    collider0 = str(pair["collider0"])
    collider1 = str(pair["collider1"])
    touches_object = _path_matches(collider0, object_path) or _path_matches(collider1, object_path)
    touches_finger = any(
        _path_matches(collider0, finger_path) or _path_matches(collider1, finger_path) for finger_path in finger_paths
    )
    return bool(touches_object and touches_finger)


def _pair_touches_finger(pair: dict[str, Any], object_path: str, finger_path: str) -> bool:
    return _pair_touches_targets(pair, object_path, [finger_path])


def _pair_touches_path(pair: dict[str, Any], path: str) -> bool:
    collider0 = str(pair["collider0"])
    collider1 = str(pair["collider1"])
    return bool(_path_matches(collider0, path) or _path_matches(collider1, path))


def _unique_pairs(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [list(pair) for pair in sorted({tuple(row["sorted_pair"]) for row in rows})]


def _phase_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        phase = str(row.get("phase", "unknown"))
        counts[phase] = counts.get(phase, 0) + 1
    return counts


def _unique_pair_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for pair in _unique_pairs(rows):
        pair_rows = [row for row in rows if list(row.get("sorted_pair") or []) == pair]
        summaries.append(
            {
                "pair": pair,
                "contact_pair_count": len(pair_rows),
                "phase_counts": _phase_counts(pair_rows),
                "first_contact_pair": pair_rows[0] if pair_rows else None,
            }
        )
    return summaries


def _contact_quality_by_phase(rows: list[dict[str, Any]]) -> dict[str, Any]:
    phases = sorted({str(row.get("phase", "unknown")) for row in rows})
    return {phase: _contact_quality_summary([row for row in rows if str(row.get("phase", "unknown")) == phase]) for phase in phases}


def _contact_quality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize contact quality across all rows, not only the first event.

    A PhysX contact report can be useful as proximity evidence even when it is
    not yet load-bearing.  Separation and impulse summaries make that distinction
    visible without changing the pass/fail gates.
    """

    separations: list[float] = []
    impulse_norms: list[float] = []
    contact_steps: set[int] = set()
    nonzero_impulse_steps: set[int] = set()
    sample_count = 0
    rows_with_samples = 0
    for row in rows:
        try:
            row_step = int(row.get("step"))
            contact_steps.add(row_step)
        except Exception:
            row_step = None
        samples = list(row.get("contact_data_sample") or [])
        if samples:
            rows_with_samples += 1
        for sample in samples:
            sample_count += 1
            for name, value in sample.items():
                lower_name = str(name).lower()
                if lower_name in {"separation", "distance"}:
                    try:
                        separations.append(float(value))
                    except Exception:
                        pass
                if "impulse" in lower_name or lower_name.endswith("force"):
                    try:
                        arr = np.asarray(value, dtype=np.float64).reshape(-1)
                        if arr.size:
                            impulse_norm = float(np.linalg.norm(arr))
                            impulse_norms.append(impulse_norm)
                            if row_step is not None and impulse_norm > 1e-8:
                                nonzero_impulse_steps.add(row_step)
                    except Exception:
                        pass
    separation_arr = np.asarray(separations, dtype=np.float64)
    impulse_arr = np.asarray(impulse_norms, dtype=np.float64)
    finite_impulse = impulse_arr[np.isfinite(impulse_arr)]
    nonzero_impulse_eps = 1e-8
    return {
        "row_count": len(rows),
        "contact_step_count": len(contact_steps),
        "contact_steps": sorted(contact_steps),
        "first_step": min(contact_steps) if contact_steps else None,
        "last_step": max(contact_steps) if contact_steps else None,
        "rows_with_contact_data_samples": rows_with_samples,
        "contact_data_sample_count": sample_count,
        "separation": _percentile_summary(separation_arr),
        "separation_min": float(np.nanmin(separation_arr)) if separation_arr.size else None,
        "separation_max": float(np.nanmax(separation_arr)) if separation_arr.size else None,
        "impulse_norm": _percentile_summary(impulse_arr),
        "max_impulse_norm": float(np.nanmax(impulse_arr)) if impulse_arr.size else None,
        "nonzero_impulse_count": int(np.sum(finite_impulse > nonzero_impulse_eps)),
        "nonzero_impulse_step_count": len(nonzero_impulse_steps),
        "first_nonzero_impulse_step": min(nonzero_impulse_steps) if nonzero_impulse_steps else None,
        "last_nonzero_impulse_step": max(nonzero_impulse_steps) if nonzero_impulse_steps else None,
        "nonzero_impulse_eps": nonzero_impulse_eps,
    }


def _first_vector_field(sample: dict[str, Any], candidate_names: tuple[str, ...]) -> list[float] | None:
    normalized = {str(name).lower(): value for name, value in sample.items()}
    value = None
    for name in candidate_names:
        if name in normalized:
            value = normalized[name]
            break
    if value is None:
        for key, candidate in normalized.items():
            if any(name in key for name in candidate_names):
                value = candidate
                break
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _first_scalar_field(sample: dict[str, Any], candidate_names: tuple[str, ...]) -> float | None:
    normalized = {str(name).lower(): value for name, value in sample.items()}
    value = None
    for name in candidate_names:
        if name in normalized:
            value = normalized[name]
            break
    if value is None:
        for key, candidate in normalized.items():
            if any(name in key for name in candidate_names):
                value = candidate
                break
    if value is None:
        return None
    try:
        number = float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
    except Exception:
        return None
    return number if np.isfinite(number) else None


def _sum_vectors(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        return np.zeros(3, dtype=np.float64)
    return np.sum(np.stack(vectors, axis=0), axis=0)


def _mean_vector(vectors: list[np.ndarray]) -> list[float] | None:
    if not vectors:
        return None
    return np.mean(np.stack(vectors, axis=0), axis=0).astype(float).tolist()


def _finger_label_for_path(path: str, index: int) -> str:
    lowered = path.lower()
    if "right_finger" in lowered or "_right_finger" in lowered or "/right_" in lowered:
        return "right_finger"
    if "left_finger" in lowered or "_left_finger" in lowered or "/left_" in lowered:
        return "left_finger"
    if "right" in lowered:
        return "right_finger"
    if "left" in lowered:
        return "left_finger"
    return f"finger_{index}"


def _lift_contact_wrench_patch_audit(
    *,
    rows: list[dict[str, Any]],
    contact_pair_rows: list[dict[str, Any]],
    object_path: str,
    expected_finger_paths: list[str],
    eval_phase: str = "post_close_lift",
) -> dict[str, Any]:
    """Diagnostic-only finger/object contact wrench and patch summary.

    PhysX reports contact impulses for a discrete solver step.  This helper
    preserves them as diagnostic vectors and relative statistics; it does not
    treat them as calibrated force measurements or change any pass/fail gate.
    """

    center_by_step: dict[int, np.ndarray] = {}
    phase_rows = [row for row in rows if row.get("phase") == eval_phase]
    for row in phase_rows:
        try:
            step = int(row.get("step"))
            center = np.asarray(
                [row.get("object_center_x"), row.get("object_center_y"), row.get("object_center_z")],
                dtype=np.float64,
            )
        except Exception:
            continue
        if np.all(np.isfinite(center)):
            center_by_step[step] = center

    finger_paths = list(expected_finger_paths)
    finger_labels = {
        path: _finger_label_for_path(path, index)
        for index, path in enumerate(finger_paths)
    }
    by_finger: dict[str, dict[str, Any]] = {
        label: {
            "finger_path": path,
            "sample_count": 0,
            "step_count": 0,
            "steps": set(),
            "_positions": [],
            "_relative_positions": [],
            "_normals": [],
            "_impulses": [],
            "_torques": [],
            "_separations": [],
        }
        for path, label in finger_labels.items()
    }

    target_contact_rows = [
        row
        for row in contact_pair_rows
        if row.get("phase") == eval_phase and _pair_touches_targets(row, object_path, finger_paths)
    ]
    usable_sample_count = 0
    missing_vector_sample_count = 0
    for row in target_contact_rows:
        try:
            step = int(row.get("step"))
        except Exception:
            continue
        object_center = center_by_step.get(step)
        touched_finger_path = next(
            (path for path in finger_paths if _pair_touches_finger(row, object_path, path)),
            None,
        )
        if touched_finger_path is None:
            continue
        label = finger_labels[touched_finger_path]
        finger_row = by_finger[label]
        for sample in list(row.get("contact_data_sample") or []):
            position = _first_vector_field(sample, ("position", "point"))
            normal = _first_vector_field(sample, ("normal",))
            impulse = _first_vector_field(sample, ("impulse", "force"))
            separation = _first_scalar_field(sample, ("separation", "distance"))
            if position is None or impulse is None:
                missing_vector_sample_count += 1
                continue
            pos = np.asarray(position, dtype=np.float64)
            imp = np.asarray(impulse, dtype=np.float64)
            rel = pos - object_center if object_center is not None else np.full(3, np.nan, dtype=np.float64)
            torque = np.cross(rel, imp) if np.all(np.isfinite(rel)) else np.full(3, np.nan, dtype=np.float64)
            finger_row["sample_count"] += 1
            finger_row["steps"].add(step)
            finger_row["_positions"].append(pos)
            finger_row["_relative_positions"].append(rel)
            if normal is not None:
                finger_row["_normals"].append(np.asarray(normal, dtype=np.float64))
            finger_row["_impulses"].append(imp)
            if np.all(np.isfinite(torque)):
                finger_row["_torques"].append(torque)
            if separation is not None:
                finger_row["_separations"].append(float(separation))
            usable_sample_count += 1

    net_impulse = np.zeros(3, dtype=np.float64)
    net_torque = np.zeros(3, dtype=np.float64)
    total_impulse_norm_by_finger: dict[str, float] = {}
    finalized_by_finger: dict[str, dict[str, Any]] = {}
    for label, data in by_finger.items():
        impulses = list(data.pop("_impulses"))
        positions = list(data.pop("_positions"))
        relative_positions = list(data.pop("_relative_positions"))
        normals = list(data.pop("_normals"))
        torques = list(data.pop("_torques"))
        separations = list(data.pop("_separations"))
        impulse_sum = _sum_vectors(impulses)
        torque_sum = _sum_vectors(torques)
        impulse_norms = np.asarray([float(np.linalg.norm(v)) for v in impulses], dtype=np.float64)
        position_spread = None
        if positions:
            pos_arr = np.stack(positions, axis=0)
            position_spread = float(np.max(np.linalg.norm(pos_arr - np.mean(pos_arr, axis=0), axis=1)))
        net_impulse += impulse_sum
        net_torque += torque_sum
        total_norm = float(np.sum(impulse_norms)) if impulse_norms.size else 0.0
        total_impulse_norm_by_finger[label] = total_norm
        finalized = {
            "finger_path": data["finger_path"],
            "sample_count": int(data["sample_count"]),
            "step_count": len(data["steps"]),
            "steps": sorted(data["steps"]),
            "total_impulse_world": impulse_sum.astype(float).tolist(),
            "total_impulse_norm_sum": total_norm,
            "vertical_impulse_z_sum": float(impulse_sum[2]),
            "lateral_impulse_xy_norm": float(np.linalg.norm(impulse_sum[:2])),
            "lateral_impulse_y_sum": float(impulse_sum[1]),
            "mean_normal_world": _mean_vector(normals),
            "mean_contact_position_world": _mean_vector(positions),
            "mean_contact_position_relative_to_object_center": _mean_vector(relative_positions),
            "total_torque_about_object_center_world": torque_sum.astype(float).tolist(),
            "contact_patch_spread_m": position_spread,
            "separation": _percentile_summary(np.asarray(separations, dtype=np.float64)),
        }
        finalized_by_finger[label] = finalized

    object_start = phase_rows[0] if phase_rows else None
    object_end = phase_rows[-1] if phase_rows else None
    object_delta = None
    if object_start is not None and object_end is not None:
        try:
            object_delta = [
                float(object_end["object_center_x"] - object_start["object_center_x"]),
                float(object_end["object_center_y"] - object_start["object_center_y"]),
                float(object_end["object_center_z"] - object_start["object_center_z"]),
            ]
        except Exception:
            object_delta = None

    positive_norms = [value for value in total_impulse_norm_by_finger.values() if value > 1e-12]
    impulse_balance_ratio = None
    if len(positive_norms) >= 2:
        impulse_balance_ratio = float(max(positive_norms) / min(positive_norms))
    y_drift = None if object_delta is None else float(object_delta[1])
    z_lift = None if object_delta is None else float(object_delta[2])
    lateral_drift_threshold_m = 0.003
    status = "PASS_CONTACT_WRENCH_AUDIT_REPORTED"
    if usable_sample_count <= 0:
        status = "INSUFFICIENT_TARGET_CONTACT_VECTOR_SAMPLES"
    elif y_drift is not None and abs(y_drift) >= lateral_drift_threshold_m and (z_lift is None or z_lift < 0.001):
        status = "DIAGNOSTIC_LATERAL_DRIFT_WITH_LOW_LIFT"

    return {
        "enabled": True,
        "diagnostic_only": True,
        "eval_phase": eval_phase,
        "status": status,
        "object_path": object_path,
        "expected_finger_paths": finger_paths,
        "target_contact_row_count": len(target_contact_rows),
        "usable_vector_sample_count": int(usable_sample_count),
        "missing_vector_sample_count": int(missing_vector_sample_count),
        "object_delta_world": object_delta,
        "net_finger_impulse_world": net_impulse.astype(float).tolist(),
        "net_vertical_impulse_z": float(net_impulse[2]),
        "net_lateral_impulse_xy_norm": float(np.linalg.norm(net_impulse[:2])),
        "net_lateral_impulse_y": float(net_impulse[1]),
        "net_torque_about_object_center_world": net_torque.astype(float).tolist(),
        "finger_total_impulse_norm_balance_ratio": impulse_balance_ratio,
        "lateral_drift_threshold_m": lateral_drift_threshold_m,
        "by_finger": finalized_by_finger,
        "interpretation_notes": (
            "Contact impulse is a PhysX per-step diagnostic impulse, not a calibrated continuous force. "
            "Use signs, relative magnitudes, contact patch offsets, and run-to-run comparisons to diagnose "
            "whether the fingers lift the bottle or push it laterally."
        ),
    }


def _box_min_z(box: dict[str, Any] | None) -> float | None:
    if not isinstance(box, dict):
        return None
    values = box.get("min")
    if not isinstance(values, list | tuple) or len(values) < 3:
        return None
    try:
        value = float(values[2])
    except Exception:
        return None
    return value if np.isfinite(value) else None


def _nested_number(row: dict[str, Any] | None, *keys: str) -> float | None:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    try:
        number = float(value)
    except Exception:
        return None
    return number if np.isfinite(number) else None


def _fixed_reference_grasp_geometry_gate(
    *,
    tabletop_reference_contract: dict[str, Any],
    object_contact_reset_box: dict[str, Any],
    object_final_contact_box: dict[str, Any],
    start_alignment: dict[str, Any],
    final_alignment: dict[str, Any],
    lift_contact_wrench_patch_audit: dict[str, Any],
    fixed_reference_required: bool,
    lift_required: bool,
    max_reset_table_gap_m: float = 0.002,
    max_final_table_penetration_m: float = 0.002,
    max_reference_contact_cross_offset_m: float = 0.005,
    max_reference_midplane_correction_m: float = 0.003,
    max_lateral_to_vertical_impulse_ratio: float = 0.15,
) -> dict[str, Any]:
    """Check whether fixed-table grasp geometry is suitable for lift claims.

    This gate is intentionally about geometry and contact direction, not about
    drive strength.  It catches cases where a finger/object contact trace exists
    but the bottle contact proxy has already penetrated the table or the grasp
    band has drifted away from the two-finger midplane.
    """

    required = bool(fixed_reference_required and lift_required)
    table_top_z = _nested_number(tabletop_reference_contract, "table_top_z_m")
    reset_min_z = _box_min_z(object_contact_reset_box)
    final_min_z = _box_min_z(object_final_contact_box)
    reset_table_gap = None if table_top_z is None or reset_min_z is None else float(reset_min_z - table_top_z)
    final_table_penetration = (
        None if table_top_z is None or final_min_z is None else float(max(0.0, table_top_z - final_min_z))
    )
    start_reference_cross_offset = _nested_number(
        start_alignment, "reference_contact_center", "cross_closing_axis_offset_norm_m"
    )
    final_reference_cross_offset = _nested_number(
        final_alignment, "reference_contact_center", "cross_closing_axis_offset_norm_m"
    )
    start_midplane_correction = _nested_number(
        start_alignment, "reference_contact_center", "correction_to_midplane_norm_m"
    )
    final_midplane_correction = _nested_number(
        final_alignment, "reference_contact_center", "correction_to_midplane_norm_m"
    )
    vertical_impulse = _nested_number(lift_contact_wrench_patch_audit, "net_vertical_impulse_z")
    lateral_impulse = _nested_number(lift_contact_wrench_patch_audit, "net_lateral_impulse_xy_norm")
    lateral_to_vertical_ratio = (
        None
        if vertical_impulse is None or abs(vertical_impulse) < 1e-12 or lateral_impulse is None
        else float(lateral_impulse / abs(vertical_impulse))
    )

    checks = {
        "reset_contact_bottom_near_table": bool(
            reset_table_gap is not None
            and abs(reset_table_gap) <= float(max_reset_table_gap_m)
        ),
        "final_contact_not_deeply_inside_table": bool(
            final_table_penetration is not None
            and final_table_penetration <= float(max_final_table_penetration_m)
        ),
        "final_reference_contact_near_two_finger_midplane": bool(
            final_reference_cross_offset is not None
            and final_reference_cross_offset <= float(max_reference_contact_cross_offset_m)
        ),
        "final_reference_contact_midplane_correction_small": bool(
            final_midplane_correction is not None
            and final_midplane_correction <= float(max_reference_midplane_correction_m)
        ),
        "net_lateral_impulse_not_dominant": bool(
            lateral_to_vertical_ratio is not None
            and lateral_to_vertical_ratio <= float(max_lateral_to_vertical_impulse_ratio)
        ),
    }
    if not required:
        return {
            "required": False,
            "pass": True,
            "status": "SKIPPED_NOT_FIXED_REFERENCE_LIFT",
            "checks": checks,
            "notes": (
                "This geometry gate is only required for fixed-reference dynamic lift validation. "
                "Contact-only or diagnostic table-shift runs keep it as metadata."
            ),
        }

    failed = [name for name, ok in checks.items() if not ok]
    status = "PASS_FIXED_REFERENCE_GRASP_GEOMETRY"
    if failed:
        if "final_contact_not_deeply_inside_table" in failed:
            status = "FAIL_FIXED_REFERENCE_CONTACT_PROXY_TABLE_PENETRATION"
        elif any(name.startswith("final_reference_contact") for name in failed):
            status = "FAIL_FIXED_REFERENCE_GRASP_BAND_LEFT_FINGER_MIDPLANE"
        elif "net_lateral_impulse_not_dominant" in failed:
            status = "FAIL_FIXED_REFERENCE_LATERAL_CONTACT_IMPULSE_DOMINANT"
        else:
            status = "FAIL_FIXED_REFERENCE_GRASP_GEOMETRY"
    return {
        "required": True,
        "pass": not failed,
        "status": status,
        "failed_checks": failed,
        "checks": checks,
        "table_top_z_m": table_top_z,
        "reset_contact_bottom_z_m": reset_min_z,
        "final_contact_bottom_z_m": final_min_z,
        "reset_contact_bottom_table_gap_m": reset_table_gap,
        "final_contact_table_penetration_m": final_table_penetration,
        "start_reference_contact_cross_offset_m": start_reference_cross_offset,
        "final_reference_contact_cross_offset_m": final_reference_cross_offset,
        "start_reference_midplane_correction_m": start_midplane_correction,
        "final_reference_midplane_correction_m": final_midplane_correction,
        "net_vertical_impulse_z": vertical_impulse,
        "net_lateral_impulse_xy_norm": lateral_impulse,
        "net_lateral_to_vertical_impulse_ratio": lateral_to_vertical_ratio,
        "thresholds": {
            "max_reset_table_gap_m": float(max_reset_table_gap_m),
            "max_final_table_penetration_m": float(max_final_table_penetration_m),
            "max_reference_contact_cross_offset_m": float(max_reference_contact_cross_offset_m),
            "max_reference_midplane_correction_m": float(max_reference_midplane_correction_m),
            "max_lateral_to_vertical_impulse_ratio": float(max_lateral_to_vertical_impulse_ratio),
        },
        "notes": (
            "Fixed-reference replay must first prove the physical bottle contact proxy sits on the real "
            "table and the intended rear-quarter grasp band stays near the two-finger midplane. Larger "
            "finger stiffness can increase impulse while still pushing the bottle laterally if this gate fails."
        ),
    }


def _summary_stat(summary: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(summary, dict):
        return None
    value = summary.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return number if np.isfinite(number) else None


def _category_phase_quality(
    contact_summary: dict[str, Any],
    *,
    category: str,
    phase: str,
) -> dict[str, Any]:
    category_row = (contact_summary.get("object_contact_categories") or {}).get(category) or {}
    phase_quality = (category_row.get("phase_quality") or {}).get(phase)
    if isinstance(phase_quality, dict):
        return phase_quality
    quality = category_row.get("contact_quality")
    return quality if isinstance(quality, dict) else {}


def _table_load_bearing_contact_gate(
    *,
    contact_summary: dict[str, Any],
    eval_phase: str,
    table_category: str = "workcell_or_environment",
    target_category: str = "target_finger",
    max_table_to_finger_impulse_mean_ratio: float = 0.10,
    max_table_nonzero_impulse_steps: int = 2,
) -> dict[str, Any]:
    """Classify table/workcell contact by load transfer, not row count alone."""

    table_quality = _category_phase_quality(contact_summary, category=table_category, phase=eval_phase)
    target_quality = _category_phase_quality(contact_summary, category=target_category, phase=eval_phase)
    table_steps = int(table_quality.get("contact_step_count") or 0)
    if table_steps <= 0:
        return {
            "pass": True,
            "status": "PASS_NO_TABLE_CONTACT_ROWS",
            "eval_phase": eval_phase,
            "table_category": table_category,
            "target_category": target_category,
            "table_contact_step_count": 0,
            "notes": "No table/workcell contact rows were reported in the evaluated lift phase.",
        }

    table_impulse = table_quality.get("impulse_norm") or {}
    table_separation = table_quality.get("separation") or {}
    target_impulse = target_quality.get("impulse_norm") or {}
    table_impulse_mean = _summary_stat(table_impulse, "mean")
    table_impulse_p50 = _summary_stat(table_impulse, "p50")
    table_impulse_max = _summary_stat(table_impulse, "max")
    table_separation_p50 = _summary_stat(table_separation, "p50")
    table_separation_p95 = _summary_stat(table_separation, "p95")
    target_impulse_mean = _summary_stat(target_impulse, "mean")
    table_nonzero_steps = int(table_quality.get("nonzero_impulse_step_count") or 0)
    ratio = (
        None
        if target_impulse_mean is None or abs(target_impulse_mean) < 1e-12 or table_impulse_mean is None
        else float(table_impulse_mean / target_impulse_mean)
    )
    median_zero_and_separated = bool(
        table_impulse_p50 is not None
        and abs(table_impulse_p50) <= 1e-12
        and table_separation_p50 is not None
        and table_separation_p50 > 0.0
    )
    sparse_impulse = table_nonzero_steps <= int(max_table_nonzero_impulse_steps)
    weak_relative_impulse = bool(ratio is not None and ratio < float(max_table_to_finger_impulse_mean_ratio))
    not_load_bearing = bool(median_zero_and_separated or sparse_impulse or weak_relative_impulse)
    return {
        "pass": not_load_bearing,
        "status": "PASS_TABLE_NEAR_CONTACT_NON_LOAD_BEARING"
        if not_load_bearing
        else "FAIL_TABLE_LOAD_BEARING_CONTACT",
        "eval_phase": eval_phase,
        "table_category": table_category,
        "target_category": target_category,
        "table_contact_step_count": table_steps,
        "table_nonzero_impulse_step_count": table_nonzero_steps,
        "table_impulse_p50": table_impulse_p50,
        "table_impulse_mean": table_impulse_mean,
        "table_impulse_max": table_impulse_max,
        "table_separation_p50": table_separation_p50,
        "table_separation_p95": table_separation_p95,
        "target_finger_impulse_mean": target_impulse_mean,
        "table_to_finger_impulse_mean_ratio": ratio,
        "max_table_to_finger_impulse_mean_ratio": float(max_table_to_finger_impulse_mean_ratio),
        "max_table_nonzero_impulse_steps": int(max_table_nonzero_impulse_steps),
        "classification_inputs": {
            "median_zero_and_separated": median_zero_and_separated,
            "sparse_impulse": sparse_impulse,
            "weak_relative_impulse": weak_relative_impulse,
        },
        "notes": (
            "PhysX contact rows can include near-contact inside contactOffset. This gate uses impulse and "
            "separation to distinguish table support from non-load-bearing proximity."
        ),
    }


def _other_collider_for_object_pair(row: dict[str, Any], object_path: str) -> str | None:
    collider0 = str(row["collider0"])
    collider1 = str(row["collider1"])
    collider0_is_object = _path_matches(collider0, object_path)
    collider1_is_object = _path_matches(collider1, object_path)
    if collider0_is_object and not collider1_is_object:
        return collider1
    if collider1_is_object and not collider0_is_object:
        return collider0
    return None


def _classify_object_contact(
    row: dict[str, Any],
    *,
    object_path: str,
    expected_finger_paths: list[str],
    same_side_robot_root: str | None,
    other_side_robot_root: str | None,
    diagnostic_contact_paths: list[str] | None,
) -> str | None:
    other_path = _other_collider_for_object_pair(row, object_path)
    if other_path is None:
        return None
    if any(_path_matches(other_path, finger_path) for finger_path in expected_finger_paths):
        return "target_finger"
    if any(_path_matches(other_path, path) for path in (diagnostic_contact_paths or [])):
        return "diagnostic_support"
    if same_side_robot_root and _path_matches(other_path, same_side_robot_root):
        return "same_side_robot_non_target"
    if other_side_robot_root and _path_matches(other_path, other_side_robot_root):
        return "other_side_robot"
    if (
        other_path.startswith("/scene/worldBody/")
        or other_path == "/scene/worldBody"
        or other_path.startswith("/colliders/")
        or other_path.startswith("/World/")
    ):
        return "workcell_or_environment"
    return "unknown"


def _summarize_contact_pairs(
    *,
    contact_pair_rows: list[dict[str, Any]],
    object_path: str,
    expected_finger_paths: list[str],
    diagnostic_contact_paths: list[str] | None = None,
    same_side_robot_root: str | None = None,
    other_side_robot_root: str | None = None,
    sample_limit: int = 80,
) -> dict[str, Any]:
    diagnostic_paths = diagnostic_contact_paths or []
    unique_pairs = sorted({tuple(row["sorted_pair"]) for row in contact_pair_rows})
    target_rows = [row for row in contact_pair_rows if _pair_touches_targets(row, object_path, expected_finger_paths)]
    wrong_rows = [
        row for row in contact_pair_rows if not _pair_touches_targets(row, object_path, expected_finger_paths)
    ]
    target_found_rows = [row for row in target_rows if row.get("type_name") == "CONTACT_FOUND"]
    target_steps = sorted({int(row["step"]) for row in target_rows})
    target_found_phases = sorted({str(row.get("phase", "unknown")) for row in target_found_rows})
    wrong_pairs = sorted({tuple(row["sorted_pair"]) for row in wrong_rows})
    finger_target_rows = {
        finger_path: [row for row in contact_pair_rows if _pair_touches_finger(row, object_path, finger_path)]
        for finger_path in expected_finger_paths
    }
    finger_target_found_rows = {
        finger_path: [row for row in rows if row.get("type_name") == "CONTACT_FOUND"]
        for finger_path, rows in finger_target_rows.items()
    }
    finger_target_quality = {
        finger_path: _contact_quality_summary(rows) for finger_path, rows in finger_target_rows.items()
    }
    diagnostic_summaries: dict[str, Any] = {}
    for path in diagnostic_paths:
        path_rows = [row for row in contact_pair_rows if _pair_touches_path(row, path)]
        object_rows = [row for row in path_rows if _pair_touches_path(row, object_path)]
        finger_rows = [
            row
            for row in path_rows
            if any(_pair_touches_path(row, finger_path) for finger_path in expected_finger_paths)
        ]
        other_rows = [
            row
            for row in path_rows
            if not _pair_touches_path(row, object_path)
            and not any(_pair_touches_path(row, finger_path) for finger_path in expected_finger_paths)
        ]
        diagnostic_summaries[path] = {
            "contact_pair_count": len(path_rows),
            "unique_contact_pairs": _unique_pairs(path_rows),
            "object_contact_pair_count": len(object_rows),
            "object_contact_pairs": _unique_pairs(object_rows),
            "expected_finger_contact_pair_count": len(finger_rows),
            "expected_finger_contact_pairs": _unique_pairs(finger_rows),
            "other_contact_pair_count": len(other_rows),
            "other_contact_pairs": _unique_pairs(other_rows),
        }
    object_contact_rows = [
        row for row in contact_pair_rows if _other_collider_for_object_pair(row, object_path) is not None
    ]
    category_rows: dict[str, list[dict[str, Any]]] = {}
    for row in object_contact_rows:
        category = _classify_object_contact(
            row,
            object_path=object_path,
            expected_finger_paths=expected_finger_paths,
            same_side_robot_root=same_side_robot_root,
            other_side_robot_root=other_side_robot_root,
            diagnostic_contact_paths=diagnostic_paths,
        )
        if category is None:
            continue
        category_rows.setdefault(category, []).append(row)
    non_target_categories = {category: rows for category, rows in category_rows.items() if category != "target_finger"}
    non_target_object_rows = [
        row for category, rows in category_rows.items() if category != "target_finger" for row in rows
    ]
    non_target_object_rows.sort(key=lambda row: (str(row.get("phase", "")), int(row.get("step", 0))))
    object_contact_categories = {
        category: {
            "contact_pair_count": len(rows),
            "unique_contact_pair_count": len(_unique_pairs(rows)),
            "unique_contact_pairs": _unique_pairs(rows),
            "unique_contact_pair_summaries": _unique_pair_summaries(rows),
            "phase_counts": _phase_counts(rows),
            "contact_quality": _contact_quality_summary(rows),
            "phase_quality": _contact_quality_by_phase(rows),
            "first_contact_pair": rows[0] if rows else None,
        }
        for category, rows in sorted(category_rows.items())
    }
    return {
        "contact_pair_count": len(contact_pair_rows),
        "unique_contact_pairs": [list(pair) for pair in unique_pairs],
        "contact_pairs_sample": contact_pair_rows[:sample_limit],
        "expected_contact_object": object_path,
        "expected_contact_fingers": expected_finger_paths,
        "target_contact_pair_found": bool(target_rows),
        "target_contact_found_event": bool(target_found_rows),
        "target_contact_finger_hits": {finger_path: bool(rows) for finger_path, rows in finger_target_rows.items()},
        "target_contact_found_finger_hits": {
            finger_path: bool(rows) for finger_path, rows in finger_target_found_rows.items()
        },
        "target_contact_quality": _contact_quality_summary(target_rows),
        "target_contact_quality_by_finger": finger_target_quality,
        "all_expected_fingers_target_contact_pair_found": all(bool(rows) for rows in finger_target_rows.values())
        if expected_finger_paths
        else False,
        "all_expected_fingers_target_contact_found_event": all(bool(rows) for rows in finger_target_found_rows.values())
        if expected_finger_paths
        else False,
        "first_target_contact_pair": target_rows[0] if target_rows else None,
        "first_target_contact_found_pair": target_found_rows[0] if target_found_rows else None,
        "first_target_contact_phase": target_rows[0].get("phase") if target_rows else None,
        "first_target_contact_found_phase": target_found_rows[0].get("phase") if target_found_rows else None,
        "first_target_contact_step": target_steps[0] if target_steps else None,
        "target_contact_found_phases": target_found_phases,
        "target_contact_found_during_settle": "settle" in target_found_phases,
        "target_contact_found_during_close": "close" in target_found_phases,
        "target_contact_found_during_post_close_hold": "post_close_hold" in target_found_phases,
        "target_contact_found_during_post_close_lift": "post_close_lift" in target_found_phases,
        "target_contact_steps": target_steps,
        "target_contact_persistence_steps": len(target_steps),
        "wrong_contact_pairs": [list(pair) for pair in wrong_pairs],
        "diagnostic_contact_summaries": diagnostic_summaries,
        "object_contact_pair_count": len(object_contact_rows),
        "object_contact_categories": object_contact_categories,
        "non_target_object_contact_pair_count": sum(len(rows) for rows in non_target_categories.values()),
        "non_target_object_contact_categories": sorted(non_target_categories),
        "non_target_object_contact_found": bool(non_target_categories),
        "first_non_target_object_contact_pair": non_target_object_rows[0] if non_target_object_rows else None,
        "first_non_target_object_contact_phase": non_target_object_rows[0].get("phase")
        if non_target_object_rows
        else None,
    }


def _cross_side_proxy_overlap_summary(
    stage: Any, paths_by_side: dict[str, dict[str, str]], side: str, tolerance: float = 1e-8
) -> dict[str, Any]:
    other_side = "right" if side == "left" else "left"
    rows: list[dict[str, Any]] = []
    for finger_name in ("left_finger", "right_finger"):
        current_path = paths_by_side[side][finger_name]
        other_path = paths_by_side[other_side][finger_name]
        current_box = _bbox_row(stage, current_path)
        other_box = _bbox_row(stage, other_path)
        center_distance = None
        size_delta = None
        overlaps = False
        if current_box.get("bbox_valid") and other_box.get("bbox_valid"):
            current_center = np.asarray(current_box["center"], dtype=np.float64)
            other_center = np.asarray(other_box["center"], dtype=np.float64)
            current_size = np.asarray(current_box["size"], dtype=np.float64)
            other_size = np.asarray(other_box["size"], dtype=np.float64)
            center_distance = float(np.linalg.norm(current_center - other_center))
            size_delta = float(np.linalg.norm(current_size - other_size))
            overlaps = bool(center_distance <= tolerance and size_delta <= tolerance)
        rows.append(
            {
                "finger": finger_name,
                "current_path": current_path,
                "other_path": other_path,
                "current_bbox_valid": bool(current_box.get("bbox_valid")),
                "other_bbox_valid": bool(other_box.get("bbox_valid")),
                "center_distance": center_distance,
                "size_delta": size_delta,
                "overlaps_with_other_side": overlaps,
            }
        )
    return {
        "side": side,
        "other_side": other_side,
        "tolerance": tolerance,
        "overlap_detected": any(row["overlaps_with_other_side"] for row in rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a local passive-object contact smoke test for ALOHA1 gripper proxies."
    )
    parser.add_argument("--stage-usd", default=str(DEFAULT_STAGE))
    parser.add_argument(
        "--stage-units-in-meters",
        type=float,
        default=DEFAULT_STAGE_UNITS_IN_METERS,
        help=(
            "World(stage_units_in_meters=...). Legacy clean-runtime stages use 0.01; "
            "Phase69 calibrated overlays must use 1.0."
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument(
        "--contact-proxy-profile",
        choices=contact_proxy_profile_names(),
        default="legacy_puppet",
        help=(
            "Namespace profile for articulation roots and fingertip bbox proxies. Use scene_base_link with "
            "Trossen/Menagerie /scene stages."
        ),
    )
    parser.add_argument("--open-offset", type=float, default=0.006)
    parser.add_argument("--close-offset", type=float, default=-0.006)
    parser.add_argument(
        "--right-finger-close-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=-1.0,
        help=(
            "Right-finger sign used only for synthetic close targets. Keep -1 for the legacy opposed-sign "
            "target convention; use +1 when validating ALOHA1 scene-base proxies that close spatially with "
            "both finger DOFs decreasing."
        ),
    )
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument(
        "--close-steps",
        type=int,
        default=None,
        help=(
            "Synthetic close step count, or optional HDF5 target-frame cap. If omitted for HDF5 replay, "
            "the full requested frame window is used."
        ),
    )
    parser.add_argument("--physics-dt", type=float, default=1.0 / 50.0)
    parser.add_argument(
        "--stage-time-codes-per-second",
        type=float,
        default=None,
        help=(
            "Optional USD stage TimeCodesPerSecond metadata. This does not set the PhysX step; "
            "it records the replay/timeline frame semantics, for example 50 for 50 Hz ALOHA HDF5 data."
        ),
    )
    parser.add_argument("--gravity", type=float, default=0.0)
    parser.add_argument(
        "--arm-kp",
        type=float,
        default=None,
        help="Optional runtime position-drive stiffness override for arm DOFs only. Defaults to the asset values.",
    )
    parser.add_argument(
        "--arm-kd",
        type=float,
        default=None,
        help="Optional runtime position-drive damping override for arm DOFs only. Defaults to the asset values.",
    )
    parser.add_argument(
        "--finger-kp",
        type=float,
        default=None,
        help="Optional runtime position-drive stiffness override for the selected side's controlled finger DOFs only.",
    )
    parser.add_argument(
        "--finger-kd",
        type=float,
        default=None,
        help="Optional runtime position-drive damping override for the selected side's controlled finger DOFs only.",
    )
    parser.add_argument("--drive-profile-name", default="runtime_override_from_cli")
    parser.add_argument("--drive-profile-provenance", default="validator_cli_arguments")
    parser.add_argument("--limit-margin", type=float, default=0.001)
    parser.add_argument("--object-fill-fraction", type=float, default=0.6)
    parser.add_argument(
        "--finger-gap-projection-model",
        choices=("world_aabb", "oriented_box"),
        default="world_aabb",
        help=(
            "Projection model for finger inner-gap checks. world_aabb preserves legacy diagnostics; "
            "oriented_box uses each finger proxy's local box support when its transform is available, "
            "which is the stricter formal model for rotated inner-pad proxies."
        ),
    )
    parser.add_argument(
        "--object-side-length",
        type=float,
        default=None,
        help=(
            "Explicit object cross-section size in stage units. When unset, the validator keeps the legacy "
            "--object-fill-fraction * open finger surface gap behavior. Use this for real object diameter "
            "checks so the physics proxy width is not inferred from a particular replay's open gap."
        ),
    )
    parser.add_argument(
        "--object-effective-contact-width",
        type=float,
        default=None,
        help=(
            "Soft-object contact width in stage units. For a real mineral-water bottle, the visible Bottle500 "
            "mesh can keep its true external diameter while the physics/contact proxy uses a smaller effective "
            "width because the bottle is compressed by the gripper. This is mutually exclusive with "
            "--object-side-length and should be reported as a soft-bottle contact model, not as a smaller "
            "visual bottle."
        ),
    )
    parser.add_argument(
        "--object-effective-contact-width-source",
        default="",
        help=(
            "Short provenance note for --object-effective-contact-width, for example measured caliper value, "
            "user observation, or controlled ablation label. This is only metadata and does not affect PhysX."
        ),
    )
    parser.add_argument(
        "--object-placement",
        choices=(
            "gap_center",
            "moving_finger_surface",
            "finger_rear_quarter",
            "hdf5_open_finger_rear_quarter",
            "hdf5_open_finger_rear_quarter_tabletop",
            "hdf5_close_finger_rear_quarter",
            "hdf5_close_finger_rear_quarter_tabletop",
            "grasp_yaml",
        ),
        default="gap_center",
    )
    parser.add_argument("--object-clearance", type=float, default=0.001)
    parser.add_argument(
        "--closing-axis-gap-solver-basis",
        choices=("open", "placement"),
        default="open",
        help=(
            "Finger boxes used by the closing-axis gap centering solver. "
            "open preserves historical behavior. placement uses the current "
            "placement frame, so hdf5_close_* placement centers the contact "
            "proxy in the loaded close-frame gap instead of the open gap."
        ),
    )
    parser.add_argument(
        "--object-width-stop-predictive-margin",
        type=float,
        default=0.0,
        help=(
            "Extra early-stop margin in meters for --enforce-object-width-finger-stop. "
            "This holds finger targets before the observed gap crosses the physical "
            "object-width threshold, preventing one-step hard over-close in zero-order "
            "hold replay."
        ),
    )
    parser.add_argument("--object-creation", choices=("dynamic_cuboid", "raw_usd"), default="raw_usd")
    parser.add_argument(
        "--disable-object-rigid-body",
        action="store_true",
        help=(
            "Create the raw USD contact object as a fixed collision body. This is for active-contact gate "
            "smoke tests; dynamic object grasp validation should leave the rigid body enabled."
        ),
    )
    parser.add_argument(
        "--object-shape",
        choices=(
            "cube",
            "cylinder",
            "capsule",
            "bottle_proxy",
            "bottle_usd",
            "bottle_usd_cylinder_proxy",
            "bottle_usd_segmented_proxy",
            "bottle_usd_grasp_band_proxy",
            "bottle_usd_grasp_box_proxy",
        ),
        default="cube",
    )
    parser.add_argument("--object-axis", choices=("X", "Y", "Z"), default="X")
    parser.add_argument(
        "--object-axis-source",
        choices=("static", "open_finger_horizontal_perpendicular"),
        default="static",
        help=(
            "How to choose the bottle long-axis direction. The open_finger_horizontal_perpendicular mode is "
            "diagnostic: it derives a horizontal bottle yaw perpendicular to the open-frame finger closing line."
        ),
    )
    parser.add_argument(
        "--object-center-offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help=(
            "World-frame offset added to the nominal object center after gap/grasp placement. "
            "Use this when the fingertip contact point is not the whole-object bbox center, "
            "for example when grasping one section of a long bottle."
        ),
    )
    parser.add_argument("--object-length-multiplier", type=float, default=4.0)
    parser.add_argument("--object-usd", default=str(DEFAULT_BOTTLE_USD))
    parser.add_argument("--object-usd-prim-path", default="/Bottle500")
    parser.add_argument(
        "--object-tabletop-reference-path",
        default="/scene/worldBody/table",
        help=(
            "Reference tabletop prim used by hdf5_*_finger_rear_quarter_tabletop. "
            "The object bbox bottom is shifted to this prim's bbox top plus --object-tabletop-clearance."
        ),
    )
    parser.add_argument(
        "--object-tabletop-top-z",
        type=float,
        default=None,
        help=(
            "Explicit world Z for the tabletop top surface used by "
            "hdf5_*_finger_rear_quarter_tabletop. When set, this overrides the reference prim bbox. "
            "Use for diagnostic table-to-robot alignment experiments."
        ),
    )
    parser.add_argument(
        "--derive-tabletop-top-z-from-open-finger-height",
        action="store_true",
        help=(
            "Gate2 diagnostic reset calibration: move the referenced tabletop collider in the composed "
            "validation stage so the soft contact proxy center of a tabletop bottle aligns with the HDF5 "
            "open-finger contact midpoint. This is not a substitute for measured workcell calibration."
        ),
    )
    parser.add_argument("--object-tabletop-clearance", type=float, default=0.001)
    parser.add_argument(
        "--object-grasp-yaml",
        default="assets/bottle_500ml/grasp/bottle_aloha_left_grasps.yaml",
        help="GraspSpec-style YAML used when --object-placement grasp_yaml is selected.",
    )
    parser.add_argument("--object-grasp-name", default="grasp_rear_quarter")
    parser.add_argument("--object-gripper-frame", default="/scene/left_base_link/left_gripper_link")
    parser.add_argument(
        "--object-rear-quarter-fraction",
        type=float,
        default=0.25,
        help=(
            "For --object-placement finger_rear_quarter or hdf5_*_finger_rear_quarter, place the fingertip "
            "gap center at this fraction from the object-axis minimum. 0.25 means the rear quarter of the "
            "bottle body."
        ),
    )
    parser.add_argument(
        "--object-rear-quarter-tolerance",
        type=float,
        default=0.07,
        help="Allowed fraction error for the finger_rear_quarter semantic gate.",
    )
    parser.add_argument(
        "--max-closing-long-axis-dot",
        type=float,
        default=0.20,
        help=(
            "Maximum absolute dot product between the gripper closing/gap axis and the bottle long axis for "
            "rear-quarter grasp semantics. 0.20 requires an almost perpendicular grasp; larger values are "
            "useful for real replay tolerance studies and are recorded in the metrics."
        ),
    )
    parser.add_argument(
        "--max-open-finger-object-center-height-error",
        type=float,
        default=0.04,
        help=(
            "Maximum allowed Z distance between the open-frame fingertip midpoint and the tabletop bottle "
            "contact proxy center for active tabletop grasp tests. Large values mean the HDF5 window is not "
            "actually at bottle-body height; the row is reported instead of hidden."
        ),
    )
    parser.add_argument(
        "--bilateral-grasp-min-contact-steps",
        type=int,
        default=10,
        help="Minimum target-contact physics steps required for each finger in a two-finger grasp.",
    )
    parser.add_argument(
        "--bilateral-grasp-min-nonzero-impulse-steps",
        type=int,
        default=3,
        help="Minimum nonzero-impulse physics steps required for each finger in a two-finger grasp.",
    )
    parser.add_argument(
        "--bilateral-grasp-max-impulse-ratio",
        type=float,
        default=4.0,
        help="Maximum allowed max-impulse ratio between fingers for two-finger grasp formation.",
    )
    parser.add_argument(
        "--bilateral-grasp-max-prelift-lateral-sweep",
        type=float,
        default=0.015,
        help="Maximum allowed object lateral sweep before the gripper begins lifting.",
    )
    parser.add_argument(
        "--bilateral-grasp-prelift-z-delta",
        type=float,
        default=0.02,
        help="Finger midpoint Z increase that ends the pre-lift phase for lateral sweep diagnostics.",
    )
    parser.add_argument(
        "--diagnostic-held-object-mode",
        choices=("none", "follow_gripper", "follow_after_bilateral_contact"),
        default="none",
        help=(
            "Diagnostic only. follow_gripper updates the object pose from the initial gripper/object relative "
            "transform at every replay step. follow_after_bilateral_contact waits until expected CONTACT_FOUND "
            "events happen during close, then follows the gripper. These modes validate carried-object "
            "trajectory semantics; they are not dynamic grasp proof."
        ),
    )
    parser.add_argument("--object-mass", type=float, default=0.01)
    parser.add_argument("--object-contact-offset", type=float, default=None)
    parser.add_argument("--object-rest-offset", type=float, default=None)
    parser.add_argument("--object-static-friction", type=float, default=None)
    parser.add_argument("--object-dynamic-friction", type=float, default=None)
    parser.add_argument("--object-restitution", type=float, default=None)
    parser.add_argument("--finger-static-friction", type=float, default=None)
    parser.add_argument("--finger-dynamic-friction", type=float, default=None)
    parser.add_argument("--finger-restitution", type=float, default=None)
    parser.add_argument(
        "--save-debug-stage",
        action="store_true",
        help=(
            "Export a copy of the composed runtime stage after object placement. "
            "This writes only into --output-dir and never saves over the source stage."
        ),
    )
    parser.add_argument("--support-plane-config", default=None)
    parser.add_argument(
        "--require-calibrated-table-frame",
        action="store_true",
        help="Reject diagnostic/not-calibrated support-plane configs before starting Isaac.",
    )
    parser.add_argument(
        "--allow-diagnostic-support-plane-config",
        action="store_true",
        help="Explicitly allow diagnostic support-plane configs. Do not use for final replay/contact validation.",
    )
    parser.add_argument(
        "--support-plane-mode",
        choices=("none", "object_bottom", "object_patch", "contact_patch", "fixed_box"),
        default="none",
    )
    parser.add_argument("--support-plane-center", type=float, nargs=3, default=None)
    parser.add_argument("--support-plane-size", type=float, default=DEFAULT_SUPPORT_PLANE_SIZE)
    parser.add_argument("--support-plane-size-x", type=float, default=None)
    parser.add_argument("--support-plane-size-y", type=float, default=None)
    parser.add_argument("--support-plane-thickness", type=float, default=DEFAULT_SUPPORT_PLANE_THICKNESS)
    parser.add_argument("--support-plane-clearance", type=float, default=0.0)
    parser.add_argument(
        "--support-plane-patch-margin",
        type=float,
        default=0.04,
        help=(
            "XY margin in meters for --support-plane-mode object_patch/contact_patch. This diagnostic patch "
            "is sized from the selected bbox and is not final full-workcell table validation."
        ),
    )
    parser.add_argument("--proxy-contact-offset", type=float, default=None)
    parser.add_argument("--proxy-rest-offset", type=float, default=None)
    parser.add_argument("--closure-profile", choices=("abrupt", "linear"), default="abrupt")
    parser.add_argument("--moving-fingers", choices=("both", "left", "right"), default="both")
    parser.add_argument("--hdf5-gripper-episode", default=None)
    parser.add_argument(
        "--hdf5-gripper-source",
        choices=("qpos", "action"),
        default="qpos",
        help=(
            "Signal used for HDF5 gripper targets. qpos preserves legacy kinematic replay; action is an "
            "explicit active-close command diagnostic for contact gates where observed gripper qpos is compressed "
            "or otherwise not in command space."
        ),
    )
    parser.add_argument(
        "--hdf5-replay-mode",
        choices=("gripper_only", "left_arm_and_gripper", HDF5_ARM_START_THEN_GRIPPER_ONLY_MODE),
        default="gripper_only",
    )
    parser.add_argument(
        "--hdf5-arm-hold-frame-offset",
        type=int,
        default=0,
        help=(
            "For hdf5_arm_start_then_gripper_only only: selected-window frame offset whose arm qpos is held "
            "constant while the gripper sequence is replayed. Default 0 preserves the historical start-frame "
            "contact-isolation diagnostic; use 28 with episode_18 frames 208:245 to hold frame 236."
        ),
    )
    parser.add_argument(
        "--hdf5-replay-actuation-mode",
        choices=("drive_target", "state_teleport"),
        default="drive_target",
        help=(
            "How to apply HDF5 replay targets. drive_target uses normal articulation drives. "
            "state_teleport is diagnostic-only and sets joint state before each step to isolate mapping/geometry "
            "from drive tracking error."
        ),
    )
    parser.add_argument(
        "--hdf5-replay-substep-mode",
        choices=("zero_order_hold", "linear_interpolation_diagnostic"),
        default="zero_order_hold",
        help=(
            "How each 50 Hz HDF5 replay target is applied across target-hold physics substeps. "
            "zero_order_hold is the formal replay path. linear_interpolation_diagnostic is an ablation only "
            "and is reported as DIAGNOSTIC_NOT_FORMAL_REPLAY."
        ),
    )
    parser.add_argument(
        "--hdf5-replay-target-hold-steps",
        type=int,
        default=1,
        help=(
            "Number of physics steps to hold each HDF5 replay target before advancing to the next 50 Hz target. "
            "Use values greater than 1 to test whether articulation-drive tracking is limited by target update rate. "
            "Default 1 preserves one recorded frame per physics step."
        ),
    )
    parser.add_argument(
        "--post-close-hold-steps",
        type=int,
        default=0,
        help=(
            "Extra physics steps to hold the final close target after all recorded HDF5 close frames. "
            "Default 0 preserves the raw replay window; positive values are a clamp-persistence diagnostic "
            "and are reported separately as post_close_hold, without deleting or smoothing recorded frames."
        ),
    )
    parser.add_argument(
        "--post-close-lift-source",
        choices=("none", "hdf5_continuation", "jacobian_vertical"),
        default="none",
        help=(
            "Diagnostic only. After the raw close window and optional post-close hold, append a lift phase. "
            "hdf5_continuation uses subsequent HDF5 arm qpos targets while holding the final close gripper "
            "target by default. jacobian_vertical generates a small local joint-space upward lift from the "
            "post-close hold state. Neither mode is formal close replay success."
        ),
    )
    parser.add_argument(
        "--post-close-lift-hdf5-start-frame",
        type=int,
        default=None,
        help=(
            "Optional HDF5 frame used as the anchor before the first post-close lift target. If omitted, "
            "defaults to hdf5_gripper_end_frame - 1. For loaded-arm diagnostics, set this to the held arm "
            "frame so the continuation does not begin with a hidden arm-target jump."
        ),
    )
    parser.add_argument(
        "--post-close-lift-hdf5-end-frame",
        type=int,
        default=None,
        help=(
            "Exclusive HDF5 end frame for --post-close-lift-source hdf5_continuation. The continuation starts "
            "from hdf5_gripper_end_frame - 1, so the first applied lift target corresponds to the next HDF5 frame."
        ),
    )
    parser.add_argument(
        "--post-close-lift-gripper-mode",
        choices=("hold_final_close", "hdf5_qpos"),
        default="hold_final_close",
        help=(
            "Gripper target semantics during post-close lift. hold_final_close isolates the arm lift path by "
            "holding the final clamp target; hdf5_qpos is a separate ablation."
        ),
    )
    parser.add_argument(
        "--post-close-lift-height",
        type=float,
        default=0.01,
        help="Diagnostic jacobian_vertical lift height in meters. Default 10mm.",
    )
    parser.add_argument(
        "--post-close-lift-steps",
        type=int,
        default=40,
        help="Diagnostic jacobian_vertical lift duration in target frames.",
    )
    parser.add_argument(
        "--post-close-lift-hold-steps",
        type=int,
        default=0,
        help=(
            "Extra diagnostic target frames to hold the final post-close lift target after the lift finishes. "
            "Use this to test whether the bottle remains grasped during a stationary hold."
        ),
    )
    parser.add_argument(
        "--post-close-lift-jacobian-eps",
        type=float,
        default=0.002,
        help="Finite-difference joint perturbation in radians for jacobian_vertical.",
    )
    parser.add_argument(
        "--post-close-lift-jacobian-damping",
        type=float,
        default=1e-6,
        help="Damping term for the one-row least-norm jacobian_vertical solve.",
    )
    parser.add_argument(
        "--post-close-lift-max-joint-delta",
        type=float,
        default=0.08,
        help="Maximum absolute joint delta in radians for jacobian_vertical diagnostic lift.",
    )
    parser.add_argument(
        "--diagnostic-loaded-clamp-squeeze-depth",
        type=float,
        default=0.0,
        help=(
            "Diagnostic-only total extra closing distance in meters, applied across both finger targets during "
            "post-close hold/lift. Each finger closes half this amount from the final HDF5 close target. "
            "The original HDF5 close sequence is not modified."
        ),
    )
    parser.add_argument(
        "--hdf5-replay-rate-hz",
        type=float,
        default=50.0,
        help="Nominal sampling rate of the HDF5 replay targets. ALOHA rollout data is expected to be 50 Hz.",
    )
    parser.add_argument(
        "--disable-hdf5-timing-alignment-check",
        action="store_true",
        help="Diagnostic only. Allows physics_dt * hold_steps to differ from the HDF5 replay period.",
    )
    parser.add_argument(
        "--max-post-step-controlled-tracking-error",
        type=float,
        default=None,
        help=(
            "Maximum allowed post-physics-step absolute joint tracking error for controlled DOFs. "
            "If omitted, HDF5 replay uses a conservative 0.02 rad/m default; non-HDF5 contact smoke tests skip "
            "this gate."
        ),
    )
    parser.add_argument(
        "--max-command-target-velocity",
        type=float,
        default=None,
        help=(
            "Optional diagnostic threshold for absolute 50 Hz replay target velocity in rad/s. "
            "If omitted, command smoothness is reported but does not fail the formal replay gate."
        ),
    )
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    parser.add_argument("--hdf5-gripper-start-frame", type=int, default=None)
    parser.add_argument("--hdf5-gripper-end-frame", type=int, default=None)
    parser.add_argument("--hdf5-gripper-max-frames", type=int, default=None)
    parser.add_argument(
        "--loaded-gripper-qpos-calibration-mode",
        choices=("none", "affine_open_to_contact"),
        default="none",
        help=(
            "Diagnostic replay interpretation only. affine_open_to_contact maps an observed open qpos anchor "
            "and a loaded soft-bottle contact qpos anchor into Isaac finger targets. It preserves the raw HDF5 "
            "sequence and must not be treated as a generic ALOHA gripper calibration."
        ),
    )
    parser.add_argument(
        "--loaded-gripper-open-qpos",
        type=float,
        default=None,
        help="Observed qpos anchor for the open gripper state, usually max/p95 over a stable open window.",
    )
    parser.add_argument(
        "--loaded-gripper-contact-qpos",
        type=float,
        default=None,
        help="Observed qpos anchor for the loaded soft-bottle contact plateau.",
    )
    parser.add_argument(
        "--loaded-gripper-effective-contact-width",
        type=float,
        default=None,
        help="Effective soft-bottle contact width in stage units used for loaded-qpos contact interpretation.",
    )
    parser.add_argument(
        "--loaded-gripper-open-standard",
        type=float,
        default=1.0,
        help="Normalized standard value assigned to the open qpos anchor. Default 1.0 means fully open.",
    )
    parser.add_argument(
        "--loaded-gripper-qpos-calibration-source",
        default="",
        help="Short provenance string for the loaded gripper calibration, stored in the output report.",
    )
    parser.add_argument("--trace-contact-pairs", action="store_true")
    parser.add_argument(
        "--fail-on-non-target-object-contact",
        action="store_true",
        help=(
            "When contact tracing is enabled, fail if the object contacts anything other than the expected "
            "finger target roots. Use for final contact-quality gates, not for broad diagnostic smoke tests."
        ),
    )
    parser.add_argument(
        "--allowed-non-target-object-contact-category",
        action="append",
        default=[],
        choices=("diagnostic_support", "workcell_or_environment", "same_side_robot_non_target", "other_side_robot", "unknown"),
        help=(
            "When --fail-on-non-target-object-contact is enabled, allow this non-target object-contact category. "
            "Use this to distinguish intentional table/workcell support from robot-body or unknown collisions."
        ),
    )
    parser.add_argument(
        "--workcell-contact-policy",
        default=None,
        help=(
            "Optional YAML path-prefix policy for non-target object contacts. Use it to split broad "
            "workcell_or_environment contacts into allowed tabletop/pipe contacts and denied frame/rail contacts."
        ),
    )
    parser.add_argument(
        "--require-active-target-contact",
        action="store_true",
        help=(
            "When contact tracing is enabled, fail unless a target finger/object CONTACT_FOUND event first appears "
            "during the close phase. Use for active-grasp claims, not for already-contacting replay references."
        ),
    )
    parser.add_argument(
        "--already-in-contact-setup",
        action="store_true",
        help=(
            "Mark this run as an already-contacting contact-candidate setup. This documents replay references like "
            "Phase97 and intentionally skips the active-contact phase gate."
        ),
    )
    parser.add_argument(
        "--trace-disable-usd-updates",
        action="store_true",
        help="Match Isaac asset-validator style contact probing. Off by default because this script needs live USD bbox readback.",
    )
    parser.add_argument(
        "--diagnostic-force-target-overlap",
        choices=("none", "nearest", "lower", "upper", "left_finger", "right_finger"),
        default="none",
        help=(
            "Diagnostic positive control only: shift the object contact proxy into one finger along the live "
            "closing axis to prove whether the selected colliders can generate a contact report. Any run using "
            "this option is explicitly non-formal and cannot pass Gate2."
        ),
    )
    parser.add_argument(
        "--diagnostic-force-target-overlap-m",
        type=float,
        default=0.001,
        help="Overlap depth for --diagnostic-force-target-overlap, in stage meters.",
    )
    parser.add_argument(
        "--disable-workcell-environment-collisions-for-diagnostic-replay",
        action="store_true",
        help=(
            "Diagnostic only: disable uncalibrated /scene/worldBody and /World/Table collisions so HDF5 arm replay "
            "can be isolated from table/base calibration errors. Do not use for final contact validation."
        ),
    )
    parser.add_argument("--min-contact-motion", type=float, default=1e-5)
    parser.add_argument(
        "--min-object-lift",
        type=float,
        default=0.0,
        help=(
            "Minimum required increase in object center Z from the start of the close phase to the final frame. "
            "Use this for dynamic grasp/lift gates; default 0 preserves older contact-only gates."
        ),
    )
    parser.add_argument(
        "--enforce-object-width-finger-stop",
        action="store_true",
        help=(
            "Stop commanding additional finger closure once the measured finger center gap is at the object body "
            "width plus --object-clearance. This records a physical target guard instead of asking the gripper "
            "to close through the bottle."
        ),
    )
    parser.add_argument("--max-object-displacement", type=float, default=0.25)
    parser.add_argument(
        "--max-finger-surface-gap-meters",
        type=float,
        default=DEFAULT_MAX_FINGER_SURFACE_GAP_METERS,
        help="Reject passive-contact setup if the open fingertip surface gap exceeds this physical distance.",
    )
    parser.add_argument(
        "--max-generated-object-side-meters",
        type=float,
        default=DEFAULT_MAX_GENERATED_OBJECT_SIDE_METERS,
        help="Reject passive-contact setup if the generated diagnostic object side exceeds this physical size.",
    )
    args = parser.parse_args()
    hdf5_target_hold_steps = int(args.hdf5_replay_target_hold_steps)
    post_close_hold_steps = int(args.post_close_hold_steps)
    if post_close_hold_steps < 0:
        raise ValueError(f"post_close_hold_steps must be non-negative, got {post_close_hold_steps}")
    post_close_lift_hold_steps = int(args.post_close_lift_hold_steps)
    if post_close_lift_hold_steps < 0:
        raise ValueError(f"post_close_lift_hold_steps must be non-negative, got {post_close_lift_hold_steps}")
    if float(args.diagnostic_loaded_clamp_squeeze_depth) < 0.0:
        raise ValueError(
            "--diagnostic-loaded-clamp-squeeze-depth must be non-negative, "
            f"got {args.diagnostic_loaded_clamp_squeeze_depth}"
        )
    if float(args.diagnostic_loaded_clamp_squeeze_depth) > 0.0 and args.moving_fingers != "both":
        raise ValueError("--diagnostic-loaded-clamp-squeeze-depth requires --moving-fingers both")
    if args.post_close_lift_source == "hdf5_continuation":
        if args.post_close_lift_hdf5_end_frame is None:
            raise ValueError("--post-close-lift-hdf5-end-frame is required for post-close lift diagnostics")
        if args.hdf5_gripper_end_frame is None:
            raise ValueError("--hdf5-gripper-end-frame is required for post-close lift diagnostics")
        if int(args.post_close_lift_hdf5_end_frame) <= int(args.hdf5_gripper_end_frame):
            raise ValueError(
                "--post-close-lift-hdf5-end-frame must be greater than --hdf5-gripper-end-frame "
                f"({args.post_close_lift_hdf5_end_frame} <= {args.hdf5_gripper_end_frame})"
            )
    elif args.post_close_lift_source == "jacobian_vertical":
        if args.hdf5_gripper_end_frame is None:
            raise ValueError("--hdf5-gripper-end-frame is required for jacobian_vertical lift diagnostics")
        if float(args.post_close_lift_height) <= 0:
            raise ValueError("--post-close-lift-height must be positive for jacobian_vertical")
        if int(args.post_close_lift_steps) <= 0:
            raise ValueError("--post-close-lift-steps must be positive for jacobian_vertical")
        if float(args.post_close_lift_jacobian_eps) <= 0:
            raise ValueError("--post-close-lift-jacobian-eps must be positive for jacobian_vertical")
        if float(args.post_close_lift_jacobian_damping) < 0:
            raise ValueError("--post-close-lift-jacobian-damping must be non-negative for jacobian_vertical")
        if float(args.post_close_lift_max_joint_delta) <= 0:
            raise ValueError("--post-close-lift-max-joint-delta must be positive for jacobian_vertical")
    effective_control_dt = float(args.physics_dt) * float(hdf5_target_hold_steps)
    requested_hdf5_dt = 1.0 / float(args.hdf5_replay_rate_hz)
    hdf5_timing_alignment = {
        "physics_dt": float(args.physics_dt),
        "physics_rate_hz": 1.0 / float(args.physics_dt) if float(args.physics_dt) > 0 else None,
        "target_hold_steps": hdf5_target_hold_steps,
        "effective_control_dt": effective_control_dt,
        "effective_control_rate_hz": 1.0 / effective_control_dt if effective_control_dt > 0 else None,
        "hdf5_replay_rate_hz": float(args.hdf5_replay_rate_hz),
        "expected_hdf5_dt": requested_hdf5_dt,
        "timing_error_s": effective_control_dt - requested_hdf5_dt,
        "pass": True,
        "status": "SKIPPED_NO_HDF5_REPLAY",
    }
    try:
        if args.require_active_target_contact and args.already_in_contact_setup:
            raise ValueError("--require-active-target-contact cannot combine with --already-in-contact-setup")
        if args.hdf5_gripper_episode:
            aligned = bool(abs(effective_control_dt - requested_hdf5_dt) <= 1e-9)
            hdf5_timing_alignment.update(
                {
                    "pass": aligned or bool(args.disable_hdf5_timing_alignment_check),
                    "status": "PASS_HDF5_TIMING_ALIGNED"
                    if aligned
                    else "SKIPPED_HDF5_TIMING_ALIGNMENT_CHECK"
                    if args.disable_hdf5_timing_alignment_check
                    else "FAIL_HDF5_TIMING_MISMATCH",
                }
            )
            if not hdf5_timing_alignment["pass"]:
                raise ValueError(
                    "HDF5 replay timing mismatch: physics_dt * hdf5_replay_target_hold_steps = "
                    f"{effective_control_dt:.9f}s, expected {requested_hdf5_dt:.9f}s for "
                    f"{args.hdf5_replay_rate_hz:g} Hz data. Use --physics-dt 0.004 "
                    "--hdf5-replay-target-hold-steps 5 for 250 Hz physics with 50 Hz ALOHA replay, "
                    "or explicitly pass --disable-hdf5-timing-alignment-check for a diagnostic ablation."
                )
        support_options = _resolve_support_plane_options(args)
        workcell_contact_policy = _load_workcell_contact_policy(args.workcell_contact_policy)
        _guard_support_plane_calibration_mode(args)
        table_frame_audit = _audit_required_table_frame(args)
        contact_stage_namespace = _guard_final_contact_stage_namespace(args)
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = Path(args.output_dir)
    json_path = output_dir / "gripper_passive_contact_metrics.json"
    csv_path = output_dir / "gripper_passive_contact_timeseries.csv"
    md_path = output_dir / "gripper_passive_contact_metrics.md"
    payload: dict[str, Any] = {
        "status": "STARTED",
        "overall_pass": False,
        "real_robot_touched": False,
        "stage_saved": False,
        "inputs": {
            "stage_usd": _rel(args.stage_usd),
            "stage_units_in_meters": args.stage_units_in_meters,
            "side": args.side,
            "contact_proxy_profile": args.contact_proxy_profile,
            "control_mode": "opposed_fingers",
            "open_offset": args.open_offset,
            "close_offset": args.close_offset,
            "right_finger_close_sign": args.right_finger_close_sign,
            "settle_steps": args.settle_steps,
            "close_steps": args.close_steps,
            "physics_dt": args.physics_dt,
            "physics_rate_hz": hdf5_timing_alignment["physics_rate_hz"],
            "stage_time_codes_per_second_requested": args.stage_time_codes_per_second,
            "drive_profile_name": args.drive_profile_name,
            "drive_profile_provenance": args.drive_profile_provenance,
            "gravity": args.gravity,
            "object_fill_fraction": args.object_fill_fraction,
            "object_placement": args.object_placement,
            "object_clearance": args.object_clearance,
            "enforce_object_width_finger_stop": bool(args.enforce_object_width_finger_stop),
            "object_width_stop_predictive_margin": args.object_width_stop_predictive_margin,
            "object_creation": args.object_creation,
            "object_rigid_body": not args.disable_object_rigid_body,
            "object_shape": args.object_shape,
            "closing_axis_gap_solver_basis": args.closing_axis_gap_solver_basis,
            "object_axis": args.object_axis,
            "object_center_offset": args.object_center_offset,
            "object_length_multiplier": args.object_length_multiplier,
            "object_usd": _rel(args.object_usd),
            "object_usd_prim_path": args.object_usd_prim_path,
            "object_tabletop_reference_path": args.object_tabletop_reference_path,
            "object_tabletop_top_z": args.object_tabletop_top_z,
            "object_tabletop_clearance": args.object_tabletop_clearance,
            "object_grasp_yaml": _rel(args.object_grasp_yaml),
            "object_grasp_name": args.object_grasp_name,
            "object_gripper_frame": args.object_gripper_frame,
            "object_rear_quarter_fraction": args.object_rear_quarter_fraction,
            "object_rear_quarter_tolerance": args.object_rear_quarter_tolerance,
            "max_closing_long_axis_dot": args.max_closing_long_axis_dot,
            "bilateral_grasp_min_contact_steps": args.bilateral_grasp_min_contact_steps,
            "bilateral_grasp_min_nonzero_impulse_steps": args.bilateral_grasp_min_nonzero_impulse_steps,
            "bilateral_grasp_max_impulse_ratio": args.bilateral_grasp_max_impulse_ratio,
            "bilateral_grasp_max_prelift_lateral_sweep": args.bilateral_grasp_max_prelift_lateral_sweep,
            "bilateral_grasp_prelift_z_delta": args.bilateral_grasp_prelift_z_delta,
            "save_debug_stage": bool(args.save_debug_stage),
            "diagnostic_held_object_mode": args.diagnostic_held_object_mode,
            "object_contact_offset": args.object_contact_offset,
            "object_rest_offset": args.object_rest_offset,
            "support_plane_config": support_options["config"],
            "support_plane_config_provenance": support_options["config_provenance"],
            "support_plane_table_frame": support_options["table_frame"],
            "require_calibrated_table_frame": args.require_calibrated_table_frame,
            "allow_diagnostic_support_plane_config": args.allow_diagnostic_support_plane_config,
            "table_frame_audit": table_frame_audit,
            "contact_stage_namespace": contact_stage_namespace,
            "support_plane_mode": support_options["mode"],
            "support_plane_center": support_options["center"],
            "support_plane_size": args.support_plane_size,
            "support_plane_size_x": support_options["size_x"],
            "support_plane_size_y": support_options["size_y"],
            "support_plane_thickness": support_options["thickness"],
            "support_plane_clearance": args.support_plane_clearance,
            "proxy_contact_offset": args.proxy_contact_offset,
            "proxy_rest_offset": args.proxy_rest_offset,
            "closure_profile": args.closure_profile,
            "moving_fingers": args.moving_fingers,
            "hdf5_gripper_episode": _rel(args.hdf5_gripper_episode) if args.hdf5_gripper_episode else None,
            "hdf5_replay_mode": args.hdf5_replay_mode,
            "hdf5_replay_actuation_mode": args.hdf5_replay_actuation_mode,
            "hdf5_replay_substep_mode": args.hdf5_replay_substep_mode,
            "formal_replay": bool(args.hdf5_replay_substep_mode == "zero_order_hold"),
            "diagnostic_interpolated_replay": bool(args.hdf5_replay_substep_mode == "linear_interpolation_diagnostic"),
            "replay_semantics_status": (
                "FORMAL_ZERO_ORDER_HOLD_REPLAY"
                if args.hdf5_replay_substep_mode == "zero_order_hold"
                else "DIAGNOSTIC_NOT_FORMAL_REPLAY"
            ),
            "hdf5_replay_rate_hz": args.hdf5_replay_rate_hz,
            "max_post_step_controlled_tracking_error": args.max_post_step_controlled_tracking_error,
            "mapping": _rel(args.mapping),
            "hdf5_gripper_start_frame": args.hdf5_gripper_start_frame,
            "hdf5_gripper_end_frame": args.hdf5_gripper_end_frame,
            "hdf5_gripper_max_frames": args.hdf5_gripper_max_frames,
            "diagnostic_loaded_clamp_squeeze_depth": args.diagnostic_loaded_clamp_squeeze_depth,
            "diagnostic_loaded_clamp_squeeze_semantics": (
                "post_close_hold_and_lift_only_total_gap_reduction_m; "
                "each_finger_closes_half; formal_hdf5_close_sequence_unchanged"
            ),
            "hdf5_effective_control_dt": hdf5_timing_alignment["effective_control_dt"],
            "hdf5_effective_control_rate_hz": hdf5_timing_alignment["effective_control_rate_hz"],
            "hdf5_timing_alignment": hdf5_timing_alignment,
            "max_command_target_velocity": args.max_command_target_velocity,
            "trace_contact_pairs": args.trace_contact_pairs,
            "fail_on_non_target_object_contact": args.fail_on_non_target_object_contact,
            "allowed_non_target_object_contact_categories": args.allowed_non_target_object_contact_category,
            "workcell_contact_policy": workcell_contact_policy,
            "require_active_target_contact": args.require_active_target_contact,
            "already_in_contact_setup": args.already_in_contact_setup,
            "trace_disable_usd_updates": args.trace_disable_usd_updates,
            "disable_workcell_environment_collisions_for_diagnostic_replay": (
                args.disable_workcell_environment_collisions_for_diagnostic_replay
            ),
            "reset_after_object_creation": False,
            "min_contact_motion": args.min_contact_motion,
            "max_object_displacement": args.max_object_displacement,
            "max_finger_surface_gap_meters": args.max_finger_surface_gap_meters,
            "max_generated_object_side_meters": args.max_generated_object_side_meters,
        },
        "outputs": {"json": _rel(json_path), "csv": _rel(csv_path), "markdown": _rel(md_path)},
    }
    _write_json(json_path, payload)

    try:
        from isaacsim import SimulationApp

        app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
        app_config["fast_shutdown"] = False
        _app = SimulationApp(app_config)
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        import isaacsim.core.utils.stage as stage_utils
        import omni.usd

        stage = _open_stage_and_wait(stage_utils, omni.usd, _app, str(Path(args.stage_usd).resolve()))
        World.clear_instance()
        world = World(stage_units_in_meters=args.stage_units_in_meters, backend="numpy", device="cpu")
        world.set_simulation_dt(physics_dt=args.physics_dt, rendering_dt=args.physics_dt)
        stage = omni.usd.get_context().get_stage() or stage
        stage_time_codes_per_second_before = float(stage.GetTimeCodesPerSecond())
        stage_frames_per_second_before = float(stage.GetFramesPerSecond())
        if args.stage_time_codes_per_second is not None:
            stage.SetTimeCodesPerSecond(float(args.stage_time_codes_per_second))
            stage.SetFramesPerSecond(float(args.stage_time_codes_per_second))
        stage_time_codes_per_second_effective = float(stage.GetTimeCodesPerSecond())
        stage_frames_per_second_effective = float(stage.GetFramesPerSecond())
        payload["inputs"]["stage_time_codes_per_second_before"] = stage_time_codes_per_second_before
        payload["inputs"]["stage_frames_per_second_before"] = stage_frames_per_second_before
        payload["inputs"]["stage_time_codes_per_second"] = stage_time_codes_per_second_effective
        payload["inputs"]["stage_frames_per_second"] = stage_frames_per_second_effective
        _write_json(json_path, payload)
        disabled_workcell_environment_collision_paths: list[str] = []
        if args.disable_workcell_environment_collisions_for_diagnostic_replay:
            disabled_workcell_environment_collision_paths = _disable_workcell_environment_collisions_for_diagnostic_replay(stage)
            payload["inputs"]["disabled_workcell_environment_collision_count"] = len(
                disabled_workcell_environment_collision_paths
            )
            payload["inputs"]["disabled_workcell_environment_collision_sample"] = disabled_workcell_environment_collision_paths[
                :20
            ]
            _write_json(json_path, payload)
        paths_by_side = _finger_proxy_paths_for_args(args)
        paths = paths_by_side[args.side]
        contact_targets = resolve_contact_target_paths(args.contact_proxy_profile)[args.side]
        finger_dof_names = finger_dof_names_for_side(args.contact_proxy_profile, args.side)
        finger_qpos_limits = finger_qpos_limits_for_side(args.contact_proxy_profile, args.side)
        payload["inputs"]["finger_qpos_limits"] = {
            "left_close": float(finger_qpos_limits.left_close),
            "left_open": float(finger_qpos_limits.left_open),
            "right_close": float(finger_qpos_limits.right_close),
            "right_open": float(finger_qpos_limits.right_open),
        }
        _write_json(json_path, payload)
        trace_state = None
        if args.trace_contact_pairs:
            # ContactReportAPI authoring changes physics schemas. Do this before
            # SingleArticulation/world.reset creates tensor views, or PhysX can
            # invalidate the articulation backend during the first state write.
            trace_state = _begin_contact_pair_trace(stage, disable_usd_updates=args.trace_disable_usd_updates)
        art = world.scene.add(SingleArticulation(prim_path=paths["articulation"], name=f"{args.side}_vx300s"))
        world.reset()
        _apply_gravity(world, args.gravity)
        _apply_arm_gains(art, args.arm_kp, args.arm_kd)
        _apply_named_dof_gains(
            art,
            [finger_dof_names["left_finger"], finger_dof_names["right_finger"]],
            args.finger_kp,
            args.finger_kd,
        )

        hdf5_target_sequence: list[np.ndarray] | None = None
        hdf5_gripper_summary: dict[str, Any] | None = None
        gripper_qpos_calibration = DEFAULT_GRIPPER_QPOS_CALIBRATION
        if args.loaded_gripper_qpos_calibration_mode == "affine_open_to_contact":
            missing = [
                name
                for name, value in (
                    ("--loaded-gripper-open-qpos", args.loaded_gripper_open_qpos),
                    ("--loaded-gripper-contact-qpos", args.loaded_gripper_contact_qpos),
                    ("--loaded-gripper-effective-contact-width", args.loaded_gripper_effective_contact_width),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "--loaded-gripper-qpos-calibration-mode affine_open_to_contact requires "
                    + ", ".join(missing)
                )
            gripper_qpos_calibration = gripper_qpos_calibration_from_loaded_contact(
                raw_open_value=float(args.loaded_gripper_open_qpos),
                raw_contact_value=float(args.loaded_gripper_contact_qpos),
                effective_contact_width=float(args.loaded_gripper_effective_contact_width),
                limits=finger_qpos_limits,
                standard_open_value=float(args.loaded_gripper_open_standard),
                source=(
                    args.loaded_gripper_qpos_calibration_source
                    or "loaded_gripper_qpos_affine_open_to_contact_cli"
                ),
            )
        if args.hdf5_gripper_episode:
            qpos = _load_hdf5_qpos(
                args.hdf5_gripper_episode,
                start=args.hdf5_gripper_start_frame,
                end=args.hdf5_gripper_end_frame,
                max_frames=args.hdf5_gripper_max_frames,
            )
            gripper_sequence = None
            gripper_source = "observations/qpos"
            if args.hdf5_gripper_source == "action":
                gripper_sequence = _load_hdf5_action(
                    args.hdf5_gripper_episode,
                    start=args.hdf5_gripper_start_frame,
                    end=args.hdf5_gripper_end_frame,
                    max_frames=args.hdf5_gripper_max_frames,
                )
                if gripper_sequence.shape[0] != qpos.shape[0]:
                    raise ValueError(
                        "Loaded qpos and action frame counts differ after slicing: "
                        f"qpos={qpos.shape}, action={gripper_sequence.shape}"
                    )
                gripper_source = "action"
            mapping = load_mapping(args.mapping) if _replay_mode_controls_arm(args.hdf5_replay_mode) else None
            hdf5_target_sequence, hdf5_gripper_summary = _targets_from_hdf5_qpos(
                art=art,
                side=args.side,
                qpos=qpos,
                gripper_sequence=gripper_sequence,
                gripper_source=gripper_source,
                mapping=mapping,
                replay_mode=args.hdf5_replay_mode,
                finger_dof_names=finger_dof_names,
                finger_qpos_limits=finger_qpos_limits,
                gripper_qpos_calibration=gripper_qpos_calibration,
                arm_hold_frame_offset=int(args.hdf5_arm_hold_frame_offset),
            )
            open_target = hdf5_target_sequence[0]
            open_values = hdf5_gripper_summary["first_target_values"]
            payload["inputs"]["control_mode"] = (
                f"hdf5_{args.hdf5_replay_mode}_{args.hdf5_gripper_source}_gripper_replay"
            )
            payload["inputs"]["hdf5_gripper_summary"] = hdf5_gripper_summary
        else:
            open_target, open_values = _finger_targets(art, args.open_offset, args.limit_margin, finger_dof_names)
        _set_full_state(art, open_target)
        _set_full_target(art, open_target)
        pre_object_update_steps = max(args.settle_steps, 1)
        payload["inputs"]["pre_object_update_steps"] = pre_object_update_steps
        _set_finger_target_and_step(world, art, open_target, pre_object_update_steps)

        left_box = _bbox_row(stage, paths["left_finger"])
        right_box = _bbox_row(stage, paths["right_finger"])
        replay_start_left_box = dict(left_box)
        replay_start_right_box = dict(right_box)
        placement_basis: dict[str, Any] = {
            "target": "open_target",
            "description": "Object placement was computed from the replay start/open fingertip bboxes.",
        }
        hdf5_close_rear_quarter_modes = {
            "hdf5_close_finger_rear_quarter",
            "hdf5_close_finger_rear_quarter_tabletop",
        }
        hdf5_open_rear_quarter_modes = {
            "hdf5_open_finger_rear_quarter",
            "hdf5_open_finger_rear_quarter_tabletop",
        }
        hdf5_tabletop_rear_quarter_modes = {
            "hdf5_open_finger_rear_quarter_tabletop",
            "hdf5_close_finger_rear_quarter_tabletop",
        }
        rear_quarter_modes = {"finger_rear_quarter", *hdf5_open_rear_quarter_modes, *hdf5_close_rear_quarter_modes}
        if args.object_placement in hdf5_open_rear_quarter_modes:
            if hdf5_target_sequence is None:
                raise ValueError(
                    f"--object-placement {args.object_placement} requires --hdf5-gripper-episode"
                )
            placement_basis = {
                "target": "hdf5_open_target",
                "description": (
                    "Object placement was computed from the first HDF5 replay target/open fingertip bboxes. "
                    "This is the active tabletop grasp placement: a PASS requires target contact to first "
                    "appear during the close phase."
                ),
                "target_values": {
                    "left_finger": float(open_target[list(art.dof_names).index(finger_dof_names["left_finger"])]),
                    "right_finger": float(open_target[list(art.dof_names).index(finger_dof_names["right_finger"])]),
                },
            }
        tabletop_object_placement_row: dict[str, Any] | None = None
        if args.object_placement in hdf5_close_rear_quarter_modes:
            if hdf5_target_sequence is None:
                raise ValueError(
                    f"--object-placement {args.object_placement} requires --hdf5-gripper-episode"
                )
            close_placement_target = hdf5_target_sequence[-1]
            _set_full_state(art, close_placement_target)
            _set_full_target(art, close_placement_target)
            _set_finger_target_and_step(world, art, close_placement_target, pre_object_update_steps)
            left_box = _bbox_row(stage, paths["left_finger"])
            right_box = _bbox_row(stage, paths["right_finger"])
            placement_basis = {
                "target": "hdf5_close_target",
                "description": (
                    "Object placement was computed from the final HDF5 close target, then the articulation was "
                    "restored to the replay start target before object creation. This is the active-grasp probe "
                    "placement: a PASS requires target contact to first appear during the close phase."
                ),
                "target_values": {
                    "left_finger": float(close_placement_target[list(art.dof_names).index(finger_dof_names["left_finger"])]),
                    "right_finger": float(close_placement_target[list(art.dof_names).index(finger_dof_names["right_finger"])]),
                },
            }
            _set_full_state(art, open_target)
            _set_full_target(art, open_target)
            _set_finger_target_and_step(world, art, open_target, pre_object_update_steps)
            replay_start_left_box = _bbox_row(stage, paths["left_finger"])
            replay_start_right_box = _bbox_row(stage, paths["right_finger"])
        placement_left_box = dict(left_box)
        placement_right_box = dict(right_box)
        cross_side_proxy_overlap = _cross_side_proxy_overlap_summary(stage, paths_by_side, args.side)
        gap = _gap_metrics(left_box, right_box)
        if not gap.get("bbox_pair_valid"):
            raise RuntimeError("Finger proxy bbox pair is invalid; cannot place contact object.")
        axis_name = str(gap["dominant_axis"])
        axis = {"x": 0, "y": 1, "z": 2}[axis_name]
        center = (
            np.asarray(left_box["center"], dtype=np.float64) + np.asarray(right_box["center"], dtype=np.float64)
        ) * 0.5
        surface_gap = _surface_gap(left_box, right_box, axis)
        if args.object_side_length is not None and args.object_effective_contact_width is not None:
            raise ValueError("--object-side-length and --object-effective-contact-width are mutually exclusive")
        side_length = max(surface_gap * args.object_fill_fraction, 1e-4)
        side_length_source = "finger_surface_gap_times_fill_fraction"
        soft_contact_model: dict[str, Any] = {
            "enabled": False,
            "visual_external_diameter_m": float(BOTTLE_RADIUS_M * 2.0),
        }
        if args.object_side_length is not None:
            side_length = max(float(args.object_side_length), 1e-4)
            side_length_source = "explicit_object_side_length"
        if args.object_effective_contact_width is not None:
            side_length = max(float(args.object_effective_contact_width), 1e-4)
            side_length_source = "soft_bottle_effective_contact_width"
            soft_contact_model.update(
                {
                    "enabled": True,
                    "effective_contact_width_m": float(side_length * args.stage_units_in_meters),
                    "effective_contact_width_stage_units": float(side_length),
                    "source": str(args.object_effective_contact_width_source),
                    "notes": (
                        "The Bottle500 visual mesh remains true-size. The contact proxy is narrower to model "
                        "a soft mineral-water bottle compressed between ALOHA gripper fingers."
                    ),
                }
            )
        geometry_sanity = _passive_contact_geometry_sanity(
            finger_surface_gap_stage_units=surface_gap,
            object_side_length_stage_units=side_length,
            stage_units_in_meters=args.stage_units_in_meters,
            max_finger_surface_gap_meters=args.max_finger_surface_gap_meters,
            max_generated_object_side_meters=args.max_generated_object_side_meters,
        )
        object_placement_row: dict[str, Any] = {
            "mode": args.object_placement,
            "axis": axis_name,
            "clearance": args.object_clearance,
            "base_center": center.tolist(),
            "placement_basis": placement_basis,
            "side_length_source": side_length_source,
            "soft_contact_model": soft_contact_model,
        }
        tabletop_reset_calibration: dict[str, Any] | None = None
        if args.derive_tabletop_top_z_from_open_finger_height:
            if args.object_placement not in hdf5_tabletop_rear_quarter_modes:
                raise ValueError(
                    "--derive-tabletop-top-z-from-open-finger-height requires an hdf5_*_rear_quarter_tabletop "
                    "object placement mode"
                )
            if args.object_tabletop_top_z is not None:
                raise ValueError(
                    "--derive-tabletop-top-z-from-open-finger-height and --object-tabletop-top-z are mutually exclusive"
                )
            derived_top = _derived_tabletop_top_z_from_open_finger(
                open_left_box=replay_start_left_box,
                open_right_box=replay_start_right_box,
                object_contact_radius=float(side_length) * 0.5 * float(args.stage_units_in_meters),
                clearance=float(args.object_tabletop_clearance),
            )
            tabletop_reset_calibration = {
                "mode": "derived_tabletop_top_z_from_open_finger_height",
                "derived_tabletop_top_z": derived_top,
                "tabletop_shift": None,
                "formal_full_scene_validation": False,
                "notes": (
                    "This calibration makes a fixed-reset Gate2 replay geometrically testable when the "
                    "current table USD height is not yet measured in the robot base frame. It must remain "
                    "visible in reports and should be replaced by measured table/base calibration later."
                ),
            }
            if not derived_top["pass"]:
                raise RuntimeError(f"derived tabletop top z failed: {derived_top['status']}")
            tabletop_shift = _calibrate_tabletop_top_z(
                stage=stage,
                table_path=args.object_tabletop_reference_path,
                target_top_z=float(derived_top["derived_table_top_z_m"]),
            )
            tabletop_reset_calibration["tabletop_shift"] = tabletop_shift
            if not tabletop_shift["pass"]:
                raise RuntimeError(f"tabletop calibration failed: {tabletop_shift['status']}")
            object_placement_row["tabletop_reset_calibration"] = tabletop_reset_calibration
        object_center_offset = _parse_vec3(args.object_center_offset, name="--object-center-offset")
        grasp_band_local_axis_offset = 0.0
        object_axis_source_row: dict[str, Any] = {
            "source": args.object_axis_source,
            "object_axis_unit_world": _axis_unit_vector(args.object_axis).tolist(),
            "provenance": "STATIC_WORLD_AXIS",
        }
        if not geometry_sanity["pass"]:
            if trace_state is not None:
                _finish_contact_pair_trace(stage, trace_state)
            payload.update(
                {
                    "status": "FAILED_GATE",
                    "overall_pass": False,
                    "contact_trace_status": geometry_sanity["status"],
                    "open_target_values": open_values,
                    "hdf5_gripper_summary": hdf5_gripper_summary,
                    "tracking_summary": None,
                    "finger_gap_axis": axis_name,
                    "finger_surface_gap_open": surface_gap,
                    "finger_surface_gap_open_meters": geometry_sanity["finger_surface_gap_open_meters"],
                    "left_finger_placement_box": placement_left_box,
                    "right_finger_placement_box": placement_right_box,
                    "cross_side_proxy_overlap": cross_side_proxy_overlap,
                    "object_placement": object_placement_row,
                    "object_side_length_stage_units": side_length,
                    "object_side_length_meters": geometry_sanity["object_side_length_meters"],
                    "soft_bottle_contact_model": soft_contact_model,
                    "contact_setup_geometry_sanity": geometry_sanity,
                    "contact_setup_geometry_sanity_status": geometry_sanity["status"],
                    "contact_pair_trace_enabled": bool(args.trace_contact_pairs),
                    "contact_trace_disable_usd_updates": bool(args.trace_disable_usd_updates),
                    "csv": _rel(csv_path),
                    "markdown": _rel(md_path),
                    "next_gate": "fix_finger_proxy_bbox_frame_before_contact_validation",
                }
            )
            _write_csv(csv_path, [])
            _write_json(json_path, payload)
            _write_markdown(md_path, _json_safe(payload))
            print(
                json.dumps(
                    {"status": payload["status"], "json": _rel(json_path), "markdown": _rel(md_path)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(3)
        if args.object_placement == "moving_finger_surface" and args.moving_fingers != "both":
            moving_box = left_box if args.moving_fingers == "left" else right_box
            other_box = right_box if args.moving_fingers == "left" else left_box
            moving_center = np.asarray(moving_box["center"], dtype=np.float64)
            other_center = np.asarray(other_box["center"], dtype=np.float64)
            direction = 1.0 if other_center[axis] >= moving_center[axis] else -1.0
            moving_surface = float(moving_box["max"][axis] if direction > 0 else moving_box["min"][axis])
            center = np.asarray(moving_box["center"], dtype=np.float64)
            center[axis] = moving_surface + direction * (side_length * 0.5 + args.object_clearance)
            object_placement_row.update(
                {
                    "moving_finger": args.moving_fingers,
                    "other_finger": "right" if args.moving_fingers == "left" else "left",
                    "direction_toward_other_finger": direction,
                    "moving_surface": moving_surface,
                    "placed_center": center.tolist(),
                }
            )
        elif args.object_placement == "moving_finger_surface":
            object_placement_row["warning"] = (
                "moving_finger_surface requires --moving-fingers left or right; used gap_center."
            )
        elif args.object_placement in rear_quarter_modes:
            if args.object_axis_source == "open_finger_horizontal_perpendicular":
                object_axis_source_row = _derive_open_finger_horizontal_perpendicular_axis(
                    left_box=placement_left_box,
                    right_box=placement_right_box,
                    preferred_axis=args.object_axis,
                )
                object_axis_unit = np.asarray(object_axis_source_row["object_axis_unit_world"], dtype=np.float64)
            else:
                object_axis_unit = _axis_unit_vector(args.object_axis)
            object_axis_length = _nominal_object_axis_length_stage_units(args, side_length)
            auto_offset = object_axis_unit * (
                (0.5 - float(args.object_rear_quarter_fraction)) * float(object_axis_length)
            )
            grasp_band_local_axis_offset = -(
                (0.5 - float(args.object_rear_quarter_fraction)) * float(object_axis_length)
            )
            center = np.asarray(center, dtype=np.float64) + auto_offset
            object_placement_row.update(
                {
                    "placement_semantics": "finger_gap_center_on_bottle_rear_quarter",
                    "object_axis": args.object_axis,
                    "object_axis_source": object_axis_source_row,
                    "object_axis_unit_world": object_axis_unit.tolist(),
                    "nominal_object_axis_length_stage_units": float(object_axis_length),
                    "rear_fraction_target": float(args.object_rear_quarter_fraction),
                    "rear_fraction_tolerance": float(args.object_rear_quarter_tolerance),
                    "auto_center_offset_world": auto_offset.tolist(),
                    "auto_placed_center": np.asarray(center, dtype=np.float64).tolist(),
                    "grasp_band_local_axis_offset": float(grasp_band_local_axis_offset),
                }
            )
        grasp_placement: dict[str, Any] | None = None
        bottle_grasp_semantics_gate: dict[str, Any] = {
            "pass": True,
            "status": "SKIPPED_NOT_GRASP_YAML_BOTTLE_PLACEMENT",
        }
        if args.object_placement == "grasp_yaml":
            from pxr import UsdGeom

            grasp_info = _load_grasp_transform(args.object_grasp_yaml, args.object_grasp_name)
            semantics = evaluate_grasp_file(args.object_grasp_yaml, selected_grasp=args.object_grasp_name)
            bottle_grasp_semantics_gate = {
                "pass": bool(semantics["pass"]),
                "status": "PASS_BOTTLE_GRASP_SEMANTICS"
                if semantics["pass"]
                else "FAIL_BOTTLE_GRASP_SEMANTICS",
                "selected_grasp": args.object_grasp_name,
                "selected_grasp_pass": semantics.get("selected_grasp_pass"),
                "bottle_semantics": semantics.get("bottle_semantics"),
                "all_grasps": semantics.get("all_grasps"),
            }
            t_w_g = _world_matrix(UsdGeom, stage, args.object_gripper_frame)
            t_o_g = np.asarray(grasp_info["t_object_gripper"], dtype=np.float64)
            t_w_o = t_w_g @ np.linalg.inv(t_o_g)
            center = np.asarray(t_w_o[:3, 3], dtype=np.float64)
            grasp_placement = {
                "grasp_yaml": _rel(grasp_info["path"]),
                "grasp_name": grasp_info["name"],
                "yaml_object_frame": grasp_info["object_frame"],
                "yaml_gripper_frame": grasp_info["gripper_frame"],
                "runtime_gripper_frame": args.object_gripper_frame,
                "t_world_object": t_w_o.tolist(),
                "t_object_gripper": t_o_g.tolist(),
            }
            object_placement_row.update(grasp_placement)
        if np.any(np.abs(object_center_offset) > 0.0):
            center = np.asarray(center, dtype=np.float64) + object_center_offset
        object_placement_row.update(
            {
                "center_offset_world": object_center_offset.tolist(),
                "placed_center_after_offset": np.asarray(center, dtype=np.float64).tolist(),
            }
        )
        object_path = "/World/phase43_passive_contact_cube"
        proxy_offset_rows = [
            _set_collision_offsets(stage, paths["left_finger"], args.proxy_contact_offset, args.proxy_rest_offset),
            _set_collision_offsets(stage, paths["right_finger"], args.proxy_contact_offset, args.proxy_rest_offset),
        ]
        finger_material_rows = {
            "left_finger": _bind_contact_physics_material(
                stage,
                prim_path=paths["left_finger"],
                material_path="/World/PhysicsMaterials/LeftFingerContactMaterial",
                static_friction=args.finger_static_friction,
                dynamic_friction=args.finger_dynamic_friction,
                restitution=args.finger_restitution,
            ),
            "right_finger": _bind_contact_physics_material(
                stage,
                prim_path=paths["right_finger"],
                material_path="/World/PhysicsMaterials/RightFingerContactMaterial",
                static_friction=args.finger_static_friction,
                dynamic_friction=args.finger_dynamic_friction,
                restitution=args.finger_restitution,
            ),
        }
        object_creation_axis = "X" if args.object_axis_source == "open_finger_horizontal_perpendicular" else args.object_axis
        _create_passive_cube(
            world=world,
            stage=stage,
            path=object_path,
            center=center,
            side_length=side_length,
            mass=args.object_mass,
            creation_mode=args.object_creation,
            shape=args.object_shape,
            axis=object_creation_axis,
            length_multiplier=args.object_length_multiplier,
            usd_path=args.object_usd,
            usd_prim_path=args.object_usd_prim_path,
            grasp_band_local_axis_offset=grasp_band_local_axis_offset,
            rigid_body=not args.disable_object_rigid_body,
        )
        object_placement_row["object_creation_axis"] = object_creation_axis
        if args.object_axis_source == "open_finger_horizontal_perpendicular":
            from pxr import Gf
            from pxr import UsdGeom

            object_prim = stage.GetPrimAtPath(object_path)
            if not object_prim or not object_prim.IsValid():
                raise RuntimeError(f"Cannot apply derived object yaw; missing prim: {object_path}")
            object_axis_unit = np.asarray(object_axis_source_row["object_axis_unit_world"], dtype=np.float64)
            _set_xform_matrix(UsdGeom, Gf, object_prim, _transform_with_local_x_axis(np.asarray(center), object_axis_unit))
            object_placement_row["object_axis_source"]["applied_root_transform"] = _world_matrix(
                UsdGeom, stage, object_path
            ).tolist()
        if grasp_placement is not None:
            from pxr import Gf
            from pxr import UsdGeom

            object_prim = stage.GetPrimAtPath(object_path)
            _set_xform_matrix(UsdGeom, Gf, object_prim, np.asarray(grasp_placement["t_world_object"]))
        object_contact_report_rows: list[dict[str, Any]] = []
        if trace_state is not None:
            object_contact_report_rows = _apply_contact_report_api(stage, [object_path])
            for row in object_contact_report_rows:
                if row.get("applied"):
                    trace_state.setdefault("late_registered_rigid_body_paths", []).append(row["path"])
        object_offset_row = _set_object_collision_offsets(
            stage, object_path, args.object_contact_offset, args.object_rest_offset
        )
        contact_geometry_path = _contact_geometry_bbox_path(args.object_shape, object_path)
        object_material_row = _bind_contact_physics_material(
            stage,
            prim_path=contact_geometry_path,
            material_path="/World/PhysicsMaterials/DynamicObjectContactMaterial",
            static_friction=args.object_static_friction,
            dynamic_friction=args.object_dynamic_friction,
            restitution=args.object_restitution,
        )
        if args.object_placement in hdf5_tabletop_rear_quarter_modes:
            tabletop_object_placement_row = _place_object_on_tabletop(
                stage=stage,
                object_path=object_path,
                table_path=args.object_tabletop_reference_path,
                clearance=float(args.object_tabletop_clearance),
                table_top_z=args.object_tabletop_top_z,
            )
            object_placement_row["tabletop_adjustment"] = tabletop_object_placement_row
            if not tabletop_object_placement_row["pass"]:
                raise RuntimeError(f"tabletop placement failed: {tabletop_object_placement_row['status']}")
            if not args.derive_tabletop_top_z_from_open_finger_height:
                object_box_after_tabletop = tabletop_object_placement_row.get("object_bbox_after") or {}
                if object_box_after_tabletop.get("bbox_valid"):
                    anchor_left_center = np.asarray(placement_left_box["center"], dtype=np.float64)
                    anchor_right_center = np.asarray(placement_right_box["center"], dtype=np.float64)
                    anchor_midpoint = (anchor_left_center + anchor_right_center) * 0.5
                    object_center_after_tabletop = np.asarray(
                        object_box_after_tabletop["center"], dtype=np.float64
                    )
                    object_placement_row["fixed_table_anchor_height_residual"] = {
                        "status": "COMPUTED_FIXED_TABLE_ANCHOR_HEIGHT_RESIDUAL",
                        "table_shift_applied": False,
                        "table_top_z_source": tabletop_object_placement_row.get("table_top_z_source"),
                        "anchor_source": placement_basis.get("target"),
                        "anchor_finger_midpoint_world_m": anchor_midpoint.tolist(),
                        "object_center_world_m": object_center_after_tabletop.tolist(),
                        "height_residual_m": float(anchor_midpoint[2] - object_center_after_tabletop[2]),
                        "notes": (
                            "Fixed-table replay diagnostic: z is constrained by the tabletop and object radius. "
                            "This residual must be reported instead of moving the table or floating the bottle "
                            "to match the replay finger midpoint."
                        ),
                    }
            if args.object_axis_source == "open_finger_horizontal_perpendicular":
                if args.closing_axis_gap_solver_basis == "placement":
                    solver_left_box = placement_left_box
                    solver_right_box = placement_right_box
                    solver_basis_target = placement_basis.get("target")
                else:
                    solver_left_box = replay_start_left_box
                    solver_right_box = replay_start_right_box
                    solver_basis_target = "hdf5_open_target"
                finger_delta = np.asarray(solver_left_box["center"], dtype=np.float64) - np.asarray(
                    solver_right_box["center"], dtype=np.float64
                )
                finger_distance = float(np.linalg.norm(finger_delta))
                solver_row: dict[str, Any] = {
                    "enabled": True,
                    "status": "SKIPPED_INVALID_OPEN_FINGER_CLOSING_AXIS",
                    "applied": False,
                    "basis": args.closing_axis_gap_solver_basis,
                    "basis_target": solver_basis_target,
                }
                if finger_distance > 1e-12 and np.isfinite(finger_distance):
                    closing_unit = finger_delta / finger_distance
                    left_proj = float(np.dot(np.asarray(solver_left_box["center"], dtype=np.float64), closing_unit))
                    right_proj = float(
                        np.dot(np.asarray(solver_right_box["center"], dtype=np.float64), closing_unit)
                    )
                    lower_box, upper_box = (
                        (solver_right_box, solver_left_box)
                        if right_proj <= left_proj
                        else (solver_left_box, solver_right_box)
                    )
                    projection_model = _contact_projection_model_for_args(
                        args=args,
                        object_box=_bbox_row(stage, contact_geometry_path),
                        object_axis_unit_world=object_axis_source_row["object_axis_unit_world"],
                        projection_unit_world=closing_unit,
                        side_length=side_length,
                    )
                    solver_row = _closing_axis_gap_centering_solver(
                        lower_box=lower_box,
                        upper_box=upper_box,
                        object_projection_model=projection_model,
                        projection_unit_world=closing_unit,
                        clearance=float(args.object_clearance),
                        use_oriented_finger_boxes=args.finger_gap_projection_model == "oriented_box",
                    )
                    solver_row["basis"] = args.closing_axis_gap_solver_basis
                    solver_row["basis_target"] = solver_basis_target
                    solver_row["object_projection_model_before_shift"] = projection_model
                    if solver_row["pass"]:
                        _shift_prim_world_translation(stage, object_path, np.asarray(solver_row["delta_world_m"]))
                        solver_row["applied"] = True
                        solver_row["object_bbox_after_shift"] = _bbox_row(stage, object_path)
                        solver_row["object_contact_bbox_after_shift"] = _bbox_row(stage, contact_geometry_path)
                        solver_row["object_projection_model_after_shift"] = _contact_projection_model_for_args(
                            args=args,
                            object_box=solver_row["object_contact_bbox_after_shift"],
                            object_axis_unit_world=object_axis_source_row["object_axis_unit_world"],
                            projection_unit_world=closing_unit,
                            side_length=side_length,
                        )
                object_placement_row["closing_axis_gap_solver"] = solver_row
        diagnostic_force_target_overlap_row: dict[str, Any] = {"enabled": False, "status": "DISABLED"}
        if args.diagnostic_force_target_overlap != "none":
            force_left_box = _bbox_row(stage, paths["left_finger"])
            force_right_box = _bbox_row(stage, paths["right_finger"])
            force_delta = np.asarray(force_left_box["center"], dtype=np.float64) - np.asarray(
                force_right_box["center"], dtype=np.float64
            )
            force_distance = float(np.linalg.norm(force_delta))
            if force_distance > 1e-12 and np.isfinite(force_distance):
                force_closing_unit = force_delta / force_distance
                left_proj = float(np.dot(np.asarray(force_left_box["center"], dtype=np.float64), force_closing_unit))
                right_proj = float(np.dot(np.asarray(force_right_box["center"], dtype=np.float64), force_closing_unit))
                lower_box, upper_box = (
                    (force_right_box, force_left_box) if right_proj <= left_proj else (force_left_box, force_right_box)
                )
                force_projection_model = _contact_projection_model_for_args(
                    args=args,
                    object_box=_bbox_row(stage, contact_geometry_path),
                    object_axis_unit_world=object_axis_source_row["object_axis_unit_world"],
                    projection_unit_world=force_closing_unit,
                    side_length=side_length,
                )
                diagnostic_force_target_overlap_row = _diagnostic_force_target_overlap_shift(
                    stage=stage,
                    object_path=object_path,
                    mode=args.diagnostic_force_target_overlap,
                    lower_box=lower_box,
                    upper_box=upper_box,
                    object_projection_model=force_projection_model,
                    projection_unit_world=force_closing_unit,
                    overlap_m=float(args.diagnostic_force_target_overlap_m),
                    use_oriented_finger_boxes=args.finger_gap_projection_model == "oriented_box",
                    named_finger_boxes={"left_finger": force_left_box, "right_finger": force_right_box},
                )
                if diagnostic_force_target_overlap_row.get("applied"):
                    diagnostic_force_target_overlap_row["object_contact_bbox_after_shift"] = _bbox_row(
                        stage, contact_geometry_path
                    )
            else:
                diagnostic_force_target_overlap_row = {
                    "enabled": True,
                    "status": "FAIL_INVALID_FORCE_OVERLAP_CLOSING_AXIS",
                    "applied": False,
                    "formal_gate_allowed": False,
                }
        debug_stage_after_object_placement = (
            _export_debug_stage(stage, output_dir / "debug_stage_after_object_placement.usda")
            if args.save_debug_stage
            else {"saved": False, "path": None, "error": None}
        )
        bottle_runtime_composition_gate = (
            _bottle_usd_runtime_composition_gate(stage, object_path)
            if args.object_shape
            in {
                "bottle_usd",
                "bottle_usd_cylinder_proxy",
                "bottle_usd_segmented_proxy",
                "bottle_usd_grasp_band_proxy",
                "bottle_usd_grasp_box_proxy",
            }
            else {
                "pass": True,
                "status": "SKIPPED_NOT_BOTTLE_USD",
                "runtime_object_path": object_path,
            }
        )
        diagnostic_held_object_row: dict[str, Any] | None = None
        diagnostic_t_gripper_object: np.ndarray | None = None
        if args.diagnostic_held_object_mode == "follow_gripper":
            from pxr import UsdGeom

            t_world_object = _world_matrix(UsdGeom, stage, object_path)
            t_world_gripper = _world_matrix(UsdGeom, stage, args.object_gripper_frame)
            diagnostic_t_gripper_object = np.linalg.inv(t_world_gripper) @ t_world_object
            diagnostic_held_object_row = {
                "mode": "follow_gripper",
                "status": "DIAGNOSTIC_NOT_DYNAMIC_GRASP_PROOF",
                "object_path": object_path,
                "object_gripper_frame": args.object_gripper_frame,
                "frame_features_at_initial_pose": _diagnostic_object_frame_features(UsdGeom, stage, object_path),
                "t_gripper_object_initial": diagnostic_t_gripper_object.tolist(),
            }
        elif args.diagnostic_held_object_mode == "follow_after_bilateral_contact":
            diagnostic_held_object_row = {
                "mode": "follow_after_bilateral_contact",
                "status": "WAITING_FOR_BILATERAL_CLOSE_CONTACT",
                "object_path": object_path,
                "object_gripper_frame": args.object_gripper_frame,
                "trigger_phase": "close",
                "trigger_step": None,
                "trigger_contact_summary": None,
                "t_gripper_object_at_trigger": None,
                "frame_features_at_trigger": None,
                "note": (
                    "Diagnostic only. The object follows the gripper only after PhysX reports expected "
                    "CONTACT_FOUND events during close. This is not a dynamic grasp proof."
                ),
            }
        support_plane_row: dict[str, Any] | None = None
        if support_options["mode"] != "none":
            support_bbox_path = contact_geometry_path if support_options["mode"] == "contact_patch" else object_path
            object_support_box = _bbox_row(stage, support_bbox_path)
            if not object_support_box.get("bbox_valid"):
                raise RuntimeError(f"Cannot place support plane because support bbox is invalid: {support_bbox_path}")
            if support_options["mode"] in {"object_bottom", "object_patch", "contact_patch"}:
                object_support_center = np.asarray(object_support_box["center"], dtype=np.float64)
                support_center = object_support_center.copy()
                support_center[2] = (
                    float(object_support_box["min"][2])
                    - float(args.support_plane_clearance)
                    - float(support_options["thickness"]) * 0.5
                )
                if support_options["mode"] in {"object_patch", "contact_patch"}:
                    support_size_x, support_size_y = _local_object_support_patch_size(
                        object_support_box,
                        margin=float(support_options["patch_margin"]),
                    )
                    support_path = (
                        "/World/phase58_contact_geometry_support_patch"
                        if support_options["mode"] == "contact_patch"
                        else "/World/phase58_local_object_support_patch"
                    )
                else:
                    support_size_x = support_options["size_x"]
                    support_size_y = support_options["size_y"]
                    support_path = "/World/phase58_static_support_plane"
            else:
                if support_options["center"] is None:
                    raise ValueError(
                        "--support-plane-mode fixed_box requires --support-plane-center X Y Z or --support-plane-config"
                    )
                support_center = np.asarray(support_options["center"], dtype=np.float64)
                support_size_x = support_options["size_x"]
                support_size_y = support_options["size_y"]
                support_path = "/World/phase58_static_support_plane"
            support_plane_row = _create_static_support_box(
                stage=stage,
                path=support_path,
                center=support_center,
                size_x=support_size_x,
                size_y=support_size_y,
                thickness=support_options["thickness"],
            )
            support_plane_row["placement_object_box"] = object_support_box
            support_plane_row["placement_bbox_path"] = support_bbox_path
            support_plane_row["mode"] = support_options["mode"]
            if support_options["mode"] in {"object_patch", "contact_patch"}:
                support_plane_row["provenance"] = (
                    "DIAGNOSTIC_CONTACT_GEOMETRY_SUPPORT_PATCH_NOT_FINAL_TABLE"
                    if support_options["mode"] == "contact_patch"
                    else "DIAGNOSTIC_LOCAL_OBJECT_SUPPORT_PATCH_NOT_FINAL_TABLE"
                )
                support_plane_row["formal_full_scene_validation"] = False
                support_plane_row["patch_margin"] = float(support_options["patch_margin"])
                support_plane_row["notes"] = (
                    "Diagnostic local support patch under the selected object/contact geometry only. It isolates "
                    "bottle-gripper closure from global table/base collision calibration and is not a final "
                    "workcell table proof."
                )
            support_plane_row["config"] = support_options["config"]
            support_plane_row["config_provenance"] = support_options["config_provenance"]
            support_plane_row["table_frame"] = support_options["table_frame"]
        first_contact_row: dict[str, Any] | None = None
        contact_pair_rows: list[dict[str, Any]] = []
        geometry_audit_window_steps = 5
        denied_workcell_geometry_audit: dict[str, Any] = {
            "status": "NO_DENIED_WORKCELL_CONTACT_OBSERVED",
            "window_radius_steps": geometry_audit_window_steps,
            "first_denied_pair": None,
            "pre_contact_snapshots": [],
            "snapshots": [],
        }
        recent_geometry_snapshots: list[dict[str, Any]] = []
        geometry_audit_follow_until: tuple[str, int] | None = None
        denied_geometry_extra_paths: list[dict[str, str]] = []
        initial_grasp_geometry_audit: dict[str, Any] = {
            "status": "NOT_CAPTURED",
            "notes": (
                "Initial object/finger/gripper geometry audit is diagnostic only. It does not move objects, "
                "change collision policy, or make a failed grasp pass."
            ),
        }

        def _append_denied_workcell_geometry_audit(
            *,
            phase: str,
            step: int,
            step_contact_rows: list[dict[str, Any]],
        ) -> None:
            nonlocal geometry_audit_follow_until, denied_geometry_extra_paths
            if not args.trace_contact_pairs or workcell_contact_policy is None:
                return
            denied_step_rows: list[dict[str, Any]] = []
            if args.moving_fingers == "both":
                audit_expected_finger_paths = [contact_targets["left_finger"], contact_targets["right_finger"]]
            else:
                audit_expected_finger_paths = [contact_targets[f"{args.moving_fingers}_finger"]]
            audit_same_side_robot_root = robot_root_for_side(args.contact_proxy_profile, args.side)
            audit_other_side_robot_root = robot_root_for_side(
                args.contact_proxy_profile,
                "right" if args.side == "left" else "left",
            )
            audit_diagnostic_paths = [support_plane_row["path"]] if support_plane_row else None
            for contact_row in step_contact_rows:
                category = _classify_object_contact(
                    contact_row,
                    object_path=object_path,
                    expected_finger_paths=audit_expected_finger_paths,
                    same_side_robot_root=audit_same_side_robot_root,
                    other_side_robot_root=audit_other_side_robot_root,
                    diagnostic_contact_paths=audit_diagnostic_paths,
                )
                if category != "workcell_or_environment":
                    continue
                other_path = _other_collider_for_object_pair(contact_row, object_path)
                if other_path is None:
                    continue
                rule = _match_workcell_contact_rule(other_path, workcell_contact_policy)
                if rule.get("decision") == "deny":
                    denied_step_rows.append(
                        {
                            "contact_pair": contact_row,
                            "other_path": other_path,
                            "matched_rule": rule,
                        }
                    )
            if denied_step_rows and denied_workcell_geometry_audit["first_denied_pair"] is None:
                first_denied = dict(denied_step_rows[0])
                denied_workcell_geometry_audit.update(
                    {
                        "status": "CAPTURED_FIRST_DENIED_WORKCELL_CONTACT",
                        "first_denied_pair": first_denied,
                        "pre_contact_snapshots": list(recent_geometry_snapshots),
                    }
                )
                matched_prefix = first_denied["matched_rule"].get("path_prefix")
                denied_geometry_extra_paths = [
                    {"label": "denied_other_collider", "path": str(first_denied["other_path"])},
                ]
                if matched_prefix:
                    denied_geometry_extra_paths.append(
                        {"label": "matched_policy_prefix", "path": str(matched_prefix)}
                    )
                geometry_audit_follow_until = (phase, int(step) + geometry_audit_window_steps)

            from pxr import UsdGeom

            include_in_denied_window = False
            if denied_workcell_geometry_audit["first_denied_pair"] is not None and geometry_audit_follow_until:
                follow_phase, follow_until_step = geometry_audit_follow_until
                include_in_denied_window = bool(phase == follow_phase and int(step) <= follow_until_step)
            snapshot = _geometry_audit_snapshot(
                UsdGeom=UsdGeom,
                stage=stage,
                phase=phase,
                step=step,
                object_path=object_path,
                contact_geometry_path=contact_geometry_path,
                object_gripper_frame=args.object_gripper_frame,
                extra_paths=denied_geometry_extra_paths if include_in_denied_window else None,
            )
            if include_in_denied_window:
                denied_workcell_geometry_audit["snapshots"].append(snapshot)
            if denied_workcell_geometry_audit["first_denied_pair"] is None:
                recent_geometry_snapshots.append(snapshot)
                del recent_geometry_snapshots[:-geometry_audit_window_steps]

        from pxr import UsdGeom

        initial_grasp_geometry_audit = _geometry_audit_snapshot(
            UsdGeom=UsdGeom,
            stage=stage,
            phase="after_object_placement",
            step=0,
            object_path=object_path,
            contact_geometry_path=contact_geometry_path,
            object_gripper_frame=args.object_gripper_frame,
            extra_paths=_grasp_geometry_audit_extra_paths(
                side=args.side,
                contact_proxy_profile=args.contact_proxy_profile,
                finger_proxy_paths=paths,
                contact_target_paths=contact_targets,
                support_plane_path=support_plane_row["path"] if support_plane_row else None,
            ),
        )
        initial_grasp_geometry_audit["status"] = "CAPTURED_AFTER_OBJECT_PLACEMENT"

        try:
            # Do not reset after object creation: object placement is computed from
            # the current open-pose fingertip bboxes, and a later reset can move
            # the articulation back under the already-placed object.
            _apply_gravity(world, args.gravity)
            _set_full_state(art, open_target)
            _set_full_target(art, open_target)
            dof_names = list(art.dof_names)
            replay_mode_for_tracking = args.hdf5_replay_mode if hdf5_target_sequence is not None else "gripper_only"
            tracking_groups = _tracking_groups(
                dof_names, replay_mode=replay_mode_for_tracking, finger_dof_names=finger_dof_names, side=args.side
            )
            runtime_limits = _get_limits(art)
            runtime_kps, runtime_kds = _get_gains(art)
            runtime_max_efforts = _get_max_efforts(art)
            runtime_max_velocities = _get_max_velocities(art)
            runtime_drive_profile = _runtime_drive_profile(
                dof_names=dof_names,
                groups=tracking_groups,
                runtime_limits=runtime_limits,
                stiffness=runtime_kps,
                damping=runtime_kds,
                max_efforts=runtime_max_efforts,
                max_velocities=runtime_max_velocities,
                profile_name=args.drive_profile_name,
                profile_provenance=args.drive_profile_provenance,
            )
            tracking_rows: list[dict[str, Any]] = []
            pre_step_tracking_rows: list[dict[str, Any]] = []
            target_limit_rows: list[dict[str, Any]] = []
            object_reset_box = _bbox_row(stage, object_path)
            object_contact_reset_box = _bbox_row(stage, contact_geometry_path)
            object_contact_projection_model: dict[str, Any] = {"source": "world_aabb"}
            object_contact_projected_interval: tuple[float, float] | None = None
            finger_delta = np.asarray(replay_start_left_box["center"], dtype=np.float64) - np.asarray(
                replay_start_right_box["center"], dtype=np.float64
            )
            finger_distance = float(np.linalg.norm(finger_delta))
            if finger_distance > 1e-12 and np.isfinite(finger_distance):
                object_contact_projection_model = _contact_projection_model_for_args(
                    args=args,
                    object_box=object_contact_reset_box,
                    object_axis_unit_world=object_axis_source_row["object_axis_unit_world"],
                    projection_unit_world=finger_delta / finger_distance,
                    side_length=side_length,
                )
                if object_contact_projection_model.get("valid"):
                    object_contact_projected_interval = tuple(
                        float(v) for v in object_contact_projection_model["object_interval_m"]
                    )
            tabletop_collision_audit = (
                _tabletop_collision_audit(stage, args.object_tabletop_reference_path)
                if tabletop_object_placement_row is not None
                else None
            )
            active_grasp_geometry_precondition = _active_grasp_geometry_precondition(
                require_active_target_contact=bool(args.trace_contact_pairs and args.require_active_target_contact),
                already_in_contact_setup=bool(args.already_in_contact_setup),
                loaded_contact_placement=bool(args.object_placement in hdf5_close_rear_quarter_modes),
                open_left_box=replay_start_left_box,
                open_right_box=replay_start_right_box,
                object_box=object_contact_reset_box,
                gap_axis=axis,
                clearance=float(args.object_clearance),
                object_projected_interval=object_contact_projected_interval,
                object_projection_model=object_contact_projection_model,
                use_oriented_finger_boxes=args.finger_gap_projection_model == "oriented_box",
            )
            open_frame_height_required = bool(args.object_placement in hdf5_open_rear_quarter_modes)
            open_finger_object_height_alignment = _open_finger_object_height_alignment(
                require_active_target_contact=bool(
                    args.trace_contact_pairs
                    and args.require_active_target_contact
                    and open_frame_height_required
                ),
                already_in_contact_setup=bool(args.already_in_contact_setup),
                open_left_box=replay_start_left_box,
                open_right_box=replay_start_right_box,
                object_box=object_contact_reset_box,
                max_error=float(args.max_open_finger_object_center_height_error),
            )
            tabletop_reference_contract = _tabletop_reference_contract(
                required=bool(tabletop_object_placement_row is not None),
                require_open_height_alignment=open_frame_height_required,
                tabletop_adjustment=tabletop_object_placement_row,
                table_collision_audit=tabletop_collision_audit,
                open_left_box=replay_start_left_box,
                open_right_box=replay_start_right_box,
                object_box=object_contact_reset_box,
                max_finger_object_center_height_error=float(args.max_open_finger_object_center_height_error),
            )
            if args.object_placement in rear_quarter_modes:
                placement_gap_vector = np.asarray(placement_left_box["center"], dtype=np.float64) - np.asarray(
                    placement_right_box["center"], dtype=np.float64
                )
                placement_gap_norm = float(np.linalg.norm(placement_gap_vector))
                if args.object_axis_source == "open_finger_horizontal_perpendicular":
                    axis_source = object_placement_row.get("object_axis_source") or object_axis_source_row
                    closing_dot = float(axis_source.get("abs_dot_closing_axis", float("inf")))
                    horizontal_abs_z = float(axis_source.get("horizontal_abs_z", float("inf")))
                    semantics = {
                        "pass": bool(
                            closing_dot <= float(args.max_closing_long_axis_dot) and horizontal_abs_z <= 0.05
                        ),
                        "status": "PASS_DERIVED_FINGER_REAR_QUARTER_PLACEMENT"
                        if closing_dot <= float(args.max_closing_long_axis_dot) and horizontal_abs_z <= 0.05
                        else "FAIL_DERIVED_FINGER_REAR_QUARTER_PLACEMENT",
                        "placement_semantics": "finger_gap_center_on_bottle_rear_quarter",
                        "object_axis_source": axis_source,
                        "finger_gap_axis_vector_world": (
                            placement_gap_vector / placement_gap_norm if placement_gap_norm > 1e-12 else None
                        ).tolist()
                        if placement_gap_norm > 1e-12
                        else None,
                        "closing_long_axis_dot_abs": closing_dot,
                        "max_closing_long_axis_dot": float(args.max_closing_long_axis_dot),
                        "horizontal_abs_z": horizontal_abs_z,
                        "rear_fraction_target": float(args.object_rear_quarter_fraction),
                        "rear_fraction_tolerance": float(args.object_rear_quarter_tolerance),
                    }
                else:
                    semantics = evaluate_axis_aligned_finger_rear_quarter(
                        finger_contact_center_world=object_placement_row["base_center"],
                        object_bbox=object_reset_box,
                        object_axis=args.object_axis,
                        finger_gap_axis=axis_name,
                        finger_gap_axis_vector_world=(
                            placement_gap_vector / placement_gap_norm if placement_gap_norm > 1e-12 else None
                        ),
                        rear_fraction_target=float(args.object_rear_quarter_fraction),
                        rear_fraction_tolerance=float(args.object_rear_quarter_tolerance),
                        max_closing_long_axis_dot=float(args.max_closing_long_axis_dot),
                    )
                bottle_grasp_semantics_gate = dict(semantics)
            object_reset_center = np.asarray(object_reset_box["center"], dtype=np.float64)
            rows: list[dict[str, Any]] = []
            object_width_stop_rows: list[dict[str, Any]] = []
            diagnostic_loaded_clamp_squeeze_rows: list[dict[str, Any]] = []
            target_contact_reachability_rows: list[dict[str, Any]] = []
            max_displacement = 0.0
            finite_motion = True
            for step in range(args.settle_steps):
                pre_step_qpos = _apply_replay_target_and_step(
                    world,
                    art,
                    open_target,
                    actuation_mode=args.hdf5_replay_actuation_mode,
                    substep_mode=args.hdf5_replay_substep_mode,
                    previous_target=open_target,
                    target_hold_steps=args.hdf5_replay_target_hold_steps,
                )
                if diagnostic_t_gripper_object is not None:
                    from pxr import Gf
                    from pxr import UsdGeom

                    _set_object_from_gripper_relative_transform(
                        stage=stage,
                        UsdGeom=UsdGeom,
                        Gf=Gf,
                        object_path=object_path,
                        object_gripper_frame=args.object_gripper_frame,
                        t_gripper_object=diagnostic_t_gripper_object,
                    )
                qpos = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
                qvel = _safe_joint_velocities(art)
                pre_step_tracking = _tracking_step_errors(target=open_target, actual=pre_step_qpos, groups=tracking_groups)
                step_tracking = _tracking_step_errors(target=open_target, actual=qpos, groups=tracking_groups)
                target_limit = _target_limit_step_violations(
                    target=open_target, limits=runtime_limits, groups=tracking_groups
                )
                pre_step_tracking_rows.append({"phase": "settle", "step": step, "groups": pre_step_tracking})
                tracking_rows.append(
                    {
                        "phase": "settle",
                        "step": step,
                        "groups": step_tracking,
                        "target": open_target.tolist(),
                        "previous_target": open_target.tolist(),
                        "next_target": open_target.tolist(),
                        "pre_qpos": pre_step_qpos.tolist(),
                        "post_qpos": qpos.tolist(),
                        "qvel": qvel,
                    }
                )
                target_limit_rows.append({"phase": "settle", "step": step, "groups": target_limit})
                left_box = _bbox_row(stage, paths["left_finger"])
                right_box = _bbox_row(stage, paths["right_finger"])
                object_box = _bbox_row(stage, object_path)
                diagnostic_frame_features = (
                    _diagnostic_object_frame_features(UsdGeom, stage, object_path)
                    if diagnostic_t_gripper_object is not None
                    else {}
                )
                object_center = np.asarray(object_box.get("center", [np.nan, np.nan, np.nan]), dtype=np.float64)
                displacement_from_reset = float(np.linalg.norm(object_center - object_reset_center))
                finite_motion = bool(
                    finite_motion and np.all(np.isfinite(object_center)) and np.isfinite(displacement_from_reset)
                )
                max_displacement = max(
                    max_displacement,
                    displacement_from_reset if np.isfinite(displacement_from_reset) else float("inf"),
                )
                step_contact_pairs = _read_contact_pairs(trace_state)
                step_contact_rows: list[dict[str, Any]] = []
                if step_contact_pairs:
                    for pair in step_contact_pairs:
                        contact_row = {"phase": "settle", "step": step, **pair}
                        contact_pair_rows.append(contact_row)
                        step_contact_rows.append(contact_row)
                    if first_contact_row is None:
                        first_contact_row = dict(contact_pair_rows[-len(step_contact_pairs)])
                _append_denied_workcell_geometry_audit(
                    phase="settle",
                    step=step,
                    step_contact_rows=step_contact_rows,
                )
                rows.append(
                    {
                        "phase": "settle",
                        "step": step,
                        "object_center_x": float(object_center[0]),
                        "object_center_y": float(object_center[1]),
                        "object_center_z": float(object_center[2]),
                        **diagnostic_frame_features,
                        "object_displacement": displacement_from_reset,
                        **_finger_qpos_values(qpos, dof_names, finger_dof_names),
                        "tracking_controlled_max_abs_error": step_tracking["controlled"]["max_abs_error"],
                        "tracking_controlled_rms_error": step_tracking["controlled"]["rms_error"],
                        "target_limit_controlled_max_violation": target_limit["controlled"]["max_violation"],
                        "pre_step_tracking_controlled_max_abs_error": pre_step_tracking["controlled"]["max_abs_error"],
                        "pre_step_tracking_controlled_rms_error": pre_step_tracking["controlled"]["rms_error"],
                        "tracking_gripper_max_abs_error": step_tracking["gripper"]["max_abs_error"],
                        "tracking_gripper_rms_error": step_tracking["gripper"]["rms_error"],
                        "target_limit_gripper_max_violation": target_limit["gripper"]["max_violation"],
                        "pre_step_tracking_gripper_max_abs_error": pre_step_tracking["gripper"]["max_abs_error"],
                        "pre_step_tracking_gripper_rms_error": pre_step_tracking["gripper"]["rms_error"],
                        "tracking_left_arm_max_abs_error": step_tracking.get("left_arm", {}).get("max_abs_error"),
                        "tracking_left_arm_rms_error": step_tracking.get("left_arm", {}).get("rms_error"),
                        "target_limit_left_arm_max_violation": target_limit.get("left_arm", {}).get("max_violation"),
                        "pre_step_tracking_left_arm_max_abs_error": pre_step_tracking.get("left_arm", {}).get(
                            "max_abs_error"
                        ),
                        "pre_step_tracking_left_arm_rms_error": pre_step_tracking.get("left_arm", {}).get("rms_error"),
                        "finger_center_distance": _gap_metrics(left_box, right_box).get("center_distance"),
                        **_finger_center_row(left_box, right_box),
                        **_axis_probe_row(
                            axis=axis,
                            left_box=left_box,
                            right_box=right_box,
                            object_box=object_box,
                            target_finger_box=left_box if args.moving_fingers != "right" else right_box,
                        ),
                    }
                )

            object_initial_box = _bbox_row(stage, object_path)
            object_contact_latest_box = _bbox_row(stage, contact_geometry_path)
            object_initial_center = np.asarray(object_initial_box["center"], dtype=np.float64)
            object_latest_box = dict(object_initial_box)
            object_latest_center = object_initial_center.copy()
            object_settle_displacement = float(np.linalg.norm(object_initial_center - object_reset_center))
            if hdf5_target_sequence is not None:
                close_target = hdf5_target_sequence[-1]
                close_values = hdf5_gripper_summary["last_target_values"] if hdf5_gripper_summary else {}
                close_sequence = hdf5_target_sequence[1:]
                if args.close_steps is not None:
                    close_sequence = close_sequence[: args.close_steps]
            else:
                close_target, close_values = _finger_targets(
                    art,
                    args.close_offset,
                    args.limit_margin,
                    finger_dof_names,
                    right_finger_sign=args.right_finger_close_sign,
                )
                close_sequence = []
            if args.moving_fingers != "both" and hdf5_target_sequence is None:
                isolated_target = open_target.copy()
                moving_finger_dof = finger_dof_names[f"{args.moving_fingers}_finger"]
                isolated_target[dof_names.index(moving_finger_dof)] = close_target[dof_names.index(moving_finger_dof)]
                close_target = isolated_target
                close_values = {
                    "left_finger": float(close_target[dof_names.index(finger_dof_names["left_finger"])]),
                    "right_finger": float(close_target[dof_names.index(finger_dof_names["right_finger"])]),
                }
            synthetic_close_steps = int(args.close_steps) if args.close_steps is not None else 180
            close_step_count = len(close_sequence) if hdf5_target_sequence is not None else synthetic_close_steps
            close_replay_step_count = int(close_step_count)
            post_close_lift_sequence: list[np.ndarray] = []
            post_close_lift_summary: dict[str, Any] = {
                "enabled": bool(args.post_close_lift_source != "none"),
                "source": args.post_close_lift_source,
                "diagnostic_only": True,
                "formal_close_replay_success": False,
                "hdf5_close_sequence_preserved": True,
                "gripper_mode": args.post_close_lift_gripper_mode,
                "step_count": 0,
                "notes": (
                    "Post-close lift is a diagnostic phase appended after the selected HDF5 close window. "
                    "It must not be used to convert the original close replay into a formal pass."
                ),
            }
            if args.post_close_lift_source == "hdf5_continuation":
                if hdf5_target_sequence is None or not args.hdf5_gripper_episode:
                    raise ValueError("post-close HDF5 lift diagnostics require --hdf5-gripper-episode")
                lift_start_frame = int(args.hdf5_gripper_end_frame) - 1
                if args.post_close_lift_hdf5_start_frame is not None:
                    lift_start_frame = int(args.post_close_lift_hdf5_start_frame)
                lift_end_frame = int(args.post_close_lift_hdf5_end_frame)
                if lift_start_frame < 0:
                    raise ValueError(f"post-close lift start frame must be non-negative, got {lift_start_frame}")
                if lift_end_frame <= lift_start_frame + 1:
                    raise ValueError(
                        "post-close lift HDF5 range must include an anchor plus at least one applied target, "
                        f"got start={lift_start_frame}, end={lift_end_frame}"
                    )
                lift_qpos = _load_hdf5_qpos(
                    args.hdf5_gripper_episode,
                    start=lift_start_frame,
                    end=lift_end_frame,
                    max_frames=None,
                )
                lift_mapping = mapping if _replay_mode_controls_arm("left_arm_and_gripper") else None
                if lift_mapping is None:
                    lift_mapping = load_mapping(args.mapping)
                lift_targets, lift_hdf5_summary = _targets_from_hdf5_qpos(
                    art=art,
                    side=args.side,
                    qpos=lift_qpos,
                    gripper_sequence=None,
                    gripper_source="observations/qpos",
                    mapping=lift_mapping,
                    replay_mode="left_arm_and_gripper",
                    finger_dof_names=finger_dof_names,
                    finger_qpos_limits=finger_qpos_limits,
                    gripper_qpos_calibration=gripper_qpos_calibration,
                    arm_hold_frame_offset=0,
                )
                post_close_lift_sequence = [np.asarray(target, dtype=np.float64).copy() for target in lift_targets[1:]]
                if args.post_close_lift_gripper_mode == "hold_final_close":
                    left_idx = dof_names.index(finger_dof_names["left_finger"])
                    right_idx = dof_names.index(finger_dof_names["right_finger"])
                    for target in post_close_lift_sequence:
                        target[left_idx] = float(close_target[left_idx])
                        target[right_idx] = float(close_target[right_idx])
                post_close_lift_summary.update(
                    {
                        "hdf5_start_frame": lift_start_frame,
                        "hdf5_first_applied_frame": lift_start_frame + 1,
                        "hdf5_end_frame": lift_end_frame,
                        "step_count": len(post_close_lift_sequence),
                        "target_hold_steps": int(args.hdf5_replay_target_hold_steps),
                        "hdf5_summary": lift_hdf5_summary,
                        "arm_target_source": "observations/qpos",
                        "gripper_target_source": (
                            "final_hdf5_close_target"
                            if args.post_close_lift_gripper_mode == "hold_final_close"
                            else "observations/qpos"
                        ),
                    }
                )
            elif args.post_close_lift_source == "jacobian_vertical":
                post_close_lift_sequence = [close_target.copy() for _ in range(int(args.post_close_lift_steps))]
                post_close_lift_summary.update(
                    {
                        "source": "jacobian_vertical",
                        "runtime_generated": True,
                        "status": "PENDING_RUNTIME_JACOBIAN_SEQUENCE",
                        "requested_lift_height_m": float(args.post_close_lift_height),
                        "step_count": int(args.post_close_lift_steps),
                        "target_hold_steps": int(args.hdf5_replay_target_hold_steps),
                        "arm_target_source": "runtime_finite_difference_jacobian",
                        "gripper_target_source": "final_hdf5_close_target",
                    }
                )
            post_close_lift_step_count = int(len(post_close_lift_sequence))
            post_close_lift_hold_step_count = (
                int(post_close_lift_hold_steps) if post_close_lift_step_count > 0 else 0
            )
            close_sequence = list(close_sequence)
            if hdf5_target_sequence is not None:
                if post_close_hold_steps > 0:
                    close_sequence.extend([close_target] * post_close_hold_steps)
                if post_close_lift_sequence:
                    close_sequence.extend(post_close_lift_sequence)
                if post_close_lift_hold_step_count > 0:
                    final_lift_target = (
                        post_close_lift_sequence[-1].copy() if post_close_lift_sequence else close_target.copy()
                    )
                    close_sequence.extend([final_lift_target] * post_close_lift_hold_step_count)
            post_close_hold_start_step = int(close_replay_step_count)
            post_close_lift_start_step = int(close_replay_step_count + post_close_hold_steps)
            post_close_lift_hold_start_step = int(post_close_lift_start_step + post_close_lift_step_count)
            close_step_count = int(
                close_replay_step_count
                + post_close_hold_steps
                + post_close_lift_step_count
                + post_close_lift_hold_step_count
            )
            for step in range(close_step_count):
                if hdf5_target_sequence is not None:
                    step_target = close_sequence[step]
                elif args.closure_profile == "linear":
                    alpha = float(step + 1) / float(max(synthetic_close_steps, 1))
                    step_target = open_target + alpha * (close_target - open_target)
                else:
                    step_target = close_target
                if hdf5_target_sequence is not None and close_sequence:
                    previous_target = open_target if step == 0 else close_sequence[step - 1]
                    next_target = close_sequence[step + 1] if step + 1 < len(close_sequence) else step_target
                else:
                    previous_target = open_target
                    next_target = close_target
                if step < close_replay_step_count:
                    phase = "close"
                    phase_step = int(step)
                elif step < post_close_lift_start_step:
                    phase = "post_close_hold"
                    phase_step = int(step - post_close_hold_start_step)
                elif step < post_close_lift_hold_start_step:
                    phase = "post_close_lift"
                    phase_step = int(step - post_close_lift_start_step)
                else:
                    phase = "post_close_lift_hold"
                    phase_step = int(step - post_close_lift_hold_start_step)
                if phase == "post_close_lift" and args.post_close_lift_source == "jacobian_vertical":
                    if post_close_lift_summary.get("status") == "PENDING_RUNTIME_JACOBIAN_SEQUENCE":
                        lift_base_qpos = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1).copy()
                        post_close_lift_sequence, jacobian_lift_summary = _build_jacobian_vertical_lift_sequence(
                            stage=stage,
                            art=art,
                            paths=paths,
                            base_qpos=lift_base_qpos,
                            close_target=close_target,
                            tracking_groups=tracking_groups,
                            side=args.side,
                            lift_height=float(args.post_close_lift_height),
                            lift_steps=int(args.post_close_lift_steps),
                            epsilon=float(args.post_close_lift_jacobian_eps),
                            damping=float(args.post_close_lift_jacobian_damping),
                            max_joint_delta=float(args.post_close_lift_max_joint_delta),
                            body_name=Path(str(args.object_gripper_frame)).name,
                        )
                        post_close_lift_summary.update(jacobian_lift_summary)
                        if len(post_close_lift_sequence) != post_close_lift_step_count:
                            raise RuntimeError(
                                "jacobian_vertical generated an unexpected sequence length: "
                                f"{len(post_close_lift_sequence)} != {post_close_lift_step_count}"
                            )
                        close_sequence[
                            post_close_lift_start_step:post_close_lift_hold_start_step
                        ] = post_close_lift_sequence
                        if post_close_lift_hold_step_count > 0:
                            close_sequence[
                                post_close_lift_hold_start_step:close_step_count
                            ] = [post_close_lift_sequence[-1].copy()] * post_close_lift_hold_step_count
                    step_target = post_close_lift_sequence[phase_step]
                    previous_target = close_target if phase_step == 0 else post_close_lift_sequence[phase_step - 1]
                    next_target = (
                        post_close_lift_sequence[phase_step + 1]
                        if phase_step + 1 < len(post_close_lift_sequence)
                        else step_target
                    )
                current_qpos_for_guard = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
                guard_left_box = _bbox_row(stage, paths["left_finger"])
                guard_right_box = _bbox_row(stage, paths["right_finger"])
                guard_object_box = _bbox_row(stage, contact_geometry_path)
                guard_closing_unit = _closing_unit_from_finger_boxes(guard_left_box, guard_right_box)
                guard_projection_model = (
                    _contact_projection_model_for_args(
                        args=args,
                        object_box=guard_object_box,
                        object_axis_unit_world=object_axis_source_row["object_axis_unit_world"],
                        projection_unit_world=guard_closing_unit,
                        side_length=side_length,
                    )
                    if guard_closing_unit is not None
                    else {"valid": False, "status": "FAIL_INVALID_GUARD_CLOSING_AXIS"}
                )
                guard_projected_interval = (
                    tuple(float(v) for v in guard_projection_model["object_interval_m"])
                    if guard_projection_model.get("valid")
                    else object_contact_projected_interval
                )
                step_target, width_stop_row = _object_width_stop_target(
                    enabled=bool(args.enforce_object_width_finger_stop),
                    current_qpos=current_qpos_for_guard,
                    target=step_target,
                    dof_names=dof_names,
                    finger_dof_names=finger_dof_names,
                    left_box=guard_left_box,
                    right_box=guard_right_box,
                    object_box=guard_object_box,
                    clearance=float(args.object_clearance),
                    predictive_margin=float(args.object_width_stop_predictive_margin),
                    object_projected_interval=guard_projected_interval,
                    use_oriented_finger_boxes=args.finger_gap_projection_model == "oriented_box",
                )
                width_stop_row.update(
                    {
                        "phase": phase,
                        "step": phase_step,
                        "global_close_step": int(step),
                        "live_object_projection_model": guard_projection_model,
                    }
                )
                object_width_stop_rows.append(width_stop_row)
                step_target, loaded_clamp_squeeze_row = _apply_diagnostic_loaded_clamp_squeeze(
                    enabled=bool(float(args.diagnostic_loaded_clamp_squeeze_depth) > 0.0),
                    target=step_target,
                    dof_names=dof_names,
                    finger_dof_names=finger_dof_names,
                    runtime_limits=runtime_limits,
                    limit_margin=float(args.limit_margin),
                    squeeze_depth=float(args.diagnostic_loaded_clamp_squeeze_depth),
                    phase=phase,
                    moving_fingers=str(args.moving_fingers),
                    reference_target=step_target,
                    reference_target_source="width_guarded_step_target",
                )
                loaded_clamp_squeeze_row.update(
                    {
                        "step": phase_step,
                        "global_close_step": int(step),
                    }
                )
                diagnostic_loaded_clamp_squeeze_rows.append(loaded_clamp_squeeze_row)
                pre_step_qpos = _apply_replay_target_and_step(
                    world,
                    art,
                    step_target,
                    actuation_mode=args.hdf5_replay_actuation_mode,
                    substep_mode=args.hdf5_replay_substep_mode,
                    previous_target=previous_target,
                    target_hold_steps=args.hdf5_replay_target_hold_steps,
                )
                if diagnostic_t_gripper_object is not None:
                    from pxr import Gf
                    from pxr import UsdGeom

                    _set_object_from_gripper_relative_transform(
                        stage=stage,
                        UsdGeom=UsdGeom,
                        Gf=Gf,
                        object_path=object_path,
                        object_gripper_frame=args.object_gripper_frame,
                        t_gripper_object=diagnostic_t_gripper_object,
                    )
                qpos = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
                qvel = _safe_joint_velocities(art)
                pre_step_tracking = _tracking_step_errors(target=step_target, actual=pre_step_qpos, groups=tracking_groups)
                step_tracking = _tracking_step_errors(target=step_target, actual=qpos, groups=tracking_groups)
                target_limit = _target_limit_step_violations(
                    target=step_target, limits=runtime_limits, groups=tracking_groups
                )
                pre_step_tracking_rows.append({"phase": phase, "step": phase_step, "groups": pre_step_tracking})
                tracking_rows.append(
                    {
                        "phase": phase,
                        "step": phase_step,
                        "global_close_step": int(step),
                        "groups": step_tracking,
                        "target": step_target.tolist(),
                        "previous_target": previous_target.tolist(),
                        "next_target": next_target.tolist(),
                        "pre_qpos": pre_step_qpos.tolist(),
                        "post_qpos": qpos.tolist(),
                        "qvel": qvel,
                    }
                )
                target_limit_rows.append({"phase": phase, "step": phase_step, "groups": target_limit})
                left_box = _bbox_row(stage, paths["left_finger"])
                right_box = _bbox_row(stage, paths["right_finger"])
                object_box = _bbox_row(stage, object_path)
                object_contact_box = _bbox_row(stage, contact_geometry_path)
                object_contact_latest_box = dict(object_contact_box)
                diagnostic_frame_features = (
                    _diagnostic_object_frame_features(UsdGeom, stage, object_path)
                    if diagnostic_t_gripper_object is not None
                    else {}
                )
                object_center = np.asarray(object_box.get("center", [np.nan, np.nan, np.nan]), dtype=np.float64)
                object_latest_box = dict(object_box)
                object_latest_center = object_center.copy()
                displacement = float(np.linalg.norm(object_center - object_initial_center))
                displacement_from_reset = float(np.linalg.norm(object_center - object_reset_center))
                finite_motion = bool(
                    finite_motion
                    and np.all(np.isfinite(object_center))
                    and np.isfinite(displacement)
                    and np.isfinite(displacement_from_reset)
                )
                max_displacement = max(
                    max_displacement,
                    displacement_from_reset if np.isfinite(displacement_from_reset) else float("inf"),
                )
                step_contact_pairs = _read_contact_pairs(trace_state)
                step_contact_rows = []
                if step_contact_pairs:
                    for pair in step_contact_pairs:
                        contact_row = {"phase": phase, "step": phase_step, "global_close_step": int(step), **pair}
                        contact_pair_rows.append(contact_row)
                        step_contact_rows.append(contact_row)
                    if first_contact_row is None:
                        first_contact_row = dict(contact_pair_rows[-len(step_contact_pairs)])
                _append_denied_workcell_geometry_audit(
                    phase=phase,
                    step=phase_step,
                    step_contact_rows=step_contact_rows,
                )
                step_closing_unit = _closing_unit_from_finger_boxes(left_box, right_box)
                step_projection_model = (
                    _contact_projection_model_for_args(
                        args=args,
                        object_box=object_contact_box,
                        object_axis_unit_world=object_axis_source_row["object_axis_unit_world"],
                        projection_unit_world=step_closing_unit,
                        side_length=side_length,
                    )
                    if step_closing_unit is not None
                    else {"valid": False, "status": "FAIL_INVALID_STEP_CLOSING_AXIS"}
                )
                step_center_offset_row: dict[str, Any] = {}
                if step_closing_unit is not None and left_box.get("center") is not None and right_box.get("center") is not None:
                    step_object_center = (
                        np.asarray(object_contact_box["center"], dtype=np.float64).reshape(3)
                        if object_contact_box.get("bbox_valid") and object_contact_box.get("center") is not None
                        else None
                    )
                    step_center_offset_row = _finger_object_center_offset_row(
                        left_center=np.asarray(left_box["center"], dtype=np.float64).reshape(3),
                        right_center=np.asarray(right_box["center"], dtype=np.float64).reshape(3),
                        object_center=step_object_center,
                        closing_unit=step_closing_unit,
                    )
                if args.moving_fingers == "both":
                    reachability_expected_fingers = [contact_targets["left_finger"], contact_targets["right_finger"]]
                else:
                    reachability_expected_fingers = [contact_targets[f"{args.moving_fingers}_finger"]]
                target_contact_reachability_rows.append(
                    _live_target_reachability_row(
                        phase=phase,
                        step=phase_step,
                        left_box=left_box,
                        right_box=right_box,
                        object_contact_box=object_contact_box,
                        object_projection_model=step_projection_model,
                        contact_rows=step_contact_rows,
                        object_path=object_path,
                        expected_finger_paths=reachability_expected_fingers,
                        table_path=args.object_tabletop_reference_path,
                        contact_distance=float(args.object_contact_offset or 0.0)
                        + float(args.proxy_contact_offset or 0.0),
                        use_oriented_finger_boxes=args.finger_gap_projection_model == "oriented_box",
                    )
                )
                if (
                    args.diagnostic_held_object_mode == "follow_after_bilateral_contact"
                    and diagnostic_t_gripper_object is None
                    and diagnostic_held_object_row is not None
                ):
                    from pxr import UsdGeom

                    if args.moving_fingers == "both":
                        expected_hold_finger_paths = [contact_targets["left_finger"], contact_targets["right_finger"]]
                    else:
                        expected_hold_finger_paths = [contact_targets[f"{args.moving_fingers}_finger"]]
                    settle_contact_row = _target_contact_hits_for_phase(
                        rows=contact_pair_rows,
                        object_path=object_path,
                        expected_finger_paths=expected_hold_finger_paths,
                        phase="settle",
                    )
                    trigger_row = _target_contact_hits_for_phase(
                        rows=contact_pair_rows,
                        object_path=object_path,
                        expected_finger_paths=expected_hold_finger_paths,
                        phase="close",
                    )
                    trigger_row["settle_contact_before_close"] = bool(settle_contact_row["triggered"])
                    trigger_row["settle_contact_summary"] = settle_contact_row
                    if trigger_row["triggered"] and not trigger_row["settle_contact_before_close"]:
                        t_world_object = _world_matrix(UsdGeom, stage, object_path)
                        t_world_gripper = _world_matrix(UsdGeom, stage, args.object_gripper_frame)
                        diagnostic_t_gripper_object = np.linalg.inv(t_world_gripper) @ t_world_object
                        diagnostic_held_object_row.update(
                            {
                                "status": "DIAGNOSTIC_NOT_DYNAMIC_GRASP_PROOF",
                                "trigger_step": int(step),
                                "trigger_contact_summary": trigger_row,
                                "frame_features_at_trigger": _diagnostic_object_frame_features(
                                    UsdGeom, stage, object_path
                                ),
                                "t_gripper_object_at_trigger": diagnostic_t_gripper_object.tolist(),
                            }
                        )
                rows.append(
                    {
                        "phase": phase,
                        "step": phase_step,
                        "global_close_step": int(step),
                        "object_center_x": float(object_center[0]),
                        "object_center_y": float(object_center[1]),
                        "object_center_z": float(object_center[2]),
                        **diagnostic_frame_features,
                        "object_displacement": displacement,
                        **_finger_qpos_values(qpos, dof_names, finger_dof_names),
                        "tracking_controlled_max_abs_error": step_tracking["controlled"]["max_abs_error"],
                        "tracking_controlled_rms_error": step_tracking["controlled"]["rms_error"],
                        "target_limit_controlled_max_violation": target_limit["controlled"]["max_violation"],
                        "pre_step_tracking_controlled_max_abs_error": pre_step_tracking["controlled"]["max_abs_error"],
                        "pre_step_tracking_controlled_rms_error": pre_step_tracking["controlled"]["rms_error"],
                        "tracking_gripper_max_abs_error": step_tracking["gripper"]["max_abs_error"],
                        "tracking_gripper_rms_error": step_tracking["gripper"]["rms_error"],
                        "target_limit_gripper_max_violation": target_limit["gripper"]["max_violation"],
                        "pre_step_tracking_gripper_max_abs_error": pre_step_tracking["gripper"]["max_abs_error"],
                        "pre_step_tracking_gripper_rms_error": pre_step_tracking["gripper"]["rms_error"],
                        "tracking_left_arm_max_abs_error": step_tracking.get("left_arm", {}).get("max_abs_error"),
                        "tracking_left_arm_rms_error": step_tracking.get("left_arm", {}).get("rms_error"),
                        "target_limit_left_arm_max_violation": target_limit.get("left_arm", {}).get("max_violation"),
                        "pre_step_tracking_left_arm_max_abs_error": pre_step_tracking.get("left_arm", {}).get(
                            "max_abs_error"
                        ),
                        "pre_step_tracking_left_arm_rms_error": pre_step_tracking.get("left_arm", {}).get("rms_error"),
                        "object_width_stop_active": bool(width_stop_row.get("active")),
                        "object_width_stop_status": width_stop_row.get("status"),
                        "object_width_stop_mode": width_stop_row.get("mode"),
                        "object_width_stop_current_center_gap_m": width_stop_row.get("current_center_gap_m"),
                        "object_width_stop_threshold_m": width_stop_row.get("stop_center_gap_m"),
                        "object_width_stop_projected_inner_gap_m": (
                            width_stop_row.get("projected_inner_gap", {}) or {}
                        ).get("finger_inner_gap_m"),
                        "object_width_stop_projected_threshold_m": width_stop_row.get("projected_stop_gap_m"),
                        "diagnostic_loaded_clamp_squeeze_active": bool(loaded_clamp_squeeze_row.get("active")),
                        "diagnostic_loaded_clamp_squeeze_status": loaded_clamp_squeeze_row.get("status"),
                        "diagnostic_loaded_clamp_squeeze_depth_m": loaded_clamp_squeeze_row.get("squeeze_depth_m"),
                        "diagnostic_loaded_clamp_squeeze_per_finger_delta_m": loaded_clamp_squeeze_row.get(
                            "per_finger_target_delta_m"
                        ),
                        "diagnostic_loaded_clamp_squeeze_clipped": loaded_clamp_squeeze_row.get("clipped"),
                        "finger_center_distance": _gap_metrics(left_box, right_box).get("center_distance"),
                        **_finger_center_row(left_box, right_box),
                        "object_offset_along_closing_axis_m": step_center_offset_row.get(
                            "object_offset_along_closing_axis_m"
                        ),
                        "object_cross_closing_axis_offset_norm_m": step_center_offset_row.get(
                            "object_cross_closing_axis_offset_norm_m"
                        ),
                        "object_cross_closing_axis_offset_x_m": step_center_offset_row.get(
                            "object_cross_closing_axis_offset_x_m"
                        ),
                        "object_cross_closing_axis_offset_y_m": step_center_offset_row.get(
                            "object_cross_closing_axis_offset_y_m"
                        ),
                        "object_cross_closing_axis_offset_z_m": step_center_offset_row.get(
                            "object_cross_closing_axis_offset_z_m"
                        ),
                        **_axis_probe_row(
                            axis=axis,
                            left_box=left_box,
                            right_box=right_box,
                            object_box=object_box,
                            target_finger_box=left_box if args.moving_fingers != "right" else right_box,
                        ),
                    }
                )
        finally:
            _finish_contact_pair_trace(stage, trace_state)

        object_final_box = object_latest_box
        object_final_contact_box = object_contact_latest_box
        object_final_center = object_latest_center
        start_finger_object_alignment = _finger_object_alignment_diagnostic(
            label="replay_start",
            left_box=replay_start_left_box,
            right_box=replay_start_right_box,
            object_box=object_contact_reset_box,
            gap_axis=axis,
            gap_axis_name=axis_name,
            reference_contact_center_world=object_placement_row.get("base_center"),
            object_long_axis_world=object_axis_source_row.get("object_axis_unit_world"),
            object_projected_interval=object_contact_projected_interval,
            object_projection_model=object_contact_projection_model,
            use_oriented_finger_boxes=args.finger_gap_projection_model == "oriented_box",
        )
        final_reference_contact_center = _translated_reference_contact_center(
            reference_contact_center_world=object_placement_row.get("base_center"),
            reset_object_box=object_contact_reset_box,
            final_object_box=object_final_contact_box,
        )
        final_closing_unit = _closing_unit_from_finger_boxes(left_box, right_box)
        final_contact_projection_model = (
            _contact_projection_model_for_args(
                args=args,
                object_box=object_final_contact_box,
                object_axis_unit_world=object_axis_source_row["object_axis_unit_world"],
                projection_unit_world=final_closing_unit,
                side_length=side_length,
            )
            if final_closing_unit is not None
            else {"valid": False, "status": "FAIL_INVALID_FINAL_CLOSING_AXIS"}
        )
        final_contact_projected_interval = (
            tuple(float(v) for v in final_contact_projection_model["object_interval_m"])
            if final_contact_projection_model.get("valid")
            else None
        )
        final_finger_object_alignment = _finger_object_alignment_diagnostic(
            label="replay_final",
            left_box=left_box,
            right_box=right_box,
            object_box=object_final_contact_box,
            gap_axis=axis,
            gap_axis_name=axis_name,
            reference_contact_center_world=final_reference_contact_center.get("reference_contact_center_world_m"),
            object_long_axis_world=object_axis_source_row.get("object_axis_unit_world"),
            object_projected_interval=final_contact_projected_interval,
            object_projection_model=final_contact_projection_model,
            use_oriented_finger_boxes=args.finger_gap_projection_model == "oriented_box",
        )
        tracking_summary = _summarize_tracking_errors(tracking_rows, tracking_groups, dof_names)
        diagnostic_phase_names = ("close", "post_close_hold", "post_close_lift", "post_close_lift_hold")
        tracking_summary_by_phase = {
            phase: _summarize_tracking_errors(
                [row for row in tracking_rows if row.get("phase") == phase],
                tracking_groups,
                dof_names,
            )
            for phase in diagnostic_phase_names
        }
        pre_step_tracking_summary = _summarize_tracking_errors(pre_step_tracking_rows, tracking_groups, dof_names)
        command_smoothness_gate = _command_delta_distribution(
            tracking_rows=tracking_rows,
            groups=tracking_groups,
            dof_names=dof_names,
            effective_target_dt=effective_control_dt,
            max_abs_target_velocity=args.max_command_target_velocity,
        )
        command_smoothness_ok = bool(command_smoothness_gate["pass"])
        target_limit_summary = _summarize_target_limit_violations(target_limit_rows, tracking_groups, dof_names)
        target_limit_ok = bool(target_limit_summary.get("controller_ready", True))
        effective_max_tracking_error = args.max_post_step_controlled_tracking_error
        if effective_max_tracking_error is None and hdf5_target_sequence is not None:
            effective_max_tracking_error = 0.02
        controller_tracking_gate = _controller_tracking_gate(
            tracking_summary=tracking_summary,
            max_controlled_error=effective_max_tracking_error,
        )
        controller_tracking_gate_by_phase = {
            phase: _controller_tracking_gate(
                tracking_summary=summary,
                max_controlled_error=effective_max_tracking_error,
            )
            for phase, summary in tracking_summary_by_phase.items()
        }
        controller_tracking_ok = bool(controller_tracking_gate["pass"])
        object_displacement = float(np.linalg.norm(object_final_center - object_initial_center))
        total_object_displacement = float(np.linalg.norm(object_final_center - object_reset_center))
        object_lift = float(object_final_center[2] - object_initial_center[2])
        object_width_stop_active_steps = sum(1 for row in object_width_stop_rows if row.get("active"))
        object_width_stop_first_active_step = next(
            (row["step"] for row in object_width_stop_rows if row.get("active")),
            None,
        )
        object_width_stop_summary = {
            "enabled": bool(args.enforce_object_width_finger_stop),
            "rows": object_width_stop_rows,
            "row_count": len(object_width_stop_rows),
            "active_steps": int(object_width_stop_active_steps),
            "first_active_step": object_width_stop_first_active_step,
        }
        diagnostic_loaded_clamp_squeeze_active_steps = sum(
            1 for row in diagnostic_loaded_clamp_squeeze_rows if row.get("active")
        )
        diagnostic_loaded_clamp_squeeze_summary = {
            "enabled": bool(float(args.diagnostic_loaded_clamp_squeeze_depth) > 0.0),
            "diagnostic_only": True,
            "formal_hdf5_close_sequence_modified": False,
            "squeeze_depth_m": float(args.diagnostic_loaded_clamp_squeeze_depth),
            "row_count": len(diagnostic_loaded_clamp_squeeze_rows),
            "active_steps": int(diagnostic_loaded_clamp_squeeze_active_steps),
            "first_active_global_step": next(
                (
                    row.get("global_close_step")
                    for row in diagnostic_loaded_clamp_squeeze_rows
                    if row.get("active")
                ),
                None,
            ),
            "any_clipped": any(bool(row.get("clipped")) for row in diagnostic_loaded_clamp_squeeze_rows),
            "rows": diagnostic_loaded_clamp_squeeze_rows,
            "notes": (
                "Diagnostic post-close loaded clamp squeeze. It changes only appended post-close hold/lift "
                "targets and keeps the original HDF5 close replay unchanged."
            ),
        }
        target_contact_reachability_audit = _summarize_target_reachability(target_contact_reachability_rows)
        target_contact_reachability_ok = bool(target_contact_reachability_audit["pass"])
        contact_motion_policy = (
            "not_required_for_bilateral_closure"
            if args.moving_fingers == "both"
            else "single_finger_push_requires_minimum_motion"
        )
        contact_motion_ok = bool(args.moving_fingers == "both" or object_displacement >= args.min_contact_motion)
        object_lift_gate = _object_lift_gate(
            object_lift=object_lift,
            min_object_lift=float(args.min_object_lift),
        )
        object_lift_ok = bool(object_lift_gate["pass"])
        no_explosion_ok = bool(finite_motion and max_displacement <= args.max_object_displacement)
        bottle_runtime_composition_ok = bool(bottle_runtime_composition_gate["pass"])
        bottle_grasp_semantics_ok = bool(bottle_grasp_semantics_gate["pass"])
        overall_pass = bool(
            bottle_runtime_composition_ok
            and bottle_grasp_semantics_ok
            and contact_motion_ok
            and object_lift_ok
            and no_explosion_ok
            and target_limit_ok
            and controller_tracking_ok
            and command_smoothness_ok
        )
        if args.moving_fingers == "both":
            expected_finger_paths = [contact_targets["left_finger"], contact_targets["right_finger"]]
        else:
            expected_finger_paths = [contact_targets[f"{args.moving_fingers}_finger"]]
        contact_summary = _summarize_contact_pairs(
            contact_pair_rows=contact_pair_rows,
            object_path=object_path,
            expected_finger_paths=expected_finger_paths,
            diagnostic_contact_paths=[support_plane_row["path"]] if support_plane_row else None,
            same_side_robot_root=robot_root_for_side(args.contact_proxy_profile, args.side),
            other_side_robot_root=robot_root_for_side(
                args.contact_proxy_profile, "right" if args.side == "left" else "left"
            ),
        )
        phase_contact_summaries = {
            phase: _summarize_contact_pairs(
                contact_pair_rows=[row for row in contact_pair_rows if row.get("phase") == phase],
                object_path=object_path,
                expected_finger_paths=expected_finger_paths,
                diagnostic_contact_paths=[support_plane_row["path"]] if support_plane_row else None,
                same_side_robot_root=robot_root_for_side(args.contact_proxy_profile, args.side),
                other_side_robot_root=robot_root_for_side(
                    args.contact_proxy_profile, "right" if args.side == "left" else "left"
                ),
            )
            for phase in diagnostic_phase_names
        }
        if args.moving_fingers == "both":
            target_contact_ok = bool(contact_summary["all_expected_fingers_target_contact_pair_found"])
        else:
            target_contact_ok = bool(contact_summary["target_contact_pair_found"])
        diagnostic_force_target_overlap_contact_pipeline_gate = {
            "enabled": bool(diagnostic_force_target_overlap_row.get("enabled")),
            "applied": bool(diagnostic_force_target_overlap_row.get("applied")),
            "formal_gate_allowed": False,
            "pass": bool(
                diagnostic_force_target_overlap_row.get("applied")
                and contact_summary.get("target_contact_pair_found")
            ),
            "status": "SKIPPED_NO_FORCED_OVERLAP_DIAGNOSTIC",
            "target_contact_pair_found": bool(contact_summary.get("target_contact_pair_found")),
            "all_expected_fingers_target_contact_pair_found": bool(
                contact_summary.get("all_expected_fingers_target_contact_pair_found")
            ),
            "first_target_contact_pair": contact_summary.get("first_target_contact_pair"),
            "first_target_contact_phase": contact_summary.get("first_target_contact_phase"),
            "first_target_contact_found_phase": contact_summary.get("first_target_contact_found_phase"),
            "notes": (
                "Positive-control result only. It can prove the selected contact pipeline can report contact, "
                "but it must never set target_contact_ok, physical_grasp_gate.pass, or overall_pass."
            ),
        }
        if diagnostic_force_target_overlap_row.get("applied"):
            diagnostic_force_target_overlap_contact_pipeline_gate["status"] = (
                "PASS_FORCED_OVERLAP_CONTACT_PIPELINE_REPORTED"
                if diagnostic_force_target_overlap_contact_pipeline_gate["pass"]
                else "FAIL_FORCED_OVERLAP_NO_TARGET_CONTACT_REPORT"
            )
        cross_side_overlap_blocks_gate = bool(
            args.moving_fingers == "both" and cross_side_proxy_overlap["overlap_detected"]
        )
        trace_pair_ok = bool(
            (not args.trace_contact_pairs) or (target_contact_ok and not cross_side_overlap_blocks_gate)
        )
        non_target_contact_gate = _non_target_contact_gate(
            contact_summary=contact_summary,
            fail_on_non_target=bool(args.trace_contact_pairs and args.fail_on_non_target_object_contact),
            allowed_categories=list(args.allowed_non_target_object_contact_category),
        )
        non_target_object_contact_ok = bool(non_target_contact_gate["pass"])
        workcell_contact_policy_gate = _workcell_contact_policy_gate(
            contact_summary=contact_summary,
            object_path=object_path,
            policy=workcell_contact_policy,
        )
        workcell_contact_policy_ok = bool(workcell_contact_policy_gate["pass"])
        active_target_contact_gate = _active_target_contact_gate(
            contact_summary=contact_summary,
            require_active_target_contact=bool(args.trace_contact_pairs and args.require_active_target_contact),
            already_in_contact_setup=bool(args.already_in_contact_setup),
        )
        bilateral_grasp_formation_gate = _bilateral_grasp_formation_gate(
            rows=rows,
            contact_summary=contact_summary,
            moving_fingers=str(args.moving_fingers),
            gap_axis_name=axis_name,
            min_contact_steps=int(args.bilateral_grasp_min_contact_steps),
            min_nonzero_impulse_steps=int(args.bilateral_grasp_min_nonzero_impulse_steps),
            max_impulse_ratio=float(args.bilateral_grasp_max_impulse_ratio),
            max_prelift_lateral_sweep=float(args.bilateral_grasp_max_prelift_lateral_sweep),
            prelift_gripper_z_delta=float(args.bilateral_grasp_prelift_z_delta),
        )
        bilateral_grasp_formation_gate_by_phase = {
            phase: _bilateral_grasp_formation_gate(
                rows=[row for row in rows if row.get("phase") == phase],
                contact_summary=summary,
                moving_fingers=str(args.moving_fingers),
                gap_axis_name=axis_name,
                min_contact_steps=int(args.bilateral_grasp_min_contact_steps),
                min_nonzero_impulse_steps=int(args.bilateral_grasp_min_nonzero_impulse_steps),
                max_impulse_ratio=float(args.bilateral_grasp_max_impulse_ratio),
                max_prelift_lateral_sweep=float(args.bilateral_grasp_max_prelift_lateral_sweep),
                prelift_gripper_z_delta=float(args.bilateral_grasp_prelift_z_delta),
            )
            for phase, summary in phase_contact_summaries.items()
        }
        post_close_hold_diagnostic_gate = {
            "enabled": bool(post_close_hold_steps > 0),
            "phase": "post_close_hold",
            "formal_replay_targets_modified": False,
            "hdf5_close_sequence_preserved": True,
            "held_target_source": "final_hdf5_close_target" if hdf5_target_sequence is not None else "synthetic_close_target",
            "hold_steps": int(post_close_hold_steps),
            "hold_physics_steps": int(post_close_hold_steps) * int(args.hdf5_replay_target_hold_steps),
            "bilateral_gate": bilateral_grasp_formation_gate_by_phase.get("post_close_hold"),
            "controller_tracking_gate": controller_tracking_gate_by_phase.get("post_close_hold"),
            "non_target_contact_ok": non_target_object_contact_ok,
            "workcell_contact_policy_ok": workcell_contact_policy_ok,
            "no_explosion_ok": no_explosion_ok,
            "pass": False,
            "status": "SKIPPED_NO_POST_CLOSE_HOLD",
            "notes": (
                "Diagnostic only. A pass means the final close target can maintain quasi-static bilateral "
                "contact after the recorded HDF5 window; it does not convert the original HDF5 close window "
                "into a formal replay pass."
            ),
        }
        if post_close_hold_steps > 0:
            post_bilateral_ok = bool(
                (post_close_hold_diagnostic_gate.get("bilateral_gate") or {}).get("pass")
            )
            post_tracking_ok = bool(
                (post_close_hold_diagnostic_gate.get("controller_tracking_gate") or {}).get("pass")
            )
            post_pass = bool(
                post_bilateral_ok
                and post_tracking_ok
                and non_target_object_contact_ok
                and workcell_contact_policy_ok
                and no_explosion_ok
            )
            post_close_hold_diagnostic_gate.update(
                {
                    "pass": post_pass,
                    "status": "PASS_QUASI_STATIC_CLAMP_HOLD_DIAGNOSTIC"
                    if post_pass
                    else "FAIL_QUASI_STATIC_CLAMP_HOLD_DIAGNOSTIC",
                    "post_bilateral_ok": post_bilateral_ok,
                    "post_tracking_ok": post_tracking_ok,
                }
            )
        prelift_static_grasp_gate = _prelift_static_grasp_gate(
            hold_rows=[row for row in rows if row.get("phase") == "post_close_hold"],
            hold_bilateral_gate=bilateral_grasp_formation_gate_by_phase.get("post_close_hold"),
            hold_tracking_gate=controller_tracking_gate_by_phase.get("post_close_hold"),
            min_hold_steps=int(post_close_hold_steps),
            min_tail_bilateral_steps=int(args.bilateral_grasp_min_contact_steps),
            min_each_finger_nonzero_impulse_steps=int(args.bilateral_grasp_min_nonzero_impulse_steps),
            max_object_xy_sweep=float(args.bilateral_grasp_max_prelift_lateral_sweep),
            max_object_z_delta=float(args.bilateral_grasp_prelift_z_delta),
        )
        post_close_hold_diagnostic_gate["prelift_static_grasp_gate"] = prelift_static_grasp_gate
        early_geometric_escape_gate = _early_geometric_escape_gate(
            object_width_stop_rows=object_width_stop_rows,
            bilateral_grasp_formation_gate=bilateral_grasp_formation_gate,
        )
        bilateral_grasp_failure_diagnostic = _bilateral_grasp_failure_diagnostic(
            bilateral_grasp_formation_gate=bilateral_grasp_formation_gate,
            contact_summary=contact_summary,
            start_alignment=start_finger_object_alignment,
            final_alignment=final_finger_object_alignment,
        )
        contact_landmark_alignment = _timeseries_gripper_object_alignment_samples(
            rows=rows,
            contact_summary=contact_summary,
            moving_fingers=str(args.moving_fingers),
        )
        lift_transport_gate = _lift_transport_gate(
            rows=rows,
            object_lift_gate=object_lift_gate,
            contact_summary=(
                phase_contact_summaries.get("post_close_lift")
                if post_close_lift_step_count > 0
                else contact_summary
            )
            or contact_summary,
            min_object_lift=float(args.min_object_lift),
            diagnostic_held_object_mode=str(args.diagnostic_held_object_mode),
        )
        lift_contact_wrench_patch_audit = _lift_contact_wrench_patch_audit(
            rows=rows,
            contact_pair_rows=contact_pair_rows,
            object_path=object_path,
            expected_finger_paths=expected_finger_paths,
            eval_phase="post_close_lift" if post_close_lift_step_count > 0 else "close",
        )
        fixed_reference_grasp_geometry_gate = _fixed_reference_grasp_geometry_gate(
            tabletop_reference_contract=tabletop_reference_contract,
            object_contact_reset_box=object_contact_reset_box,
            object_final_contact_box=object_final_contact_box,
            start_alignment=start_finger_object_alignment,
            final_alignment=final_finger_object_alignment,
            lift_contact_wrench_patch_audit=lift_contact_wrench_patch_audit,
            fixed_reference_required=bool(args.object_tabletop_reference_path),
            lift_required=bool(float(args.min_object_lift) > 0.0 and post_close_lift_step_count > 0),
        )
        post_close_lift_diagnostic_gate = {
            "enabled": bool(post_close_lift_step_count > 0),
            "phase": "post_close_lift",
            "source": args.post_close_lift_source,
            "diagnostic_only": True,
            "formal_close_replay_success": False,
            "hdf5_close_sequence_preserved": True,
            "lift_summary": post_close_lift_summary,
            "pre_lift_hold_gate": post_close_hold_diagnostic_gate,
            "prelift_static_grasp_gate": prelift_static_grasp_gate,
            "bilateral_gate": bilateral_grasp_formation_gate_by_phase.get("post_close_lift"),
            "controller_tracking_gate": controller_tracking_gate_by_phase.get("post_close_lift"),
            "lift_transport_gate": lift_transport_gate,
            "fixed_reference_grasp_geometry_gate": fixed_reference_grasp_geometry_gate,
            "object_lift_gate": object_lift_gate,
            "non_target_contact_ok": non_target_object_contact_ok,
            "workcell_contact_policy_ok": workcell_contact_policy_ok,
            "no_explosion_ok": no_explosion_ok,
            "pass": False,
            "status": "SKIPPED_NO_POST_CLOSE_LIFT",
            "notes": (
                "Diagnostic only. A pass means the already-held final clamp can transport the bottle through "
                "the appended lift phase. It does not convert the original HDF5 close window into a formal pass."
            ),
        }
        if post_close_lift_step_count > 0:
            lift_bilateral_ok = bool(
                (post_close_lift_diagnostic_gate.get("bilateral_gate") or {}).get("pass")
            )
            lift_tracking_ok = bool(
                (post_close_lift_diagnostic_gate.get("controller_tracking_gate") or {}).get("pass")
            )
            lift_transport_ok = bool(lift_transport_gate.get("pass"))
            pre_lift_hold_ok = bool(post_close_hold_diagnostic_gate.get("pass"))
            fixed_reference_geometry_ok = bool(fixed_reference_grasp_geometry_gate.get("pass"))
            lift_pass = bool(
                pre_lift_hold_ok
                and lift_bilateral_ok
                and lift_tracking_ok
                and lift_transport_ok
                and fixed_reference_geometry_ok
                and object_lift_ok
                and non_target_object_contact_ok
                and workcell_contact_policy_ok
                and no_explosion_ok
            )
            if not pre_lift_hold_ok:
                lift_status = "FAIL_LIFT_HOLD_DID_NOT_PASS"
            elif not lift_bilateral_ok:
                lift_status = "FAIL_LIFT_BILATERAL_CONTACT_NOT_PERSISTENT"
            elif str(lift_transport_gate.get("status")) == "FAIL_STRICT_OBJECT_LIFT_CLEARANCE" or not object_lift_ok:
                lift_status = "FAIL_LIFT_STRICT_OBJECT_CLEARANCE"
            elif str(lift_transport_gate.get("status")) == "FAIL_TABLE_STILL_LOAD_BEARING":
                lift_status = "FAIL_LIFT_TABLE_STILL_LOAD_BEARING"
            elif not lift_transport_ok:
                lift_status = "FAIL_LIFT_OBJECT_DID_NOT_FOLLOW_GRIPPER"
            elif not fixed_reference_geometry_ok:
                lift_status = str(fixed_reference_grasp_geometry_gate.get("status"))
            elif not lift_tracking_ok:
                lift_status = "FAIL_LIFT_CONTROLLER_TRACKING"
            elif not non_target_object_contact_ok or not workcell_contact_policy_ok:
                lift_status = "FAIL_LIFT_FORBIDDEN_NON_TARGET_CONTACT"
            elif not no_explosion_ok:
                lift_status = "FAIL_LIFT_OBJECT_MOTION_NOT_FINITE_OR_TOO_LARGE"
            else:
                lift_status = "PASS_LIFT_FROM_STABLE_CLAMP_DIAGNOSTIC"
            post_close_lift_diagnostic_gate.update(
                {
                    "pass": lift_pass,
                    "status": lift_status,
                    "pre_lift_hold_ok": pre_lift_hold_ok,
                    "lift_bilateral_ok": lift_bilateral_ok,
                    "lift_tracking_ok": lift_tracking_ok,
                    "lift_transport_ok": lift_transport_ok,
                    "fixed_reference_geometry_ok": fixed_reference_geometry_ok,
                    "object_lift_ok": object_lift_ok,
                }
            )
        active_target_contact_ok = bool(active_target_contact_gate["pass"])
        bilateral_grasp_formation_ok = bool(bilateral_grasp_formation_gate["pass"])
        early_geometric_escape_ok = bool(early_geometric_escape_gate["pass"])
        active_grasp_geometry_precondition_ok = bool(active_grasp_geometry_precondition["pass"])
        open_finger_object_height_alignment_ok = bool(open_finger_object_height_alignment["pass"])
        tabletop_reference_contract_ok = bool(tabletop_reference_contract["pass"])
        fixed_reference_grasp_geometry_ok = bool(fixed_reference_grasp_geometry_gate.get("pass"))
        trace_pair_ok = bool(
            trace_pair_ok and non_target_object_contact_ok and workcell_contact_policy_ok and active_target_contact_ok
            and bilateral_grasp_formation_ok
            and active_grasp_geometry_precondition_ok and open_finger_object_height_alignment_ok
            and tabletop_reference_contract_ok
            and fixed_reference_grasp_geometry_ok
        )
        tracking_spike_packet = _tracking_spike_packet(
            tracking_rows=tracking_rows,
            tracking_summary=tracking_summary,
            contact_pair_rows=contact_pair_rows,
            dof_names=dof_names,
            runtime_limits=runtime_limits,
            physics_dt=float(args.physics_dt),
            target_hold_steps=int(args.hdf5_replay_target_hold_steps),
            arm_gain_override={"kp": args.arm_kp, "kd": args.arm_kd},
            finger_gain_override={"kp": args.finger_kp, "kd": args.finger_kd},
            finger_dof_names=finger_dof_names,
        )
        drive_authority_audit = _drive_authority_audit(
            tracking_spike=tracking_spike_packet,
            runtime_drive_profile=runtime_drive_profile,
        )
        physical_grasp_gate = {
            "pass": bool(
                bottle_runtime_composition_ok
                and bottle_grasp_semantics_ok
                and contact_motion_ok
                and object_lift_ok
                and no_explosion_ok
                and target_limit_ok
                and target_contact_ok
                and non_target_object_contact_ok
                and workcell_contact_policy_ok
                and active_target_contact_ok
                and bilateral_grasp_formation_ok
                and active_grasp_geometry_precondition_ok
                and open_finger_object_height_alignment_ok
                and tabletop_reference_contract_ok
            ),
            "status": "PASS_PHYSICAL_GRASP_SEMANTICS" if (
                bottle_runtime_composition_ok
                and bottle_grasp_semantics_ok
                and contact_motion_ok
                and object_lift_ok
                and no_explosion_ok
                and target_limit_ok
                and target_contact_ok
                and non_target_object_contact_ok
                and workcell_contact_policy_ok
                and active_target_contact_ok
                and bilateral_grasp_formation_ok
                and active_grasp_geometry_precondition_ok
                and open_finger_object_height_alignment_ok
                and tabletop_reference_contract_ok
                and fixed_reference_grasp_geometry_ok
            ) else "FAIL_PHYSICAL_GRASP_SEMANTICS",
            "notes": (
                "This gate intentionally excludes controller replay fidelity. It answers whether the "
                "object/contact/lift/workcell semantics passed for the executed physics run."
            ),
            "target_contact_ok": target_contact_ok,
            "contact_motion_ok": contact_motion_ok,
            "object_lift_ok": object_lift_ok,
            "object_lift_gate": object_lift_gate,
            "no_explosion_ok": no_explosion_ok,
            "target_limit_ok": target_limit_ok,
            "workcell_contact_policy_ok": workcell_contact_policy_ok,
            "active_target_contact_ok": active_target_contact_ok,
            "bilateral_grasp_formation_ok": bilateral_grasp_formation_ok,
            "bilateral_grasp_formation_gate": bilateral_grasp_formation_gate,
            "bilateral_grasp_formation_gate_by_phase": bilateral_grasp_formation_gate_by_phase,
            "bottle_runtime_composition_ok": bottle_runtime_composition_ok,
            "bottle_grasp_semantics_ok": bottle_grasp_semantics_ok,
            "active_grasp_geometry_precondition_ok": active_grasp_geometry_precondition_ok,
            "open_finger_object_height_alignment_ok": open_finger_object_height_alignment_ok,
            "tabletop_reference_contract_ok": tabletop_reference_contract_ok,
            "fixed_reference_grasp_geometry_ok": fixed_reference_grasp_geometry_ok,
        }
        tabletop_grasp_contact_gate = {
            "pass": bool(
                bottle_runtime_composition_ok
                and bottle_grasp_semantics_ok
                and contact_motion_ok
                and no_explosion_ok
                and target_limit_ok
                and target_contact_ok
                and non_target_object_contact_ok
                and workcell_contact_policy_ok
                and active_target_contact_ok
                and bilateral_grasp_formation_ok
                and active_grasp_geometry_precondition_ok
                and open_finger_object_height_alignment_ok
                and tabletop_reference_contract_ok
                and fixed_reference_grasp_geometry_ok
            ),
            "status": "PASS_TABLETOP_GRASP_CONTACT" if (
                bottle_runtime_composition_ok
                and bottle_grasp_semantics_ok
                and contact_motion_ok
                and no_explosion_ok
                and target_limit_ok
                and target_contact_ok
                and non_target_object_contact_ok
                and workcell_contact_policy_ok
                and active_target_contact_ok
                and bilateral_grasp_formation_ok
                and active_grasp_geometry_precondition_ok
                and open_finger_object_height_alignment_ok
                and tabletop_reference_contract_ok
                and fixed_reference_grasp_geometry_ok
            ) else "FAIL_TABLETOP_GRASP_CONTACT",
            "lift_required": bool(object_lift_gate["required"]),
            "lift_gate_status": object_lift_gate["status"],
            "notes": (
                "This gate validates tabletop open-to-close grasp contact and placement semantics. "
                "It intentionally does not require object lift; use a positive --min-object-lift and a "
                "trajectory with a lift phase for dynamic lift validation."
            ),
            "target_contact_ok": target_contact_ok,
            "contact_motion_ok": contact_motion_ok,
            "no_explosion_ok": no_explosion_ok,
            "target_limit_ok": target_limit_ok,
            "workcell_contact_policy_ok": workcell_contact_policy_ok,
            "active_target_contact_ok": active_target_contact_ok,
            "bilateral_grasp_formation_ok": bilateral_grasp_formation_ok,
                "bilateral_grasp_formation_gate": bilateral_grasp_formation_gate,
                "bilateral_grasp_formation_gate_by_phase": bilateral_grasp_formation_gate_by_phase,
                "bottle_runtime_composition_ok": bottle_runtime_composition_ok,
            "bottle_grasp_semantics_ok": bottle_grasp_semantics_ok,
            "active_grasp_geometry_precondition_ok": active_grasp_geometry_precondition_ok,
            "open_finger_object_height_alignment_ok": open_finger_object_height_alignment_ok,
            "tabletop_reference_contract_ok": tabletop_reference_contract_ok,
            "fixed_reference_grasp_geometry_ok": fixed_reference_grasp_geometry_ok,
        }
        controller_replay_fidelity_gate = {
            **controller_tracking_gate,
            "timing_alignment_pass": bool(hdf5_timing_alignment["pass"]),
            "target_limit_ok": target_limit_ok,
            "command_smoothness_gate": command_smoothness_gate,
            "drive_authority_audit": drive_authority_audit,
            "tracking_spike": tracking_spike_packet,
            "notes": (
                "This gate checks whether PhysX articulation drives actually tracked the formal replay "
                "targets. It remains a hard overall gate; do not hide tracking failure by state-setting."
            ),
        }
        formal_replay_feasibility_ok = bool(
            hdf5_timing_alignment["pass"]
            and target_limit_ok
            and controller_tracking_ok
            and command_smoothness_ok
        )
        if not hdf5_timing_alignment["pass"]:
            formal_replay_feasibility_status = "FAIL_HDF5_TIMING_ALIGNMENT"
        elif not target_limit_ok:
            formal_replay_feasibility_status = "FAIL_TARGET_OUTSIDE_RUNTIME_LIMITS"
        elif not command_smoothness_ok:
            formal_replay_feasibility_status = str(command_smoothness_gate["status"])
        elif not controller_tracking_ok:
            formal_replay_feasibility_status = str(controller_tracking_gate["status"])
        else:
            formal_replay_feasibility_status = "PASS_FORMAL_REPLAY_FEASIBILITY"
        formal_replay_feasibility_gate = {
            "pass": formal_replay_feasibility_ok,
            "status": formal_replay_feasibility_status,
            "timing_alignment_pass": bool(hdf5_timing_alignment["pass"]),
            "target_limit_ok": target_limit_ok,
            "controller_tracking_pass": controller_tracking_ok,
            "command_smoothness_pass": command_smoothness_ok,
            "formal_replay_uses_raw_zero_order_hold": args.hdf5_replay_substep_mode == "zero_order_hold",
            "diagnostic_smoothing_used_for_pass": False,
            "notes": (
                "This gate combines formal replay timing, target limits, command smoothness, and controller "
                "tracking. It reports feasibility only; it does not delete frames or smooth targets."
            ),
        }
        diagnostic_force_target_overlap_active = bool(diagnostic_force_target_overlap_row.get("applied"))
        overall_pass = bool(
            overall_pass
            and trace_pair_ok
            and formal_replay_feasibility_ok
            and not diagnostic_force_target_overlap_active
        )
        failure_reasons = []
        if diagnostic_force_target_overlap_active:
            failure_reasons.append("diagnostic_forced_overlap_not_formal_gate")
        if not contact_motion_ok:
            failure_reasons.append("contact_motion_below_threshold")
        if not object_lift_ok:
            failure_reasons.append("object_lift_below_threshold")
        if not bottle_runtime_composition_ok:
            failure_reasons.append("bottle_usd_runtime_composition_gate_failed")
        if not bottle_grasp_semantics_ok:
            failure_reasons.append("bottle_grasp_semantics_gate_failed")
        if not no_explosion_ok:
            failure_reasons.append("object_motion_exceeded_limit")
        if not trace_pair_ok:
            failure_reasons.append("contact_trace_gate_failed")
            if args.trace_contact_pairs and not target_contact_ok:
                if target_contact_reachability_ok:
                    failure_reasons.append("expected_finger_target_contact_incomplete")
                else:
                    failure_reasons.append("target_contact_reachability_audit_failed")
                    if (
                        target_contact_reachability_audit.get("status")
                        == "FAIL_1D_PROJECTION_OVERLAP_BUT_3D_AABB_SEPARATION"
                    ):
                        failure_reasons.append("closing_axis_projection_false_positive_3d_aabb_separation")
        if not active_target_contact_ok:
            failure_reasons.append("active_target_contact_gate_failed")
        if not bilateral_grasp_formation_ok:
            failure_reasons.append("bilateral_grasp_formation_failed")
        if not early_geometric_escape_ok:
            failure_reasons.append("object_escaped_capture_region_before_bilateral_contact")
        if not active_grasp_geometry_precondition_ok:
            failure_reasons.append("active_grasp_geometry_precondition_failed")
            if (
                active_grasp_geometry_precondition.get("status")
                == "FAIL_ACTIVE_FREE_SPACE_TRUE_CLOSING_AXIS_GEOMETRY_PRECONDITION"
            ):
                failure_reasons.append("true_closing_axis_gap_precondition_failed")
        if not open_finger_object_height_alignment_ok:
            failure_reasons.append("open_finger_object_height_alignment_failed")
        if not tabletop_reference_contract_ok:
            failure_reasons.append("tabletop_reference_contract_failed")
        if not fixed_reference_grasp_geometry_gate.get("pass") and fixed_reference_grasp_geometry_gate.get("required"):
            failure_reasons.append("fixed_reference_grasp_geometry_failed")
        if not workcell_contact_policy_ok:
            failure_reasons.append("workcell_contact_policy_gate_failed")
        if not target_limit_ok:
            failure_reasons.append("target_outside_runtime_limits")
        if not controller_tracking_ok:
            failure_reasons.append("post_step_controller_tracking_exceeded_threshold")
        if not command_smoothness_ok:
            failure_reasons.append("command_target_velocity_exceeded_threshold")
        if args.trace_contact_pairs:
            if cross_side_overlap_blocks_gate:
                contact_trace_status = "FAIL_CROSS_SIDE_PROXY_OVERLAP"
            elif not tabletop_reference_contract_ok:
                contact_trace_status = str(tabletop_reference_contract["status"])
            elif not fixed_reference_grasp_geometry_ok:
                contact_trace_status = str(fixed_reference_grasp_geometry_gate["status"])
            elif (
                args.moving_fingers == "both"
                and contact_summary.get("target_contact_pair_found")
                and not bilateral_grasp_formation_ok
            ):
                contact_trace_status = str(
                    early_geometric_escape_gate["status"]
                    if not early_geometric_escape_ok
                    else bilateral_grasp_formation_gate["status"]
                )
            elif not target_contact_ok:
                contact_trace_status = str(target_contact_reachability_audit["status"])
            elif not non_target_object_contact_ok:
                contact_trace_status = str(non_target_contact_gate["status"])
            elif not workcell_contact_policy_ok:
                contact_trace_status = str(workcell_contact_policy_gate["status"])
            elif not active_grasp_geometry_precondition_ok:
                contact_trace_status = str(active_grasp_geometry_precondition["status"])
            elif not open_finger_object_height_alignment_ok:
                contact_trace_status = str(open_finger_object_height_alignment["status"])
            elif not active_target_contact_ok:
                contact_trace_status = str(active_target_contact_gate["status"])
            elif not no_explosion_ok:
                contact_trace_status = "FAIL_OBJECT_EJECTION"
            else:
                contact_trace_status = (
                    "PASS_SINGLE_FINGER_CONTACT_ISOLATION"
                    if args.moving_fingers != "both"
                    else "PASS_BILATERAL_CONTACT_CANDIDATE"
                )
        else:
            contact_trace_status = "NOT_TRACED"
        diagnostic_held_object_gate = {
            "enabled": bool(args.diagnostic_held_object_mode != "none"),
            "mode": args.diagnostic_held_object_mode,
            "dynamic_grasp_proof": False,
            "status": "SKIPPED_NO_DIAGNOSTIC_HELD_OBJECT_MODE",
            "triggered": False,
            "same_side_robot_non_target_contact": False,
            "same_side_robot_non_target_contact_pairs": [],
            "same_side_robot_non_target_first_contact_pair": None,
                "object_lift_ok": object_lift_ok,
                "object_lift_gate": object_lift_gate,
                "no_explosion_ok": no_explosion_ok,
            "workcell_contact_policy_ok": workcell_contact_policy_ok,
            "controller_tracking_ok": controller_tracking_ok,
            "notes": (
                "This gate checks carried-object trajectory semantics only. It must not be used as proof "
                "that passive two-finger contact can lift the object."
            ),
        }
        if args.diagnostic_held_object_mode != "none":
            triggered = bool(
                diagnostic_t_gripper_object is not None
                or (diagnostic_held_object_row or {}).get("status") == "DIAGNOSTIC_NOT_DYNAMIC_GRASP_PROOF"
            )
            same_side_payload = (contact_summary.get("object_contact_categories") or {}).get(
                "same_side_robot_non_target"
            ) or {}
            diagnostic_pass = bool(triggered and object_lift_ok and no_explosion_ok and workcell_contact_policy_ok and controller_tracking_ok)
            diagnostic_held_object_gate.update(
                {
                    "status": "PASS_CARRIED_OBJECT_PATH_DIAGNOSTIC"
                    if diagnostic_pass
                    else "FAIL_CARRIED_OBJECT_PATH_DIAGNOSTIC",
                    "triggered": triggered,
                    "trigger_step": (diagnostic_held_object_row or {}).get("trigger_step"),
                    "trigger_status": (diagnostic_held_object_row or {}).get("status"),
                    "same_side_robot_non_target_contact": bool(same_side_payload),
                    "same_side_robot_non_target_contact_pairs": same_side_payload.get("unique_contact_pairs") or [],
                    "same_side_robot_non_target_first_contact_pair": same_side_payload.get("first_contact_pair"),
                    "object_lift_m": object_lift,
                    "object_displacement_m": object_displacement,
                }
            )
        loaded_gripper_soft_bottle_calibration = _loaded_gripper_soft_bottle_calibration_diagnostic(
            final_alignment=final_finger_object_alignment,
            hdf5_gripper_summary=hdf5_gripper_summary,
            reachability_audit=target_contact_reachability_audit,
            contact_distance_m=float(args.object_contact_offset or 0.0) + float(args.proxy_contact_offset or 0.0),
            object_effective_contact_width_m=soft_contact_model.get("effective_contact_width_m"),
            visual_bottle_outer_diameter_m=soft_contact_model.get("visual_external_diameter_m")
            or geometry_sanity.get("object_side_length_meters"),
            moving_fingers=str(args.moving_fingers),
            controller_tracking_gate=controller_tracking_gate,
            positive_control_gate=diagnostic_force_target_overlap_contact_pipeline_gate,
        )
        payload.update(
            {
                "status": "PASS" if overall_pass else "FAILED_GATE",
                "overall_pass": overall_pass,
                "contact_trace_status": contact_trace_status,
                "open_target_values": open_values,
                "close_target_values": close_values,
                "hdf5_gripper_summary": hdf5_gripper_summary,
                "hdf5_gripper_replay_steps": close_replay_step_count if hdf5_target_sequence is not None else None,
                "post_close_hold_steps": int(post_close_hold_steps),
                "post_close_hold_phase": "post_close_hold" if post_close_hold_steps > 0 else None,
                "post_close_lift_source": args.post_close_lift_source,
                "post_close_lift_steps": int(post_close_lift_step_count),
                "post_close_lift_phase": "post_close_lift" if post_close_lift_step_count > 0 else None,
                "post_close_lift_hold_steps": int(post_close_lift_hold_step_count),
                "post_close_lift_hold_phase": (
                    "post_close_lift_hold" if post_close_lift_hold_step_count > 0 else None
                ),
                "post_close_lift_summary": post_close_lift_summary,
                "total_close_and_hold_steps": int(close_replay_step_count + post_close_hold_steps),
                "total_close_hold_lift_steps": int(close_step_count),
                "hdf5_replay_target_hold_steps": int(args.hdf5_replay_target_hold_steps),
                "hdf5_replay_rate_hz": float(args.hdf5_replay_rate_hz),
                "physics_dt": float(args.physics_dt),
                "physics_rate_hz": hdf5_timing_alignment["physics_rate_hz"],
                "stage_time_codes_per_second": stage_time_codes_per_second_effective,
                "stage_frames_per_second": stage_frames_per_second_effective,
                "hdf5_effective_control_dt": hdf5_timing_alignment["effective_control_dt"],
                "hdf5_effective_control_rate_hz": hdf5_timing_alignment["effective_control_rate_hz"],
                "hdf5_timing_alignment": hdf5_timing_alignment,
                "hdf5_replay_target_physics_steps": (
                    close_replay_step_count * int(args.hdf5_replay_target_hold_steps)
                    if hdf5_target_sequence is not None
                    else None
                ),
                "post_close_hold_physics_steps": int(post_close_hold_steps)
                * int(args.hdf5_replay_target_hold_steps),
                "post_close_lift_physics_steps": int(post_close_lift_step_count)
                * int(args.hdf5_replay_target_hold_steps),
                "post_close_lift_hold_physics_steps": int(post_close_lift_hold_step_count)
                * int(args.hdf5_replay_target_hold_steps),
                "hdf5_replay_physics_steps": (
                    args.settle_steps + close_step_count * int(args.hdf5_replay_target_hold_steps)
                    if hdf5_target_sequence is not None
                    else None
                ),
                "hdf5_replay_segment_duration_s": (
                    close_replay_step_count / float(args.hdf5_replay_rate_hz)
                    if hdf5_target_sequence is not None
                    else None
                ),
                "runtime_arm_gain_override": {"kp": args.arm_kp, "kd": args.arm_kd},
                "runtime_finger_gain_override": {"kp": args.finger_kp, "kd": args.finger_kd},
                "pre_step_tracking_summary": pre_step_tracking_summary,
                "tracking_summary": tracking_summary,
                "tracking_summary_by_phase": tracking_summary_by_phase,
                "controller_tracking_gate": controller_tracking_gate,
                "controller_tracking_gate_by_phase": controller_tracking_gate_by_phase,
                "controller_replay_fidelity_gate": controller_replay_fidelity_gate,
                "command_smoothness_gate": command_smoothness_gate,
                "formal_replay_feasibility_gate": formal_replay_feasibility_gate,
                "runtime_drive_profile": runtime_drive_profile,
                "drive_authority_audit": drive_authority_audit,
                "tracking_spike_packet": tracking_spike_packet,
                "physical_grasp_gate": physical_grasp_gate,
                "tabletop_grasp_contact_gate": tabletop_grasp_contact_gate,
                "post_close_hold_diagnostic_gate": post_close_hold_diagnostic_gate,
                "prelift_static_grasp_gate": prelift_static_grasp_gate,
                "post_close_lift_diagnostic_gate": post_close_lift_diagnostic_gate,
                "lift_transport_gate": lift_transport_gate,
                "lift_contact_wrench_patch_audit": lift_contact_wrench_patch_audit,
                "fixed_reference_grasp_geometry_gate": fixed_reference_grasp_geometry_gate,
                "active_target_contact_gate": active_target_contact_gate,
                "early_geometric_escape_gate": early_geometric_escape_gate,
                "tabletop_reference_contract": tabletop_reference_contract,
                "workcell_contact_policy_gate": workcell_contact_policy_gate,
                "initial_grasp_geometry_audit": initial_grasp_geometry_audit,
                "denied_workcell_geometry_audit": denied_workcell_geometry_audit,
                "target_limit_summary": target_limit_summary,
                "target_limit_gate_ok": target_limit_ok,
                "failure_reasons": failure_reasons,
                "finger_gap_axis": axis_name,
                "finger_surface_gap_open": surface_gap,
                "finger_surface_gap_open_meters": geometry_sanity["finger_surface_gap_open_meters"],
                "left_finger_placement_box": placement_left_box,
                "right_finger_placement_box": placement_right_box,
                "left_finger_replay_start_box": replay_start_left_box,
                "right_finger_replay_start_box": replay_start_right_box,
                "start_finger_object_alignment": start_finger_object_alignment,
                "final_finger_object_alignment": final_finger_object_alignment,
                "final_reference_contact_center_transport": final_reference_contact_center,
                "cross_side_proxy_overlap": cross_side_proxy_overlap,
                "left_finger_final_box": left_box,
                "right_finger_final_box": right_box,
                "object_path": object_path,
                "object_contact_geometry_path": contact_geometry_path,
                "object_shape": args.object_shape,
                "bottle_runtime_composition_gate": bottle_runtime_composition_gate,
                "bottle_grasp_semantics_gate": bottle_grasp_semantics_gate,
                "visible_bottle_runtime_path": (
                    bottle_runtime_composition_gate.get("runtime_object_path")
                    if args.object_shape
                    in {
                        "bottle_usd",
                        "bottle_usd_cylinder_proxy",
                        "bottle_usd_segmented_proxy",
                        "bottle_usd_grasp_band_proxy",
                        "bottle_usd_grasp_box_proxy",
                    }
                    else None
                ),
                "object_axis": args.object_axis,
                "object_length_multiplier": args.object_length_multiplier,
                "object_usd": _rel(args.object_usd),
                "object_usd_prim_path": args.object_usd_prim_path,
                "object_placement": object_placement_row,
                "diagnostic_force_target_overlap": diagnostic_force_target_overlap_row,
                "diagnostic_force_target_overlap_contact_pipeline_gate": (
                    diagnostic_force_target_overlap_contact_pipeline_gate
                ),
                "loaded_gripper_soft_bottle_calibration_diagnostic": (
                    loaded_gripper_soft_bottle_calibration
                ),
                "object_side_length_stage_units": side_length,
                "object_side_length_meters": geometry_sanity["object_side_length_meters"],
                "soft_bottle_contact_model": soft_contact_model,
                "contact_setup_geometry_sanity": geometry_sanity,
                "contact_setup_geometry_sanity_status": geometry_sanity["status"],
                "support_plane": support_plane_row,
                "proxy_collision_offsets": proxy_offset_rows,
                "finger_contact_materials": finger_material_rows,
                "object_collision_offsets": object_offset_row,
                "object_contact_material": object_material_row,
                "debug_stage_after_object_placement": debug_stage_after_object_placement,
                "diagnostic_held_object": diagnostic_held_object_row,
                "diagnostic_held_object_gate": diagnostic_held_object_gate,
                "object_reset_box": object_reset_box,
                "object_contact_reset_box": object_contact_reset_box,
                "object_initial_box": object_initial_box,
                "object_final_box": object_final_box,
                "object_final_contact_box": object_final_contact_box,
                "object_reset_center": object_reset_center.tolist(),
                "object_initial_center": object_initial_center.tolist(),
                "object_final_center": object_final_center.tolist(),
                "object_settle_displacement": object_settle_displacement,
                "object_displacement": object_displacement,
                "object_lift": object_lift,
                "min_object_lift": float(args.min_object_lift),
                "object_lift_ok": object_lift_ok,
                "object_lift_gate": object_lift_gate,
                "object_width_stop_summary": object_width_stop_summary,
                "diagnostic_loaded_clamp_squeeze_summary": diagnostic_loaded_clamp_squeeze_summary,
                "target_contact_reachability_audit": target_contact_reachability_audit,
                "total_object_displacement": total_object_displacement,
                "max_object_displacement": max_displacement,
                "object_motion_finite": finite_motion,
                "contact_motion_policy": contact_motion_policy,
                "contact_motion_ok": contact_motion_ok,
                "no_explosion_ok": no_explosion_ok,
                "contact_pair_trace_enabled": bool(args.trace_contact_pairs),
                "contact_trace_disable_usd_updates": bool(args.trace_disable_usd_updates),
                "fail_on_non_target_object_contact": bool(args.fail_on_non_target_object_contact),
                "allowed_non_target_object_contact_categories": list(args.allowed_non_target_object_contact_category),
                "non_target_contact_gate": non_target_contact_gate,
                "non_target_object_contact_ok": non_target_object_contact_ok,
                "require_active_target_contact": bool(args.require_active_target_contact),
                "already_in_contact_setup": bool(args.already_in_contact_setup),
                "active_target_contact_gate": active_target_contact_gate,
                "active_target_contact_ok": active_target_contact_ok,
                "bilateral_grasp_formation_gate": bilateral_grasp_formation_gate,
                "bilateral_grasp_formation_gate_by_phase": bilateral_grasp_formation_gate_by_phase,
                "bilateral_grasp_failure_diagnostic": bilateral_grasp_failure_diagnostic,
                "bilateral_grasp_formation_ok": bilateral_grasp_formation_ok,
                "contact_summary_by_phase": phase_contact_summaries,
                "early_geometric_escape_ok": early_geometric_escape_ok,
                "contact_landmark_alignment": contact_landmark_alignment,
                "active_grasp_geometry_precondition": active_grasp_geometry_precondition,
                "active_grasp_geometry_precondition_ok": active_grasp_geometry_precondition_ok,
                "open_finger_object_height_alignment": open_finger_object_height_alignment,
                "open_finger_object_height_alignment_ok": open_finger_object_height_alignment_ok,
                "contact_trace_rigid_body_paths": trace_state["rigid_body_paths"] if trace_state else [],
                "contact_trace_late_registered_rigid_bodies": object_contact_report_rows,
                "first_contact_pair": first_contact_row,
                **contact_summary,
                "csv": _rel(csv_path),
                "markdown": _rel(md_path),
                "next_gate": "gripper_contact_with_task_shape"
                if overall_pass
                else "inspect_contact_geometry_or_finger_control",
            }
        )
        _write_csv(csv_path, rows)
        _write_json(json_path, payload)
        _write_markdown(md_path, _json_safe(payload))
        print(
            json.dumps(
                {"status": payload["status"], "json": _rel(json_path), "markdown": _rel(md_path)}, ensure_ascii=False
            ),
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if overall_pass else 3)
    except BaseException as exc:
        payload.update(
            {
                "status": "EXCEPTION",
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-25:],
            }
        )
        _write_json(json_path, payload)
        print(
            json.dumps(
                {"status": payload["status"], "json": _rel(json_path), "exception": payload["exception"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
