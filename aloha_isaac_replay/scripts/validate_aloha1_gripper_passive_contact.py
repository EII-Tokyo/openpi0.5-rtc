from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np
import yaml

from aloha_isaac_replay.adapters.gripper_mapping import standard_gripper_qpos_to_isaac_fingers
from aloha_isaac_replay.adapters.isaac_dof_adapter import load_mapping
from aloha_isaac_replay.replay.arm_only_mapping import arm_only_targets_from_standard_qpos
from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG
from aloha_isaac_replay.scripts.audit_table_frame_candidate import audit_table_frame
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _apply_arm_gains
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _apply_gravity
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _apply_named_dof_gains
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _get_limits
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


def _rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n")


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
        raise ValueError("--support-plane-config cannot be combined with --support-plane-mode object_bottom")
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
    diagnostic_contacts = payload.get("diagnostic_contact_summaries") or {}
    support_size = support_plane.get("size")
    tracking_gate = payload.get("controller_tracking_gate") or {}
    non_target_gate = payload.get("non_target_contact_gate") or {}
    active_target_gate = payload.get("active_target_contact_gate") or {}
    failure_reasons = payload.get("failure_reasons") or []
    lines = [
        "# Gripper Passive Contact Smoke",
        "",
        f"- status: `{payload['status']}`",
        f"- contact trace status: `{payload.get('contact_trace_status')}`",
        f"- failure reasons: `{failure_reasons}`",
        f"- stage: `{payload['inputs']['stage_usd']}`",
        f"- control mode: `{payload['inputs']['control_mode']}`",
        f"- moving fingers: `{payload['inputs'].get('moving_fingers')}`",
        f"- object side length: `{payload.get('object_side_length_stage_units')}` stage units",
        f"- object side length: `{payload.get('object_side_length_meters')}` m",
        f"- finger surface gap open: `{payload.get('finger_surface_gap_open')}` stage units",
        f"- finger surface gap open: `{payload.get('finger_surface_gap_open_meters')}` m",
        f"- contact setup geometry sanity: `{payload.get('contact_setup_geometry_sanity_status')}`",
        f"- object settle displacement: `{payload.get('object_settle_displacement')}` stage units",
        f"- object close displacement: `{payload.get('object_displacement')}` stage units",
        f"- object total displacement: `{payload.get('total_object_displacement')}` stage units",
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
        f"- active target contact gate: `{active_target_gate}`",
        f"- active target contact gate ok: `{payload.get('active_target_contact_ok')}`",
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
        f"- controller tracking gate: `{tracking_gate}`",
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
        "This is a local contact smoke test. It only checks whether a small passive cube between the gripper proxies remains numerically stable and moves within a bounded range during finger closure.",
        "A non-zero contact count is not a success condition. The trace must show that the target fingertip proxy contacts the target object collider, and object motion must remain bounded.",
        "It does not validate grasp success, bottle geometry, friction realism, or full-arm task behavior.",
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


def _target_from_standard_qpos(
    *,
    art: Any,
    side: str,
    qpos_frame: np.ndarray,
    mapping: dict[str, Any] | None,
    replay_mode: str,
    finger_dof_names: dict[str, str],
    finger_qpos_limits: Any,
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
            target[dof_names.index(dof_name)] = float(arm_target.value)
    channel = 6 if side == "left" else 13
    fingers = standard_gripper_qpos_to_isaac_fingers(float(qpos_frame[channel]), side=side, limits=finger_qpos_limits)
    target[dof_names.index(finger_dof_names["left_finger"])] = float(fingers[f"{side}/left_finger"])
    target[dof_names.index(finger_dof_names["right_finger"])] = float(fingers[f"{side}/right_finger"])
    return target


def _targets_from_hdf5_qpos(
    *,
    art: Any,
    side: str,
    qpos: np.ndarray,
    mapping: dict[str, Any] | None,
    replay_mode: str,
    finger_dof_names: dict[str, str],
    finger_qpos_limits: Any,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    dof_names = list(art.dof_names)
    left_idx = dof_names.index(finger_dof_names["left_finger"])
    right_idx = dof_names.index(finger_dof_names["right_finger"])
    channel = 6 if side == "left" else 13
    gripper_qpos = np.asarray(qpos[:, channel], dtype=np.float64)
    targets: list[np.ndarray] = []
    for frame in qpos:
        targets.append(
            _target_from_standard_qpos(
                art=art,
                side=side,
                qpos_frame=frame,
                mapping=mapping,
                replay_mode=replay_mode,
                finger_dof_names=finger_dof_names,
                finger_qpos_limits=finger_qpos_limits,
            )
        )
    arm_delta = None
    if replay_mode == "left_arm_and_gripper":
        indices = slice(0, 6) if side == "left" else slice(7, 13)
        arm_qpos = np.asarray(qpos[:, indices], dtype=np.float64)
        arm_delta = {
            "max_abs_frame_delta": float(np.max(np.abs(np.diff(arm_qpos, axis=0)))) if len(arm_qpos) > 1 else 0.0,
            "max_abs_net_delta": float(np.max(np.abs(arm_qpos[-1] - arm_qpos[0]))),
        }
    return targets, {
        "source": "observations/qpos",
        "side": side,
        "replay_mode": replay_mode,
        "sample_count": int(gripper_qpos.size),
        "raw_start": float(gripper_qpos[0]),
        "raw_end": float(gripper_qpos[-1]),
        "raw_min": float(np.min(gripper_qpos)),
        "raw_max": float(np.max(gripper_qpos)),
        "raw_range": float(np.max(gripper_qpos) - np.min(gripper_qpos)),
        "raw_net": float(gripper_qpos[-1] - gripper_qpos[0]),
        "first_target_values": {
            "left_finger": float(targets[0][left_idx]),
            "right_finger": float(targets[0][right_idx]),
        },
        "last_target_values": {
            "left_finger": float(targets[-1][left_idx]),
            "right_finger": float(targets[-1][right_idx]),
        },
        "arm_qpos_delta": arm_delta,
    }


def _tracking_groups(
    dof_names: list[str], *, replay_mode: str, finger_dof_names: dict[str, str], side: str = "left"
) -> dict[str, list[int]]:
    finger_indices = [
        dof_names.index(finger_dof_names["left_finger"]),
        dof_names.index(finger_dof_names["right_finger"]),
    ]
    groups: dict[str, list[int]] = {"gripper": finger_indices}
    if replay_mode == "left_arm_and_gripper":
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
        "active_phases": ["close"],
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
    if contact_summary.get("first_target_contact_found_phase") == "close":
        row.update({"pass": True, "status": "PASS_ACTIVE_TARGET_CONTACT_FOUND_DURING_CLOSE"})
    else:
        row.update({"pass": False, "status": "FAIL_NO_ACTIVE_TARGET_CONTACT_DURING_CLOSE"})
    return row


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

    pre_step_qpos: np.ndarray | None = None
    for _ in range(target_hold_steps):
        if actuation_mode == "drive_target":
            _set_full_target(art, target)
        elif actuation_mode == "state_teleport":
            _set_full_state(art, target)
            _set_full_target(art, target)
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


def _bbox_center(stage: Any, path: str) -> np.ndarray:
    box = _bbox_row(stage, path)
    if not box.get("bbox_valid"):
        raise RuntimeError(f"Cannot compute bbox center for {path}")
    return np.asarray(box["center"], dtype=np.float64)


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
    rigid_body: bool = True,
) -> None:
    from pxr import Gf
    from pxr import UsdGeom
    from pxr import UsdPhysics

    if shape != "cube" and creation_mode != "raw_usd":
        raise ValueError(f"{shape} object shape requires raw_usd creation; got {creation_mode}")
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


def _read_contact_pairs(trace_state: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not trace_state:
        return []
    from omni.physx.bindings._physx import ContactEventType
    from pxr import PhysicsSchemaTools

    contact_headers, _contact_data = trace_state["physx_interface"].get_contact_report()
    rows: list[dict[str, Any]] = []
    for contact_header in contact_headers:
        collider0 = str(PhysicsSchemaTools.intToSdfPath(contact_header.collider0))
        collider1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.collider1))
        rows.append(
            {
                "type": int(contact_header.type),
                "type_name": "CONTACT_FOUND"
                if contact_header.type == ContactEventType.CONTACT_FOUND
                else str(contact_header.type),
                "collider0": collider0,
                "collider1": collider1,
                "sorted_pair": sorted([collider0, collider1]),
            }
        )
    return rows


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
    diagnostic_contact_paths: list[str],
) -> str | None:
    other_path = _other_collider_for_object_pair(row, object_path)
    if other_path is None:
        return None
    if any(_path_matches(other_path, finger_path) for finger_path in expected_finger_paths):
        return "target_finger"
    if any(_path_matches(other_path, path) for path in diagnostic_contact_paths):
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
            "phase_counts": _phase_counts(rows),
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
    parser.add_argument("--close-steps", type=int, default=180)
    parser.add_argument("--physics-dt", type=float, default=1.0 / 50.0)
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
    parser.add_argument("--limit-margin", type=float, default=0.001)
    parser.add_argument("--object-fill-fraction", type=float, default=0.6)
    parser.add_argument("--object-placement", choices=("gap_center", "moving_finger_surface"), default="gap_center")
    parser.add_argument("--object-clearance", type=float, default=0.001)
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
        "--object-shape", choices=("cube", "cylinder", "capsule", "bottle_proxy", "bottle_usd"), default="cube"
    )
    parser.add_argument("--object-axis", choices=("X", "Y", "Z"), default="X")
    parser.add_argument("--object-length-multiplier", type=float, default=4.0)
    parser.add_argument("--object-usd", default=str(DEFAULT_BOTTLE_USD))
    parser.add_argument("--object-usd-prim-path", default="/Bottle500")
    parser.add_argument("--object-mass", type=float, default=0.01)
    parser.add_argument("--object-contact-offset", type=float, default=None)
    parser.add_argument("--object-rest-offset", type=float, default=None)
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
    parser.add_argument("--support-plane-mode", choices=("none", "object_bottom", "fixed_box"), default="none")
    parser.add_argument("--support-plane-center", type=float, nargs=3, default=None)
    parser.add_argument("--support-plane-size", type=float, default=DEFAULT_SUPPORT_PLANE_SIZE)
    parser.add_argument("--support-plane-size-x", type=float, default=None)
    parser.add_argument("--support-plane-size-y", type=float, default=None)
    parser.add_argument("--support-plane-thickness", type=float, default=DEFAULT_SUPPORT_PLANE_THICKNESS)
    parser.add_argument("--support-plane-clearance", type=float, default=0.0)
    parser.add_argument("--proxy-contact-offset", type=float, default=None)
    parser.add_argument("--proxy-rest-offset", type=float, default=None)
    parser.add_argument("--closure-profile", choices=("abrupt", "linear"), default="abrupt")
    parser.add_argument("--moving-fingers", choices=("both", "left", "right"), default="both")
    parser.add_argument("--hdf5-gripper-episode", default=None)
    parser.add_argument("--hdf5-replay-mode", choices=("gripper_only", "left_arm_and_gripper"), default="gripper_only")
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
        "--max-post-step-controlled-tracking-error",
        type=float,
        default=None,
        help=(
            "Maximum allowed post-physics-step absolute joint tracking error for controlled DOFs. "
            "If omitted, HDF5 replay uses a conservative 0.02 rad/m default; non-HDF5 contact smoke tests skip "
            "this gate."
        ),
    )
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    parser.add_argument("--hdf5-gripper-start-frame", type=int, default=None)
    parser.add_argument("--hdf5-gripper-end-frame", type=int, default=None)
    parser.add_argument("--hdf5-gripper-max-frames", type=int, default=None)
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
        "--disable-workcell-environment-collisions-for-diagnostic-replay",
        action="store_true",
        help=(
            "Diagnostic only: disable uncalibrated /scene/worldBody and /World/Table collisions so HDF5 arm replay "
            "can be isolated from table/base calibration errors. Do not use for final contact validation."
        ),
    )
    parser.add_argument("--min-contact-motion", type=float, default=1e-5)
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
    try:
        if args.require_active_target_contact and args.already_in_contact_setup:
            raise ValueError("--require-active-target-contact cannot combine with --already-in-contact-setup")
        support_options = _resolve_support_plane_options(args)
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
            "gravity": args.gravity,
            "object_fill_fraction": args.object_fill_fraction,
            "object_placement": args.object_placement,
            "object_clearance": args.object_clearance,
            "object_creation": args.object_creation,
            "object_rigid_body": not args.disable_object_rigid_body,
            "object_shape": args.object_shape,
            "object_axis": args.object_axis,
            "object_length_multiplier": args.object_length_multiplier,
            "object_usd": _rel(args.object_usd),
            "object_usd_prim_path": args.object_usd_prim_path,
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
            "max_post_step_controlled_tracking_error": args.max_post_step_controlled_tracking_error,
            "mapping": _rel(args.mapping),
            "hdf5_gripper_start_frame": args.hdf5_gripper_start_frame,
            "hdf5_gripper_end_frame": args.hdf5_gripper_end_frame,
            "hdf5_gripper_max_frames": args.hdf5_gripper_max_frames,
            "trace_contact_pairs": args.trace_contact_pairs,
            "fail_on_non_target_object_contact": args.fail_on_non_target_object_contact,
            "allowed_non_target_object_contact_categories": args.allowed_non_target_object_contact_category,
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

        stage_utils.open_stage(str(Path(args.stage_usd).resolve()))
        World.clear_instance()
        world = World(stage_units_in_meters=args.stage_units_in_meters, backend="numpy", device="cpu")
        world.set_simulation_dt(physics_dt=args.physics_dt, rendering_dt=args.physics_dt)
        stage = omni.usd.get_context().get_stage()
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
        if args.hdf5_gripper_episode:
            qpos = _load_hdf5_qpos(
                args.hdf5_gripper_episode,
                start=args.hdf5_gripper_start_frame,
                end=args.hdf5_gripper_end_frame,
                max_frames=args.hdf5_gripper_max_frames,
            )
            mapping = load_mapping(args.mapping) if args.hdf5_replay_mode == "left_arm_and_gripper" else None
            hdf5_target_sequence, hdf5_gripper_summary = _targets_from_hdf5_qpos(
                art=art,
                side=args.side,
                qpos=qpos,
                mapping=mapping,
                replay_mode=args.hdf5_replay_mode,
                finger_dof_names=finger_dof_names,
                finger_qpos_limits=finger_qpos_limits,
            )
            open_target = hdf5_target_sequence[0]
            open_values = hdf5_gripper_summary["first_target_values"]
            payload["inputs"]["control_mode"] = f"hdf5_{args.hdf5_replay_mode}_qpos_replay"
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
        side_length = max(surface_gap * args.object_fill_fraction, 1e-4)
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
        object_path = "/World/phase43_passive_contact_cube"
        proxy_offset_rows = [
            _set_collision_offsets(stage, paths["left_finger"], args.proxy_contact_offset, args.proxy_rest_offset),
            _set_collision_offsets(stage, paths["right_finger"], args.proxy_contact_offset, args.proxy_rest_offset),
        ]
        _create_passive_cube(
            world=world,
            stage=stage,
            path=object_path,
            center=center,
            side_length=side_length,
            mass=args.object_mass,
            creation_mode=args.object_creation,
            shape=args.object_shape,
            axis=args.object_axis,
            length_multiplier=args.object_length_multiplier,
            usd_path=args.object_usd,
            usd_prim_path=args.object_usd_prim_path,
            rigid_body=not args.disable_object_rigid_body,
        )
        object_offset_row = _set_object_collision_offsets(
            stage, object_path, args.object_contact_offset, args.object_rest_offset
        )
        support_plane_row: dict[str, Any] | None = None
        if support_options["mode"] != "none":
            object_support_box = _bbox_row(stage, object_path)
            if not object_support_box.get("bbox_valid"):
                raise RuntimeError("Cannot place support plane because object bbox is invalid.")
            if support_options["mode"] == "object_bottom":
                object_support_center = np.asarray(object_support_box["center"], dtype=np.float64)
                support_center = object_support_center.copy()
                support_center[2] = (
                    float(object_support_box["min"][2])
                    - float(args.support_plane_clearance)
                    - float(support_options["thickness"]) * 0.5
                )
            else:
                if support_options["center"] is None:
                    raise ValueError(
                        "--support-plane-mode fixed_box requires --support-plane-center X Y Z or --support-plane-config"
                    )
                support_center = np.asarray(support_options["center"], dtype=np.float64)
            support_plane_row = _create_static_support_box(
                stage=stage,
                path="/World/phase58_static_support_plane",
                center=support_center,
                size_x=support_options["size_x"],
                size_y=support_options["size_y"],
                thickness=support_options["thickness"],
            )
            support_plane_row["placement_object_box"] = object_support_box
            support_plane_row["mode"] = support_options["mode"]
            support_plane_row["config"] = support_options["config"]
            support_plane_row["config_provenance"] = support_options["config_provenance"]
            support_plane_row["table_frame"] = support_options["table_frame"]
        first_contact_row: dict[str, Any] | None = None
        contact_pair_rows: list[dict[str, Any]] = []
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
            tracking_rows: list[dict[str, Any]] = []
            pre_step_tracking_rows: list[dict[str, Any]] = []
            target_limit_rows: list[dict[str, Any]] = []
            object_reset_box = _bbox_row(stage, object_path)
            object_reset_center = np.asarray(object_reset_box["center"], dtype=np.float64)
            rows: list[dict[str, Any]] = []
            max_displacement = 0.0
            finite_motion = True
            for step in range(args.settle_steps):
                pre_step_qpos = _apply_replay_target_and_step(
                    world,
                    art,
                    open_target,
                    actuation_mode=args.hdf5_replay_actuation_mode,
                    target_hold_steps=args.hdf5_replay_target_hold_steps,
                )
                qpos = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
                pre_step_tracking = _tracking_step_errors(target=open_target, actual=pre_step_qpos, groups=tracking_groups)
                step_tracking = _tracking_step_errors(target=open_target, actual=qpos, groups=tracking_groups)
                target_limit = _target_limit_step_violations(
                    target=open_target, limits=runtime_limits, groups=tracking_groups
                )
                pre_step_tracking_rows.append({"phase": "settle", "step": step, "groups": pre_step_tracking})
                tracking_rows.append({"phase": "settle", "step": step, "groups": step_tracking})
                target_limit_rows.append({"phase": "settle", "step": step, "groups": target_limit})
                left_box = _bbox_row(stage, paths["left_finger"])
                right_box = _bbox_row(stage, paths["right_finger"])
                object_box = _bbox_row(stage, object_path)
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
                if step_contact_pairs:
                    for pair in step_contact_pairs:
                        contact_row = {"phase": "settle", "step": step, **pair}
                        contact_pair_rows.append(contact_row)
                    if first_contact_row is None:
                        first_contact_row = dict(contact_pair_rows[-len(step_contact_pairs)])
                rows.append(
                    {
                        "phase": "settle",
                        "step": step,
                        "object_center_x": float(object_center[0]),
                        "object_center_y": float(object_center[1]),
                        "object_center_z": float(object_center[2]),
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
            close_step_count = len(close_sequence) if hdf5_target_sequence is not None else args.close_steps
            for step in range(close_step_count):
                if hdf5_target_sequence is not None:
                    step_target = close_sequence[step]
                elif args.closure_profile == "linear":
                    alpha = float(step + 1) / float(max(args.close_steps, 1))
                    step_target = open_target + alpha * (close_target - open_target)
                else:
                    step_target = close_target
                pre_step_qpos = _apply_replay_target_and_step(
                    world,
                    art,
                    step_target,
                    actuation_mode=args.hdf5_replay_actuation_mode,
                    target_hold_steps=args.hdf5_replay_target_hold_steps,
                )
                qpos = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
                pre_step_tracking = _tracking_step_errors(target=step_target, actual=pre_step_qpos, groups=tracking_groups)
                step_tracking = _tracking_step_errors(target=step_target, actual=qpos, groups=tracking_groups)
                target_limit = _target_limit_step_violations(
                    target=step_target, limits=runtime_limits, groups=tracking_groups
                )
                pre_step_tracking_rows.append({"phase": "close", "step": step, "groups": pre_step_tracking})
                tracking_rows.append({"phase": "close", "step": step, "groups": step_tracking})
                target_limit_rows.append({"phase": "close", "step": step, "groups": target_limit})
                left_box = _bbox_row(stage, paths["left_finger"])
                right_box = _bbox_row(stage, paths["right_finger"])
                object_box = _bbox_row(stage, object_path)
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
                if step_contact_pairs:
                    for pair in step_contact_pairs:
                        contact_row = {"phase": "close", "step": step, **pair}
                        contact_pair_rows.append(contact_row)
                    if first_contact_row is None:
                        first_contact_row = dict(contact_pair_rows[-len(step_contact_pairs)])
                rows.append(
                    {
                        "phase": "close",
                        "step": step,
                        "object_center_x": float(object_center[0]),
                        "object_center_y": float(object_center[1]),
                        "object_center_z": float(object_center[2]),
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
                        "finger_center_distance": _gap_metrics(left_box, right_box).get("center_distance"),
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
        object_final_center = object_latest_center
        tracking_summary = _summarize_tracking_errors(tracking_rows, tracking_groups, dof_names)
        pre_step_tracking_summary = _summarize_tracking_errors(pre_step_tracking_rows, tracking_groups, dof_names)
        target_limit_summary = _summarize_target_limit_violations(target_limit_rows, tracking_groups, dof_names)
        target_limit_ok = bool(target_limit_summary.get("controller_ready", True))
        effective_max_tracking_error = args.max_post_step_controlled_tracking_error
        if effective_max_tracking_error is None and hdf5_target_sequence is not None:
            effective_max_tracking_error = 0.02
        controller_tracking_gate = _controller_tracking_gate(
            tracking_summary=tracking_summary,
            max_controlled_error=effective_max_tracking_error,
        )
        controller_tracking_ok = bool(controller_tracking_gate["pass"])
        object_displacement = float(np.linalg.norm(object_final_center - object_initial_center))
        total_object_displacement = float(np.linalg.norm(object_final_center - object_reset_center))
        contact_motion_policy = (
            "not_required_for_bilateral_closure"
            if args.moving_fingers == "both"
            else "single_finger_push_requires_minimum_motion"
        )
        contact_motion_ok = bool(args.moving_fingers == "both" or object_displacement >= args.min_contact_motion)
        no_explosion_ok = bool(finite_motion and max_displacement <= args.max_object_displacement)
        overall_pass = bool(contact_motion_ok and no_explosion_ok and target_limit_ok and controller_tracking_ok)
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
        if args.moving_fingers == "both":
            target_contact_ok = bool(contact_summary["all_expected_fingers_target_contact_pair_found"])
        else:
            target_contact_ok = bool(contact_summary["target_contact_pair_found"])
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
        active_target_contact_gate = _active_target_contact_gate(
            contact_summary=contact_summary,
            require_active_target_contact=bool(args.trace_contact_pairs and args.require_active_target_contact),
            already_in_contact_setup=bool(args.already_in_contact_setup),
        )
        active_target_contact_ok = bool(active_target_contact_gate["pass"])
        trace_pair_ok = bool(trace_pair_ok and non_target_object_contact_ok and active_target_contact_ok)
        overall_pass = bool(overall_pass and trace_pair_ok)
        failure_reasons = []
        if not contact_motion_ok:
            failure_reasons.append("contact_motion_below_threshold")
        if not no_explosion_ok:
            failure_reasons.append("object_motion_exceeded_limit")
        if not trace_pair_ok:
            failure_reasons.append("contact_trace_gate_failed")
        if not active_target_contact_ok:
            failure_reasons.append("active_target_contact_gate_failed")
        if not target_limit_ok:
            failure_reasons.append("target_outside_runtime_limits")
        if not controller_tracking_ok:
            failure_reasons.append("post_step_controller_tracking_exceeded_threshold")
        if args.trace_contact_pairs:
            if cross_side_overlap_blocks_gate:
                contact_trace_status = "FAIL_CROSS_SIDE_PROXY_OVERLAP"
            elif not target_contact_ok:
                contact_trace_status = "FAIL_NO_TARGET_CONTACT"
            elif not non_target_object_contact_ok:
                contact_trace_status = str(non_target_contact_gate["status"])
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
        payload.update(
            {
                "status": "PASS" if overall_pass else "FAILED_GATE",
                "overall_pass": overall_pass,
                "contact_trace_status": contact_trace_status,
                "open_target_values": open_values,
                "close_target_values": close_values,
                "hdf5_gripper_summary": hdf5_gripper_summary,
                "hdf5_gripper_replay_steps": len(close_sequence) if hdf5_target_sequence is not None else None,
                "hdf5_replay_target_hold_steps": int(args.hdf5_replay_target_hold_steps),
                "hdf5_replay_physics_steps": (
                    (args.settle_steps + len(close_sequence)) * int(args.hdf5_replay_target_hold_steps)
                    if hdf5_target_sequence is not None
                    else None
                ),
                "runtime_arm_gain_override": {"kp": args.arm_kp, "kd": args.arm_kd},
                "runtime_finger_gain_override": {"kp": args.finger_kp, "kd": args.finger_kd},
                "pre_step_tracking_summary": pre_step_tracking_summary,
                "tracking_summary": tracking_summary,
                "controller_tracking_gate": controller_tracking_gate,
                "active_target_contact_gate": active_target_contact_gate,
                "target_limit_summary": target_limit_summary,
                "target_limit_gate_ok": target_limit_ok,
                "failure_reasons": failure_reasons,
                "finger_gap_axis": axis_name,
                "finger_surface_gap_open": surface_gap,
                "finger_surface_gap_open_meters": geometry_sanity["finger_surface_gap_open_meters"],
                "left_finger_placement_box": placement_left_box,
                "right_finger_placement_box": placement_right_box,
                "cross_side_proxy_overlap": cross_side_proxy_overlap,
                "left_finger_final_box": left_box,
                "right_finger_final_box": right_box,
                "object_path": object_path,
                "object_shape": args.object_shape,
                "object_axis": args.object_axis,
                "object_length_multiplier": args.object_length_multiplier,
                "object_usd": _rel(args.object_usd),
                "object_usd_prim_path": args.object_usd_prim_path,
                "object_placement": object_placement_row,
                "object_side_length_stage_units": side_length,
                "object_side_length_meters": geometry_sanity["object_side_length_meters"],
                "contact_setup_geometry_sanity": geometry_sanity,
                "contact_setup_geometry_sanity_status": geometry_sanity["status"],
                "support_plane": support_plane_row,
                "proxy_collision_offsets": proxy_offset_rows,
                "object_collision_offsets": object_offset_row,
                "object_reset_box": object_reset_box,
                "object_initial_box": object_initial_box,
                "object_final_box": object_final_box,
                "object_reset_center": object_reset_center.tolist(),
                "object_initial_center": object_initial_center.tolist(),
                "object_final_center": object_final_center.tolist(),
                "object_settle_displacement": object_settle_displacement,
                "object_displacement": object_displacement,
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
                "contact_trace_rigid_body_paths": trace_state["rigid_body_paths"] if trace_state else [],
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
