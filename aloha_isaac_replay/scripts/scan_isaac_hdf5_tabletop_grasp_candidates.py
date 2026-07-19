from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from aloha_isaac_replay.adapters.isaac_dof_adapter import load_mapping
from aloha_isaac_replay.data.grasp_candidate_scan import inspect_grasp_candidate
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import LIGHTWEIGHT_SIMULATION_APP_CONFIG
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _apply_gravity
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _bbox_row
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _finger_proxy_paths_for_args
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _load_hdf5_qpos
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _set_full_state
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _set_full_target
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _targets_from_hdf5_qpos
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import finger_dof_names_for_side
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import finger_qpos_limits_for_side
from aloha_isaac_replay.validation.bottle_grasp_semantics import BOTTLE_RADIUS_M
from aloha_isaac_replay.validation.contact_proxy_profiles import CONTACT_PROXY_PROFILES


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE = (
    REPO_ROOT
    / "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose"
    / "aloha2_menagerie_scene_deep_black_real_start_pose_proxy_runtime.usda"
)
DEFAULT_MAPPING = REPO_ROOT / "configs/aloha/trossen_scene_base_link_aloha1_left_controller_mapping.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/hdf5_tabletop_grasp_candidate_scan_20260719"
DEFAULT_ROOTS = [
    Path("/home/eii/project/high_level/video/main_s01_L_pick_bottle_capped"),
    REPO_ROOT / "local_rlt_data/raw_from_103/rollouts/key_regions",
]


def _iter_hdf5_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix in {".hdf5", ".h5"}:
            files.append(root)
            continue
        files.extend(sorted(root.rglob("*.hdf5")))
        files.extend(sorted(root.rglob("*.h5")))
    return files


def _scan_candidates(roots: list[Path], *, top_k: int) -> list[Any]:
    rows = []
    for path in _iter_hdf5_files(roots):
        try:
            row = inspect_grasp_candidate(path)
        except Exception:
            continue
        if row.close_frame is None:
            continue
        rows.append(row)
    rows.sort(key=lambda row: float(row.score), reverse=True)
    return rows[:top_k]


def _qpos_for_scoring(qpos: np.ndarray, *, side: str) -> np.ndarray:
    """Return a scanner-only qpos copy with tiny gripper overshoots clipped.

    Some HDF5 files contain normalized gripper observations such as 1.000017.
    That is useful to record as raw data, but it should not abort a broad
    candidate scan whose purpose is only to rank possible tabletop grasps.
    """

    qpos_for_targets = np.array(qpos, copy=True)
    channel = 6 if side == "left" else 13
    qpos_for_targets[:, channel] = np.clip(qpos_for_targets[:, channel], 0.0, 1.0)
    return qpos_for_targets


def _open_then_close_frame_indices(
    gripper: np.ndarray,
    *,
    open_threshold: float = 0.65,
    close_threshold: float = 0.35,
    lookahead: int = 120,
    max_indices: int = 4,
) -> list[int]:
    """Find open pre-grasp frames that are followed by a close event."""

    g = np.asarray(gripper, dtype=np.float64).reshape(-1)
    if len(g) == 0:
        return []
    rows: list[tuple[float, int]] = []
    for index in range(len(g) - 1):
        if g[index] < open_threshold:
            continue
        future = g[index + 1 : min(len(g), index + 1 + lookahead)]
        if len(future) == 0 or float(np.nanmin(future)) > close_threshold:
            continue
        # Prefer frames where the gripper is widely open and the close happens
        # soon afterwards; those are better active-tabletop grasp starts.
        close_offsets = np.nonzero(future <= close_threshold)[0]
        first_close_distance = int(close_offsets[0] + 1) if len(close_offsets) else lookahead
        score = float(g[index]) - 0.002 * float(first_close_distance)
        rows.append((score, index))
    selected: list[int] = []
    for _score, index in sorted(rows, reverse=True):
        if all(abs(index - prior) >= 10 for prior in selected):
            selected.append(int(index))
        if len(selected) >= max_indices:
            break
    return sorted(selected)


def _row_for_frame(
    *,
    world: Any,
    stage: Any,
    art: Any,
    paths: dict[str, str],
    target: np.ndarray,
    table_top_z: float,
    file_path: Path,
    frame: int,
    raw_gripper: float,
    source_score: float,
) -> dict[str, Any]:
    _set_full_state(art, target)
    _set_full_target(art, target)
    world.step(render=False)
    left_box = _bbox_row(stage, paths["left_finger"])
    right_box = _bbox_row(stage, paths["right_finger"])
    if not left_box.get("bbox_valid") or not right_box.get("bbox_valid"):
        return {
            "path": str(file_path),
            "frame": int(frame),
            "raw_gripper": float(raw_gripper),
            "source_score": float(source_score),
            "bbox_valid": False,
            "status": "FAIL_FINGER_BBOX_INVALID",
        }

    left_center = np.asarray(left_box["center"], dtype=np.float64)
    right_center = np.asarray(right_box["center"], dtype=np.float64)
    center_delta = left_center - right_center
    center_distance = float(np.linalg.norm(center_delta))
    midpoint = (left_center + right_center) * 0.5
    closing_axis = center_delta / center_distance if center_distance > 1e-12 else np.asarray([np.nan, np.nan, np.nan])
    expected_bottle_center_z = table_top_z + BOTTLE_RADIUS_M
    height_error = abs(float(midpoint[2]) - float(expected_bottle_center_z))
    closing_dot_x_abs = abs(float(closing_axis[0])) if np.all(np.isfinite(closing_axis)) else float("nan")
    score = (
        float(height_error) * 8.0
        + float(closing_dot_x_abs) * 2.0
        + abs(float(center_distance) - 2.0 * BOTTLE_RADIUS_M) * 2.0
        + max(float(raw_gripper) - 0.25, 0.0)
        - float(source_score) * 0.02
    )
    return {
        "path": str(file_path),
        "frame": int(frame),
        "raw_gripper": float(raw_gripper),
        "source_score": float(source_score),
        "bbox_valid": True,
        "status": "OK",
        "midpoint_world": midpoint.tolist(),
        "finger_center_delta_world": center_delta.tolist(),
        "finger_center_distance_m": center_distance,
        "closing_axis_world": closing_axis.tolist(),
        "closing_dot_object_x_abs": closing_dot_x_abs,
        "table_top_z_m": float(table_top_z),
        "expected_bottle_center_z_m": float(expected_bottle_center_z),
        "midpoint_tabletop_height_error_m": float(height_error),
        "left_finger_box": left_box,
        "right_finger_box": right_box,
        "rank_score": float(score),
    }


def _write_outputs(output_dir: Path, rows: list[dict[str, Any]], roots: list[Path], stage_usd: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "isaac_tabletop_grasp_candidates.json"
    csv_path = output_dir / "isaac_tabletop_grasp_candidates.csv"
    md_path = output_dir / "isaac_tabletop_grasp_candidates.md"
    payload = {
        "stage_usd": str(stage_usd),
        "scan_roots": [str(path) for path in roots],
        "row_count": len(rows),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames = [
        "rank_score",
        "path",
        "frame",
        "raw_gripper",
        "source_score",
        "midpoint_tabletop_height_error_m",
        "closing_dot_object_x_abs",
        "finger_center_distance_m",
        "status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    lines = [
        "# Isaac HDF5 Tabletop Grasp Candidate Scan",
        "",
        f"- stage: `{stage_usd}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Top Candidates",
        "",
        "| rank | score | frame | height error m | closing dot x | center distance m | raw gripper | path |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for idx, row in enumerate(rows[:30], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    f"{row.get('rank_score', float('nan')):.4f}",
                    str(row.get("frame")),
                    f"{row.get('midpoint_tabletop_height_error_m', float('nan')):.4f}",
                    f"{row.get('closing_dot_object_x_abs', float('nan')):.4f}",
                    f"{row.get('finger_center_distance_m', float('nan')):.4f}",
                    f"{row.get('raw_gripper', float('nan')):.4f}",
                    f"`{Path(str(row.get('path'))).name}`",
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score HDF5 grasp candidates by Isaac finger/tabletop geometry.")
    parser.add_argument("--root", action="append", type=Path, default=None)
    parser.add_argument("--stage-usd", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--side", choices=("left",), default="left")
    parser.add_argument("--contact-proxy-profile", choices=tuple(CONTACT_PROXY_PROFILES), default="scene_base_link")
    parser.add_argument("--table-path", default="/scene/worldBody/table")
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--max-frame-offsets", type=int, nargs="*", default=[0, 10, 20, 40])
    parser.add_argument("--open-threshold", type=float, default=0.65)
    parser.add_argument("--close-threshold", type=float, default=0.35)
    parser.add_argument("--open-close-lookahead", type=int, default=120)
    parser.add_argument("--max-open-close-frames-per-file", type=int, default=4)
    parser.add_argument(
        "--sample-frame-step",
        type=int,
        default=0,
        help="Also sample every Nth frame for each selected HDF5. 0 disables dense frame sampling.",
    )
    parser.add_argument("--gravity", type=float, default=-9.81)
    args = parser.parse_args()

    roots = args.root or DEFAULT_ROOTS
    candidates = _scan_candidates(roots, top_k=args.top_k)

    from isaacsim import SimulationApp

    app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
    app_config["fast_shutdown"] = False
    _app = SimulationApp(app_config)
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        import isaacsim.core.utils.stage as stage_utils
        import omni.usd

        stage_utils.open_stage(str(args.stage_usd.resolve()))
        World.clear_instance()
        world = World(stage_units_in_meters=1.0, backend="numpy", device="cpu")
        world.set_simulation_dt(physics_dt=0.02, rendering_dt=0.02)
        stage = omni.usd.get_context().get_stage()
        profile_args = argparse.Namespace(contact_proxy_profile=args.contact_proxy_profile, side=args.side)
        paths = _finger_proxy_paths_for_args(profile_args)[args.side]
        art = world.scene.add(SingleArticulation(prim_path=paths["articulation"], name=f"{args.side}_vx300s"))
        world.reset()
        _apply_gravity(world, args.gravity)
        table_box = _bbox_row(stage, args.table_path)
        if not table_box.get("bbox_valid"):
            raise RuntimeError(f"table bbox invalid: {args.table_path}")
        table_top_z = float(table_box["max"][2])
        mapping = load_mapping(args.mapping)
        finger_dof_names = finger_dof_names_for_side(args.contact_proxy_profile, args.side)
        finger_qpos_limits = finger_qpos_limits_for_side(args.contact_proxy_profile, args.side)
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            qpos = _load_hdf5_qpos(str(candidate.path), start=None, end=None, max_frames=None)
            qpos_for_targets = _qpos_for_scoring(qpos, side=args.side)
            targets, _summary = _targets_from_hdf5_qpos(
                art=art,
                side=args.side,
                qpos=qpos_for_targets,
                mapping=mapping,
                replay_mode="left_arm_and_gripper",
                finger_dof_names=finger_dof_names,
                finger_qpos_limits=finger_qpos_limits,
            )
            frame_indices = {candidate.close_frame}
            frame_indices.update(
                _open_then_close_frame_indices(
                    qpos[:, 6 if args.side == "left" else 13],
                    open_threshold=float(args.open_threshold),
                    close_threshold=float(args.close_threshold),
                    lookahead=int(args.open_close_lookahead),
                    max_indices=int(args.max_open_close_frames_per_file),
                )
            )
            for offset in args.max_frame_offsets:
                frame_indices.add(min(len(targets) - 1, candidate.close_frame + int(offset)))
            frame_indices.add(len(targets) - 1)
            if int(args.sample_frame_step) > 0:
                frame_indices.update(range(0, len(targets), int(args.sample_frame_step)))
            for frame in sorted(index for index in frame_indices if 0 <= index < len(targets)):
                raw = float(qpos[frame, 6])
                rows.append(
                    _row_for_frame(
                        world=world,
                        stage=stage,
                        art=art,
                        paths=paths,
                        target=targets[frame],
                        table_top_z=table_top_z,
                        file_path=candidate.path,
                        frame=frame,
                        raw_gripper=raw,
                        source_score=candidate.score,
                    )
                )
        rows.sort(key=lambda row: float(row.get("rank_score", float("inf"))))
        _write_outputs(args.output_dir, rows, [Path(path) for path in roots], args.stage_usd)
        print(json.dumps({"status": "PASS", "rows": len(rows), "output_dir": str(args.output_dir)}, ensure_ascii=False))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except Exception:
        sys.stdout.flush()
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
