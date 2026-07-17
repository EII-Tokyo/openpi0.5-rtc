from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from aloha_isaac_replay.adapters.standard_aloha import STANDARD_ALOHA_14D_NAMES
from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import DEFAULT_HDF5_ROOT
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import DEFAULT_LEFT_URDF
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import DEFAULT_SCAFFOLD_USD
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import _indices
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import _kabsch_align
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import _rel
from aloha_isaac_replay.scripts.compare_aloha_fk import _link_transform
from aloha_isaac_replay.scripts.compare_aloha_fk import _pin_ee
from aloha_isaac_replay.scripts.compare_aloha_fk import _pin_model


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE9_JSON = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase9_fk_mapping_holdout_20260717/fk_mapping_holdout.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase12_trossen_terminal_body_scan_20260718"


def _discover_episodes(root: Path, limit: int) -> list[Path]:
    paths = sorted(root.rglob("episode.hdf5"))[:limit]
    if not paths:
        raise FileNotFoundError(f"No episode.hdf5 files under {root}")
    return paths


def _load_qpos(paths: list[Path], max_frames_per_episode: int, stride: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    arrays = []
    rows = []
    for path in paths:
        with h5py.File(path, "r") as h5:
            if "observations/qpos" not in h5:
                rows.append({"path": _rel(path), "status": "SKIP_NO_QPOS"})
                continue
            qpos = np.asarray(h5["observations/qpos"][:], dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] < 14:
            rows.append({"path": _rel(path), "status": "SKIP_BAD_QPOS_SHAPE", "shape": list(qpos.shape)})
            continue
        sampled = qpos[::stride, :14][:max_frames_per_episode]
        if len(sampled) < 2:
            rows.append({"path": _rel(path), "status": "SKIP_TOO_FEW_SAMPLED_FRAMES", "frames": int(len(sampled))})
            continue
        arrays.append(sampled)
        rows.append({"path": _rel(path), "status": "OK", "frames": int(len(sampled))})
    if not arrays:
        raise ValueError("No valid qpos samples")
    return np.concatenate(arrays, axis=0), rows


def _load_combo(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["best"]["combo"]


def _left_mapping_values(frame: np.ndarray, combo: dict[str, dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    names = []
    values = []
    for joint_idx, canonical in enumerate(
        (
            "left_waist",
            "left_shoulder",
            "left_elbow",
            "left_forearm_roll",
            "left_wrist_angle",
            "left_wrist_rotate",
        )
    ):
        option = combo[canonical]
        source_idx = STANDARD_ALOHA_14D_NAMES.index(canonical)
        names.append(f"follower_left_joint_{joint_idx}")
        values.append(float(option["sign"]) * float(frame[source_idx]) + float(option["offset"]))
    return np.asarray(values, dtype=np.float64), names


def _rotation_from_wxyz(quat: np.ndarray) -> Rotation:
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])


def _orientation_metrics(ref_quats_wxyz: list[np.ndarray], candidate_quats_wxyz: list[np.ndarray]) -> dict[str, Any]:
    ref_rots = [_rotation_from_wxyz(quat) for quat in ref_quats_wxyz]
    cand_rots = [_rotation_from_wxyz(quat) for quat in candidate_quats_wxyz]
    fixed_delta = ref_rots[0] * cand_rots[0].inv()
    residuals = []
    for ref_rot, cand_rot in zip(ref_rots, cand_rots, strict=True):
        delta = ref_rot * cand_rot.inv()
        residuals.append((fixed_delta.inv() * delta).magnitude() * 180.0 / np.pi)
    residuals_array = np.asarray(residuals, dtype=np.float64)
    return {
        "mean_residual_deg": float(np.mean(residuals_array)),
        "p95_residual_deg": float(np.quantile(residuals_array, 0.95)),
        "max_residual_deg": float(np.max(residuals_array)),
        "fixed_delta_quat_xyzw": fixed_delta.as_quat().tolist(),
    }


def _position_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    raw = np.linalg.norm(candidate - reference, axis=1)
    aligned, rotation, translation = _kabsch_align(candidate, reference)
    aligned_error = np.linalg.norm(aligned - reference, axis=1)
    return {
        "raw_rmse_m": float(np.sqrt(np.mean(np.square(raw)))),
        "raw_max_m": float(np.max(raw)),
        "rigid_aligned_rmse_m": float(np.sqrt(np.mean(np.square(aligned_error)))),
        "rigid_aligned_max_m": float(np.max(aligned_error)),
        "rigid_alignment_rotation": rotation.tolist(),
        "rigid_alignment_translation": translation.tolist(),
    }


def _candidate_body_names(body_names: list[str]) -> list[str]:
    candidates = [
        name
        for name in body_names
        if name.startswith("follower_left_")
        and (
            "link_5" in name
            or "link_6" in name
            or "camera" in name
            or "carriage" in name
            or "gripper" in name
        )
    ]
    preferred = [
        "follower_left_link_6",
        "follower_left_camera_mount_d405",
        "follower_left_camera_link",
        "follower_left_carriage_left",
        "follower_left_carriage_right",
        "follower_left_gripper_left",
        "follower_left_gripper_right",
    ]
    ordered = [name for name in preferred if name in candidates]
    ordered.extend(name for name in candidates if name not in ordered)
    return ordered


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 12 - Trossen Terminal Body Scan - 2026-07-18",
        "",
        "## Scope",
        "",
        "This offline diagnostic scans Trossen left-arm body candidates to find which body frame best matches the trusted ALOHA1 end-effector FK under the Phase 9 mapping candidate.",
        "",
        "It does not touch the real robot, save the stage, or validate controller execution.",
        "",
        "## Dataset",
        "",
        f"- valid episodes: `{payload['summary']['valid_episode_count']}`",
        f"- sampled frames: `{payload['summary']['frame_count']}`",
        f"- scanned bodies: `{len(payload['body_results'])}`",
        "",
        "## Gates",
        "",
    ]
    for key, value in payload["gates"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Body Results", ""])
    lines.append("| rank | body | pos RMSE m | pos max m | ori p95 deg | ori max deg |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for rank, row in enumerate(payload["body_results"], start=1):
        pos = row["position_metrics"]
        ori = row["orientation_metrics"]
        lines.append(
            "| "
            f"{rank} | "
            f"`{row['body_name']}` | "
            f"{pos['rigid_aligned_rmse_m']:.6f} | "
            f"{pos['rigid_aligned_max_m']:.6f} | "
            f"{ori['p95_residual_deg']:.6f} | "
            f"{ori['max_residual_deg']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A useful terminal frame should have both low rigid-aligned position error and low orientation residual.",
            "",
            "If every candidate body has large orientation residual, the mapping or frame semantics are still not controller-ready.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan Trossen left-arm terminal body candidates against trusted ALOHA1 FK.")
    parser.add_argument("--hdf5-root", type=Path, default=DEFAULT_HDF5_ROOT)
    parser.add_argument("--phase9-json", type=Path, default=DEFAULT_PHASE9_JSON)
    parser.add_argument("--scaffold-usd", type=Path, default=DEFAULT_SCAFFOLD_USD)
    parser.add_argument("--left-urdf", type=Path, default=DEFAULT_LEFT_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode-limit", type=int, default=16)
    parser.add_argument("--max-frames-per-episode", type=int, default=8)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--normal-close", action="store_true")
    args = parser.parse_args()

    paths = _discover_episodes(args.hdf5_root, args.episode_limit)
    qpos, episode_rows = _load_qpos(paths, args.max_frames_per_episode, args.stride)
    combo = _load_combo(args.phase9_json)

    from isaacsim import SimulationApp

    app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
    app_config["fast_shutdown"] = False
    app = SimulationApp(app_config)
    try:
        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation

        left_model, left_data = _pin_model(args.left_urdf)
        World.clear_instance()
        stage_utils.create_new_stage()
        world = World(stage_units_in_meters=1.0, backend="numpy", device="cpu")
        root = "/World/trossen_stationary_ai"
        stage_utils.add_reference_to_stage(usd_path=str(args.scaffold_usd.resolve()), prim_path=root)
        art = world.scene.add(
            SingleArticulation(
                prim_path=f"{root}/Aloha1TrossenBackedScaffold/root_joint",
                name="aloha1_trossen_backed",
            )
        )
        world.reset()
        dof_names = list(art.dof_names)
        body_names = list(art._articulation_view.body_names)
        candidate_bodies = _candidate_body_names(body_names)

        ref_positions = []
        ref_quats = []
        body_positions = {name: [] for name in candidate_bodies}
        body_quats = {name: [] for name in candidate_bodies}
        for frame in qpos:
            ref_pos, ref_quat = _pin_ee(left_model, left_data, "left", frame)
            values, names = _left_mapping_values(frame, combo)
            art.set_joint_positions(values, joint_indices=_indices(dof_names, names))
            art.set_joint_velocities(np.zeros_like(values), joint_indices=_indices(dof_names, names))
            world.step(render=False)
            ref_positions.append(ref_pos)
            ref_quats.append(ref_quat)
            for body in candidate_bodies:
                pos, quat = _link_transform(art, body)
                body_positions[body].append(pos)
                body_quats[body].append(quat)

        ref_positions_array = np.asarray(ref_positions)
        results = []
        for body in candidate_bodies:
            results.append(
                {
                    "body_name": body,
                    "position_metrics": _position_metrics(ref_positions_array, np.asarray(body_positions[body])),
                    "orientation_metrics": _orientation_metrics(ref_quats, body_quats[body]),
                }
            )
        results.sort(
            key=lambda row: (
                row["orientation_metrics"]["p95_residual_deg"],
                row["position_metrics"]["rigid_aligned_rmse_m"],
            )
        )
        best = results[0]
        orientation_ok = best["orientation_metrics"]["p95_residual_deg"] <= 5.0 and best["orientation_metrics"]["max_residual_deg"] <= 10.0
        payload = {
            "inputs": {
                "hdf5_root": _rel(args.hdf5_root),
                "phase9_json": _rel(args.phase9_json),
                "scaffold_usd": _rel(args.scaffold_usd),
                "left_urdf": _rel(args.left_urdf),
            },
            "summary": {
                "valid_episode_count": sum(1 for row in episode_rows if row["status"] == "OK"),
                "frame_count": int(qpos.shape[0]),
                "candidate_bodies": candidate_bodies,
            },
            "episode_rows": episode_rows,
            "body_results": results,
            "gates": {
                "real_robot_touched": "PASS_FALSE",
                "stage_saved": "PASS_FALSE",
                "isaac_runtime_started": "PASS",
                "body_scan_executed": "PASS",
                "best_orientation_consistency": "PASS_DIAGNOSTIC" if orientation_ok else "FAIL_NO_BODY_WITH_STABLE_ORIENTATION",
                "controller": "BLOCKED_NOT_ATTEMPTED",
            },
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.output_dir / "terminal_body_scan.json"
        md_path = args.output_dir / "terminal_body_scan.md"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(_render_markdown(payload), encoding="utf-8")
        print(
            json.dumps(
                {
                    "json": _rel(json_path),
                    "markdown": _rel(md_path),
                    "best": best,
                    "gates": payload["gates"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        if not args.normal_close:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        return 0 if orientation_ok else 2
    finally:
        if args.normal_close:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
