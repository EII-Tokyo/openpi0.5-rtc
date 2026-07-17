from __future__ import annotations

import argparse
import itertools
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
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import TROSSEN_EE_BODY
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import _indices
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import _kabsch_align
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import _load_candidate_rows
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import _rel
from aloha_isaac_replay.scripts.compare_aloha_fk import _link_transform
from aloha_isaac_replay.scripts.compare_aloha_fk import _pin_ee
from aloha_isaac_replay.scripts.compare_aloha_fk import _pin_model
from aloha_isaac_replay.scripts.search_trossen_fk_mapping_candidates import DEFAULT_CANDIDATES_JSON


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE9_JSON = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase9_fk_mapping_holdout_20260717/fk_mapping_holdout.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase14_orientation_aware_mapping_20260718"

LEFT_CHAIN = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
)


def _discover_episodes(root: Path, limit: int) -> list[Path]:
    paths = sorted(root.rglob("episode.hdf5"))[:limit]
    if len(paths) < 2:
        raise FileNotFoundError(f"Need at least two episode.hdf5 files under {root}, got {len(paths)}")
    return paths


def _load_episode_qpos(path: Path, max_frames: int, stride: int) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        qpos = np.asarray(h5["observations/qpos"][:], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] < 14:
        raise ValueError(f"{path} has bad observations/qpos shape {qpos.shape}")
    sampled = qpos[::stride, :14][:max_frames]
    if len(sampled) < 3:
        raise ValueError(f"{path} has too few sampled frames: {len(sampled)}")
    return sampled


def _stack_qpos(paths: list[Path], max_frames: int, stride: int) -> np.ndarray:
    return np.concatenate([_load_episode_qpos(path, max_frames, stride) for path in paths], axis=0)


def _load_phase9_combo(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["best"]["combo"]


def _dedupe(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for option in options:
        key = (option["sign"], round(float(option["offset"]), 8), option.get("source"))
        if key in seen:
            continue
        seen.add(key)
        output.append(option)
    return output


def _phase6_plus_minus(row: dict[str, Any]) -> list[dict[str, Any]]:
    return _dedupe(
        [
            {
                "sign": int(row["candidate_plus"]["sign"]),
                "offset": float(row["candidate_plus"]["offset"]),
                "source": "phase6_plus",
            },
            {
                "sign": int(row["candidate_minus"]["sign"]),
                "offset": float(row["candidate_minus"]["offset"]),
                "source": "phase6_minus",
            },
        ]
    )


def _forearm_options(row: dict[str, Any], qpos: np.ndarray, grid: int) -> list[dict[str, Any]]:
    idx = STANDARD_ALOHA_14D_NAMES.index("left_forearm_roll")
    samples = qpos[:, idx]
    # Trossen VX arm forearm roll is constrained in the runtime asset. Use the
    # same conservative interval used by earlier limit-validation phases.
    lower, upper = -np.pi / 2.0, np.pi / 2.0
    options: list[dict[str, Any]] = []
    for sign in (1, -1):
        lo = float(np.max(lower - sign * samples))
        hi = float(np.min(upper - sign * samples))
        if lo <= hi:
            for offset in np.linspace(lo, hi, grid):
                options.append({"sign": sign, "offset": float(offset), "source": "forearm_limit_grid"})
    options.extend(_phase6_plus_minus(row))
    return _dedupe(options)


def _make_options(candidate_rows: dict[str, dict[str, Any]], phase9_combo: dict[str, dict[str, Any]], qpos: np.ndarray, grid: int) -> dict[str, list[dict[str, Any]]]:
    options: dict[str, list[dict[str, Any]]] = {}
    for name in LEFT_CHAIN:
        row = candidate_rows[name]
        if name in ("left_waist", "left_wrist_angle", "left_wrist_rotate"):
            options[name] = _phase6_plus_minus(row)
        elif name == "left_forearm_roll":
            options[name] = _forearm_options(row, qpos, grid)
        elif name == "left_elbow":
            # The direct axis comparison makes elbow sign suspicious, but the
            # Phase 6 plus candidate violates the previous limit-fit gate.
            # Keep both as diagnostic options and let full-dataset limits block
            # any invalid winner later.
            options[name] = _phase6_plus_minus(row)
        else:
            options[name] = [phase9_combo[name]]
    return options


def _make_combinations(options: dict[str, list[dict[str, Any]]]) -> list[dict[str, dict[str, Any]]]:
    keys = list(LEFT_CHAIN)
    return [dict(zip(keys, values)) for values in itertools.product(*(options[key] for key in keys))]


def _left_values(frame: np.ndarray, candidate_rows: dict[str, dict[str, Any]], combo: dict[str, dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    values = []
    names = []
    for canonical in LEFT_CHAIN:
        row = candidate_rows[canonical]
        option = combo[canonical]
        source_idx = STANDARD_ALOHA_14D_NAMES.index(canonical)
        values.append(float(option["sign"]) * float(frame[source_idx]) + float(option["offset"]))
        names.append(row["trossen_dof"])
    return np.asarray(values, dtype=np.float64), names


def _rotation_from_wxyz(quat: np.ndarray) -> Rotation:
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])


def _orientation_residuals_deg(ref_quats_wxyz: list[np.ndarray], candidate_quats_wxyz: list[np.ndarray]) -> np.ndarray:
    ref_rots = [_rotation_from_wxyz(quat) for quat in ref_quats_wxyz]
    cand_rots = [_rotation_from_wxyz(quat) for quat in candidate_quats_wxyz]
    fixed_delta = ref_rots[0] * cand_rots[0].inv()
    residuals = []
    for ref_rot, cand_rot in zip(ref_rots, cand_rots, strict=True):
        delta = ref_rot * cand_rot.inv()
        residuals.append((fixed_delta.inv() * delta).magnitude() * 180.0 / np.pi)
    return np.asarray(residuals, dtype=np.float64)


def _score(reference_pos: np.ndarray, candidate_pos: np.ndarray, reference_quats: list[np.ndarray], candidate_quats: list[np.ndarray], orientation_weight_m_per_deg: float) -> dict[str, Any]:
    _, _, _ = _kabsch_align(candidate_pos, reference_pos)
    aligned, rotation, translation = _kabsch_align(candidate_pos, reference_pos)
    aligned_errors = np.linalg.norm(aligned - reference_pos, axis=1)
    raw_errors = np.linalg.norm(candidate_pos - reference_pos, axis=1)
    ori = _orientation_residuals_deg(reference_quats, candidate_quats)
    pos_rmse = float(np.sqrt(np.mean(np.square(aligned_errors))))
    ori_p95 = float(np.quantile(ori, 0.95))
    return {
        "composite_score": pos_rmse + orientation_weight_m_per_deg * ori_p95,
        "position_rigid_aligned_rmse_m": pos_rmse,
        "position_rigid_aligned_max_m": float(np.max(aligned_errors)),
        "position_raw_rmse_m": float(np.sqrt(np.mean(np.square(raw_errors)))),
        "orientation_mean_deg": float(np.mean(ori)),
        "orientation_p95_deg": ori_p95,
        "orientation_max_deg": float(np.max(ori)),
        "rigid_alignment_rotation": rotation.tolist(),
        "rigid_alignment_translation": translation.tolist(),
    }


def _evaluate_combo(
    *,
    world: Any,
    art: Any,
    dof_names: list[str],
    candidate_rows: dict[str, dict[str, Any]],
    combo: dict[str, dict[str, Any]],
    qpos: np.ndarray,
    left_model: Any,
    left_data: Any,
    orientation_weight_m_per_deg: float,
) -> dict[str, Any]:
    ref_pos = []
    ref_quats = []
    cand_pos = []
    cand_quats = []
    for frame in qpos:
        pos, quat = _pin_ee(left_model, left_data, "left", frame)
        values, names = _left_values(frame, candidate_rows, combo)
        art.set_joint_positions(values, joint_indices=_indices(dof_names, names))
        art.set_joint_velocities(np.zeros_like(values), joint_indices=_indices(dof_names, names))
        world.step(render=False)
        trossen_pos, trossen_quat = _link_transform(art, TROSSEN_EE_BODY["left"])
        ref_pos.append(pos)
        ref_quats.append(quat)
        cand_pos.append(trossen_pos)
        cand_quats.append(trossen_quat)
    return _score(np.asarray(ref_pos), np.asarray(cand_pos), ref_quats, cand_quats, orientation_weight_m_per_deg)


def _render_markdown(payload: dict[str, Any]) -> str:
    best = payload["best"]
    lines = [
        "# Phase 14 - Orientation-Aware Mapping Search - 2026-07-18",
        "",
        "## Scope",
        "",
        "This is an offline search over ALOHA1-to-Trossen sign/offset candidates using both end-effector position and orientation.",
        "",
        "It does not touch the real robot, save a stage, execute a controller, or validate contact/gripper behavior.",
        "",
        "## Dataset",
        "",
        f"- search episodes: `{len(payload['inputs']['search_episodes'])}`",
        f"- holdout episodes: `{len(payload['inputs']['holdout_episodes'])}`",
        f"- combinations tested: `{payload['summary']['combination_count']}`",
        f"- orientation weight: `{payload['summary']['orientation_weight_m_per_deg']}` m per degree",
        "",
        "## Gates",
        "",
    ]
    for key, value in payload["gates"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Best Candidate",
            "",
            "| split | composite | pos RMSE m | pos max m | orientation p95 deg | orientation max deg |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ("search", "holdout"):
        score = best[f"{split}_score"]
        lines.append(
            "| "
            f"{split} | {score['composite_score']:.6f} | {score['position_rigid_aligned_rmse_m']:.6f} | "
            f"{score['position_rigid_aligned_max_m']:.6f} | {score['orientation_p95_deg']:.6f} | {score['orientation_max_deg']:.6f} |"
        )
    lines.extend(["", "Mapping:", ""])
    for name, option in best["combo"].items():
        lines.append(f"- `{name}`: sign `{option['sign']}`, offset `{option['offset']:.6f}`, source `{option['source']}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This phase is the first gate that penalizes a candidate for matching position while rotating the wrist/forearm chain incorrectly.",
            "",
            "A controller remains blocked unless the holdout orientation p95 is small enough and the mapping also passes full-dataset joint limits.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Search ALOHA1-to-Trossen mappings with position and orientation constraints.")
    parser.add_argument("--hdf5-root", type=Path, default=DEFAULT_HDF5_ROOT)
    parser.add_argument("--scaffold-usd", type=Path, default=DEFAULT_SCAFFOLD_USD)
    parser.add_argument("--candidates-json", type=Path, default=DEFAULT_CANDIDATES_JSON)
    parser.add_argument("--phase9-json", type=Path, default=DEFAULT_PHASE9_JSON)
    parser.add_argument("--left-urdf", type=Path, default=DEFAULT_LEFT_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode-limit", type=int, default=12)
    parser.add_argument("--holdout-count", type=int, default=4)
    parser.add_argument("--max-frames-per-episode", type=int, default=8)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--forearm-grid", type=int, default=7)
    parser.add_argument("--orientation-weight-m-per-deg", type=float, default=0.002)
    parser.add_argument("--orientation-p95-threshold-deg", type=float, default=8.0)
    parser.add_argument("--position-rmse-threshold-m", type=float, default=0.03)
    parser.add_argument("--normal-close", action="store_true")
    args = parser.parse_args()

    episodes = _discover_episodes(args.hdf5_root, args.episode_limit)
    holdout_count = min(args.holdout_count, max(1, len(episodes) // 3))
    search_paths = episodes[:-holdout_count]
    holdout_paths = episodes[-holdout_count:]
    search_qpos = _stack_qpos(search_paths, args.max_frames_per_episode, args.stride)
    holdout_qpos = _stack_qpos(holdout_paths, args.max_frames_per_episode, args.stride)
    _, candidate_rows = _load_candidate_rows(args.candidates_json)
    phase9_combo = _load_phase9_combo(args.phase9_json)
    options = _make_options(candidate_rows, phase9_combo, search_qpos, args.forearm_grid)
    combinations = _make_combinations(options)

    from isaacsim import SimulationApp

    app = SimulationApp(dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG))
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

        scored = []
        for combo in combinations:
            scored.append(
                {
                    "combo": combo,
                    "search_score": _evaluate_combo(
                        world=world,
                        art=art,
                        dof_names=dof_names,
                        candidate_rows=candidate_rows,
                        combo=combo,
                        qpos=search_qpos,
                        left_model=left_model,
                        left_data=left_data,
                        orientation_weight_m_per_deg=args.orientation_weight_m_per_deg,
                    ),
                }
            )
        scored.sort(
            key=lambda item: (
                item["search_score"]["composite_score"],
                item["search_score"]["orientation_p95_deg"],
                item["search_score"]["position_rigid_aligned_rmse_m"],
            )
        )
        best = scored[0]
        best["holdout_score"] = _evaluate_combo(
            world=world,
            art=art,
            dof_names=dof_names,
            candidate_rows=candidate_rows,
            combo=best["combo"],
            qpos=holdout_qpos,
            left_model=left_model,
            left_data=left_data,
            orientation_weight_m_per_deg=args.orientation_weight_m_per_deg,
        )
        holdout_orientation_ok = best["holdout_score"]["orientation_p95_deg"] <= args.orientation_p95_threshold_deg
        holdout_position_ok = best["holdout_score"]["position_rigid_aligned_rmse_m"] <= args.position_rmse_threshold_m
        payload = {
            "inputs": {
                "search_episodes": [_rel(path) for path in search_paths],
                "holdout_episodes": [_rel(path) for path in holdout_paths],
                "scaffold_usd": _rel(args.scaffold_usd),
                "candidates_json": _rel(args.candidates_json),
                "phase9_json": _rel(args.phase9_json),
                "left_urdf": _rel(args.left_urdf),
            },
            "summary": {
                "search_frames": int(search_qpos.shape[0]),
                "holdout_frames": int(holdout_qpos.shape[0]),
                "combination_count": len(scored),
                "options_per_joint": {key: len(value) for key, value in options.items()},
                "orientation_weight_m_per_deg": args.orientation_weight_m_per_deg,
                "orientation_p95_threshold_deg": args.orientation_p95_threshold_deg,
                "position_rmse_threshold_m": args.position_rmse_threshold_m,
            },
            "best": best,
            "top_search_candidates": scored[:10],
            "gates": {
                "real_robot_touched": "PASS_FALSE",
                "stage_saved": "PASS_FALSE",
                "isaac_runtime_started": "PASS",
                "search_executed": "PASS",
                "holdout_executed": "PASS",
                "holdout_position": "PASS_DIAGNOSTIC" if holdout_position_ok else "FAIL_POSITION",
                "holdout_orientation": "PASS_DIAGNOSTIC" if holdout_orientation_ok else "FAIL_ORIENTATION",
                "controller": "BLOCKED_NOT_ATTEMPTED",
            },
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.output_dir / "orientation_aware_mapping.json"
        md_path = args.output_dir / "orientation_aware_mapping.md"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(_render_markdown(payload), encoding="utf-8")
        print(
            json.dumps(
                {
                    "json": _rel(json_path),
                    "markdown": _rel(md_path),
                    "best_search_score": best["search_score"],
                    "best_holdout_score": best["holdout_score"],
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
        return 0
    finally:
        if args.normal_close:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
