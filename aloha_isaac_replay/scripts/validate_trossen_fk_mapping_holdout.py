from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import DEFAULT_CANDIDATES_JSON
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import DEFAULT_HDF5_ROOT
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import DEFAULT_LEFT_URDF
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import DEFAULT_SCAFFOLD_USD
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import TROSSEN_EE_BODY
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import _indices
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import _load_candidate_rows
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import _rel
from aloha_isaac_replay.scripts.compare_aloha_fk import _link_transform
from aloha_isaac_replay.scripts.compare_aloha_fk import _pin_ee
from aloha_isaac_replay.scripts.compare_aloha_fk import _pin_model
from aloha_isaac_replay.scripts.search_trossen_fk_mapping_candidates import LEFT_FIXED_JOINTS
from aloha_isaac_replay.scripts.search_trossen_fk_mapping_candidates import LEFT_SEARCH_JOINTS
from aloha_isaac_replay.scripts.search_trossen_fk_mapping_candidates import _candidate_options
from aloha_isaac_replay.scripts.search_trossen_fk_mapping_candidates import _left_mapping_values
from aloha_isaac_replay.scripts.search_trossen_fk_mapping_candidates import _score


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase9_fk_mapping_holdout_20260717"


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
    qpos = qpos[::stride, :14]
    qpos = qpos[:max_frames]
    if len(qpos) < 3:
        raise ValueError(f"{path} has too few sampled frames: {len(qpos)}")
    return qpos


def _stack_qpos(paths: list[Path], max_frames: int, stride: int) -> np.ndarray:
    return np.concatenate([_load_episode_qpos(path, max_frames, stride) for path in paths], axis=0)


def _make_combinations(candidate_rows: dict[str, dict[str, Any]], search_qpos: np.ndarray) -> list[dict[str, dict[str, Any]]]:
    import itertools

    keys = ["left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate"]
    options = {name: _candidate_options(name, candidate_rows[name], search_qpos) for name in keys}
    return [dict(zip(keys, values)) for values in itertools.product(*(options[key] for key in keys))]


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
) -> dict[str, Any]:
    reference = []
    candidate = []
    for frame in qpos:
        ref_pos, _ = _pin_ee(left_model, left_data, "left", frame)
        values, names = _left_mapping_values(frame, candidate_rows, combo)
        art.set_joint_positions(values, joint_indices=_indices(dof_names, names))
        art.set_joint_velocities(np.zeros_like(values), joint_indices=_indices(dof_names, names))
        world.step(render=False)
        trossen_pos, _ = _link_transform(art, TROSSEN_EE_BODY["left"])
        reference.append(ref_pos)
        candidate.append(trossen_pos)
    return _score(np.asarray(reference), np.asarray(candidate))


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 9 - FK Mapping Holdout Validation - 2026-07-17",
        "",
        "## Scope",
        "",
        "This is an offline search/holdout validation for the left-arm ALOHA1-to-Trossen FK mapping candidate.",
        "",
        "It does not touch the real robot, does not save the USD stage, and does not validate controller execution.",
        "",
        "## Dataset",
        "",
        f"- search episodes: `{len(payload['inputs']['search_episodes'])}`",
        f"- holdout episodes: `{len(payload['inputs']['holdout_episodes'])}`",
        f"- frames per episode: `{payload['summary']['max_frames_per_episode']}`",
        f"- stride: `{payload['summary']['stride']}`",
        f"- combinations tested: `{payload['summary']['combination_count']}`",
        "",
        "## Gates",
        "",
    ]
    for key, value in payload["gates"].items():
        lines.append(f"- {key}: `{value}`")
    best = payload["best"]
    lines.extend(["", "## Best Candidate", ""])
    lines.append("| set | rigid-aligned RMSE m | rigid-aligned max m | raw RMSE m |")
    lines.append("|---|---:|---:|---:|")
    for split in ("search", "holdout"):
        score = best[f"{split}_score"]
        lines.append(
            "| "
            f"{split} | {score['rigid_aligned_rmse_m']:.6f} | {score['rigid_aligned_max_m']:.6f} | {score['raw_rmse_m']:.6f} |"
        )
    lines.extend(["", "Mapping:", ""])
    for name, option in best["combo"].items():
        lines.append(f"- `{name}`: sign `{option['sign']}`, offset `{option['offset']:.6f}`, source `{option['source']}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is stronger evidence than Phase 8 because the selected candidate is scored on episodes that did not participate in the search.",
            "",
            "It is still not controller-ready unless the holdout error is low enough and the same mapping is supported by positive-direction or matched-pose evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Search ALOHA1-to-Trossen FK mapping on one split and validate on holdout.")
    parser.add_argument("--hdf5-root", type=Path, default=DEFAULT_HDF5_ROOT)
    parser.add_argument("--scaffold-usd", type=Path, default=DEFAULT_SCAFFOLD_USD)
    parser.add_argument("--candidates-json", type=Path, default=DEFAULT_CANDIDATES_JSON)
    parser.add_argument("--left-urdf", type=Path, default=DEFAULT_LEFT_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode-limit", type=int, default=12)
    parser.add_argument("--holdout-count", type=int, default=4)
    parser.add_argument("--max-frames-per-episode", type=int, default=8)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--normal-close", action="store_true")
    args = parser.parse_args()

    episodes = _discover_episodes(args.hdf5_root, args.episode_limit)
    holdout_count = min(args.holdout_count, max(1, len(episodes) // 3))
    search_paths = episodes[:-holdout_count]
    holdout_paths = episodes[-holdout_count:]
    search_qpos = _stack_qpos(search_paths, args.max_frames_per_episode, args.stride)
    holdout_qpos = _stack_qpos(holdout_paths, args.max_frames_per_episode, args.stride)
    _, candidate_rows = _load_candidate_rows(args.candidates_json)
    combinations = _make_combinations(candidate_rows, search_qpos)

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

        scored = []
        for combo in combinations:
            search_score = _evaluate_combo(
                world=world,
                art=art,
                dof_names=dof_names,
                candidate_rows=candidate_rows,
                combo=combo,
                qpos=search_qpos,
                left_model=left_model,
                left_data=left_data,
            )
            scored.append({"combo": combo, "search_score": search_score})
        scored.sort(key=lambda item: (item["search_score"]["rigid_aligned_rmse_m"], item["search_score"]["rigid_aligned_max_m"]))
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
        )

        holdout_threshold_m = 0.03
        holdout_ok = best["holdout_score"]["rigid_aligned_rmse_m"] <= holdout_threshold_m
        payload = {
            "inputs": {
                "search_episodes": [_rel(path) for path in search_paths],
                "holdout_episodes": [_rel(path) for path in holdout_paths],
                "scaffold_usd": _rel(args.scaffold_usd),
                "candidates_json": _rel(args.candidates_json),
                "left_urdf": _rel(args.left_urdf),
            },
            "summary": {
                "search_frames": int(search_qpos.shape[0]),
                "holdout_frames": int(holdout_qpos.shape[0]),
                "max_frames_per_episode": int(args.max_frames_per_episode),
                "stride": int(args.stride),
                "combination_count": len(scored),
                "searched_joints": list(LEFT_SEARCH_JOINTS),
                "fixed_joints": list(LEFT_FIXED_JOINTS),
                "holdout_threshold_m": holdout_threshold_m,
            },
            "best": best,
            "top_search_candidates": scored[:10],
            "gates": {
                "real_robot_touched": "PASS_FALSE",
                "stage_saved": "PASS_FALSE",
                "isaac_runtime_started": "PASS",
                "search_executed": "PASS",
                "holdout_executed": "PASS",
                "holdout_fk_shape": "PASS_DIAGNOSTIC" if holdout_ok else "FAIL_HOLDOUT_FK_SHAPE",
                "controller": "BLOCKED_NOT_ATTEMPTED",
            },
        }

        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.output_dir / "fk_mapping_holdout.json"
        md_path = args.output_dir / "fk_mapping_holdout.md"
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
        return 0
    finally:
        if args.normal_close:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
