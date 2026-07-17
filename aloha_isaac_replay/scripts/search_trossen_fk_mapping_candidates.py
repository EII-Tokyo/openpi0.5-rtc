from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from aloha_isaac_replay.adapters.standard_aloha import STANDARD_ALOHA_14D_NAMES
from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import DEFAULT_CANDIDATES_JSON
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import DEFAULT_HDF5_ROOT
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import DEFAULT_LEFT_URDF
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import DEFAULT_OUTPUT_DIR as PHASE7_OUTPUT_DIR
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import DEFAULT_SCAFFOLD_USD
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import TROSSEN_EE_BODY
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import _discover_episode
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import _indices
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import _kabsch_align
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import _load_candidate_rows
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import _load_qpos
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import _rel
from aloha_isaac_replay.scripts.compare_aloha_fk import _link_transform
from aloha_isaac_replay.scripts.compare_aloha_fk import _pin_ee
from aloha_isaac_replay.scripts.compare_aloha_fk import _pin_model


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase8_fk_mapping_search_20260717"

LEFT_SEARCH_JOINTS = (
    "left_waist",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
)
LEFT_FIXED_JOINTS = (
    "left_shoulder",
    "left_elbow",
)


def _candidate_options(name: str, row: dict[str, Any], qpos: np.ndarray) -> list[dict[str, Any]]:
    if name == "left_forearm_roll":
        source_idx = STANDARD_ALOHA_14D_NAMES.index(name)
        samples = qpos[:, source_idx]
        lower = float(row["candidate_plus"]["mapped_min"] - row["candidate_plus"]["min_limit_margin"])
        upper = float(row["candidate_plus"]["mapped_max"] + row["candidate_plus"]["min_limit_margin"])
        # The formula above reconstructs poorly when min margin is negative.
        # Prefer the known Phase 5 runtime limits if present in candidate rows
        # after preserving compatibility with older output.
        if "trossen_runtime_limit" in row:
            lower, upper = map(float, row["trossen_runtime_limit"])
        else:
            lower, upper = -np.pi / 2.0, np.pi / 2.0
        options = []
        for sign in (1.0, -1.0):
            lo = float(np.max(lower - sign * samples))
            hi = float(np.min(upper - sign * samples))
            if lo > hi:
                continue
            for offset in (lo, (lo + hi) / 2.0, hi):
                options.append(
                    {
                        "sign": int(sign),
                        "offset": float(offset),
                        "source": "limit_interval_from_current_episode",
                    }
                )
        return _dedupe_options(options)
    if name in LEFT_SEARCH_JOINTS:
        return _dedupe_options(
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
    return [
        {
            "sign": int(row["selected_sign"]),
            "offset": float(row["selected_offset"]),
            "source": "phase6_selected",
        }
    ]


def _dedupe_options(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for option in options:
        key = (int(option["sign"]), round(float(option["offset"]), 8))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(option)
    return deduped


def _left_mapping_values(frame: np.ndarray, rows: dict[str, dict[str, Any]], combo: dict[str, dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    values = []
    names = []
    for canonical in (
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
    ):
        row = rows[canonical]
        option = combo[canonical]
        source_idx = STANDARD_ALOHA_14D_NAMES.index(canonical)
        values.append(float(option["sign"]) * float(frame[source_idx]) + float(option["offset"]))
        names.append(row["trossen_dof"])
    return np.asarray(values, dtype=np.float64), names


def _score(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    raw_errors = np.linalg.norm(candidate - reference, axis=1)
    aligned, rotation, translation = _kabsch_align(candidate, reference)
    aligned_errors = np.linalg.norm(aligned - reference, axis=1)
    return {
        "raw_rmse_m": float(np.sqrt(np.mean(np.square(raw_errors)))),
        "raw_max_m": float(np.max(raw_errors)),
        "rigid_aligned_rmse_m": float(np.sqrt(np.mean(np.square(aligned_errors)))),
        "rigid_aligned_max_m": float(np.max(aligned_errors)),
        "rigid_alignment_rotation": rotation.tolist(),
        "rigid_alignment_translation": translation.tolist(),
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 8 - FK Mapping Candidate Search - 2026-07-17",
        "",
        "## Scope",
        "",
        "This is an offline diagnostic search over a small set of ALOHA1-to-Trossen joint sign and offset candidates.",
        "",
        "It does not touch the real robot, does not save the USD stage, and does not validate a controller.",
        "",
        "## Inputs",
        "",
        f"- episode: `{payload['inputs']['episode']}`",
        f"- frames: `{payload['summary']['frames']}`",
        f"- combinations tested: `{payload['summary']['combination_count']}`",
        "",
        "## Gates",
        "",
    ]
    for key, value in payload["gates"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Best Candidates", ""])
    lines.append("| rank | RMSE m | max m | left_forearm_roll sign | left_forearm_roll offset | mapping summary |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    for rank, row in enumerate(payload["top_candidates"][:10], start=1):
        forearm = row["combo"]["left_forearm_roll"]
        summary = ", ".join(
            f"{name}:{opt['sign']}@{opt['offset']:.3f}" for name, opt in row["combo"].items() if name in LEFT_SEARCH_JOINTS
        )
        lines.append(
            "| "
            f"{rank} | "
            f"{row['score']['rigid_aligned_rmse_m']:.6f} | "
            f"{row['score']['rigid_aligned_max_m']:.6f} | "
            f"{forearm['sign']} | "
            f"{forearm['offset']:.6f} | "
            f"`{summary}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A low rigid-aligned FK error is useful evidence for a candidate mapping, but it is not sufficient for real control.",
            "",
            "The candidate must still be checked against independent trajectories, orientation, gripper semantics, joint limits over the full dataset, and real positive-direction evidence before any controller work.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Search small FK-based ALOHA1-to-Trossen mapping candidates.")
    parser.add_argument("--episode", type=Path, default=None)
    parser.add_argument("--hdf5-root", type=Path, default=DEFAULT_HDF5_ROOT)
    parser.add_argument("--scaffold-usd", type=Path, default=DEFAULT_SCAFFOLD_USD)
    parser.add_argument("--candidates-json", type=Path, default=DEFAULT_CANDIDATES_JSON)
    parser.add_argument("--left-urdf", type=Path, default=DEFAULT_LEFT_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--normal-close", action="store_true")
    args = parser.parse_args()

    episode = args.episode if args.episode is not None else _discover_episode(args.hdf5_root)
    qpos = _load_qpos(episode, args.max_frames, args.stride)
    _, candidate_rows = _load_candidate_rows(args.candidates_json)

    options = {
        name: _candidate_options(name, candidate_rows[name], qpos)
        for name in ("left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate")
    }
    combinations = []
    keys = list(options)
    for values in itertools.product(*(options[key] for key in keys)):
        combinations.append(dict(zip(keys, values)))

    from isaacsim import SimulationApp

    app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
    app_config["fast_shutdown"] = False
    app = SimulationApp(app_config)
    try:
        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation

        left_model, left_data = _pin_model(args.left_urdf)
        reference = []
        for frame in qpos:
            ref_pos, _ = _pin_ee(left_model, left_data, "left", frame)
            reference.append(ref_pos)
        reference_arr = np.asarray(reference)

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
            positions = []
            for frame in qpos:
                values, names = _left_mapping_values(frame, candidate_rows, combo)
                art.set_joint_positions(values, joint_indices=_indices(dof_names, names))
                art.set_joint_velocities(np.zeros_like(values), joint_indices=_indices(dof_names, names))
                world.step(render=False)
                pos, _ = _link_transform(art, TROSSEN_EE_BODY["left"])
                positions.append(pos)
            score = _score(reference_arr, np.asarray(positions))
            scored.append({"combo": combo, "score": score})
        scored.sort(key=lambda item: (item["score"]["rigid_aligned_rmse_m"], item["score"]["rigid_aligned_max_m"]))

        best = scored[0]
        payload = {
            "inputs": {
                "episode": _rel(episode),
                "scaffold_usd": _rel(args.scaffold_usd),
                "candidates_json": _rel(args.candidates_json),
                "left_urdf": _rel(args.left_urdf),
            },
            "summary": {
                "frames": int(qpos.shape[0]),
                "stride": int(args.stride),
                "combination_count": len(scored),
                "searched_joints": list(LEFT_SEARCH_JOINTS),
                "fixed_joints": list(LEFT_FIXED_JOINTS),
            },
            "options": options,
            "top_candidates": scored[:10],
            "gates": {
                "real_robot_touched": "PASS_FALSE",
                "stage_saved": "PASS_FALSE",
                "isaac_runtime_started": "PASS",
                "fk_search_executed": "PASS",
                "best_rigid_aligned_rmse_m": best["score"]["rigid_aligned_rmse_m"],
                "candidate_for_next_validation": "PASS_DIAGNOSTIC_ONLY_NOT_CONTROLLER_READY",
                "controller": "BLOCKED_NOT_ATTEMPTED",
            },
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.output_dir / "fk_mapping_search.json"
        md_path = args.output_dir / "fk_mapping_search.md"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(_render_markdown(payload), encoding="utf-8")
        print(
            json.dumps(
                {
                    "json": _rel(json_path),
                    "markdown": _rel(md_path),
                    "best": {
                        "score": best["score"],
                        "combo": best["combo"],
                    },
                    "combination_count": len(scored),
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
