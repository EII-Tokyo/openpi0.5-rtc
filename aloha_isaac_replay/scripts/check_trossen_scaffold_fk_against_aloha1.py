from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from aloha_isaac_replay.adapters.standard_aloha import STANDARD_ALOHA_14D_NAMES
from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG
from aloha_isaac_replay.scripts.compare_aloha_fk import _link_transform
from aloha_isaac_replay.scripts.compare_aloha_fk import _pin_ee
from aloha_isaac_replay.scripts.compare_aloha_fk import _pin_model
from aloha_isaac_replay.scripts.infer_trossen_aloha1_affine_candidates import ARM_CANONICAL_NAMES


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCAFFOLD_USD = (
    REPO_ROOT
    / "local_eval_assets/aloha1_trossen_backed_scaffold_20260717/aloha1_trossen_backed_scaffold.usda"
)
DEFAULT_CANDIDATES_JSON = (
    REPO_ROOT / "reports/aloha1_isaac_adaptation/phase6_affine_candidate_inference_20260717/affine_candidates.json"
)
DEFAULT_LEFT_URDF = REPO_ROOT / "assets/isaac/original_stationary_aloha/generated/puppet_left_vx300s_resolved.urdf"
DEFAULT_RIGHT_URDF = REPO_ROOT / "assets/isaac/original_stationary_aloha/generated/puppet_right_vx300s_resolved.urdf"
DEFAULT_HDF5_ROOT = REPO_ROOT / "local_rlt_data/raw_from_103/rollouts/key_regions"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase7_trossen_fk_candidate_check_20260717"

TROSSEN_EE_BODY = {
    "left": "follower_left_link_6",
    "right": "follower_right_link_6",
}


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _discover_episode(root: Path) -> Path:
    episodes = sorted(root.rglob("episode.hdf5"))
    if not episodes:
        raise FileNotFoundError(f"No episode.hdf5 found under {root}")
    return episodes[0]


def _load_qpos(path: Path, max_frames: int | None, stride: int) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if "observations/qpos" not in h5:
            raise KeyError(f"{path} is missing observations/qpos")
        qpos = np.asarray(h5["observations/qpos"][:], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] < 14:
        raise ValueError(f"Expected /observations/qpos shape (T, >=14), got {qpos.shape}")
    qpos = qpos[::stride, :14]
    if max_frames is not None:
        qpos = qpos[:max_frames]
    if len(qpos) < 3:
        raise ValueError(f"Need at least 3 sampled frames for FK shape comparison, got {len(qpos)}")
    return qpos


def _load_candidate_rows(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {row["canonical_name"]: row for row in payload["candidates"]}
    missing = [name for name in ARM_CANONICAL_NAMES if name not in rows]
    if missing:
        raise ValueError(f"Candidate JSON missing rows: {missing}")
    return payload, rows


def _candidate_completeness(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fail = [name for name, row in rows.items() if str(row["status"]).startswith("FAIL")]
    ambiguous = [name for name, row in rows.items() if str(row["status"]).startswith("AMBIGUOUS")]
    unique = [name for name, row in rows.items() if str(row["status"]).startswith("PASS_LIMIT_FIT_UNIQUE")]
    if fail or ambiguous:
        status = f"BLOCKED_{len(fail)}_FAIL_{len(ambiguous)}_AMBIGUOUS"
    else:
        status = "PASS_ALL_UNIQUE_LIMIT_FIT_CANDIDATES"
    return {"status": status, "fail": fail, "ambiguous": ambiguous, "unique": unique}


def _mapped_trossen_q(qpos_frame: np.ndarray, rows: dict[str, dict[str, Any]], dof_names: list[str]) -> tuple[np.ndarray, list[str]]:
    values = []
    names = []
    for canonical in ARM_CANONICAL_NAMES:
        row = rows[canonical]
        source_idx = STANDARD_ALOHA_14D_NAMES.index(canonical)
        dof_name = row["trossen_dof"]
        if dof_name not in dof_names:
            raise ValueError(f"Trossen DOF {dof_name!r} is missing from runtime DOFs: {dof_names}")
        value = float(row["selected_sign"]) * float(qpos_frame[source_idx]) + float(row["selected_offset"])
        values.append(value)
        names.append(dof_name)
    return np.asarray(values, dtype=np.float64), names


def _indices(actual_dof_names: list[str], names: list[str]) -> np.ndarray:
    return np.asarray([actual_dof_names.index(name) for name in names], dtype=np.int64)


def _kabsch_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source aligned to target with the best rigid transform.

    This removes unknown fixed base-frame offsets between the original ALOHA1
    URDF asset and the Trossen scaffold. It does not hide joint-sign mistakes:
    wrong signs or offsets change the trajectory shape and remain visible after
    this rigid alignment.
    """

    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    h = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = target_centroid - r @ source_centroid
    aligned = (r @ source.T).T + t
    return aligned, r, t


def _position_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "frame",
        "side",
        "aloha_fk_x",
        "aloha_fk_y",
        "aloha_fk_z",
        "trossen_fk_x",
        "trossen_fk_y",
        "trossen_fk_z",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 7 - Trossen Scaffold FK Candidate Check - 2026-07-17",
        "",
        "## Scope",
        "",
        "This is an offline Isaac headless check. It does not command the real robot and does not save the stage.",
        "",
        "It compares:",
        "",
        "- trusted ALOHA1 FK from the generated VX300S URDF resolved from the archived 103 robot description;",
        "- Trossen `stationary_ai` scaffold FK after applying the Phase 6 affine candidate mapping.",
        "",
        "The check reports both raw position error and rigid-aligned trajectory-shape error. Rigid alignment removes an unknown fixed base transform; it does not prove a bad mapping correct.",
        "",
        "## Inputs",
        "",
        f"- episode: `{payload['inputs']['episode']}`",
        f"- frames: `{payload['summary']['frames']}`",
        f"- scaffold USD: `{payload['inputs']['scaffold_usd']}`",
        f"- candidate JSON: `{payload['inputs']['candidates_json']}`",
        "",
        "## Gates",
        "",
    ]
    for key, value in payload["gates"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Candidate Completeness", ""])
    comp = payload["candidate_completeness"]
    lines.append(f"- unique: `{len(comp['unique'])}`")
    lines.append(f"- ambiguous: `{len(comp['ambiguous'])}` {comp['ambiguous']}")
    lines.append(f"- fail: `{len(comp['fail'])}` {comp['fail']}")
    lines.extend(["", "## FK Position Metrics", ""])
    lines.append("| side | raw RMSE m | raw max m | rigid-aligned RMSE m | rigid-aligned max m |")
    lines.append("|---|---:|---:|---:|---:|")
    for side in ("left", "right"):
        metrics = payload["fk_metrics"][side]
        lines.append(
            "| "
            f"{side} | "
            f"{metrics['raw_rmse_m']:.6f} | "
            f"{metrics['raw_max_m']:.6f} | "
            f"{metrics['rigid_aligned_rmse_m']:.6f} | "
            f"{metrics['rigid_aligned_max_m']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This run is not allowed to pass the final ALOHA1-to-Trossen mapping gate because Phase 6 still has one failed joint candidate and seven ambiguous candidates.",
            "",
            "The FK numbers are therefore diagnostic evidence only. The next valid step is to collect stronger geometric evidence, especially for `left_forearm_roll` and the ambiguous sign rows, not to drive a controller.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Trossen scaffold FK under candidate mapping against trusted ALOHA1 FK.")
    parser.add_argument("--episode", type=Path, default=None)
    parser.add_argument("--hdf5-root", type=Path, default=DEFAULT_HDF5_ROOT)
    parser.add_argument("--scaffold-usd", type=Path, default=DEFAULT_SCAFFOLD_USD)
    parser.add_argument("--candidates-json", type=Path, default=DEFAULT_CANDIDATES_JSON)
    parser.add_argument("--left-urdf", type=Path, default=DEFAULT_LEFT_URDF)
    parser.add_argument("--right-urdf", type=Path, default=DEFAULT_RIGHT_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--normal-close", action="store_true")
    args = parser.parse_args()

    episode = args.episode if args.episode is not None else _discover_episode(args.hdf5_root)
    qpos = _load_qpos(episode, args.max_frames, args.stride)
    candidate_payload, candidate_rows = _load_candidate_rows(args.candidates_json)
    candidate_completeness = _candidate_completeness(candidate_rows)

    from isaacsim import SimulationApp

    app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
    app_config["fast_shutdown"] = False
    app = SimulationApp(app_config)
    try:
        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation

        left_model, left_data = _pin_model(args.left_urdf)
        right_model, right_data = _pin_model(args.right_urdf)

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

        reference = {"left": [], "right": []}
        trossen = {"left": [], "right": []}
        csv_rows: list[dict[str, Any]] = []
        for frame_idx, frame in enumerate(qpos):
            mapped_q, mapped_names = _mapped_trossen_q(frame, candidate_rows, dof_names)
            art.set_joint_positions(mapped_q, joint_indices=_indices(dof_names, mapped_names))
            art.set_joint_velocities(np.zeros_like(mapped_q), joint_indices=_indices(dof_names, mapped_names))
            world.step(render=False)

            for side, model, data in (("left", left_model, left_data), ("right", right_model, right_data)):
                ref_pos, _ = _pin_ee(model, data, side, frame)
                trossen_pos, _ = _link_transform(art, TROSSEN_EE_BODY[side])
                reference[side].append(ref_pos)
                trossen[side].append(trossen_pos)
                csv_rows.append(
                    {
                        "frame": frame_idx,
                        "side": side,
                        "aloha_fk_x": float(ref_pos[0]),
                        "aloha_fk_y": float(ref_pos[1]),
                        "aloha_fk_z": float(ref_pos[2]),
                        "trossen_fk_x": float(trossen_pos[0]),
                        "trossen_fk_y": float(trossen_pos[1]),
                        "trossen_fk_z": float(trossen_pos[2]),
                    }
                )

        fk_metrics = {
            side: _position_metrics(np.asarray(reference[side]), np.asarray(trossen[side]))
            for side in ("left", "right")
        }
        mapping_complete = candidate_completeness["status"].startswith("PASS_")
        shape_threshold_m = 0.02
        shape_ok = all(metrics["rigid_aligned_rmse_m"] <= shape_threshold_m for metrics in fk_metrics.values())
        fk_gate = (
            "BLOCKED_MAPPING_CANDIDATES_INCOMPLETE"
            if not mapping_complete
            else ("PASS_RIGID_ALIGNED_POSITION_SHAPE" if shape_ok else "FAIL_RIGID_ALIGNED_POSITION_SHAPE")
        )
        payload = {
            "inputs": {
                "episode": _rel(episode),
                "scaffold_usd": _rel(args.scaffold_usd),
                "candidates_json": _rel(args.candidates_json),
                "left_urdf": _rel(args.left_urdf),
                "right_urdf": _rel(args.right_urdf),
                "phase6_summary": candidate_payload.get("summary", {}),
            },
            "summary": {
                "frames": int(qpos.shape[0]),
                "stride": int(args.stride),
                "trossen_ee_body": TROSSEN_EE_BODY,
                "shape_threshold_m": shape_threshold_m,
            },
            "candidate_completeness": candidate_completeness,
            "fk_metrics": fk_metrics,
            "gates": {
                "real_robot_touched": "PASS_FALSE",
                "stage_saved": "PASS_FALSE",
                "isaac_runtime_started": "PASS",
                "trusted_aloha1_fk_loaded": "PASS",
                "trossen_scaffold_fk_loaded": "PASS",
                "candidate_mapping_complete": candidate_completeness["status"],
                "fk_position_shape": fk_gate,
                "orientation": "BLOCKED_FRAME_ALIGNMENT_NOT_ESTABLISHED",
                "controller": "BLOCKED_NOT_ATTEMPTED",
                "gripper": "BLOCKED_NOT_ATTEMPTED",
            },
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.output_dir / "trossen_fk_candidate_check.json"
        md_path = args.output_dir / "trossen_fk_candidate_check.md"
        csv_path = args.output_dir / "trossen_fk_candidate_points.csv"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(_render_markdown(payload), encoding="utf-8")
        _write_csv(csv_path, csv_rows)
        print(
            json.dumps(
                {
                    "json": _rel(json_path),
                    "markdown": _rel(md_path),
                    "csv": _rel(csv_path),
                    "gates": payload["gates"],
                    "fk_metrics": fk_metrics,
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
