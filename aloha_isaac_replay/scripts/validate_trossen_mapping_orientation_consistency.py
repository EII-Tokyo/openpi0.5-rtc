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
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import TROSSEN_EE_BODY
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import _indices
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import _rel
from aloha_isaac_replay.scripts.compare_aloha_fk import _link_transform
from aloha_isaac_replay.scripts.compare_aloha_fk import _pin_ee
from aloha_isaac_replay.scripts.compare_aloha_fk import _pin_model


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE9_JSON = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase9_fk_mapping_holdout_20260717/fk_mapping_holdout.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase11_orientation_consistency_20260718"


def _discover_episodes(root: Path, limit: int) -> list[Path]:
    paths = sorted(root.rglob("episode.hdf5"))[:limit]
    if not paths:
        raise FileNotFoundError(f"No episode.hdf5 files under {root}")
    return paths


def _load_qpos(paths: list[Path], max_frames_per_episode: int, stride: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    parts = []
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
        parts.append(sampled)
        rows.append({"path": _rel(path), "status": "OK", "frames": int(len(sampled))})
    if not parts:
        raise ValueError("No valid qpos samples")
    return np.concatenate(parts, axis=0), rows


def _load_combo(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["best"]["combo"]


def _left_mapping_values(frame: np.ndarray, combo: dict[str, dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    names = []
    values = []
    for canonical in (
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
    ):
        source_idx = STANDARD_ALOHA_14D_NAMES.index(canonical)
        option = combo[canonical]
        names.append(f"follower_left_joint_{len(names)}")
        values.append(float(option["sign"]) * float(frame[source_idx]) + float(option["offset"]))
    return np.asarray(values, dtype=np.float64), names


def _rotation_from_wxyz(quat: np.ndarray) -> Rotation:
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])


def _orientation_residuals_deg(ref_quats_wxyz: list[np.ndarray], candidate_quats_wxyz: list[np.ndarray]) -> tuple[np.ndarray, Rotation]:
    ref_rots = [_rotation_from_wxyz(quat) for quat in ref_quats_wxyz]
    cand_rots = [_rotation_from_wxyz(quat) for quat in candidate_quats_wxyz]
    fixed_delta = ref_rots[0] * cand_rots[0].inv()
    residuals = []
    for ref_rot, cand_rot in zip(ref_rots, cand_rots, strict=True):
        delta = ref_rot * cand_rot.inv()
        residual = fixed_delta.inv() * delta
        residuals.append(residual.magnitude() * 180.0 / np.pi)
    return np.asarray(residuals, dtype=np.float64), fixed_delta


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 11 - Orientation Consistency Check - 2026-07-18",
        "",
        "## Scope",
        "",
        "This is an offline orientation consistency check for the Phase 9 left-arm mapping candidate.",
        "",
        "It does not touch the real robot, does not save the USD stage, and does not validate controller execution.",
        "",
        "The test asks whether the orientation difference between trusted ALOHA1 EE FK and Trossen `follower_left_link_6` is approximately constant over sampled qpos.",
        "",
        "## Dataset",
        "",
        f"- valid episodes: `{payload['summary']['valid_episode_count']}`",
        f"- sampled frames: `{payload['summary']['frame_count']}`",
        "",
        "## Gates",
        "",
    ]
    for key, value in payload["gates"].items():
        lines.append(f"- {key}: `{value}`")
    metrics = payload["orientation_metrics"]
    lines.extend(
        [
            "",
            "## Orientation Residual Metrics",
            "",
            f"- mean residual deg: `{metrics['mean_residual_deg']:.6f}`",
            f"- p95 residual deg: `{metrics['p95_residual_deg']:.6f}`",
            f"- max residual deg: `{metrics['max_residual_deg']:.6f}`",
            "",
            "## Interpretation",
            "",
            "If the candidate link frame and ALOHA1 EE frame differ only by a fixed transform, the residual after calibrating the first frame should stay small.",
            "",
            "A large residual means the joint mapping, link choice, or frame definition is still inconsistent.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check orientation consistency for the Phase 9 left-arm mapping candidate.")
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

        ref_quats = []
        candidate_quats = []
        for frame in qpos:
            _, ref_quat = _pin_ee(left_model, left_data, "left", frame)
            values, names = _left_mapping_values(frame, combo)
            art.set_joint_positions(values, joint_indices=_indices(dof_names, names))
            art.set_joint_velocities(np.zeros_like(values), joint_indices=_indices(dof_names, names))
            world.step(render=False)
            _, candidate_quat = _link_transform(art, TROSSEN_EE_BODY["left"])
            ref_quats.append(ref_quat)
            candidate_quats.append(candidate_quat)

        residuals, fixed_delta = _orientation_residuals_deg(ref_quats, candidate_quats)
        p95 = float(np.quantile(residuals, 0.95))
        max_residual = float(np.max(residuals))
        threshold_p95_deg = 5.0
        threshold_max_deg = 10.0
        orientation_ok = p95 <= threshold_p95_deg and max_residual <= threshold_max_deg
        payload = {
            "inputs": {
                "hdf5_root": _rel(args.hdf5_root),
                "phase9_json": _rel(args.phase9_json),
                "scaffold_usd": _rel(args.scaffold_usd),
                "left_urdf": _rel(args.left_urdf),
            },
            "summary": {
                "episode_count_total": len(paths),
                "valid_episode_count": sum(1 for row in episode_rows if row["status"] == "OK"),
                "frame_count": int(qpos.shape[0]),
                "trossen_body": TROSSEN_EE_BODY["left"],
            },
            "episode_rows": episode_rows,
            "orientation_metrics": {
                "mean_residual_deg": float(np.mean(residuals)),
                "p95_residual_deg": p95,
                "max_residual_deg": max_residual,
                "threshold_p95_deg": threshold_p95_deg,
                "threshold_max_deg": threshold_max_deg,
                "fixed_delta_quat_xyzw": fixed_delta.as_quat().tolist(),
            },
            "gates": {
                "real_robot_touched": "PASS_FALSE",
                "stage_saved": "PASS_FALSE",
                "isaac_runtime_started": "PASS",
                "qpos_loaded": "PASS",
                "orientation_consistency": "PASS_DIAGNOSTIC" if orientation_ok else "FAIL_ORIENTATION_INCONSISTENT",
                "controller": "BLOCKED_NOT_ATTEMPTED",
            },
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.output_dir / "orientation_consistency.json"
        md_path = args.output_dir / "orientation_consistency.md"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(_render_markdown(payload), encoding="utf-8")
        print(
            json.dumps(
                {
                    "json": _rel(json_path),
                    "markdown": _rel(md_path),
                    "orientation_metrics": payload["orientation_metrics"],
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
