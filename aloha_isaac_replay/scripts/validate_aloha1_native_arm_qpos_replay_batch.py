from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from aloha_isaac_replay.adapters.isaac_dof_adapter import load_mapping
from aloha_isaac_replay.replay.arm_only_mapping import arm_only_targets_from_standard_qpos
from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG
from aloha_isaac_replay.scripts.replay_aloha_qpos_arm_only import _load_qpos
from aloha_isaac_replay.scripts.replay_aloha_qpos_arm_only import _resolve_indices


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HDF5_ROOT = REPO_ROOT / "local_rlt_data/raw_from_103/rollouts/key_regions/twist_off_the_bottle_cap/2026-06-17/rl"
DEFAULT_MAPPING = REPO_ROOT / "configs/aloha/original_stationary_aloha_mapping.yaml"
DEFAULT_LEFT_USD = REPO_ROOT / "assets/isaac/aloha1_native_physics_wrapper/aloha1_left.usda"
DEFAULT_RIGHT_USD = REPO_ROOT / "assets/isaac/aloha1_native_physics_wrapper/aloha1_right.usda"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase22_arm_qpos_replay_batch_20260718"


def _rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n")


def _is_valid_episode(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as h5:
            if "observations/qpos" not in h5:
                return False
            qpos = h5["observations/qpos"]
            return qpos.ndim == 2 and qpos.shape[1] == 14 and qpos.shape[0] > 0
    except Exception:
        return False


def _discover_episodes(root: Path, limit: int) -> list[Path]:
    paths = [path for path in sorted(root.rglob("episode.hdf5")) if _is_valid_episode(path)]
    if not paths:
        raise FileNotFoundError(f"No valid episode.hdf5 with observations/qpos shape (T, 14) under {root}")
    return paths[:limit]


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 22 Arm-Only Qpos Replay Batch",
        "",
        f"- status: `{payload['status']}`",
        f"- overall pass: `{payload['overall_pass']}`",
        f"- episodes tested: `{payload['summary']['episodes_tested']}`",
        f"- total frames: `{payload['summary']['total_frames']}`",
        f"- max abs readback error: `{payload['summary']['max_abs_readback_error']}`",
        f"- gate max abs error rad: `{payload['summary']['gate_max_abs_error_rad']}`",
        "",
        "## Inputs",
        "",
        f"- hdf5 root: `{payload['inputs']['hdf5_root']}`",
        f"- mapping: `{payload['inputs']['mapping']}`",
        f"- stage USD: `{payload['inputs']['stage_usd']}`",
        f"- stage units in meters: `{payload['inputs']['stage_units_in_meters']}`",
        f"- left USD: `{payload['inputs']['left_usd']}`",
        f"- right USD: `{payload['inputs']['right_usd']}`",
        f"- left prim path: `{payload['inputs']['left_prim_path']}`",
        f"- right prim path: `{payload['inputs']['right_prim_path']}`",
        "",
        "## Episode Results",
        "",
        "| idx | frames | status | max abs error | episode |",
        "| ---: | ---: | --- | ---: | --- |",
    ]
    for idx, item in enumerate(payload["episodes"], start=1):
        lines.append(
            f"| {idx} | {item['frames']} | `{item['status']}` | "
            f"{item['max_abs_readback_error']} | `{item['episode']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This gate validates deterministic arm-joint set/readback consistency across multiple real ALOHA1 HDF5 episodes.",
            "It still does not validate dynamic tracking, contact, gripper semantics, visual mesh completeness, or bottle manipulation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-validate ALOHA1 native wrapper arm-only qpos set/readback.")
    parser.add_argument("--hdf5-root", default=str(DEFAULT_HDF5_ROOT))
    parser.add_argument("--episode-limit", type=int, default=6)
    parser.add_argument("--max-frames-per-episode", type=int, default=80)
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    parser.add_argument("--stage-usd", default=None)
    parser.add_argument("--stage-units-in-meters", type=float, default=None)
    parser.add_argument("--left-usd", default=str(DEFAULT_LEFT_USD))
    parser.add_argument("--right-usd", default=str(DEFAULT_RIGHT_USD))
    parser.add_argument("--left-prim-path", default="/World/left/root_joint")
    parser.add_argument("--right-prim-path", default="/World/right/root_joint")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    if args.stage_usd:
        if args.left_prim_path == "/World/left/root_joint":
            args.left_prim_path = "/puppet_left_vx300s/root_joint"
        if args.right_prim_path == "/World/right/root_joint":
            args.right_prim_path = "/puppet_right_vx300s/root_joint"
    if args.stage_units_in_meters is None:
        args.stage_units_in_meters = 0.01 if args.stage_usd else 1.0

    output_dir = Path(args.output_dir)
    json_path = output_dir / "batch_replay_metrics.json"
    md_path = output_dir / "batch_replay_metrics.md"
    payload: dict[str, Any] = {
        "status": "STARTED",
        "overall_pass": False,
        "real_robot_touched": False,
        "stage_saved": False,
        "inputs": {
            "hdf5_root": _rel(args.hdf5_root),
            "episode_limit": args.episode_limit,
            "max_frames_per_episode": args.max_frames_per_episode,
            "mapping": _rel(args.mapping),
            "stage_usd": _rel(args.stage_usd) if args.stage_usd else None,
            "stage_units_in_meters": args.stage_units_in_meters,
            "left_usd": _rel(args.left_usd),
            "right_usd": _rel(args.right_usd),
            "left_prim_path": args.left_prim_path,
            "right_prim_path": args.right_prim_path,
        },
    }
    _write_json(json_path, payload)

    try:
        episodes = _discover_episodes(Path(args.hdf5_root), args.episode_limit)
        mapping = load_mapping(args.mapping)

        from isaacsim import SimulationApp

        app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
        app_config["fast_shutdown"] = False
        _app = SimulationApp(app_config)
        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation

        World.clear_instance()
        if args.stage_usd:
            stage_utils.open_stage(str(Path(args.stage_usd).resolve()))
        else:
            stage_utils.create_new_stage()
        world = World(stage_units_in_meters=args.stage_units_in_meters, backend="numpy", device="cpu")
        if not args.stage_usd:
            stage_utils.add_reference_to_stage(usd_path=str(Path(args.left_usd).resolve()), prim_path="/World/left")
            stage_utils.add_reference_to_stage(usd_path=str(Path(args.right_usd).resolve()), prim_path="/World/right")
        left = world.scene.add(SingleArticulation(prim_path=args.left_prim_path, name="left_vx300s"))
        right = world.scene.add(SingleArticulation(prim_path=args.right_prim_path, name="right_vx300s"))
        world.reset()

        first_qpos = _load_qpos(episodes[0], max_frames=1)[0]
        first_targets = arm_only_targets_from_standard_qpos(first_qpos, mapping)
        left_target_names = [target.isaac_dof_name for target in first_targets if target.isaac_dof_name.startswith("left/")]
        right_target_names = [target.isaac_dof_name for target in first_targets if target.isaac_dof_name.startswith("right/")]
        left_indices = _resolve_indices(list(left.dof_names), left_target_names, "left")
        right_indices = _resolve_indices(list(right.dof_names), right_target_names, "right")

        episode_rows = []
        all_abs_errors: list[np.ndarray] = []
        for episode in episodes:
            qpos = _load_qpos(episode, max_frames=args.max_frames_per_episode)
            errors = []
            for frame in qpos:
                targets = arm_only_targets_from_standard_qpos(frame, mapping)
                left_values = np.array([target.value for target in targets if target.isaac_dof_name.startswith("left/")], dtype=np.float64)
                right_values = np.array([target.value for target in targets if target.isaac_dof_name.startswith("right/")], dtype=np.float64)
                left.set_joint_positions(left_values, joint_indices=np.array(left_indices, dtype=np.int64))
                right.set_joint_positions(right_values, joint_indices=np.array(right_indices, dtype=np.int64))
                readback = np.concatenate(
                    [
                        left.get_joint_positions(joint_indices=np.array(left_indices, dtype=np.int64)),
                        right.get_joint_positions(joint_indices=np.array(right_indices, dtype=np.int64)),
                    ]
                )
                expected = np.concatenate([left_values, right_values])
                errors.append(readback - expected)
            error_arr = np.asarray(errors, dtype=np.float64)
            abs_error = np.abs(error_arr)
            all_abs_errors.append(abs_error)
            max_abs = float(abs_error.max()) if abs_error.size else 0.0
            episode_rows.append(
                {
                    "episode": _rel(episode),
                    "frames": int(qpos.shape[0]),
                    "status": "PASS" if max_abs < 1e-5 else "FAIL",
                    "max_abs_readback_error": max_abs,
                    "mean_abs_readback_error": float(abs_error.mean()) if abs_error.size else 0.0,
                }
            )

        stacked = np.concatenate([arr.reshape(-1) for arr in all_abs_errors]) if all_abs_errors else np.asarray([], dtype=np.float64)
        overall_max = float(stacked.max()) if stacked.size else 0.0
        payload.update(
            {
                "status": "PASS" if overall_max < 1e-5 and all(row["status"] == "PASS" for row in episode_rows) else "FAILED_GATE",
                "overall_pass": bool(overall_max < 1e-5 and all(row["status"] == "PASS" for row in episode_rows)),
                "left_dof_names": list(left.dof_names),
                "right_dof_names": list(right.dof_names),
                "left_indices": left_indices,
                "right_indices": right_indices,
                "episodes": episode_rows,
                "summary": {
                    "episodes_tested": len(episode_rows),
                    "total_frames": int(sum(row["frames"] for row in episode_rows)),
                    "max_abs_readback_error": overall_max,
                    "mean_abs_readback_error": float(stacked.mean()) if stacked.size else 0.0,
                    "gate_max_abs_error_rad": 1e-5,
                },
            }
        )
        _write_json(json_path, payload)
        _write_markdown(md_path, _json_safe(payload))
        print(json.dumps({"json": _rel(json_path), "markdown": _rel(md_path), "status": payload["status"]}, ensure_ascii=False), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if payload["overall_pass"] else 3)
    except BaseException as exc:
        payload.update(
            {
                "status": "EXCEPTION",
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-25:],
            }
        )
        _write_json(json_path, payload)
        print(json.dumps({"json": _rel(json_path), "status": payload["status"], "exception": payload["exception"]}, ensure_ascii=False), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
