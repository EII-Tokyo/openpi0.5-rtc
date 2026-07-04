from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


DEFAULT_DATASET_ROOT = Path("/home/eii/.cache/huggingface/lerobot/lyl472324464")
DEFAULT_CROP_ROOT = Path("/home/eii/data/openpi0.5-rtc-reward-learning/replay/discriminator_expert_crops")
EXPECTED_CONFIG = "eii_rinse_11repo_cam4_fullft_rl_token_lower_right_query_4layer"
EXPECTED_CHECKPOINT_FRAGMENT = "rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint"
CAMERA_PATHS = {
    "cam_high": "observation.images.cam_high",
    "cam_low": "observation.images.cam_low",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
}


@dataclasses.dataclass(frozen=True)
class AuditArgs:
    dataset_root: Path = DEFAULT_DATASET_ROOT
    crop_root: Path = DEFAULT_CROP_ROOT
    z_cache_root: Path | None = None
    q_replay_root: Path | None = None
    expected_z_dim: int = 2048
    expected_config: str = EXPECTED_CONFIG
    expected_checkpoint_fragment: str = EXPECTED_CHECKPOINT_FRAGMENT
    require_camera: tuple[str, ...] = ("cam_low", "cam_right_wrist")
    output_path: Path | None = None


def audit_expert_crops(args: AuditArgs) -> dict[str, Any]:
    crop_paths = sorted(args.crop_root.glob("*/*.json"))
    issues: Counter[str] = Counter()
    by_dataset: Counter[str] = Counter()
    by_reward: Counter[str] = Counter()
    episode_to_crops: dict[tuple[str, int], list[Path]] = defaultdict(list)
    crop_records: list[dict[str, Any]] = []

    dataset_cache: dict[str, dict[str, Any]] = {}
    for crop_path in crop_paths:
        try:
            crop = json.loads(crop_path.read_text(encoding="utf-8"))
            dataset_id = str(crop["dataset_id"])
            episode_index = int(crop["episode_index"])
            start_sec = float(crop["start_sec"])
            end_sec = float(crop["end_sec"])
            reward = int(crop.get("reward", 1))
        except Exception:
            issues["invalid_crop_json"] += 1
            continue
        by_dataset[dataset_id] += 1
        by_reward[str(reward)] += 1
        episode_to_crops[(dataset_id, episode_index)].append(crop_path)
        dataset = dataset_cache.get(dataset_id)
        if dataset is None:
            dataset = _load_dataset_summary(args.dataset_root / dataset_id, args.require_camera)
            dataset_cache[dataset_id] = dataset
            issues.update(dataset["issues"])
        frames_in_crop = _frames_in_range(dataset, episode_index, start_sec, end_sec)
        if frames_in_crop < 20:
            issues["crop_too_short_for_horizon10"] += 1
        crop_records.append(
            {
                "crop_path": str(crop_path),
                "dataset_id": dataset_id,
                "episode_index": episode_index,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "reward": reward,
                "frames_in_crop": frames_in_crop,
            }
        )

    z_cache_summary = _audit_z_cache(args, crop_records, issues) if args.z_cache_root else None
    q_replay_summary = _audit_q_replay(args, issues) if args.q_replay_root else None
    duplicate_episodes = {
        f"{dataset_id}:{episode_index:06d}": len(paths)
        for (dataset_id, episode_index), paths in sorted(episode_to_crops.items())
        if len(paths) > 1
    }
    summary = {
        "crop_count": len(crop_paths),
        "unique_episode_count": len(episode_to_crops),
        "duplicate_episode_crops": duplicate_episodes,
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_reward": dict(sorted(by_reward.items())),
        "datasets": dataset_cache,
        "z_cache": z_cache_summary,
        "q_replay": q_replay_summary,
        "issues": dict(sorted(issues.items())),
        "is_usable": not issues,
    }
    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def _load_dataset_summary(dataset_dir: Path, require_camera: tuple[str, ...]) -> dict[str, Any]:
    issues: Counter[str] = Counter()
    parquet_paths = sorted((dataset_dir / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_paths:
        return {"path": str(dataset_dir), "issues": {"missing_dataset_parquet": 1}}
    episode_frames: Counter[str] = Counter()
    max_global_index = -1
    state_dim = None
    action_dim = None
    timestamps: dict[int, list[float]] = defaultdict(list)
    for parquet_path in parquet_paths:
        table = pq.read_table(
            parquet_path,
            columns=["episode_index", "frame_index", "index", "timestamp", "observation.state", "action"],
        )
        for row in range(table.num_rows):
            episode_index = int(table["episode_index"][row].as_py())
            episode_frames[str(episode_index)] += 1
            max_global_index = max(max_global_index, int(table["index"][row].as_py()))
            timestamps[episode_index].append(float(table["timestamp"][row].as_py()))
            if state_dim is None:
                state_dim = len(table["observation.state"][row].as_py())
                action_dim = len(table["action"][row].as_py())
    if state_dim != 14:
        issues["unexpected_state_dim"] += 1
    if action_dim != 14:
        issues["unexpected_action_dim"] += 1

    videos = {}
    for camera in require_camera:
        camera_dir = dataset_dir / "videos" / CAMERA_PATHS[camera]
        files = sorted(camera_dir.glob("chunk-*/file-*.mp4"))
        frames = _video_frame_count(files)
        videos[camera] = {"files": len(files), "frames": frames}
        if not files:
            issues[f"missing_video_{camera}"] += 1
        if frames is not None and max_global_index >= 0 and frames <= max_global_index:
            issues[f"video_too_short_{camera}"] += 1
    return {
        "path": str(dataset_dir),
        "parquet_files": len(parquet_paths),
        "episode_count": len(episode_frames),
        "episode_frames": dict(sorted(episode_frames.items(), key=lambda kv: int(kv[0]))),
        "max_global_index": max_global_index,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "videos": videos,
        "timestamps": {
            str(key): {"start": values[0], "end": values[-1], "count": len(values)}
            for key, values in sorted(timestamps.items())
        },
        "issues": dict(sorted(issues.items())),
    }


def _video_frame_count(files: list[Path]) -> int | None:
    try:
        import av

        total = 0
        for path in files:
            with av.open(str(path)) as container:
                total += int(container.streams.video[0].frames)
        return total
    except Exception:
        return None


def _frames_in_range(dataset: dict[str, Any], episode_index: int, start_sec: float, end_sec: float) -> int:
    episode = dataset.get("timestamps", {}).get(str(episode_index))
    if not episode:
        return 0
    # The detailed per-frame timestamps are intentionally not retained in the JSON summary.
    # For audit purposes the caller only needs to catch missing/obviously too-short episodes;
    # exact transition counts are verified by convert_expert_crops_to_q_replay.
    duration = float(episode["end"]) - float(episode["start"])
    count = int(episode["count"])
    if duration <= 0:
        return 0
    fps = count / duration
    return int(max(0.0, end_sec - start_sec) * fps)


def _audit_z_cache(args: AuditArgs, crop_records: list[dict[str, Any]], issues: Counter[str]) -> dict[str, Any]:
    cache_files = sorted(args.z_cache_root.glob("*/*.npz")) if args.z_cache_root else []
    by_episode = {(row["dataset_id"], int(row["episode_index"])) for row in crop_records}
    cache_by_episode: dict[tuple[str, int], Path] = {}
    bad_dim = 0
    bad_metadata = 0
    for path in cache_files:
        dataset_id = path.parent.name
        try:
            episode_index = int(path.stem.split("_")[1])
        except Exception:
            continue
        cache_by_episode[(dataset_id, episode_index)] = path
        with np.load(path, allow_pickle=False) as data:
            z_rl = np.asarray(data["z_rl"])
            if z_rl.ndim != 2 or int(z_rl.shape[-1]) != int(args.expected_z_dim):
                bad_dim += 1
            metadata = _load_metadata(data, "metadata")
        if metadata.get("rl_token_config_name") != args.expected_config:
            bad_metadata += 1
        if args.expected_checkpoint_fragment not in str(metadata.get("rl_token_checkpoint_path", "")):
            bad_metadata += 1
    missing = by_episode - set(cache_by_episode)
    if missing:
        issues["missing_z_cache"] += len(missing)
    if bad_dim:
        issues["bad_z_cache_dim"] += bad_dim
    if bad_metadata:
        issues["bad_z_cache_metadata"] += bad_metadata
    return {
        "root": str(args.z_cache_root),
        "files": len(cache_files),
        "episodes_required": len(by_episode),
        "missing_episode_count": len(missing),
        "bad_dim_files": bad_dim,
        "bad_metadata_count": bad_metadata,
    }


def _audit_q_replay(args: AuditArgs, issues: Counter[str]) -> dict[str, Any]:
    files = sorted(args.q_replay_root.glob("*/*.npz")) if args.q_replay_root else []
    bad_dim = 0
    bad_metadata = 0
    transitions = 0
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            z_rl = np.asarray(data["z_rl"])
            next_z_rl = np.asarray(data["next_z_rl"])
            transitions += int(z_rl.shape[0])
            if z_rl.ndim != 2 or next_z_rl.ndim != 2 or z_rl.shape[-1] != args.expected_z_dim or next_z_rl.shape[-1] != args.expected_z_dim:
                bad_dim += 1
            manifest = _load_metadata(data, "manifest")
        if manifest.get("rl_token_config_name") != args.expected_config:
            bad_metadata += 1
        if args.expected_checkpoint_fragment not in str(manifest.get("rl_token_checkpoint_path", "")):
            bad_metadata += 1
    if bad_dim:
        issues["bad_q_replay_dim"] += bad_dim
    if bad_metadata:
        issues["bad_q_replay_metadata"] += bad_metadata
    return {
        "root": str(args.q_replay_root),
        "files": len(files),
        "transitions": transitions,
        "bad_dim_files": bad_dim,
        "bad_metadata_count": bad_metadata,
    }


def _load_metadata(data: np.lib.npyio.NpzFile, key: str) -> dict[str, Any]:
    if key not in data.files:
        return {}
    raw = data[key]
    value = raw.item() if raw.shape == () else raw.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        decoded = json.loads(value)
        return decoded if isinstance(decoded, dict) else {}
    return value if isinstance(value, dict) else {}


def _parse_args() -> AuditArgs:
    parser = argparse.ArgumentParser(description="Audit Expert-for-D crop, z-cache, and Q replay consistency.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--crop-root", type=Path, default=DEFAULT_CROP_ROOT)
    parser.add_argument("--z-cache-root", type=Path, default=None)
    parser.add_argument("--q-replay-root", type=Path, default=None)
    parser.add_argument("--expected-z-dim", type=int, default=2048)
    parser.add_argument("--expected-config", default=EXPECTED_CONFIG)
    parser.add_argument("--expected-checkpoint-fragment", default=EXPECTED_CHECKPOINT_FRAGMENT)
    parser.add_argument("--require-camera", action="append", default=["cam_low", "cam_right_wrist"])
    parser.add_argument("--output-path", type=Path, default=None)
    ns = parser.parse_args()
    return AuditArgs(
        dataset_root=ns.dataset_root,
        crop_root=ns.crop_root,
        z_cache_root=ns.z_cache_root,
        q_replay_root=ns.q_replay_root,
        expected_z_dim=ns.expected_z_dim,
        expected_config=ns.expected_config,
        expected_checkpoint_fragment=ns.expected_checkpoint_fragment,
        require_camera=tuple(ns.require_camera),
        output_path=ns.output_path,
    )


def main() -> None:
    print(json.dumps(audit_expert_crops(_parse_args()), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
