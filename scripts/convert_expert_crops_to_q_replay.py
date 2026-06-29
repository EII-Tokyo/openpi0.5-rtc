from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


@dataclasses.dataclass(frozen=True)
class ConversionArgs:
    dataset_root: Path
    crop_root: Path
    output_root: Path
    manifest_path: Path
    train_horizon: int = 10
    chunk_stride: int = 2
    proprio_dim: int = 32
    z_dim: int = 512
    z_cache_root: Path | None = None
    allow_dummy_z: bool = False
    overwrite: bool = False


@dataclasses.dataclass(frozen=True)
class ConversionSummary:
    converted: int
    skipped: dict[str, int]
    manifest_path: Path


@dataclasses.dataclass(frozen=True)
class _EpisodeData:
    dataset_id: str
    episode_index: int
    frame_index: np.ndarray
    timestamp: np.ndarray
    state: np.ndarray
    action: np.ndarray


@dataclasses.dataclass(frozen=True)
class _ZCacheData:
    z_rl: np.ndarray
    cache_path: Path
    metadata: dict[str, Any]


def convert_expert_crops(args: ConversionArgs) -> ConversionSummary:
    if args.train_horizon <= 0:
        raise ValueError("train_horizon must be positive")
    if args.chunk_stride <= 0:
        raise ValueError("chunk_stride must be positive")
    if args.proprio_dim <= 0:
        raise ValueError("proprio_dim must be positive")
    if args.z_dim <= 0:
        raise ValueError("z_dim must be positive")
    if args.z_cache_root is None and not args.allow_dummy_z:
        raise ValueError(
            "z_cache_root is required for trainable conversion. "
            "Pass --allow-dummy-z only for pipeline tests; dummy z shards are not valid for critic/actor training."
        )

    crop_paths = sorted(args.crop_root.glob("*/*.json"))
    skipped: Counter[str] = Counter()
    manifest_rows: list[dict[str, Any]] = []
    converted = 0
    seen_outputs: set[Path] = set()

    for crop_path in crop_paths:
        try:
            crop = _load_crop(crop_path)
            episode = _load_episode(args.dataset_root, crop["dataset_id"], int(crop["episode_index"]))
            arrays, manifest = _convert_one_crop(args, crop_path, crop, episode)
        except _SkipCrop as exc:
            skipped[exc.reason] += 1
            continue

        shard_path = Path(manifest["shard_path"])
        if shard_path in seen_outputs:
            skipped["duplicate_output"] += 1
            continue
        seen_outputs.add(shard_path)
        if shard_path.exists() and not args.overwrite:
            skipped["output_exists"] += 1
            continue
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        _write_npz(shard_path, {**arrays, "manifest": np.asarray(json.dumps(manifest, sort_keys=True))})
        manifest_rows.append(manifest)
        converted += 1

    if manifest_rows:
        args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest_path.open("w", encoding="utf-8") as file:
            for row in manifest_rows:
                file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    elif args.manifest_path.exists() and args.overwrite:
        args.manifest_path.unlink()
    return ConversionSummary(converted=converted, skipped=dict(sorted(skipped.items())), manifest_path=args.manifest_path)


class _SkipCrop(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _load_crop(path: Path) -> dict[str, Any]:
    try:
        crop = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _SkipCrop("invalid_crop_json") from exc
    for key in ("dataset_id", "episode_index", "start_sec", "end_sec"):
        if key not in crop:
            raise _SkipCrop(f"missing_{key}")
    reward = int(crop.get("reward", 1))
    if reward not in (0, 1):
        raise _SkipCrop("invalid_reward")
    if float(crop["end_sec"]) <= float(crop["start_sec"]):
        raise _SkipCrop("invalid_range")
    crop["reward"] = reward
    return crop


def _load_episode(dataset_root: Path, dataset_id: str, episode_index: int) -> _EpisodeData:
    dataset_dir = dataset_root / dataset_id
    parquet_paths = sorted((dataset_dir / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise _SkipCrop("missing_dataset_parquet")

    frame_index_parts: list[np.ndarray] = []
    timestamp_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    for parquet_path in parquet_paths:
        table = pq.read_table(
            parquet_path,
            columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"],
        )
        episodes = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        mask = episodes == int(episode_index)
        if not np.any(mask):
            continue
        frame_index_parts.append(np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)[mask])
        timestamp_parts.append(np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)[mask])
        state_parts.append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[mask])
        action_parts.append(np.asarray(table["action"].to_pylist(), dtype=np.float32)[mask])

    if not frame_index_parts:
        raise _SkipCrop("missing_episode")
    frame_index = np.concatenate(frame_index_parts, axis=0)
    order = np.argsort(frame_index)
    return _EpisodeData(
        dataset_id=dataset_id,
        episode_index=episode_index,
        frame_index=frame_index[order],
        timestamp=np.concatenate(timestamp_parts, axis=0)[order],
        state=np.concatenate(state_parts, axis=0)[order],
        action=np.concatenate(action_parts, axis=0)[order],
    )


def _convert_one_crop(
    args: ConversionArgs,
    crop_path: Path,
    crop: dict[str, Any],
    episode: _EpisodeData,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    start_sec = float(crop["start_sec"])
    end_sec = float(crop["end_sec"])
    rel_time = episode.timestamp - float(episode.timestamp[0])
    mask = (rel_time >= start_sec) & (rel_time < end_sec)
    indices = np.flatnonzero(mask)
    if len(indices) < 2 * args.train_horizon:
        raise _SkipCrop("too_short")

    state = episode.state[indices].astype(np.float32)
    action = episode.action[indices].astype(np.float32)
    frame_index = episode.frame_index[indices].astype(np.int64)
    z_cache_data: _ZCacheData | None = None
    if args.z_cache_root is not None:
        z_cache_data = _load_z_cache(args.z_cache_root, episode.dataset_id, episode.episode_index, frame_index, args.z_dim)
        z_rl = z_cache_data.z_rl
        z_source = "precomputed_frame_cache"
    else:
        z_rl = _dummy_z(episode.dataset_id, episode.episode_index, frame_index, args.z_dim)
        z_source = "dummy_deterministic_not_for_training"
    proprio = _pad_or_trim(state, args.proprio_dim)

    starts = _transition_starts(len(indices), train_horizon=args.train_horizon, chunk_stride=args.chunk_stride)
    if not starts:
        raise _SkipCrop("too_short")

    action_chunks = np.stack([action[start : start + args.train_horizon] for start in starts], axis=0).astype(np.float32)
    next_reference = np.stack(
        [action[start + args.train_horizon : start + 2 * args.train_horizon] for start in starts],
        axis=0,
    ).astype(np.float32)
    z_samples = np.stack([z_rl[start] for start in starts], axis=0).astype(np.float32)
    next_z = np.stack([z_rl[start + args.train_horizon] for start in starts], axis=0).astype(np.float32)
    proprio_samples = np.stack([proprio[start] for start in starts], axis=0).astype(np.float32)
    next_proprio = np.stack([proprio[start + args.train_horizon] for start in starts], axis=0).astype(np.float32)

    reward = int(crop.get("reward", 1))
    reward_seq = np.zeros((len(starts), args.train_horizon), dtype=np.float32)
    reward_seq[-1, args.train_horizon - 1] = float(reward)
    done = np.zeros((len(starts),), dtype=np.bool_)
    done[-1] = True

    output_name = f"{episode.dataset_id}_episode_{episode.episode_index:06d}_{_stable_id(crop_path)}.npz"
    shard_path = (args.output_root / episode.dataset_id / output_name).resolve()
    delta = np.abs(action_chunks - action_chunks)
    manifest = {
        "schema_version": 1,
        "source_type": "human_expert",
        "source_crop_path": str(crop_path.resolve()),
        "source_dataset": episode.dataset_id,
        "episode_index": int(episode.episode_index),
        "user_crop_start_sec": start_sec,
        "user_crop_end_sec": end_sec,
        "crop_start_frame": int(frame_index[0]),
        "crop_end_frame": int(frame_index[-1]),
        "reward": reward,
        "label": "expert_no_actor_q",
        "phase": "human_expert",
        "batch": "human_expert",
        "shard_path": str(shard_path),
        "num_replay_transitions": int(len(starts)),
        "train_horizon": int(args.train_horizon),
        "train_chunk_horizon": int(args.train_horizon),
        "policy_horizon": int(args.train_horizon),
        "chunk_stride": int(args.chunk_stride),
        "action_dim": int(action_chunks.shape[-1]),
        "action_space": "aloha_exec",
        "reward_placement": "terminal_last_train_step",
        "train_eligible": True,
        "voided": False,
        "replay_ready": True,
        "replay_status": "ready",
        "actor_enabled": False,
        "actor_checkpoint_path": None,
        "actor_checkpoint_step": None,
        "intervention_scale": 0.0,
        "rlt_actor_applied_ratio": 0.0,
        "reference_action_source": "human_action_no_actor",
        "proprio_source": f"observation.state,pad_or_trim_to_{args.proprio_dim}",
        "z_rl_source": z_source,
        "z_rl_dim": int(z_samples.shape[-1]),
        "action_reference_delta": {
            "all_max_abs": float(np.max(delta)) if delta.size else 0.0,
            "all_p95_abs": float(np.percentile(delta, 95)) if delta.size else 0.0,
        },
        "replay_array_shapes": {
            "z_rl": list(z_samples.shape),
            "proprio": list(proprio_samples.shape),
            "action": list(action_chunks.shape),
            "reference_action": list(action_chunks.shape),
            "reward_seq": list(reward_seq.shape),
            "next_z_rl": list(next_z.shape),
            "next_proprio": list(next_proprio.shape),
            "next_reference_action": list(next_reference.shape),
            "done": list(done.shape),
        },
    }
    if z_cache_data is not None:
        manifest.update(
            {
                "z_cache_path": str(z_cache_data.cache_path.resolve()),
                "z_cache_root": str(args.z_cache_root.resolve()),
                "z_cache_metadata": z_cache_data.metadata,
                "rl_token_checkpoint_path": z_cache_data.metadata.get("rl_token_checkpoint_path"),
                "rl_token_config_name": z_cache_data.metadata.get("rl_token_config_name"),
            }
        )
    arrays = {
        "z_rl": z_samples,
        "proprio": proprio_samples,
        "action": action_chunks,
        "reference_action": action_chunks.copy(),
        "reward_seq": reward_seq,
        "next_z_rl": next_z,
        "next_proprio": next_proprio,
        "next_reference_action": next_reference,
        "done": done,
    }
    return arrays, manifest


def _transition_starts(num_frames: int, *, train_horizon: int, chunk_stride: int) -> list[int]:
    last_start = num_frames - (2 * train_horizon)
    if last_start < 0:
        return []
    starts = list(range(0, last_start + 1, chunk_stride))
    if starts and starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _pad_or_trim(values: np.ndarray, dim: int) -> np.ndarray:
    if values.shape[-1] == dim:
        return values.astype(np.float32)
    if values.shape[-1] > dim:
        return values[:, :dim].astype(np.float32)
    pad = dim - values.shape[-1]
    return np.pad(values, ((0, 0), (0, pad)), mode="constant").astype(np.float32)


def _dummy_z(dataset_id: str, episode_index: int, frame_index: np.ndarray, z_dim: int) -> np.ndarray:
    seed_bytes = hashlib.sha256(f"{dataset_id}:{episode_index}".encode("utf-8")).digest()
    seed = int.from_bytes(seed_bytes[:4], "little")
    rng = np.random.default_rng(seed)
    projection = rng.normal(loc=0.0, scale=0.01, size=(4, z_dim)).astype(np.float32)
    features = np.stack(
        [
            np.ones_like(frame_index, dtype=np.float32),
            frame_index.astype(np.float32) / 1000.0,
            np.sin(frame_index.astype(np.float32) / 10.0),
            np.cos(frame_index.astype(np.float32) / 10.0),
        ],
        axis=-1,
    )
    return (features @ projection).astype(np.float32)


def _load_z_cache(
    z_cache_root: Path,
    dataset_id: str,
    episode_index: int,
    frame_index: np.ndarray,
    z_dim: int,
) -> _ZCacheData:
    candidates = (
        z_cache_root / dataset_id / f"episode_{episode_index:06d}_z_rl.npz",
        z_cache_root / f"{dataset_id}_episode_{episode_index:06d}_z_rl.npz",
    )
    cache_path = next((path for path in candidates if path.exists()), None)
    if cache_path is None:
        raise _SkipCrop("missing_z_cache")
    with np.load(cache_path) as data:
        if "frame_index" not in data or "z_rl" not in data:
            raise _SkipCrop("invalid_z_cache")
        cached_frames = np.asarray(data["frame_index"], dtype=np.int64)
        cached_z = np.asarray(data["z_rl"], dtype=np.float32)
        metadata = _load_z_cache_metadata(data)
    if cached_z.ndim != 2 or int(cached_z.shape[-1]) != int(z_dim):
        raise _SkipCrop("z_cache_shape_mismatch")
    by_frame = {int(frame): cached_z[index] for index, frame in enumerate(cached_frames)}
    try:
        z_rl = np.stack([by_frame[int(frame)] for frame in frame_index], axis=0).astype(np.float32)
    except KeyError as exc:
        raise _SkipCrop("z_cache_missing_frame") from exc
    return _ZCacheData(z_rl=z_rl, cache_path=cache_path, metadata=metadata)


def _load_z_cache_metadata(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata" not in data:
        return {}
    raw = data["metadata"]
    value = raw.item() if raw.shape == () else raw.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise _SkipCrop("invalid_z_cache_metadata") from exc
        if not isinstance(decoded, dict):
            raise _SkipCrop("invalid_z_cache_metadata")
        return decoded
    if isinstance(value, dict):
        return dict(value)
    raise _SkipCrop("invalid_z_cache_metadata")


def _stable_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    tmp_path.replace(path)


def _parse_args() -> ConversionArgs:
    parser = argparse.ArgumentParser(description="Convert saved expert demo crops into no-actor Q replay shards.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/eii/.cache/huggingface/lerobot/lyl472324464"))
    parser.add_argument(
        "--crop-root",
        type=Path,
        default=Path("/home/eii/data/openpi0.5-rtc-reward-learning/replay/discriminator_expert_crops"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/eii/data/openpi0.5-rtc-reward-learning/replay/human_expert_no_actor_q_cam4_provenance"),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("local_rlt_manifests/human_expert_no_actor_q_cam4_provenance_20260629.jsonl"),
    )
    parser.add_argument("--train-horizon", type=int, default=10)
    parser.add_argument("--chunk-stride", type=int, default=2)
    parser.add_argument("--proprio-dim", type=int, default=32)
    parser.add_argument("--z-dim", type=int, default=512)
    parser.add_argument(
        "--z-cache-root",
        type=Path,
        default=None,
        help="Directory containing precomputed frame-level z_rl caches: <root>/<dataset_id>/episode_000000_z_rl.npz.",
    )
    parser.add_argument("--allow-dummy-z", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    ns = parser.parse_args()
    return ConversionArgs(
        dataset_root=ns.dataset_root,
        crop_root=ns.crop_root,
        output_root=ns.output_root,
        manifest_path=ns.manifest_path,
        train_horizon=ns.train_horizon,
        chunk_stride=ns.chunk_stride,
        proprio_dim=ns.proprio_dim,
        z_dim=ns.z_dim,
        z_cache_root=ns.z_cache_root,
        allow_dummy_z=ns.allow_dummy_z,
        overwrite=ns.overwrite,
    )


def main() -> None:
    summary = convert_expert_crops(_parse_args())
    print(json.dumps(dataclasses.asdict(summary), ensure_ascii=False, default=str, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
