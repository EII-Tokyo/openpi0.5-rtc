"""Compute normalization statistics for a config.

This script supports two paths:
1. The original dataloader path, which calls dataset `__getitem__` and therefore decodes videos.
2. A faster parquet path for LeRobot datasets that reads only `state/action` columns from parquet
   and reconstructs the action horizon numerically, bypassing image/video decoding entirely.
"""

import dataclasses
from pathlib import Path
import time
from typing import Literal

import lerobot.datasets.lerobot_dataset as lerobot_dataset
from huggingface_hub import snapshot_download
import numpy as np
import pyarrow.parquet as pq
import torch
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
from openpi.data import dataloaders as _data_loader
from openpi.data import datasets as _datasets
from openpi.data import transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.LeRobotAlohaDataConfig,
    action_horizon: int,
    batch_size: int,
    num_workers: int,
    max_frames: int | None = None,
    shuffle_if_truncated: bool = True,
) -> tuple[_datasets.Dataset, int]:
    if not data_config.repo_ids:
        raise ValueError("Data config must have non-empty repo_ids")
    dataset = _datasets.create_torch_dataset(data_config, action_horizon)
    if data_config.transform_pipeline is None:
        raise ValueError("A transform pipeline is required to compute norm stats.")
    dataset = _datasets.TransformedDataset(
        dataset,
        [
            *data_config.transform_pipeline.stats_input_transforms(),
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = shuffle_if_truncated
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def _compute_stats_from_data_loader(data_loader, num_batches: int) -> dict[str, normalize.NormStats]:
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    return {key: stats.get_statistics() for key, stats in stats.items()}


def _get_repo_ids(data_config: _config.LeRobotAlohaDataConfig) -> list[str]:
    if not data_config.repo_ids:
        raise ValueError("Data config must have non-empty repo_ids")
    return list(data_config.repo_ids)


def _repo_parquet_files(repo_id: str, meta: lerobot_dataset.LeRobotDatasetMetadata) -> list[Path]:
    snapshot_root = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["data/**/*.parquet"],
        )
    )
    parquet_files = sorted(snapshot_root.glob("data/**/*.parquet"))
    if parquet_files:
        return parquet_files

    return sorted(Path(meta.root).glob("data/**/*.parquet"))


def _load_repo_arrays(repo_id: str) -> tuple[lerobot_dataset.LeRobotDatasetMetadata, dict[str, np.ndarray]]:
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, force_cache_sync=True)
    parquet_files = _repo_parquet_files(repo_id, meta)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for repo: {repo_id}")

    state_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    trainable_parts: list[np.ndarray] = []

    for parquet_file in parquet_files:
        columns = pq.ParquetFile(parquet_file).schema.names
        read_columns = ["observation.state", "action", "episode_index"]
        if "is_for_training" in columns:
            read_columns.append("is_for_training")
        table = pq.read_table(
            parquet_file,
            columns=read_columns,
        )
        state_parts.append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32))
        action_parts.append(np.asarray(table["action"].to_pylist(), dtype=np.float32))
        episode_parts.append(np.asarray(table["episode_index"].to_pylist(), dtype=np.int64))
        if "is_for_training" in read_columns:
            trainable_parts.append(np.asarray(table["is_for_training"].to_pylist(), dtype=bool))
        else:
            trainable_parts.append(np.ones(table.num_rows, dtype=bool))

    arrays = {
        "state": np.concatenate(state_parts, axis=0),
        "action": np.concatenate(action_parts, axis=0),
        "episode_index": np.concatenate(episode_parts, axis=0),
        "is_for_training": np.concatenate(trainable_parts, axis=0),
    }
    return meta, arrays


def _episode_bounds(episode_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.empty_like(episode_index)
    ends = np.empty_like(episode_index)
    boundaries = np.concatenate(([0], np.flatnonzero(np.diff(episode_index)) + 1, [len(episode_index)]))
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        starts[start:end] = start
        ends[start:end] = end
    return starts, ends


def _sample_effective_indices(is_for_training: np.ndarray, generator: torch.Generator) -> np.ndarray:
    effective_indices = np.arange(len(is_for_training), dtype=np.int64)
    invalid = ~is_for_training
    if not invalid.any():
        return effective_indices

    trainable_indices = np.flatnonzero(is_for_training)
    if len(trainable_indices) == 0:
        raise ValueError("Dataset has no samples with is_for_training=true.")
    sampled = torch.randint(len(trainable_indices), (int(invalid.sum()),), generator=generator).numpy()
    effective_indices[invalid] = trainable_indices[sampled]
    return effective_indices


def _apply_state_action_transforms(
    data_config: _config.LeRobotAlohaDataConfig,
    state_batch: np.ndarray,
    actions_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if data_config.transform_pipeline is None:
        raise ValueError("A transform pipeline is required to compute norm stats.")
    # RepackTransform expects raw LeRobot keys even though norm stats only need state/actions.
    data = {
        "observation.images.cam_high": np.zeros((len(state_batch), 1, 1, 3), dtype=np.uint8),
        "observation.images.cam_low": np.zeros((len(state_batch), 1, 1, 3), dtype=np.uint8),
        "observation.images.cam_left_wrist": np.zeros((len(state_batch), 1, 1, 3), dtype=np.uint8),
        "observation.images.cam_right_wrist": np.zeros((len(state_batch), 1, 1, 3), dtype=np.uint8),
        "observation.state": state_batch,
        "action": actions_batch,
        "task": "compute norm stats",
        "subtask": "compute norm stats",
    }
    for transform in data_config.transform_pipeline.raw_state_action_transforms():
        data = transform(data)
    return np.asarray(data["state"]), np.asarray(data["actions"])


def compute_parquet_norm_stats(
    data_config: _config.LeRobotAlohaDataConfig,
    action_horizon: int,
    *,
    max_frames: int | None = None,
    chunk_size: int = 8192,
    seed: int = 0,
) -> dict[str, normalize.NormStats]:
    repo_ids = _get_repo_ids(data_config)
    base_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_ids[0], force_cache_sync=True)
    base_fps = float(base_meta.fps)
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    generator = torch.Generator().manual_seed(seed)

    total_frames = 0
    repo_lengths: list[int] = []
    for repo_id in repo_ids:
        _, arrays = _load_repo_arrays(repo_id)
        repo_len = len(arrays["state"])
        repo_lengths.append(repo_len)
        total_frames += repo_len
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    remaining = total_frames
    progress = tqdm.tqdm(total=total_frames, desc="Computing stats (parquet)")
    for repo_id, repo_len in zip(repo_ids, repo_lengths, strict=True):
        if remaining <= 0:
            break

        meta, arrays = _load_repo_arrays(repo_id)
        repo_fps = float(meta.fps)
        delta_indices = np.asarray([round((t / base_fps) * repo_fps) for t in range(action_horizon)], dtype=np.int64)
        episode_start, episode_end = _episode_bounds(arrays["episode_index"])
        effective_indices = _sample_effective_indices(arrays["is_for_training"], generator)

        num_rows = min(repo_len, remaining)
        effective_indices = effective_indices[:num_rows]

        for start in range(0, num_rows, chunk_size):
            end = min(start + chunk_size, num_rows)
            batch_indices = effective_indices[start:end]
            state_batch = arrays["state"][batch_indices].copy()
            action_indices = batch_indices[:, None] + delta_indices[None, :]
            action_indices = np.clip(
                action_indices,
                episode_start[batch_indices][:, None],
                episode_end[batch_indices][:, None] - 1,
            )
            actions_batch = arrays["action"][action_indices].copy()
            state_batch, actions_batch = _apply_state_action_transforms(data_config, state_batch, actions_batch)
            stats["state"].update(state_batch)
            stats["actions"].update(actions_batch)
            progress.update(end - start)

        remaining -= num_rows

    progress.close()
    return {key: value.get_statistics() for key, value in stats.items()}


def _single_repo_data_config(data_config: _config.LeRobotAlohaDataConfig, repo_id: str) -> _config.LeRobotAlohaDataConfig:
    return dataclasses.replace(data_config, repo_ids=[repo_id])


def _max_stat_diff(lhs: normalize.NormStats, rhs: normalize.NormStats) -> float:
    diffs = [
        np.max(np.abs(np.asarray(lhs.mean) - np.asarray(rhs.mean))),
        np.max(np.abs(np.asarray(lhs.std) - np.asarray(rhs.std))),
    ]
    if lhs.q01 is not None and rhs.q01 is not None:
        diffs.append(np.max(np.abs(np.asarray(lhs.q01) - np.asarray(rhs.q01))))
    if lhs.q99 is not None and rhs.q99 is not None:
        diffs.append(np.max(np.abs(np.asarray(lhs.q99) - np.asarray(rhs.q99))))
    return float(max(diffs))


def benchmark_methods(
    config: _config.TrainConfig,
    data_config: _config.LeRobotAlohaDataConfig,
    repo_id: str,
    *,
    max_frames: int | None = None,
    parquet_chunk_size: int = 8192,
    compare_num_workers: int = 0,
    seed: int = 0,
) -> None:
    compare_data_config = _single_repo_data_config(data_config, repo_id)

    start = time.perf_counter()
    data_loader, num_batches = create_torch_dataloader(
        compare_data_config,
        config.model.action_horizon,
        config.batch_size,
        compare_num_workers,
        max_frames,
        shuffle_if_truncated=False,
    )
    dataloader_stats = _compute_stats_from_data_loader(data_loader, num_batches)
    dataloader_seconds = time.perf_counter() - start

    start = time.perf_counter()
    parquet_stats = compute_parquet_norm_stats(
        compare_data_config,
        config.model.action_horizon,
        max_frames=max_frames,
        chunk_size=parquet_chunk_size,
        seed=seed,
    )
    parquet_seconds = time.perf_counter() - start

    state_diff = _max_stat_diff(dataloader_stats["state"], parquet_stats["state"])
    action_diff = _max_stat_diff(dataloader_stats["actions"], parquet_stats["actions"])
    speedup = dataloader_seconds / parquet_seconds if parquet_seconds > 0 else float("inf")

    print(f"[benchmark] repo={repo_id}")
    print(f"[benchmark] dataloader_seconds={dataloader_seconds:.3f}")
    print(f"[benchmark] parquet_seconds={parquet_seconds:.3f}")
    print(f"[benchmark] speedup={speedup:.2f}x")
    print(f"[benchmark] max_abs_diff.state={state_diff:.8e}")
    print(f"[benchmark] max_abs_diff.actions={action_diff:.8e}")


def main(
    config_name: str,
    max_frames: int | None = None,
    method: Literal["parquet", "dataloader"] = "parquet",
    compare_repo: str | None = None,
    parquet_chunk_size: int = 8192,
    compare_num_workers: int = 0,
    seed: int = 0,
):
    config = _config.get_config(config_name)
    data_config = config.data

    if compare_repo is not None:
        benchmark_methods(
            config,
            data_config,
            compare_repo,
            max_frames=max_frames,
            parquet_chunk_size=parquet_chunk_size,
            compare_num_workers=compare_num_workers,
            seed=seed,
        )

    if method == "dataloader":
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.num_workers,
            max_frames,
        )
        norm_stats = _compute_stats_from_data_loader(data_loader, num_batches)
    else:
        norm_stats = compute_parquet_norm_stats(
            data_config,
            config.model.action_horizon,
            max_frames=max_frames,
            chunk_size=parquet_chunk_size,
            seed=seed,
        )

    output_path = Path(data_config.transform_pipeline.assets.assets_dir) / data_config.transform_pipeline.assets.asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
