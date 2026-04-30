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
import numpy as np
import pyarrow.parquet as pq
import torch
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
    shuffle_if_truncated: bool = True,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None and not data_config.repo_ids:
        raise ValueError("Data config must have a repo_id or non-empty repo_ids")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *([transforms.PromptFromLeRobotTask()] if data_config.prompt_from_task else []),
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    # print(data_config.data_transforms.inputs)
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


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *([transforms.PromptFromLeRobotTask()] if data_config.prompt_from_task else []),
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
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


def _get_repo_ids(data_config: _config.DataConfig) -> list[str]:
    if data_config.repo_ids:
        return list(data_config.repo_ids)
    if data_config.repo_id:
        return [data_config.repo_id]
    raise ValueError("Data config must have a repo_id or non-empty repo_ids")


def _load_repo_arrays(repo_id: str) -> tuple[lerobot_dataset.LeRobotDatasetMetadata, dict[str, np.ndarray]]:
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, force_cache_sync=True)
    parquet_files = sorted(Path(meta.root).glob("data/**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for repo: {repo_id}")

    state_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    trainable_parts: list[np.ndarray] = []

    for parquet_file in parquet_files:
        table = pq.read_table(
            parquet_file,
            columns=["observation.state", "action", "episode_index", "is_for_training"],
        )
        state_parts.append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32))
        action_parts.append(np.asarray(table["action"].to_pylist(), dtype=np.float32))
        episode_parts.append(np.asarray(table["episode_index"].to_pylist(), dtype=np.int64))
        trainable_parts.append(np.asarray(table["is_for_training"].to_pylist(), dtype=bool))

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
    data_config: _config.DataConfig,
    state_batch: np.ndarray,
    actions_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # AlohaInputs expects images to exist even though norm stats only need state/actions.
    dummy_images = {"cam_high": np.zeros((len(state_batch), 1, 1, 3), dtype=np.uint8)}
    data = {"images": dummy_images, "state": state_batch, "actions": actions_batch}
    for transform in data_config.data_transforms.inputs:
        data = transform(data)
    return np.asarray(data["state"]), np.asarray(data["actions"])


def compute_parquet_norm_stats(
    data_config: _config.DataConfig,
    action_horizon: int,
    *,
    max_frames: int | None = None,
    chunk_size: int = 8192,
    seed: int = 0,
) -> dict[str, normalize.NormStats]:
    if data_config.rlds_data_dir is not None:
        raise ValueError("Parquet fast path only supports LeRobot datasets, not RLDS datasets.")
    if tuple(data_config.action_sequence_keys) != ("action",):
        raise ValueError(
            "Parquet fast path currently supports action_sequence_keys=('action',) only. "
            f"Got: {data_config.action_sequence_keys}"
        )

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


def _single_repo_data_config(data_config: _config.DataConfig, repo_id: str) -> _config.DataConfig:
    return dataclasses.replace(data_config, repo_id=repo_id, repo_ids=None, asset_id=None, norm_stats=None)


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
    data_config: _config.DataConfig,
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
        config.model,
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
    data_config = config.data.create(config.assets_dirs, config.model)

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
        if data_config.rlds_data_dir is not None:
            data_loader, num_batches = create_rlds_dataloader(
                data_config, config.model.action_horizon, config.batch_size, max_frames
            )
        else:
            data_loader, num_batches = create_torch_dataloader(
                data_config,
                config.model.action_horizon,
                config.batch_size,
                config.model,
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

    out_id = data_config.repo_id or data_config.asset_id
    if out_id is None:
        raise ValueError("Cannot determine output directory: repo_id and asset_id are both None")
    output_path = config.assets_dirs / out_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
