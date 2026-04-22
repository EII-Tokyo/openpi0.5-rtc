"""Compute normalization statistics for a config.

For ALOHA LeRobot configs, this script can use a parquet fast path that reads
only numeric state/action columns and avoids image decoding.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tqdm
import tyro

import openpi.models.model as _model
from openpi.policies import aloha_policy
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _disable_norm(transforms_seq: list[transforms.DataTransformFn]) -> list[transforms.DataTransformFn]:
    disabled: list[transforms.DataTransformFn] = []
    for transform in transforms_seq:
        if isinstance(transform, (transforms.Normalize, transforms.Unnormalize)):
            continue
        if hasattr(transform, "apply_norm"):
            disabled.append(dataclasses.replace(transform, apply_norm=False))
        else:
            disabled.append(transform)
    return disabled


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None and not data_config.repo_ids:
        raise ValueError("Data config must have repo_id or repo_ids")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *_disable_norm(list(data_config.data_transforms.inputs)),
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
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


def _load_lerobot_arrays(repo_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id, revision="main", force_cache_sync=False)
    parquet_files = sorted(Path(meta.root).glob("data/**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {meta.root}/data")

    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    global_indices: list[np.ndarray] = []
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file, columns=["observation.state", "action", "episode_index", "index"])
        actions.append(np.asarray(table["action"].to_pylist(), dtype=np.float32))
        states.append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32))
        episode_indices.append(np.asarray(table["episode_index"].to_pylist(), dtype=np.int64))
        global_indices.append(np.asarray(table["index"].to_pylist(), dtype=np.int64))

    actions_np = np.concatenate(actions, axis=0)
    states_np = np.concatenate(states, axis=0)
    episode_index_np = np.concatenate(episode_indices, axis=0)
    global_index_np = np.concatenate(global_indices, axis=0)
    order = np.argsort(global_index_np)
    return actions_np[order], states_np[order], episode_index_np[order]


def _action_chunks_by_episode(actions: np.ndarray, states: np.ndarray, episode_index: np.ndarray, horizon: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    chunk_states: list[np.ndarray] = []
    change = np.flatnonzero(np.diff(episode_index) != 0) + 1
    starts = np.concatenate([np.array([0]), change])
    ends = np.concatenate([change, np.array([len(episode_index)])])
    for start, end in zip(starts, ends, strict=True):
        episode_actions = actions[start:end]
        episode_states = states[start:end]
        for offset in range(len(episode_actions)):
            chunk = episode_actions[offset : offset + horizon].copy()
            if len(chunk) < horizon:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], horizon - len(chunk), axis=0)], axis=0)
            chunks.append(chunk)
            chunk_states.append(episode_states[offset])

    action_chunks = np.stack(chunks, axis=0)
    state_chunks = np.stack(chunk_states, axis=0)
    delta_mask = np.asarray(transforms.make_bool_mask(6, -1, 6, -1), dtype=bool)
    dims = delta_mask.shape[-1]
    action_chunks[..., :dims] -= np.expand_dims(np.where(delta_mask, state_chunks[..., :dims], 0), axis=-2)
    return action_chunks


def _compute_aloha_lerobot_fast_stats(
    repo_ids: list[str],
    *,
    action_horizon: int,
    adapt_to_pi: bool,
    max_frames: int | None,
) -> dict[str, normalize.NormStats]:
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    remaining = max_frames
    for repo_id in repo_ids:
        raw_actions, raw_states, episode_index = _load_lerobot_arrays(repo_id)
        if remaining is not None:
            raw_actions = raw_actions[:remaining]
            raw_states = raw_states[:remaining]
            episode_index = episode_index[:remaining]
            remaining -= len(raw_actions)
        print(f"[{repo_id}] frames={len(raw_actions)}")

        states = np.asarray(aloha_policy._decode_state(raw_states, adapt_to_pi=adapt_to_pi), dtype=np.float32)
        actions = np.asarray(aloha_policy._encode_actions_inv(raw_actions, adapt_to_pi=adapt_to_pi), dtype=np.float32)
        action_chunks = _action_chunks_by_episode(actions, states, episode_index, action_horizon)
        print(f"[{repo_id}] action_chunks={action_chunks.shape}")

        stats["state"].update(states)
        stats["actions"].update(action_chunks)
        if remaining is not None and remaining <= 0:
            break

    return {key: running.get_statistics() for key, running in stats.items()}


def _write_stats(output_path: Path, norm_stats: dict[str, normalize.NormStats]) -> None:
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


def main(
    config_name: str,
    max_frames: int | None = None,
    output_dir: Path | None = None,
    repo_ids: list[str] | None = None,
    use_fast_lerobot_aloha_path: bool = True,
):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    selected_repo_ids = repo_ids or data_config.repo_ids or ([data_config.repo_id] if data_config.repo_id else [])

    if use_fast_lerobot_aloha_path and selected_repo_ids:
        norm_stats = _compute_aloha_lerobot_fast_stats(
            selected_repo_ids,
            action_horizon=config.model.action_horizon,
            adapt_to_pi=getattr(config.data, "adapt_to_pi", True),
            max_frames=max_frames,
        )
        if output_dir is None:
            if data_config.asset_id is None:
                raise ValueError("Data config must define asset_id to write norm stats.")
            output_dir = config.assets_dirs / data_config.asset_id
        _write_stats(output_dir, norm_stats)
        return

    if data_config.rlds_data_dir is not None:
        raise ValueError(
            "This script only supports LeRobot (torch) configs. "
            "RLDS/DROID norm stats are not supported here; use a config without rlds_data_dir."
        )

    data_loader, num_batches = create_torch_dataloader(
        data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    if data_config.asset_id is None:
        raise ValueError("Data config must define asset_id to write norm stats.")
    output_path = output_dir or (config.assets_dirs / data_config.asset_id)
    _write_stats(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
