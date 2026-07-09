from __future__ import annotations

from pathlib import Path

from flax import struct

import jax
import jax.numpy as jnp
import numpy as np
from torch.utils import data as torch_data


@struct.dataclass
class ReplayBatch:
    rlt_token: jax.Array
    next_rlt_token: jax.Array
    normalized_state: jax.Array
    normalized_next_state: jax.Array
    normalized_reference_action_chunk: jax.Array
    normalized_next_reference_action_chunk: jax.Array
    normalized_executed_action_chunk: jax.Array
    executed_action_mask: jax.Array
    td_reward: jax.Array
    done: jax.Array
    episode_id: jax.Array
    step_index: jax.Array
    sample_index: jax.Array
    is_intervention: jax.Array


REPLAY_BATCH_FIELDS = tuple(ReplayBatch.__dataclass_fields__)


class ReplayDataset(torch_data.Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        files = sorted(self.path.glob("*.npz")) if self.path.is_dir() else [self.path]
        if not files:
            raise FileNotFoundError(f"No replay .npz files found under {self.path}")
        arrays = [np.load(file) for file in files]
        optional_defaults = {"sample_index": lambda arr: np.zeros((arr[REPLAY_BATCH_FIELDS[0]].shape[0],), dtype=np.int32)}
        missing = [key for key in REPLAY_BATCH_FIELDS if key not in arrays[0].files and key not in optional_defaults]
        if missing:
            raise KeyError("Replay shard " + str(files[0]) + " missing required fields: " + str(tuple(missing)))

        def _field_array(arr, key: str) -> np.ndarray:
            if key in arr.files:
                return arr[key]
            if key in optional_defaults:
                return optional_defaults[key](arr)
            raise KeyError(key)

        self.data = {key: np.concatenate([_field_array(arr, key) for arr in arrays], axis=0) for key in REPLAY_BATCH_FIELDS}
        self.source_files = [str(file) for file in files]
        self.split_episode_id = np.concatenate(
            [np.full((arr[REPLAY_BATCH_FIELDS[0]].shape[0],), file_idx, dtype=np.int32) for file_idx, arr in enumerate(arrays)],
            axis=0,
        )
        self.size = int(next(iter(self.data.values())).shape[0])

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return {key: value[idx] for key, value in self.data.items()}


def collate_replay_batch(samples: list[dict[str, np.ndarray]]) -> ReplayBatch:
    # This runs inside PyTorch DataLoader workers when num_workers > 0.
    # Keep it numpy-only; touching JAX in forked workers can crash CUDA.
    data = {}
    for key in REPLAY_BATCH_FIELDS:
        data[key] = np.stack([sample[key] for sample in samples], axis=0)
    return ReplayBatch(**data)


def batch_to_jax(batch: ReplayBatch) -> ReplayBatch:
    return ReplayBatch(**{key: jnp.asarray(getattr(batch, key)) for key in REPLAY_BATCH_FIELDS})
