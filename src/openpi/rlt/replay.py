from __future__ import annotations

from pathlib import Path

from flax import struct

import jax
import jax.numpy as jnp
import numpy as np


@struct.dataclass
class ReplayBatch:
    state: jax.Array
    next_state: jax.Array
    reference_action_chunk: jax.Array
    next_reference_action_chunk: jax.Array
    executed_action_chunk: jax.Array
    reward: jax.Array
    done: jax.Array
    episode_id: jax.Array
    step_index: jax.Array
    is_intervention: jax.Array


def create_synthetic_replay(path: str | Path, *, num_samples: int = 1024, action_horizon: int = 50, action_dim: int = 32, state_dim: int = 32, seed: int = 0) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    reference = rng.normal(size=(num_samples, action_horizon, action_dim)).astype(np.float32)
    executed = reference + 0.05 * rng.normal(size=reference.shape).astype(np.float32)
    done = np.zeros((num_samples,), dtype=np.float32)
    done[-1] = 1.0
    reward = np.zeros((num_samples,), dtype=np.float32)
    reward[-1] = 1.0
    shard = path / "shard_00000.npz"
    np.savez_compressed(
        shard,
        state=rng.normal(size=(num_samples, state_dim)).astype(np.float32),
        next_state=rng.normal(size=(num_samples, state_dim)).astype(np.float32),
        reference_action_chunk=reference,
        next_reference_action_chunk=np.roll(reference, -1, axis=0),
        executed_action_chunk=executed,
        reward=reward,
        done=done,
        episode_id=np.zeros((num_samples,), dtype=np.int32),
        step_index=np.arange(num_samples, dtype=np.int32),
        is_intervention=np.zeros((num_samples,), dtype=np.float32),
    )
    return shard


class ReplayDataset:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        files = sorted(self.path.glob("*.npz")) if self.path.is_dir() else [self.path]
        if not files:
            raise FileNotFoundError(f"No replay .npz files found under {self.path}")
        arrays = [np.load(file) for file in files]
        self.data = {key: np.concatenate([arr[key] for arr in arrays], axis=0) for key in arrays[0].files}
        self.size = int(next(iter(self.data.values())).shape[0])

    def sample(self, rng: np.random.Generator, batch_size: int) -> ReplayBatch:
        idx = rng.integers(0, self.size, size=batch_size)
        data = {key: jnp.asarray(value[idx]) for key, value in self.data.items()}
        return ReplayBatch(**data)
