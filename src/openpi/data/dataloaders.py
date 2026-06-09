from collections.abc import Iterator
import logging
import multiprocessing
import os
from typing import Protocol, TypeVar

import jax
import numpy as np

import openpi.models.model as _model
from openpi.data import datasets as _datasets
import openpi.training.config as _config

T_co = TypeVar("T_co", covariant=True)
_WORKER_DATASET = None


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.LeRobotAlohaDataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training."""
    data_config = config.data
    logging.info("data_config: %s", data_config)

    dataset = _datasets.create_dataset(data_config, config.model.action_horizon)
    dataset = _datasets.transform_dataset(dataset, data_config)

    local_batch_size = config.batch_size // jax.process_count()

    logging.info("dataset length: %d", len(dataset))
    logging.info("local_batch_size: %d", local_batch_size)
    data_loader = BatchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    return DataLoaderImpl(data_config, data_loader)


def _init_worker(dataset, seed: int) -> None:
    global _WORKER_DATASET
    _WORKER_DATASET = dataset
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    np.random.seed(seed + os.getpid())


def _worker_get_item(index: int):
    if _WORKER_DATASET is None:
        raise RuntimeError("Data loader worker was not initialized.")
    return _WORKER_DATASET[index]


class BatchDataLoader:
    """Small numpy/JAX batch loader for random-access datasets."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        self._dataset = dataset
        self._local_batch_size = local_batch_size
        self._shuffle = shuffle
        self._num_batches = num_batches
        self._rng = np.random.default_rng(seed)
        self._num_workers = num_workers
        self._pool = None
        self._worker_context = None

        self._sharding = sharding
        if sharding is None:
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

    def _indices_for_epoch(self) -> np.ndarray:
        indices = np.arange(len(self._dataset), dtype=np.int64)
        if self._shuffle:
            self._rng.shuffle(indices)
        return indices

    def _get_pool(self):
        if self._num_workers <= 0:
            return None
        if self._pool is None:
            self._worker_context = multiprocessing.get_context("spawn")
            self._pool = self._worker_context.Pool(
                processes=self._num_workers,
                initializer=_init_worker,
                initargs=(self._dataset, int(self._rng.integers(0, 2**31 - 1))),
            )
        return self._pool

    def _load_items(self, indices: np.ndarray):
        pool = self._get_pool()
        if pool is None:
            return [self._dataset[int(index)] for index in indices]
        return pool.map(_worker_get_item, [int(index) for index in indices])

    def __iter__(self):
        num_items = 0
        while True:
            indices = self._indices_for_epoch()
            for start in range(0, len(indices) - self._local_batch_size + 1, self._local_batch_size):
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                batch_indices = indices[start : start + self._local_batch_size]
                batch = _collate_fn(self._load_items(batch_indices))
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.LeRobotAlohaDataConfig, data_loader: BatchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.LeRobotAlohaDataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
