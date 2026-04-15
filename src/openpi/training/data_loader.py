from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import time
import os
from pathlib import Path
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import datasets
import jax
import jax.numpy as jnp
import lerobot.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import pyarrow.parquet as pq
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index.__index__()])

    def __len__(self) -> int:
        return len(self._dataset)


class IsForTrainingWrapper(Dataset[T_co]):
    """Redirect indices marked as not-for-training to random trainable samples."""

    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._trainable_mask = self._build_trainable_mask(dataset)
        self._trainable_indices = np.flatnonzero(self._trainable_mask)
        if len(self._trainable_indices) == 0:
            raise ValueError("Dataset has no samples with is_for_training=true.")

    def __getitem__(self, index: SupportsIndex) -> T_co:
        idx = index.__index__()
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self)}.")
        if not self._trainable_mask[idx]:
            idx = int(self._trainable_indices[torch.randint(len(self._trainable_indices), (1,)).item()])
        return self._dataset[idx]

    def __len__(self) -> int:
        return len(self._dataset)

    @classmethod
    def _build_trainable_mask(cls, dataset: Dataset) -> np.ndarray:
        if isinstance(dataset, lerobot_dataset.MultiLeRobotDataset):
            masks = [cls._build_lerobot_mask(ds) for ds in dataset._datasets]
            return np.concatenate(masks, axis=0) if masks else np.zeros(0, dtype=bool)
        if isinstance(dataset, lerobot_dataset.LeRobotDataset):
            return cls._build_lerobot_mask(dataset)
        return np.ones(len(dataset), dtype=bool)

    @staticmethod
    def _build_lerobot_mask(dataset: lerobot_dataset.LeRobotDataset) -> np.ndarray:
        mask = np.ones(len(dataset), dtype=bool)
        found_column = False
        parquet_files = sorted(Path(dataset.root).glob("data/**/*.parquet"))
        for parquet_file in parquet_files:
            pf = pq.ParquetFile(parquet_file)
            if "is_for_training" not in pf.schema.names:
                continue
            found_column = True
            table = pf.read(columns=["index", "is_for_training"])
            indices = np.asarray(table.column("index").to_pylist(), dtype=np.int64)
            values = np.asarray(table.column("is_for_training").to_pylist(), dtype=bool)
            mask[indices] = values
        if found_column:
            logging.info(
                "Loaded is_for_training mask for %s: %d/%d trainable",
                dataset.repo_id,
                int(mask.sum()),
                len(mask),
            )
        else:
            logging.info(
                "Dataset %s has no is_for_training column locally; treating all %d samples as trainable",
                dataset.repo_id,
                len(mask),
            )
        return mask


class TemporalFrameStackDataset(Dataset[dict]):
    """Stacks image observations from earlier timesteps in the same episode.

    Frames are sampled at fixed second intervals and returned in oldest->newest order.
    If the requested history is not available inside the episode, the earliest valid frame
    is repeated.
    """

    IMAGE_PREFIX = "observation.images."

    def __init__(
        self,
        dataset: Dataset,
        *,
        fps: float,
        num_frames: int,
        stride_seconds: float,
        trainable_mask: np.ndarray | None = None,
    ):
        self._dataset = dataset
        self._fps = fps
        self._num_frames = num_frames
        self._stride_seconds = stride_seconds
        self._stride_frames = max(1, int(round(fps * stride_seconds)))
        self._trainable_mask = trainable_mask
        self._trainable_indices = None if trainable_mask is None else np.flatnonzero(trainable_mask)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__()
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self)}.")
        if self._trainable_mask is not None and not self._trainable_mask[idx]:
            idx = int(self._trainable_indices[torch.randint(len(self._trainable_indices), (1,)).item()])
        current = self._dataset[idx]
        if self._num_frames <= 1:
            return current

        episode_index = self._to_int(current["episode_index"])
        frame_index = self._to_int(current["frame_index"])
        episode_start_idx = idx - frame_index
        history_indices: list[int] = []
        for offset in reversed(range(self._num_frames)):
            target_frame = max(frame_index - offset * self._stride_frames, 0)
            candidate_idx = episode_start_idx + target_frame
            if candidate_idx < 0 or candidate_idx >= len(self._dataset):
                candidate_idx = idx
            else:
                candidate = self._dataset[candidate_idx]
                if self._to_int(candidate["episode_index"]) != episode_index:
                    candidate_idx = idx
            history_indices.append(candidate_idx)

        frames = [self._dataset[hist_idx] for hist_idx in history_indices]
        result = dict(current)
        for key in current:
            if key.startswith(self.IMAGE_PREFIX):
                stacked = [np.asarray(frame[key]) for frame in frames]
                result[key] = np.stack(stacked, axis=0)
        return result

    @staticmethod
    def _to_int(value) -> int:
        if hasattr(value, "item"):
            return int(value.item())
        return int(value)


class MultiLeRobotDatasetWithOptionalKeys(lerobot_dataset.MultiLeRobotDataset):
    """Multi-dataset wrapper that preserves selected optional keys with defaults.

    LeRobot's MultiLeRobotDataset drops any key that is not shared by all datasets.
    For mixed training we still want fields like `subtask` to survive, defaulting to
    an empty value on datasets that do not provide them.
    """

    def __init__(self, *args, optional_defaults: dict[str, typing.Any] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._optional_defaults = dict(optional_defaults or {})

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")

        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")

        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_features:
            if data_key in self._optional_defaults:
                item.setdefault(data_key, self._optional_defaults[data_key])
            elif data_key in item:
                del item[data_key]

        for data_key, default_value in self._optional_defaults.items():
            item.setdefault(data_key, default_value)
        return item

    @property
    def features(self) -> datasets.Features:
        features: dict[str, typing.Any] = {}
        for dataset in self._datasets:
            features.update(
                {
                    k: v
                    for k, v in dataset.hf_features.items()
                    if k not in self.disabled_features or k in self._optional_defaults
                }
            )
        for key in self._optional_defaults:
            features.setdefault(key, datasets.Value("string"))
        return datasets.Features(features)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        obs_dict = observation.to_dict()
        # Model transforms (e.g. `TokenizePrompt`) populate these; drop spec placeholders so collate is numeric-only.
        for k in list(obs_dict):
            if k.startswith("tokenized_") or k in ("token_ar_mask", "token_loss_mask"):
                obs_dict.pop(k, None)

        return {
            **obs_dict,
            "actions": action,
            # `transform_dataset` always runs `PromptFromLeRobotTask`, which needs a LeRobot-style task string.
            "task": "debug",
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    repo_ids = data_config.repo_ids
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    def _delta_timestamps(fps: float) -> dict[str, list[float]]:
        return {
            key: [t / fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        }

    fps_meta: lerobot_dataset.LeRobotDatasetMetadata | None = None
    if repo_ids is not None and len(repo_ids) > 0:
        if len(repo_ids) == 1:
            fps_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_ids[0], revision="main", force_cache_sync=False)
            dataset = lerobot_dataset.LeRobotDataset(
                repo_ids[0],
                revision="main",
                force_cache_sync=True,
                delta_timestamps=_delta_timestamps(fps_meta.fps),
            )
        else:
            fps_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_ids[0], force_cache_sync=False)
            dataset = MultiLeRobotDatasetWithOptionalKeys(
                repo_ids,
                delta_timestamps=_delta_timestamps(fps_meta.fps),
                optional_defaults={"subtask": "", "train_action": True},
            )
    elif repo_id is not None:
        fps_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, revision="main", force_cache_sync=False)
        dataset = lerobot_dataset.LeRobotDataset(
            repo_id,
            revision="main",
            force_cache_sync=True,
            delta_timestamps=_delta_timestamps(fps_meta.fps),
        )
    else:
        raise ValueError("Repo ID or non-empty repo_ids is required. Cannot create dataset.")

    trainable_mask = IsForTrainingWrapper._build_trainable_mask(dataset)
    if getattr(data_config, "video_memory_num_frames", 1) > 1:
        dataset = TemporalFrameStackDataset(
            dataset,
            fps=float(fps_meta.fps),
            num_frames=data_config.video_memory_num_frames,
            stride_seconds=data_config.video_memory_stride_seconds,
            trainable_mask=trainable_mask,
        )
    else:
        dataset = IsForTrainingWrapper(dataset)
    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )

    return TransformedDataset(
        dataset,
        [
            _transforms.PromptFromLeRobotTask(),
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions, np.ndarray]]:
    """Build LeRobot dataset + transforms + PyTorch DataLoader."""
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info("data_config: %s", data_config)
    if data_config.rlds_data_dir is not None:
        raise ValueError(
            "RLDS data loading was removed from this fork; use a LeRobot config with repo_id / repo_ids only."
        )

    model_config = config.model
    action_horizon = model_config.action_horizon
    batch_size = config.batch_size
    num_workers = config.num_workers
    seed = config.seed

    t0 = time.perf_counter()
    base_dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(base_dataset, data_config, skip_norm_stats=skip_norm_stats)

    subtask_eval_outer = None
    subtask_eval_split = None
    subtask_eval_index_to_class: dict[int, int] | None = None
    if config.subtask_eval_enabled:
        logging.info(
            "data_loader: subtask_eval holdout disabled; training uses full dataset and no automatic val split is created"
        )

    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(
        "data_loader: len=%d local_batch_size=%d build_time=%.2fs",
        len(dataset),
        local_batch_size,
        time.perf_counter() - t0,
    )
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(
        data_config,
        data_loader,
        subtask_eval_outer_dataset=subtask_eval_outer,
        subtask_eval_split=subtask_eval_split,
        subtask_eval_index_to_class=subtask_eval_index_to_class,
    )


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_WorkerInitFn(seed),
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    def _stack_with_context(path: str, values):
        arrays = [np.asarray(x) for x in values]
        try:
            return np.stack(arrays, axis=0)
        except ValueError as e:
            shapes = [arr.shape for arr in arrays]
            dtypes = [str(arr.dtype) for arr in arrays]
            raise ValueError(f"Failed to collate '{path}': shapes={shapes}, dtypes={dtypes}") from e

    def _collate(path: str, values):
        first = values[0]
        if isinstance(first, dict):
            return {k: _collate(f"{path}.{k}" if path else k, [v[k] for v in values]) for k in first}
        if hasattr(first, "_asdict"):
            data = {
                k: _collate(f"{path}.{k}" if path else k, [getattr(v, k) for v in values])
                for k in first._asdict()
            }
            return type(first)(**data)
        if isinstance(first, tuple):
            return type(first)(_collate(f"{path}[{i}]", [v[i] for v in values]) for i in range(len(first)))
        return _stack_with_context(path or "<root>", values)

    return _collate("", items)


class _WorkerInitFn:
    """Worker initialization function that can be pickled for multiprocessing."""

    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int) -> None:
        """Initialize worker process with JAX settings and numpy random seed."""
        # Tell JAX inside the worker process not to preallocate the GPU memory.
        # NOTE: This is called after jax is imported inside the worker process. This
        # means that this approach will not work for selecting the backend.
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        # Set numpy random seed for each worker to ensure different random states
        # Use seed + worker_id to ensure each worker has a different seed
        np.random.seed(self.seed + worker_id)


class DataLoaderImpl(DataLoader):
    def __init__(
        self,
        data_config: _config.DataConfig,
        data_loader: TorchDataLoader,
        *,
        subtask_eval_outer_dataset: typing.Any = None,
        subtask_eval_split: typing.Any = None,
        subtask_eval_index_to_class: dict[int, int] | None = None,
    ):
        self._data_config = data_config
        self._data_loader = data_loader
        self.subtask_eval_outer_dataset = subtask_eval_outer_dataset
        self.subtask_eval_split = subtask_eval_split
        self.subtask_eval_index_to_class = subtask_eval_index_to_class or {}

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            actions_mask = batch.get("actions_mask")
            if actions_mask is None:
                batch_size = batch["actions"].shape[0]
                actions_mask = jnp.ones((batch_size,), dtype=jnp.bool_)
            else:
                actions_mask = jnp.asarray(actions_mask, dtype=jnp.bool_)
            batch["actions_mask"] = actions_mask
            yield _model.Observation.from_dict(batch), batch["actions"], actions_mask
