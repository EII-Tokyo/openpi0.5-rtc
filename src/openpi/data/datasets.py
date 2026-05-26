from collections.abc import Iterator, Sequence
import logging
from pathlib import Path
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import pyarrow.parquet as pq
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.data import transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


def _patch_lerobot_numpy_bool_compat() -> None:
    """Make LeRobot padding masks robust to numpy bool scalars under numpy 2.x."""

    original = getattr(lerobot_dataset.LeRobotDataset, "_get_query_indices", None)
    if original is None or getattr(original, "_openpi_numpy_bool_compat", False):
        return

    def _patched_get_query_indices(self, idx: int, ep_idx: int):
        ep = self.meta.episodes[ep_idx]
        ep_start = int(ep["dataset_from_index"])
        ep_end = int(ep["dataset_to_index"])
        idx = int(idx)
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, idx + int(delta))) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.tensor(
                [bool((idx + int(delta) < ep_start) or (idx + int(delta) >= ep_end)) for delta in delta_idx],
                dtype=torch.bool,
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    _patched_get_query_indices._openpi_numpy_bool_compat = True
    lerobot_dataset.LeRobotDataset._get_query_indices = _patched_get_query_indices


_patch_lerobot_numpy_bool_compat()


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

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
    """Stacks image observations from earlier timesteps in the same episode."""

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
        self._episode_indices, self._frame_indices, self._timestamps, self._dataset_indices = self._load_temporal_metadata(
            dataset
        )

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

        episode_index = self._episode_indices[idx]
        frame_index = self._frame_indices[idx]
        episode_start_idx = idx - frame_index
        history_indices: list[int] = []
        for offset in reversed(range(self._num_frames)):
            target_frame = max(frame_index - offset * self._stride_frames, 0)
            candidate_idx = episode_start_idx + target_frame
            if candidate_idx < 0 or candidate_idx >= len(self._dataset):
                raise AssertionError(
                    f"Temporal history index {candidate_idx} out of bounds for dataset length {len(self._dataset)} "
                    f"(idx={idx}, episode_index={episode_index}, frame_index={frame_index}, "
                    f"offset={offset}, stride_frames={self._stride_frames})"
                )
            if self._episode_indices[candidate_idx] != episode_index:
                raise AssertionError(
                    f"Temporal history crossed episode boundary: current episode {episode_index}, "
                    f"candidate episode {self._episode_indices[candidate_idx]} "
                    f"(idx={idx}, candidate_idx={candidate_idx}, frame_index={frame_index}, "
                    f"offset={offset}, stride_frames={self._stride_frames})"
                )
            if self._dataset_indices is not None and self._dataset_indices[candidate_idx] != self._dataset_indices[idx]:
                raise AssertionError(
                    f"Temporal history crossed dataset boundary: current dataset {self._dataset_indices[idx]}, "
                    f"candidate dataset {self._dataset_indices[candidate_idx]} "
                    f"(idx={idx}, candidate_idx={candidate_idx}, frame_index={frame_index}, "
                    f"offset={offset}, stride_frames={self._stride_frames})"
                )
            history_indices.append(candidate_idx)

        result = dict(current)
        frame_cache = {idx: current}
        frames = []
        for hist_idx in history_indices:
            frame = frame_cache.get(hist_idx)
            if frame is None:
                frame = self._dataset[hist_idx]
                frame_cache[hist_idx] = frame
            frames.append(frame)
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

    @classmethod
    def _load_temporal_metadata(
        cls, dataset: Dataset
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        episode_indices = cls._get_column(dataset, "episode_index")
        frame_indices = cls._get_column(dataset, "frame_index")
        timestamps = cls._get_column(dataset, "timestamp", dtype=np.float64)
        dataset_indices = cls._get_index_maps(dataset)
        return episode_indices, frame_indices, timestamps, dataset_indices

    @classmethod
    def _get_column(cls, dataset: Dataset, column: str, dtype=np.int64) -> np.ndarray:
        if isinstance(dataset, lerobot_dataset.MultiLeRobotDataset):
            parts = [cls._get_column(ds, column, dtype=dtype) for ds in dataset._datasets]
            return np.concatenate(parts, axis=0) if parts else np.zeros(0, dtype=dtype)
        if isinstance(dataset, lerobot_dataset.LeRobotDataset):
            values = dataset.hf_dataset[column]
            if np.issubdtype(np.dtype(dtype), np.floating):
                return np.asarray([float(value.item() if hasattr(value, "item") else value) for value in values], dtype=dtype)
            return np.asarray([cls._to_int(value) for value in values], dtype=dtype)
        if np.issubdtype(np.dtype(dtype), np.floating):
            return np.asarray(
                [float(dataset[i][column].item() if hasattr(dataset[i][column], "item") else dataset[i][column]) for i in range(len(dataset))],
                dtype=dtype,
            )
        return np.asarray([cls._to_int(dataset[i][column]) for i in range(len(dataset))], dtype=dtype)

    @staticmethod
    def _get_index_maps(dataset: Dataset) -> np.ndarray | None:
        if isinstance(dataset, lerobot_dataset.MultiLeRobotDataset):
            dataset_indices = []
            for dataset_index, subdataset in enumerate(dataset._datasets):
                dataset_indices.append(np.full(len(subdataset), dataset_index, dtype=np.int32))
            return np.concatenate(dataset_indices, axis=0)
        if isinstance(dataset, lerobot_dataset.LeRobotDataset):
            return np.zeros(len(dataset), dtype=np.int32)
        return None


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


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

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    repo_ids = data_config.repo_ids
    if repo_id is None and repo_ids is None:
        raise ValueError("Repo ID or repo IDs is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)
    fps_meta: lerobot_dataset.LeRobotDatasetMetadata | None = None
    if repo_ids is not None and len(repo_ids) > 0:
        fps_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_ids[0], force_cache_sync=True)
        delta_timestamps = {key: [t / fps_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys}
        dataset = lerobot_dataset.MultiLeRobotDataset(repo_ids, delta_timestamps=delta_timestamps)
    elif repo_id is not None:
        fps_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, force_cache_sync=True)
        delta_timestamps = {key: [t / fps_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys}
        dataset = lerobot_dataset.LeRobotDataset(
            data_config.repo_id,
            delta_timestamps=delta_timestamps,
        )
    else:
        raise ValueError("Repo ID or non-empty repo_ids is required. Cannot create dataset.")
    # print(data_config)
    # if data_config.prompt_from_task:
    #     # dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
    #     dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask()])

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
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *([_transforms.PromptFromLeRobotTask()] if data_config.prompt_from_task else []),
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )
