import dataclasses

import jax
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


class _BadVideoSampleDataset:
    def __init__(self, bad_indices: set[int], size: int = 8):
        self._bad_indices = bad_indices
        self._size = size

    def __getitem__(self, index: int) -> dict:
        if index in self._bad_indices:
            raise AssertionError(
                "One or several query timestamps unexpectedly violate the tolerance "
                "(tensor([0.0667]) > tolerance_s=0.0001)."
                "It means that the closest frame that can be loaded from the video is too far away in time."
                "This might be due to synchronization issues with timestamps during data collection."
                "To be safe, we advise to ignore this item during training."
            )
        return {"value": index}

    def get_sample_context(self, index: int) -> dict:
        return {
            "repo_id": "test/repo",
            "episode_index": index // 2,
            "frame_index": index,
        }

    def __len__(self) -> int:
        return self._size


def test_retry_on_error_dataset_skips_bad_sample():
    dataset = _data_loader.RetryOnErrorDataset(
        _BadVideoSampleDataset({1}),
        tracker=_data_loader.SkippedItemTracker(),
    )

    assert dataset[0]["value"] == 0
    assert dataset[1]["value"] == 2
    assert dataset._tracker.report() == (
        "Skipped 1 frame(s) due to recoverable video timestamp mismatches:\n"
        "  - dataset=test/repo, episode=0, skipped_frames=1"
    )


def test_retry_on_error_dataset_raises_after_retry_budget():
    dataset = _data_loader.RetryOnErrorDataset(
        _BadVideoSampleDataset(set(range(8))),
        max_retries=2,
        tracker=_data_loader.SkippedItemTracker(),
    )

    with pytest.raises(RuntimeError, match="Exceeded retry budget"):
        _ = dataset[0]


def test_torch_data_loader_parallel_skips_recoverable_errors():
    dataset = _data_loader.RetryOnErrorDataset(_BadVideoSampleDataset({1, 5}, size=10))

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert batch["value"].shape == (4,)
    assert loader.skipped_items_report() == (
        "Skipped 2 frame(s) due to recoverable video timestamp mismatches:\n"
        "  - dataset=test/repo, episode=0, skipped_frames=1\n"
        "  - dataset=test/repo, episode=2, skipped_frames=1"
    )
