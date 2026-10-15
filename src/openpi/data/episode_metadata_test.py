"""Episode annotations must remain correct across samples, repos and batching."""

from types import SimpleNamespace

import numpy as np
import pytest
from torch.utils.data import default_collate

from openpi.data.episode_metadata import attach_episode_metadata, install_episode_metadata_reader


def test_episode_join_uses_local_episode_id():
    """A frame index is not an episode index and must never select the operator."""
    metadata = SimpleNamespace(episodes=[{"episode_index": 0, "operator": "Zhaonan Ma", "rating": None},
                                        {"episode_index": 1, "operator": "Jia Long", "rating": 5}])
    action = np.zeros((50, 14), dtype=np.float32)
    item = {"episode_index": 1, "index": 700, "action": action, "is_for_training": False}
    result = attach_episode_metadata(item, metadata)
    assert result["operator"] == "Jia Long" and result["rating"] == 5 and result["rating_is_valid"]
    assert result["action"] is action and not result["is_for_training"]
    assert "operator" not in item


def test_unrated_collates_without_none():
    """Null scores stay distinct from the lowest real score and support default batching."""
    metadata = SimpleNamespace(episodes=[{"episode_index": 0, "operator": "Xinzheng He", "rating": None},
                                        {"episode_index": 1, "operator": "Xinzheng He", "rating": 1}])
    batch = default_collate([attach_episode_metadata({"episode_index": i}, metadata) for i in range(2)])
    assert batch["rating"].tolist() == [0, 1]
    assert batch["rating_is_valid"].tolist() == [False, True]


@pytest.mark.parametrize("rating", [True, 0, 6, 2.5, "3"])
def test_invalid_rating_is_rejected(rating):
    """Do not silently coerce bad episode scores into valid training metadata."""
    with pytest.raises(ValueError, match="rating"):
        attach_episode_metadata({"episode_index": 0}, SimpleNamespace(episodes=[{"episode_index": 0, "rating": rating}]))


def test_old_frame_operator_remains_supported():
    """Older immutable snapshots still provide their existing per-frame operator."""
    result = attach_episode_metadata({"episode_index": 0, "operator": "Zhaonan Ma"},
                                     SimpleNamespace(episodes=[{"episode_index": 0}]))
    assert result["operator"] == "Zhaonan Ma" and result["rating"] == 0 and not result["rating_is_valid"]


def test_multi_repo_join_and_idempotence():
    """Identical local episode IDs in different repos must not mix their annotations."""
    class Single:
        """Minimal single-repo reader preserving an existing action reference."""
        def __init__(self, name, rating):
            self.meta = SimpleNamespace(episodes=[{"episode_index": 0, "operator": name, "rating": rating}])

        def __getitem__(self, index):
            return {"episode_index": 0, "index": index}

    class Multi:
        """Model upstream's removal of disabled non-common frame features."""
        def __init__(self):
            self._datasets = [Single("Zhaonan Ma", None), Single("Jia Long", 4)]

        def __getitem__(self, index):
            item = self._datasets[index][10]
            item.pop("operator", None)
            return {**item, "dataset_index": index}

    module = SimpleNamespace(LeRobotDataset=Single, MultiLeRobotDataset=Multi)
    install_episode_metadata_reader(module)
    first = Single.__getitem__
    install_episode_metadata_reader(module)
    assert Single.__getitem__ is first
    data = Multi()
    assert data[0]["operator"] == "Zhaonan Ma" and data[0]["rating"] == 0
    assert data[1]["operator"] == "Jia Long" and data[1]["rating"] == 4
