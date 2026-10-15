"""Expose episode annotations as sample fields without duplicating stored frames."""

from functools import wraps
from numbers import Integral


def attach_episode_metadata(item: dict, metadata) -> dict:
    """Join by the sample's local episode ID; encode unrated as zero plus a mask."""
    episode_index = item["episode_index"]
    if hasattr(episode_index, "item"):
        episode_index = episode_index.item()
    episode = metadata.episodes[int(episode_index)]
    if int(episode["episode_index"]) != int(episode_index):
        raise ValueError("Episode metadata index does not match sample")
    result = dict(item)
    operator = episode.get("operator", result.get("operator", ""))
    if operator is None:
        operator = result.get("operator")
    if operator is None:
        operator = ""
    if not isinstance(operator, str):
        raise ValueError("Episode operator must be a string")
    result["operator"] = operator
    rating = episode.get("rating")
    if rating is not None and (isinstance(rating, bool) or not isinstance(rating, Integral) or not 1 <= rating <= 5):
        raise ValueError("Episode rating must be null or an integer from 1 to 5")
    result["rating"] = int(rating) if rating is not None else 0
    result["rating_is_valid"] = rating is not None
    return result


def install_episode_metadata_reader(module=None) -> None:
    """Extend both single and multi-repo readers, following the existing loader patches."""
    if module is None:
        import lerobot.datasets.lerobot_dataset as module

    single = module.LeRobotDataset
    if not getattr(single.__getitem__, "_openpi_episode_metadata", False):
        original_single = single.__getitem__

        @wraps(original_single)
        def single_item(self, index):
            """Preserve video decoding, action windows, padding and frame eligibility."""
            return attach_episode_metadata(original_single(self, index), self.meta)

        single_item._openpi_episode_metadata = True
        single.__getitem__ = single_item

    multi = module.MultiLeRobotDataset
    if not getattr(multi.__getitem__, "_openpi_episode_metadata", False):
        original_multi = multi.__getitem__

        @wraps(original_multi)
        def multi_item(self, index):
            """Reattach after LeRobot drops non-common features in mixed old/new repos."""
            item = original_multi(self, index)
            dataset_index = item["dataset_index"]
            if hasattr(dataset_index, "item"):
                dataset_index = dataset_index.item()
            return attach_episode_metadata(item, self._datasets[int(dataset_index)].meta)

        multi_item._openpi_episode_metadata = True
        multi.__getitem__ = multi_item
