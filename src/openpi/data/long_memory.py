"""Join annotation-derived causal memory to the ACTUAL returned LeRobot sample."""

import random

import numpy as np

from openpi.shared.long_memory import MemoryTimeline, condition_at, encode_memory, episode_split


class LongMemoryDataset:
    def __init__(self, dataset, *, raw_dataset, settings):
        self._dataset = dataset
        self.settings = settings
        self.timelines = {}
        masks = []
        bases = getattr(raw_dataset, "_datasets", [raw_dataset])
        self.repo_ids = [base.repo_id for base in bases]
        for dataset_index, base in enumerate(bases):
            table = base.hf_dataset
            episodes = np.asarray(table["episode_index"], dtype=np.int64)
            times = np.asarray(table["timestamp"], dtype=float)
            labels = list(table["subtask"])
            eligible = np.asarray(table["is_for_training"], dtype=bool) if "is_for_training" in table.column_names else np.ones(len(episodes), bool)
            mask = eligible.copy()
            # Consecutive runs, not episode ID alone, protect against malformed concatenations.
            boundaries = np.r_[0, np.flatnonzero(np.diff(episodes)) + 1, len(episodes)]
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
                key = (dataset_index, int(episodes[start]))
                if key in self.timelines:
                    raise ValueError("An episode appears in disjoint blocks")
                self.timelines[key] = MemoryTimeline(times[start:end], labels[start:end], eligible[start:end], reset_on_home=settings.reset_on_home)
                if episode_split(base.repo_id, key[1], settings.split_seed) != "train":
                    mask[start:end] = False
            masks.append(mask)
        self.train_mask = np.concatenate(masks)
        self.train_indices = np.flatnonzero(self.train_mask)
        if not len(self.train_indices):
            raise ValueError("No eligible training episodes remain after the shared episode split")

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        # Existing eligibility wrappers may redirect index. Use returned provenance,
        # NEVER the originally requested index, for the memory join.
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if not self.train_mask[index]:
            index = int(self.train_indices[random.randrange(len(self.train_indices))])
        sample = dict(self._dataset[index])
        key = (int(sample.get("dataset_index", 0)), int(sample["episode_index"]))
        if episode_split(self.repo_ids[key[0]], key[1], self.settings.split_seed) != "train":
            raise ValueError("Underlying wrapper returned an evaluation episode during training")
        time_s = float(sample["timestamp"])
        memory = condition_at(self.timelines[key], time_s, self.settings, training=True, rng=random)
        sample["long_memory"] = encode_memory(memory)
        sample["memory_time_s"] = time_s
        return sample
