"""Join annotation-derived causal memory to the ACTUAL returned LeRobot sample."""

import random
import dataclasses
import json
from pathlib import Path

import numpy as np

from openpi.shared.completed_step_memory import INSERT_AND_RINSE, RETIRED_SUBTASKS, normalize_subtask, validate_rinse_memory
from openpi.shared.long_memory import MemoryTimeline, encode_memory, episode_split, parse_annotation, memory_augmentation


class LongMemoryDataset:
    def __init__(self, dataset, *, raw_dataset, settings, insert_action_quality_prompt=False):
        self._dataset = dataset
        self.settings = settings
        self.insert_action_quality_prompt = insert_action_quality_prompt
        self.timelines = {}
        cache=json.loads(Path(settings.summary_cache_path).read_text())
        if cache.get('schema_version')!=2:raise ValueError('Summary cache v2 required')
        from openpi.shared.completed_memory_labels import TASK_CATALOG_VERSION
        if cache.get('task_catalog_version')!=TASK_CATALOG_VERSION:
            raise ValueError('Regenerate the memory cache for the current rinse task catalog')
        self.summary_cache=cache['episodes']
        masks = []
        seen=set()
        bases = getattr(raw_dataset, "_datasets", [raw_dataset])
        self.repo_ids = [base.repo_id for base in bases]
        for dataset_index, base in enumerate(bases):
            table = base.hf_dataset
            episodes = np.asarray(table["episode_index"], dtype=np.int64)
            times = np.asarray(table["timestamp"], dtype=float)
            labels = list(table["subtask"])
            eligible = np.asarray(table["is_for_training"], dtype=bool) if "is_for_training" in table.column_names else np.ones(len(episodes), bool)
            mask = eligible & np.asarray([
                fields is not None and fields[1] not in RETIRED_SUBTASKS
                for fields in map(parse_annotation, labels)
            ], dtype=bool)
            # Consecutive runs, not episode ID alone, protect against malformed concatenations.
            boundaries = np.r_[0, np.flatnonzero(np.diff(episodes)) + 1, len(episodes)]
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
                key = (dataset_index, int(episodes[start]))
                if key in seen:
                    raise ValueError("An episode appears in disjoint blocks")
                seen.add(key)
                updates=self.summary_cache.get(base.repo_id,{}).get(str(key[1]))
                if updates is not None:self.timelines[key]=MemoryTimeline(updates)
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
        if key not in self.timelines:
            raise ValueError('This EP has no verified summary labels; prepare its cache before training')
        clean=self.timelines[key].at(time_s)
        # The instruction is this frame's actual annotation, independent of memory corruption.
        fields=parse_annotation(sample['subtask'])
        if fields is None or fields[1] in RETIRED_SUBTASKS:raise ValueError('Sample has no annotated current task')
        task=normalize_subtask(fields[1],return_context=self.timelines[key].context_at(time_s))
        clean=dataclasses.replace(clean,bottle_description=fields[0] or clean.bottle_description,current_task=task)
        clean,memory,augmentation=memory_augmentation(self.timelines[key],time_s,self.settings,training=True,rng=random,clean=clean)
        if self.insert_action_quality_prompt and task == INSERT_AND_RINSE:
            annotation = sample['subtask']
            annotation = json.loads(annotation) if isinstance(annotation, str) else annotation
            quality = annotation.get('good_bad_action', 'normal')
            if quality not in ('normal', 'good action', 'bad action', None, ''):
                raise ValueError(f'Invalid insertion action quality: {quality!r}')
            sample['insert_action_quality'] = 'bad' if quality == 'bad action' else 'good'
        sample['long_memory_before_augmentation']=encode_memory(clean)
        sample['memory_augmentation']=augmentation
        validate_rinse_memory(task, memory.memory)
        sample["long_memory"] = encode_memory(memory)
        return sample
