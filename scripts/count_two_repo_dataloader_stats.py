import argparse
import dataclasses
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from openpi.models.tokenizer import get_good_bad_action_label
from openpi.training import config as training_config
from openpi.training import data_loader

DEFAULT_REPOS = [
    "lyl472324464/2026-03-09-no-cap-inference",
    "lyl472324464/2026-02-03-no-cap-and-direction",
]


def _normalize_subtask_values(values: Any) -> list[Any]:
    if hasattr(values, "numpy"):
        values = values.numpy()
    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            return [values.item()]
        return values.tolist()
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


def _count_labels(subtasks: list[Any]) -> Counter:
    counts = Counter()
    for subtask in subtasks:
        label = get_good_bad_action_label(subtask)
        counts[label] += 1
    return counts


def _count_local_parquet(repo_id: str) -> dict[str, int]:
    root = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    total = 0
    trainable = 0
    skipped = 0
    labels = Counter()

    for parquet_file in sorted(root.glob("data/**/*.parquet")):
        pf = pq.ParquetFile(parquet_file)
        columns = pf.schema.names
        total += pf.metadata.num_rows

        read_columns = [c for c in ["subtask", "is_for_training"] if c in columns]
        table = pf.read(columns=read_columns) if read_columns else None

        if "is_for_training" in columns:
            flags = table.column("is_for_training").to_pylist()
            trainable += sum(v is True for v in flags)
            skipped += sum(v is False for v in flags)
        else:
            trainable += pf.metadata.num_rows

        if "subtask" in columns:
            labels.update(_count_labels(table.column("subtask").to_pylist()))

    return {
        "total": total,
        "trainable": trainable,
        "not_for_training": skipped,
        "good_action": labels["good action"],
        "bad_action": labels["bad action"],
        "normal": labels["normal"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="twist_off_the_bottle_cap_lora")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo", action="append", dest="repos")
    args = parser.parse_args()

    repos = args.repos or DEFAULT_REPOS

    config = training_config.get_config(args.config_name)
    config = dataclasses.replace(
        config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data=dataclasses.replace(config.data, repo_ids=repos),
    )
    data_config = config.data.create(config.assets_dirs, config.model)

    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    loader = data_loader.TorchDataLoader(
        dataset,
        local_batch_size=args.batch_size,
        shuffle=False,
        num_batches=len(dataset) // args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        framework="pytorch",
    )

    source_counts = {repo: _count_local_parquet(repo) for repo in repos}
    source_total = Counter()
    for repo_counts in source_counts.values():
        source_total.update(repo_counts)

    yielded = 0
    yielded_labels = Counter()
    for batch in loader.torch_loader:
        subtasks = _normalize_subtask_values(batch["subtask"])
        yielded += len(subtasks)
        yielded_labels.update(_count_labels(subtasks))

    requested_trainable = int(dataset._trainable_mask[:yielded].sum())
    requested_not_for_training = int(yielded - requested_trainable)

    report = {
        "repos": repos,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "source_counts_per_repo": source_counts,
        "source_counts_total": dict(source_total),
        "dataloader_one_pass": {
            "yielded_samples": yielded,
            "drop_last": len(dataset) - yielded,
            "requested_trainable": requested_trainable,
            "requested_not_for_training": requested_not_for_training,
            "effective_trainable_yielded": yielded,
            "effective_not_for_training_yielded": 0,
            "good_action": yielded_labels["good action"],
            "bad_action": yielded_labels["bad action"],
            "normal": yielded_labels["normal"],
        },
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
