#!/usr/bin/env python3
"""Build a fixed, balanced twist nine-class evaluation manifest.

The output is a JSON file containing frame references into LeRobot repos:
- repo_id
- frame_index
- episode_index
- bottle_state
- subtask
- class_id

Design goals:
- only score the canonical 9 twist classes
- avoid leakage from the twist training subset source repo by default
- reduce near-duplicate domination by capping samples per episode per class
- keep one fixed manifest for checkpoint-vs-checkpoint comparisons
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tyro
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from openpi.evaluation import subtask_obs_utils as _obs
from openpi.training import config as _config
from openpi.training import subtask_eval as _subtask_eval


@dataclass(frozen=True)
class EvalRow:
    repo_id: str
    frame_index: int
    episode_index: int
    class_id: int
    bottle_state: str
    subtask: str


@dataclass
class Args:
    config_name: str = "twist_only_lora"
    output: Path = Path("assets/twist_nine_class_eval_manifest.json")
    max_per_class: int = 300
    max_per_episode_per_class: int = 5
    seed: int = 0
    exclude_repo_ids: tuple[str, ...] = ("lyl472324464/2025-12-10-twist-one-bottle",)


def _open_repo(repo_id: str) -> LeRobotDataset:
    return LeRobotDataset(
        repo_id,
        revision="main",
        force_cache_sync=False,
        download_videos=False,
        delta_timestamps=None,
    )


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    train_cfg = _config.get_config(args.config_name)
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    repo_ids = [r for r in (data_cfg.repo_ids or []) if r not in set(args.exclude_repo_ids)]
    if not repo_ids:
        raise SystemExit("No repo_ids left after exclude_repo_ids filtering.")

    canonical = _subtask_eval.resolve_canonical_pairs(train_cfg)
    n_classes = len(canonical)
    rng = np.random.default_rng(args.seed)

    pools: dict[int, dict[tuple[str, int], list[EvalRow]]] = {
        c: defaultdict(list) for c in range(n_classes)
    }
    scanned_rows = 0

    for repo_i, repo_id in enumerate(repo_ids, start=1):
        logging.info("Scanning repo %d/%d: %s", repo_i, len(repo_ids), repo_id)
        ds = _open_repo(repo_id)
        subtask_col = ds.hf_dataset.select_columns(["subtask", "episode_index"])
        for idx in range(len(ds)):
            row = subtask_col[int(idx)]
            parsed = _obs.parse_json_bottle_state_subtask(_obs.subtask_cell_to_str(row["subtask"]))
            if parsed is None:
                continue
            bs, st = parsed
            cid = _subtask_eval.class_id_for_pair(bs, st, canonical)
            if cid is None:
                continue
            ep = int(row["episode_index"])
            pools[cid][(repo_id, ep)].append(
                EvalRow(
                    repo_id=repo_id,
                    frame_index=int(idx),
                    episode_index=ep,
                    class_id=cid,
                    bottle_state=bs,
                    subtask=st,
                )
            )
            scanned_rows += 1

    selected: list[EvalRow] = []
    per_class_counts: dict[int, int] = {c: 0 for c in range(n_classes)}
    per_class_episode_counts: dict[int, int] = {c: 0 for c in range(n_classes)}

    for cid in range(n_classes):
        episode_keys = list(pools[cid].keys())
        rng.shuffle(episode_keys)
        chosen: list[EvalRow] = []
        used_eps = 0
        for ep_key in episode_keys:
            rows = pools[cid][ep_key]
            if not rows:
                continue
            order = rng.permutation(len(rows))
            take_n = min(len(rows), args.max_per_episode_per_class)
            for j in order[:take_n]:
                chosen.append(rows[int(j)])
                if len(chosen) >= args.max_per_class:
                    break
            used_eps += 1
            if len(chosen) >= args.max_per_class:
                break
        selected.extend(chosen)
        per_class_counts[cid] = len(chosen)
        per_class_episode_counts[cid] = used_eps

    selected.sort(key=lambda r: (r.class_id, r.repo_id, r.episode_index, r.frame_index))

    payload = {
        "config_name": args.config_name,
        "seed": args.seed,
        "max_per_class": args.max_per_class,
        "max_per_episode_per_class": args.max_per_episode_per_class,
        "exclude_repo_ids": list(args.exclude_repo_ids),
        "canonical_pairs": [
            {"class_id": i, "bottle_state": bs, "subtask": st} for i, (bs, st) in enumerate(canonical)
        ],
        "summary": {
            "repo_ids": repo_ids,
            "num_repos": len(repo_ids),
            "scanned_canonical_rows": scanned_rows,
            "selected_rows": len(selected),
            "per_class_counts": per_class_counts,
            "per_class_episode_counts": per_class_episode_counts,
        },
        "rows": [asdict(r) for r in selected],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"\nSaved manifest to: {args.output}")


if __name__ == "__main__":
    main(tyro.cli(Args))
