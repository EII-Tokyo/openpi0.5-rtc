#!/usr/bin/env python3
"""Build unified same-forward paper-anchor replay manifests.

The final all-data manifest is intended for unified training. The eval-split
manifests keep source-specific holdouts available for diagnostics, but those
holdouts are deliberately also present in the all-data training manifest.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import tyro

DEFAULT_MANIFEST_DIR = Path("local_rlt_manifests/unified_same_forward_20260706")
DEFAULT_ORIGINAL_TRAIN = Path(
    "local_rlt_manifests/bootstrap146_vla_same_forward_exact_split_20260706/"
    "train_bootstrap117_vla_same_forward_exact_split.jsonl"
)
DEFAULT_ORIGINAL_HOLDOUT = Path(
    "local_rlt_manifests/bootstrap146_vla_same_forward_exact_split_20260706/"
    "holdout_bootstrap29_vla_same_forward_exact_split.jsonl"
)
DEFAULT_BASE142 = Path("local_rlt_manifests/iterative_same_forward_20260706/base142_20260706_same_forward.jsonl")
DEFAULT_ACTOR93 = Path("local_rlt_manifests/iterative_same_forward_20260706/actor93_20260706_same_forward.jsonl")

FORMAL_GRAIN = "paper_subsampled_anchor"
SAME_FORWARD_SOURCE = "vla_same_forward_low_right_tokens_then_lower_right_rl_token_encoder"


@dataclasses.dataclass
class Args:
    original_train_manifest: Path = DEFAULT_ORIGINAL_TRAIN
    original_holdout_manifest: Path = DEFAULT_ORIGINAL_HOLDOUT
    base142_manifest: Path = DEFAULT_BASE142
    actor93_manifest: Path = DEFAULT_ACTOR93
    output_dir: Path = DEFAULT_MANIFEST_DIR
    holdout_ratio: float = 0.2
    seed: int = 20260706


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _read_npz_manifest(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "manifest" not in data:
        return {}
    raw = data["manifest"].item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    raise TypeError(f"Unsupported npz manifest payload: {type(raw)!r}")


def _key_region_id(row: dict[str, Any], embedded: dict[str, Any], shard_path: Path) -> str:
    raw = row.get("key_region_id") or embedded.get("key_region_id") or shard_path.stem.removeprefix("key_region_")
    return str(raw).removeprefix("key_region_")


def _reward(row: dict[str, Any], embedded: dict[str, Any]) -> int:
    return int(float(row.get("reward", embedded.get("reward", 0))) > 0.0)


def _stable_sort_key(row: dict[str, Any], seed: int) -> str:
    key = f"{seed}:{row['source_group']}:{row['reward']}:{row['key_region_id']}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def normalize_manifest_entry(row: dict[str, Any], *, source_group: str, expected_z_dim: int = 2048) -> dict[str, Any]:
    shard_path = Path(row["shard_path"]).expanduser()
    if not shard_path.exists():
        raise FileNotFoundError(shard_path)

    with np.load(shard_path, allow_pickle=False) as data:
        embedded = _read_npz_manifest(data)
        if "z_rl" not in data or "next_z_rl" not in data or "action" not in data:
            raise KeyError(f"{shard_path} is missing z_rl/next_z_rl/action arrays")
        z_dim = int(data["z_rl"].shape[-1])
        next_z_dim = int(data["next_z_rl"].shape[-1])
        num_transitions = int(data["action"].shape[0])
        action_horizon = int(data["action"].shape[1])

    grain = row.get("replay_state_grain") or embedded.get("replay_state_grain") or embedded.get("formal_replay_state_grain")
    if grain != FORMAL_GRAIN:
        raise ValueError(f"{shard_path} has replay_state_grain={grain!r}, expected {FORMAL_GRAIN!r}")
    if z_dim != expected_z_dim or next_z_dim != expected_z_dim:
        raise ValueError(f"{shard_path} has z dims z_rl={z_dim} next_z_rl={next_z_dim}, expected {expected_z_dim}")

    z_rl_source = row.get("z_rl_source") or embedded.get("z_rl_source")
    if z_rl_source != SAME_FORWARD_SOURCE:
        raise ValueError(f"{shard_path} has z_rl_source={z_rl_source!r}, expected {SAME_FORWARD_SOURCE!r}")

    normalized = dict(row)
    normalized.update(
        {
            "key_region_id": _key_region_id(row, embedded, shard_path),
            "reward": _reward(row, embedded),
            "shard_path": str(shard_path),
            "source_group": source_group,
            "replay_state_grain": FORMAL_GRAIN,
            "z_dim": z_dim,
            "z_rl_source": z_rl_source,
            "num_transitions": num_transitions,
            "action_horizon": action_horizon,
            "formal_replay_ready": True,
        }
    )
    return normalized


def load_source_manifest(path: Path, *, source_group: str) -> list[dict[str, Any]]:
    return [normalize_manifest_entry(row, source_group=source_group) for row in read_jsonl(path)]


def stratified_holdout_split(
    entries: list[dict[str, Any]],
    *,
    holdout_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be in [0.0, 1.0)")

    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for reward in (0, 1):
        group = [row for row in entries if int(row["reward"]) == reward]
        group = sorted(group, key=lambda row: _stable_sort_key(row, seed))
        if not group:
            continue
        holdout_count = math.ceil(len(group) * holdout_ratio) if holdout_ratio > 0 else 0
        if len(group) > 1 and holdout_count == 0:
            holdout_count = 1
        holdout.extend(group[:holdout_count])
        train.extend(group[holdout_count:])
    return sorted(train, key=_manifest_sort_key), sorted(holdout, key=_manifest_sort_key)


def _manifest_sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (str(row["source_group"]), int(row["reward"]), str(row["key_region_id"]))


def _dedupe_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for row in entries:
        key = str(row["key_region_id"])
        if key in deduped:
            previous = deduped[key]
            if previous["shard_path"] != row["shard_path"]:
                raise ValueError(f"Duplicate key_region_id with different shards: {key}")
            continue
        deduped[key] = row
    return sorted(deduped.values(), key=_manifest_sort_key)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, dict[str, int]] = {}
    for row in rows:
        source = str(row["source_group"])
        item = by_source.setdefault(source, {"rows": 0, "success": 0, "failure": 0, "transitions": 0})
        item["rows"] += 1
        item["success"] += int(row["reward"] == 1)
        item["failure"] += int(row["reward"] == 0)
        item["transitions"] += int(row.get("num_transitions", 0))
    return {
        "rows": len(rows),
        "success": sum(int(row["reward"] == 1) for row in rows),
        "failure": sum(int(row["reward"] == 0) for row in rows),
        "transitions": sum(int(row.get("num_transitions", 0)) for row in rows),
        "by_source": by_source,
    }


def _assert_disjoint(left: list[dict[str, Any]], right: list[dict[str, Any]], *, label: str) -> None:
    overlap = {row["key_region_id"] for row in left} & {row["key_region_id"] for row in right}
    if overlap:
        sample = sorted(overlap)[:5]
        raise ValueError(f"{label} key_region_id overlap count={len(overlap)} sample={sample}")


def build_unified_manifests(
    *,
    original_train_manifest: Path,
    original_holdout_manifest: Path,
    base142_manifest: Path,
    actor93_manifest: Path,
    output_dir: Path,
    holdout_ratio: float,
    seed: int,
) -> dict[str, Path]:
    original_train = load_source_manifest(original_train_manifest, source_group="original_train")
    original_holdout = load_source_manifest(original_holdout_manifest, source_group="original_holdout")
    base142 = load_source_manifest(base142_manifest, source_group="base142")
    actor93 = load_source_manifest(actor93_manifest, source_group="actor93")

    base_train, base_holdout = stratified_holdout_split(base142, holdout_ratio=holdout_ratio, seed=seed)
    actor_train, actor_holdout = stratified_holdout_split(actor93, holdout_ratio=holdout_ratio, seed=seed)

    all_entries = _dedupe_entries([*original_train, *original_holdout, *base142, *actor93])
    eval_train = _dedupe_entries([*original_train, *base_train, *actor_train])
    combined_holdout = _dedupe_entries([*original_holdout, *base_holdout, *actor_holdout])
    _assert_disjoint(eval_train, combined_holdout, label="eval_train vs combined_holdout")

    paths = {
        "inventory": output_dir / "inventory_all_same_forward_paper_anchor.jsonl",
        "train_all": output_dir / "train_all_same_forward_paper_anchor.jsonl",
        "train_eval": output_dir / "train_eval_split_same_forward_paper_anchor.jsonl",
        "holdout_original": output_dir / "holdout_original_bootstrap29.jsonl",
        "holdout_base142": output_dir / "holdout_base142_stratified.jsonl",
        "holdout_actor93": output_dir / "holdout_actor93_stratified.jsonl",
        "holdout_combined": output_dir / "holdout_combined_stratified.jsonl",
        "summary": output_dir / "summary.json",
    }

    write_jsonl(paths["inventory"], all_entries)
    write_jsonl(paths["train_all"], all_entries)
    write_jsonl(paths["train_eval"], eval_train)
    write_jsonl(paths["holdout_original"], sorted(original_holdout, key=_manifest_sort_key))
    write_jsonl(paths["holdout_base142"], base_holdout)
    write_jsonl(paths["holdout_actor93"], actor_holdout)
    write_jsonl(paths["holdout_combined"], combined_holdout)

    summary = {
        "schema": "unified_same_forward_paper_anchor_v1",
        "holdout_ratio": holdout_ratio,
        "seed": seed,
        "inputs": {
            "original_train_manifest": str(original_train_manifest),
            "original_holdout_manifest": str(original_holdout_manifest),
            "base142_manifest": str(base142_manifest),
            "actor93_manifest": str(actor93_manifest),
        },
        "outputs": {key: str(path) for key, path in paths.items() if key != "summary"},
        "splits": {
            "inventory_all": _summarize(all_entries),
            "train_all": _summarize(all_entries),
            "train_eval": _summarize(eval_train),
            "holdout_original": _summarize(original_holdout),
            "holdout_base142": _summarize(base_holdout),
            "holdout_actor93": _summarize(actor_holdout),
            "holdout_combined": _summarize(combined_holdout),
        },
        "notes": [
            "train_all intentionally includes original/base142/actor93 holdout rows for final all-data training.",
            "train_eval excludes holdout_* rows and is only for independent diagnostics.",
        ],
    }
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def main(args: Args) -> None:
    paths = build_unified_manifests(
        original_train_manifest=args.original_train_manifest,
        original_holdout_manifest=args.original_holdout_manifest,
        base142_manifest=args.base142_manifest,
        actor93_manifest=args.actor93_manifest,
        output_dir=args.output_dir,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
