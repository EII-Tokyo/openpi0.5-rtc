#!/usr/bin/env python3
"""Convert the 2026-07-01 online replay batch to VLA-same-forward 2048 z_rl.

This intentionally follows the strict TD3 "B group" path:
1. run the cam4 VLA forward on the replay anchor observation;
2. extract low/right image token blocks from that same forward pass;
3. encode those token blocks with the lower+right 4-layer RLToken autoencoder.

It does not use the sidecar image encoder path.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from scripts.compare_vla_same_forward_vs_sidecar_tokens import (
    DEFAULT_CAM4_CHECKPOINT,
    DEFAULT_CAM4_CONFIG,
    DEFAULT_SIDECAR_CHECKPOINT,
    DEFAULT_SIDECAR_CONFIG,
)
from scripts.prepare_strict_td3_z_ablation_replay import (
    REPLAY_KEYS,
    _encode_blocks,
    extract_vla_token_blocks,
    stratified_split,
)
from scripts.reencode_clean_no_actor_z_rl import find_rollout_dir, load_manifest_from_npz


DEFAULT_SOURCE_ROOT = Path(
    "/home/eii/data/openpi0.5-rtc-reward-learning/replay/"
    "rlt_key_regions/twist_off_the_bottle_cap/2026-07-01"
)
DEFAULT_ROLLOUT_ROOT = Path("/home/eii/data/openpi0.5-rtc-reward-learning/rollouts/key_regions")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/eii/data/openpi0.5-rtc-reward-learning/replay/"
    "rlt_key_regions_vla_same_forward_z2048_4layer/twist_off_the_bottle_cap/2026-07-01"
)
PROMPT = "Twist off the bottle cap."


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _candidate_source_path(entry: dict[str, Any], source_root: Path) -> Path | None:
    key_region_id = entry.get("key_region_id")
    raw_path = entry.get("shard_path") or entry.get("path") or entry.get("local_path") or entry.get("replay_path")
    if raw_path:
        path = Path(str(raw_path))
        if path.exists():
            return path
        by_name = source_root / "shards" / path.name
        if by_name.exists():
            return by_name
    if key_region_id:
        by_key = source_root / "shards" / f"key_region_{key_region_id}.npz"
        if by_key.exists():
            return by_key
    return None


def _reward(path: Path, manifest: dict[str, Any]) -> int:
    if "reward" in manifest:
        return int(float(manifest["reward"]) > 0.0)
    with np.load(path, allow_pickle=False) as data:
        reward_seq = np.asarray(data["reward_seq"], dtype=np.float32)
        done = np.asarray(data["done"], dtype=bool)
    terminal = float(reward_seq[done].sum()) if np.any(done) else float(reward_seq.sum())
    return int(terminal > 0.0)


def discover_20260701_candidates(
    *,
    source_root: Path,
    rollout_root: Path,
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    manifest_path = source_root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    candidates: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    seen: set[Path] = set()
    for entry in _read_jsonl(manifest_path):
        source = _candidate_source_path(entry, source_root)
        if source is None:
            skipped["missing_source_shard"] = skipped.get("missing_source_shard", 0) + 1
            continue
        source = source.resolve()
        if source in seen:
            skipped["duplicate_source_shard"] = skipped.get("duplicate_source_shard", 0) + 1
            continue
        seen.add(source)
        try:
            manifest = load_manifest_from_npz(source)
            with np.load(source, allow_pickle=False) as data:
                rows = int(data["z_rl"].shape[0])
                z_dim = int(data["z_rl"].shape[-1])
                next_z_dim = int(data["next_z_rl"].shape[-1])
                if z_dim != 512 or next_z_dim != 512:
                    raise ValueError(f"expected old z_dim=512, got z_rl={z_dim}, next_z_rl={next_z_dim}")
                if int(data["action"].shape[1]) < 10:
                    raise ValueError(f"expected action horizon >=10, got {data['action'].shape[1]}")
            rollout_dir = find_rollout_dir(rollout_root, manifest)
            required = ("episode.hdf5", "cam_high.mp4", "cam_low.mp4", "cam_left_wrist.mp4", "cam_right_wrist.mp4")
            missing = [name for name in required if not (rollout_dir / name).exists()]
            if missing:
                raise FileNotFoundError(f"missing rollout files: {missing}")
            key_region_id = str(manifest.get("key_region_id") or entry.get("key_region_id") or source.stem.removeprefix("key_region_"))
            candidates.append(
                {
                    "source_shard_path": str(source),
                    "rollout_dir": str(rollout_dir),
                    "key_region_id": key_region_id,
                    "reward": _reward(source, manifest),
                    "rows": rows,
                    "z_dim": z_dim,
                    "phase": str(manifest.get("phase") or entry.get("phase") or ""),
                    "task": str(manifest.get("task") or entry.get("task") or ""),
                }
            )
            if limit is not None and len(candidates) >= limit:
                break
        except Exception as exc:
            skipped[type(exc).__name__] = skipped.get(type(exc).__name__, 0) + 1
            logging.warning("skip %s: %s", source, exc)
    return candidates, skipped


def encode_vla_replay_to_output_root(
    candidates: list[dict[str, Any]],
    split: dict[str, Any],
    output_root: Path,
    work_dir: Path,
    *,
    overwrite: bool,
    batch_size: int,
) -> None:
    from openpi.policies import policy_config
    from openpi.training import config as train_config

    shard_dir = output_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    token_dir = work_dir / "vla_token_blocks"
    cfg = train_config.get_config(DEFAULT_SIDECAR_CONFIG)
    policy = policy_config.create_trained_policy(cfg, DEFAULT_SIDECAR_CHECKPOINT, default_prompt=PROMPT)
    autoencoder = policy._model.rl_token_autoencoder  # noqa: SLF001
    manifests = {"train": [], "holdout": [], "all": []}
    holdout = set(split["holdout_key_region_ids"])

    for index, row in enumerate(candidates, start=1):
        source = Path(row["source_shard_path"])
        out = shard_dir / source.name
        if out.exists() and not overwrite:
            logging.info("skip existing converted replay %s", out)
        else:
            token_path = token_dir / f"key_region_{row['key_region_id']}.npz"
            with np.load(token_path, allow_pickle=False) as token_data:
                low = np.asarray(token_data["low_tokens"], dtype=np.float32)
                right = np.asarray(token_data["right_tokens"], dtype=np.float32)
                next_low = np.asarray(token_data["next_low_tokens"], dtype=np.float32)
                next_right = np.asarray(token_data["next_right_tokens"], dtype=np.float32)
            z_rl = _encode_blocks(autoencoder, low, right, batch_size=batch_size)
            next_z_rl = _encode_blocks(autoencoder, next_low, next_right, batch_size=batch_size)
            with np.load(source, allow_pickle=False) as data:
                arrays = {key: np.asarray(data[key]) for key in REPLAY_KEYS if key in data}
                manifest = load_manifest_from_npz(source)
            if z_rl.shape[0] != arrays["z_rl"].shape[0]:
                raise ValueError(f"{source} token rows {z_rl.shape[0]} != replay rows {arrays['z_rl'].shape[0]}")
            previous_shapes = {key: list(value.shape) for key, value in arrays.items()}
            arrays["z_rl"] = z_rl.astype(np.float32)
            arrays["next_z_rl"] = next_z_rl.astype(np.float32)
            replay_array_shapes = dict(manifest.get("replay_array_shapes") or {})
            replay_array_shapes["z_rl"] = list(arrays["z_rl"].shape)
            replay_array_shapes["next_z_rl"] = list(arrays["next_z_rl"].shape)
            manifest.update(
                {
                    "z_rl_source": "vla_same_forward_low_right_tokens_then_lower_right_rl_token_encoder",
                    "z_rl_dim": int(z_rl.shape[-1]),
                    "previous_replay_array_shapes": previous_shapes,
                    "replay_array_shapes": replay_array_shapes,
                    "source_512_shard_path": str(source),
                    "vla_base_config": DEFAULT_CAM4_CONFIG,
                    "vla_base_checkpoint": str(DEFAULT_CAM4_CHECKPOINT),
                    "rl_token_encoder_config": DEFAULT_SIDECAR_CONFIG,
                    "rl_token_encoder_checkpoint": str(DEFAULT_SIDECAR_CHECKPOINT),
                    "replay_state_grain": "paper_subsampled_anchor",
                    "requires_offline_reencode": False,
                    "formal_replay_state_grain": "paper_subsampled_anchor",
                    "formal_replay_ready": True,
                    "train_eligible": True,
                    "subsampled_transition_semantics": "x_i_action_i_to_i_plus_c_next_x_i_plus_c",
                    "conversion_script": Path(__file__).name,
                }
            )
            arrays["manifest"] = np.asarray(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
            tmp_path = out.with_suffix(out.suffix + ".tmp")
            with tmp_path.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            tmp_path.replace(out)
            logging.info("encoded converted replay %d/%d %s rows=%d", index, len(candidates), row["key_region_id"], z_rl.shape[0])

        payload = {
            "shard_path": str(out.resolve()),
            "batch": "20260701_online_vla_same_forward_z2048",
            "key_region_id": row["key_region_id"],
            "reward": row["reward"],
            "num_replay_transitions": row["rows"],
            "z_rl_dim": 2048,
            "z_rl_source": "vla_same_forward_low_right_tokens_then_lower_right_rl_token_encoder",
            "source_512_shard_path": row["source_shard_path"],
        }
        split_name = "holdout" if row["key_region_id"] in holdout else "train"
        manifests[split_name].append(payload)
        manifests["all"].append(payload)

    for name, rows in manifests.items():
        (output_root / f"{name}_manifest.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in rows),
            encoding="utf-8",
        )


def write_audit(
    *,
    candidates: list[dict[str, Any]],
    skipped: dict[str, int],
    split: dict[str, Any],
    source_root: Path,
    rollout_root: Path,
    output_root: Path,
    work_dir: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "source_root": str(source_root),
        "rollout_root": str(rollout_root),
        "output_root": str(output_root),
        "work_dir": str(work_dir),
        "num_candidates": len(candidates),
        "num_transitions": int(sum(int(row["rows"]) for row in candidates)),
        "num_success_episodes": int(sum(int(row["reward"]) == 1 for row in candidates)),
        "num_failure_episodes": int(sum(int(row["reward"]) == 0 for row in candidates)),
        "skipped": skipped,
        "split": split,
        "method": "VLA same-forward low/right tokens -> lower+right 4-layer RLToken autoencoder",
        "vla_base_config": DEFAULT_CAM4_CONFIG,
        "vla_base_checkpoint": str(DEFAULT_CAM4_CHECKPOINT),
        "rl_token_encoder_config": DEFAULT_SIDECAR_CONFIG,
        "rl_token_encoder_checkpoint": str(DEFAULT_SIDECAR_CHECKPOINT),
        "candidates": candidates,
    }
    (output_root / "conversion_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if candidates:
        with (output_root / "conversion_candidates.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(candidates[0].keys()))
            writer.writeheader()
            writer.writerows(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "_work")
    parser.add_argument("--phase", choices=("audit", "extract-vla", "encode-vla", "all"), default="audit")
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--encode-batch-size", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    candidates, skipped = discover_20260701_candidates(
        source_root=args.source_root,
        rollout_root=args.rollout_root,
        limit=args.limit,
    )
    if not candidates:
        raise RuntimeError(f"No candidates discovered. skipped={skipped}")
    split = stratified_split(candidates, holdout_ratio=args.holdout_ratio, seed=args.seed)
    write_audit(
        candidates=candidates,
        skipped=skipped,
        split=split,
        source_root=args.source_root,
        rollout_root=args.rollout_root,
        output_root=args.output_root,
        work_dir=args.work_dir,
    )
    if args.phase in {"extract-vla", "all"}:
        extract_vla_token_blocks(candidates, args.work_dir, overwrite=args.overwrite)
    if args.phase in {"encode-vla", "all"}:
        encode_vla_replay_to_output_root(
            candidates,
            split,
            args.output_root,
            args.work_dir,
            overwrite=args.overwrite,
            batch_size=args.encode_batch_size,
        )
    print(
        json.dumps(
            {
                "source_root": str(args.source_root),
                "output_root": str(args.output_root),
                "work_dir": str(args.work_dir),
                "phase": args.phase,
                "candidates": len(candidates),
                "transitions": int(sum(int(row["rows"]) for row in candidates)),
                "skipped": skipped,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
