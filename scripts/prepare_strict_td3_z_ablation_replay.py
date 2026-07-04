#!/usr/bin/env python3
"""Prepare full-replay A/B data for sidecar z_rl vs VLA-token z_rl TD3 critic tests."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from scripts.compare_vla_same_forward_vs_sidecar_tokens import (
    DEFAULT_CAM4_CHECKPOINT,
    DEFAULT_CAM4_CONFIG,
    DEFAULT_SIDECAR_CHECKPOINT,
    DEFAULT_SIDECAR_CONFIG,
    PrefixFeatureExtractor,
    _build_lower_right_prefix_from_blocks,
    _load_observation,
)
from scripts.reencode_clean_no_actor_z_rl import (
    _VideoFrameReader,
    _load_qpos,
    compute_replay_frame_indices,
    find_rollout_dir,
    load_manifest_from_npz,
)


DEFAULT_MANIFESTS = (
    Path("local_rlt_manifests/paper_anchor_bootstrap_expert_20260703/holdout_bootstrap29.jsonl"),
    Path("local_rlt_manifests/paper_anchor_bootstrap_expert_20260703/train_bootstrap117_expert59.jsonl"),
)
DEFAULT_ROLLOUT_ROOT = Path("/home/eii/data/openpi0.5-rtc-reward-learning/rollouts/key_regions")
DEFAULT_OUTPUT_DIR = Path("local_rlt_runs/strict_td3_z_ablation_20260704")
REPLAY_KEYS = (
    "z_rl",
    "proprio",
    "action",
    "reference_action",
    "reward_seq",
    "next_z_rl",
    "next_proprio",
    "next_reference_action",
    "done",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _entry_path(row: dict[str, Any]) -> Path | None:
    for key in ("shard_path", "path", "local_path", "replay_path"):
        if row.get(key):
            return Path(str(row[key]))
    return None


def _is_robot_shard(path: Path, manifest: dict[str, Any]) -> bool:
    source = str(manifest.get("source_shard_path", ""))
    text = f"{path} {source}"
    return "key_region_" in text and "human_expert" not in text and "lerobot" not in text


def _reward(path: Path, manifest: dict[str, Any]) -> int:
    if "reward" in manifest:
        return int(float(manifest["reward"]) > 0.0)
    with np.load(path, allow_pickle=False) as data:
        reward_seq = np.asarray(data["reward_seq"], dtype=np.float32)
        done = np.asarray(data["done"], dtype=bool)
        terminal = float(reward_seq[done].sum()) if np.any(done) else float(reward_seq.sum())
    return int(terminal > 0.0)


def _validate_replay_shape(path: Path) -> tuple[int, int]:
    with np.load(path, allow_pickle=False) as data:
        z_dim = int(data["z_rl"].shape[-1])
        rows = int(data["z_rl"].shape[0])
        if z_dim != 2048:
            raise ValueError(f"expected z_dim=2048, got {z_dim}")
        if int(data["action"].shape[1]) < 10:
            raise ValueError(f"expected action horizon >=10, got {data['action'].shape[1]}")
        if data["next_z_rl"].shape != data["z_rl"].shape:
            raise ValueError("next_z_rl shape must match z_rl")
    return rows, z_dim


def discover_eligible_shards(manifests: tuple[Path, ...], rollout_root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    seen: set[str] = set()
    for manifest_path in manifests:
        for entry in _read_jsonl(manifest_path):
            path = _entry_path(entry)
            if path is None:
                skipped["missing_manifest_path"] = skipped.get("missing_manifest_path", 0) + 1
                continue
            if str(path) in seen:
                skipped["duplicate_path"] = skipped.get("duplicate_path", 0) + 1
                continue
            seen.add(str(path))
            try:
                if not path.exists():
                    raise FileNotFoundError(path)
                manifest = load_manifest_from_npz(path)
                if not _is_robot_shard(path, manifest):
                    skipped["not_robot_key_region"] = skipped.get("not_robot_key_region", 0) + 1
                    continue
                rollout_dir = find_rollout_dir(rollout_root, manifest)
                required = ("episode.hdf5", "cam_high.mp4", "cam_low.mp4", "cam_left_wrist.mp4", "cam_right_wrist.mp4")
                missing = [name for name in required if not (rollout_dir / name).exists()]
                if missing:
                    raise FileNotFoundError(f"missing rollout files: {missing}")
                rows, z_dim = _validate_replay_shape(path)
                reward = _reward(path, manifest)
                candidates.append(
                    {
                        "source_shard_path": str(path),
                        "rollout_dir": str(rollout_dir),
                        "key_region_id": str(manifest.get("key_region_id") or path.stem.replace("key_region_", "").split(".")[0]),
                        "reward": reward,
                        "rows": rows,
                        "z_dim": z_dim,
                    }
                )
            except Exception as exc:
                skipped[type(exc).__name__] = skipped.get(type(exc).__name__, 0) + 1
                logging.warning("skip %s: %s", path, exc)
    return candidates, skipped


def stratified_split(candidates: list[dict[str, Any]], *, holdout_ratio: float, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    holdout: set[str] = set()
    for reward in (0, 1):
        keys = [row["key_region_id"] for row in candidates if int(row["reward"]) == reward]
        rng.shuffle(keys)
        if len(keys) >= 2:
            count = max(1, round(len(keys) * holdout_ratio))
            count = min(count, len(keys) - 1)
            holdout.update(keys[:count])
    train = [row["key_region_id"] for row in candidates if row["key_region_id"] not in holdout]
    holdout_list = [row["key_region_id"] for row in candidates if row["key_region_id"] in holdout]
    return {
        "seed": seed,
        "holdout_ratio": holdout_ratio,
        "train_key_region_ids": train,
        "holdout_key_region_ids": holdout_list,
    }


def build_sidecar_replay(candidates: list[dict[str, Any]], split: dict[str, Any], output_dir: Path) -> None:
    sidecar_root = output_dir / "replay" / "sidecar_z"
    shard_dir = sidecar_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifests = {"train": [], "holdout": [], "all": []}
    holdout = set(split["holdout_key_region_ids"])
    for row in candidates:
        source = Path(row["source_shard_path"])
        dest = shard_dir / source.name
        if not dest.exists():
            shutil.copy2(source, dest)
        payload = {"shard_path": str(dest.resolve()), "batch": "strict_td3_sidecar_z", "key_region_id": row["key_region_id"], "reward": row["reward"]}
        split_name = "holdout" if row["key_region_id"] in holdout else "train"
        manifests[split_name].append(payload)
        manifests["all"].append(payload)
    for name, rows in manifests.items():
        (sidecar_root / f"{name}_manifest.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )


def extract_vla_token_blocks(candidates: list[dict[str, Any]], output_dir: Path, *, overwrite: bool) -> None:
    token_dir = output_dir / "vla_token_blocks"
    token_dir.mkdir(parents=True, exist_ok=True)
    extractor = PrefixFeatureExtractor(config_name=DEFAULT_CAM4_CONFIG, checkpoint=DEFAULT_CAM4_CHECKPOINT, prompt="Twist off the bottle cap.")
    for index, row in enumerate(candidates, start=1):
        out = token_dir / f"key_region_{row['key_region_id']}.npz"
        if out.exists() and not overwrite:
            logging.info("skip existing VLA token block %s", out)
            continue
        source = Path(row["source_shard_path"])
        rollout_dir = Path(row["rollout_dir"])
        manifest = load_manifest_from_npz(source)
        qpos = _load_qpos(rollout_dir / "episode.hdf5")
        clean_rows = int(row["rows"])
        current_frames, next_frames = compute_replay_frame_indices(manifest, clean_rows=clean_rows, episode_frames=len(qpos))
        reader = _VideoFrameReader(rollout_dir, convert_bgr_to_rgb=False)
        low_tokens: list[np.ndarray] = []
        right_tokens: list[np.ndarray] = []
        next_low_tokens: list[np.ndarray] = []
        next_right_tokens: list[np.ndarray] = []
        try:
            for frame, next_frame in zip(current_frames, next_frames, strict=True):
                for target_frame, low_list, right_list in (
                    (int(frame), low_tokens, right_tokens),
                    (int(next_frame), next_low_tokens, next_right_tokens),
                ):
                    obs = {
                        "images": reader.read_all(target_frame),
                        "state": np.asarray(qpos[target_frame], dtype=np.float32),
                        "prompt": "Twist off the bottle cap.",
                    }
                    result = extractor.extract(obs)
                    low_list.append(result["token_blocks"]["base_1_rgb"])
                    right_list.append(result["token_blocks"]["right_wrist_0_rgb"])
        finally:
            reader.close()
        np.savez_compressed(
            out,
            low_tokens=np.stack(low_tokens).astype(np.float16),
            right_tokens=np.stack(right_tokens).astype(np.float16),
            next_low_tokens=np.stack(next_low_tokens).astype(np.float16),
            next_right_tokens=np.stack(next_right_tokens).astype(np.float16),
            current_frames=current_frames,
            next_frames=next_frames,
            source_shard_path=str(source),
            key_region_id=str(row["key_region_id"]),
        )
        logging.info("extracted VLA token blocks %d/%d %s rows=%d", index, len(candidates), row["key_region_id"], clean_rows)


def encode_vla_replay(candidates: list[dict[str, Any]], split: dict[str, Any], output_dir: Path, *, overwrite: bool, batch_size: int) -> None:
    from openpi.policies import policy_config
    from openpi.training import config as train_config

    vla_root = output_dir / "replay" / "vla_token_z"
    shard_dir = vla_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    token_dir = output_dir / "vla_token_blocks"
    cfg = train_config.get_config(DEFAULT_SIDECAR_CONFIG)
    policy = policy_config.create_trained_policy(cfg, DEFAULT_SIDECAR_CHECKPOINT, default_prompt="Twist off the bottle cap.")
    autoencoder = policy._model.rl_token_autoencoder  # noqa: SLF001
    manifests = {"train": [], "holdout": [], "all": []}
    holdout = set(split["holdout_key_region_ids"])
    for index, row in enumerate(candidates, start=1):
        source = Path(row["source_shard_path"])
        out = shard_dir / source.name
        if out.exists() and not overwrite:
            logging.info("skip existing VLA-token replay %s", out)
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
            arrays["z_rl"] = z_rl.astype(np.float32)
            arrays["next_z_rl"] = next_z_rl.astype(np.float32)
            manifest.update(
                {
                    "z_rl_source": "vla_same_forward_low_right_tokens_then_lower_right_rl_token_encoder",
                    "z_rl_dim": int(z_rl.shape[-1]),
                    "source_sidecar_shard_path": str(source),
                    "vla_base_checkpoint": str(DEFAULT_CAM4_CHECKPOINT),
                    "rl_token_encoder_checkpoint": str(DEFAULT_SIDECAR_CHECKPOINT),
                    "replay_state_grain": "paper_subsampled_anchor",
                }
            )
            arrays["manifest"] = np.asarray(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
            np.savez_compressed(out, **arrays)
            logging.info("encoded VLA-token replay %d/%d %s rows=%d", index, len(candidates), row["key_region_id"], z_rl.shape[0])
        payload = {"shard_path": str(out.resolve()), "batch": "strict_td3_vla_token_z", "key_region_id": row["key_region_id"], "reward": row["reward"]}
        split_name = "holdout" if row["key_region_id"] in holdout else "train"
        manifests[split_name].append(payload)
        manifests["all"].append(payload)
    for name, rows in manifests.items():
        (vla_root / f"{name}_manifest.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )


def _encode_blocks(autoencoder, low: np.ndarray, right: np.ndarray, *, batch_size: int) -> np.ndarray:
    out: list[np.ndarray] = []
    for start in range(0, low.shape[0], batch_size):
        end = min(start + batch_size, low.shape[0])
        prefix, mask = _build_lower_right_prefix_from_blocks(low[start:end], right[start:end])
        z = autoencoder.encode(jax.lax.stop_gradient(jnp.asarray(prefix)), jnp.asarray(mask))
        out.append(np.asarray(jax.device_get(z), dtype=np.float32))
    return np.concatenate(out, axis=0)


def write_audit(candidates: list[dict[str, Any]], skipped: dict[str, int], split: dict[str, Any], output_dir: Path) -> None:
    audit = {
        "num_candidates": len(candidates),
        "num_transitions": int(sum(int(row["rows"]) for row in candidates)),
        "num_success_episodes": int(sum(int(row["reward"]) == 1 for row in candidates)),
        "num_failure_episodes": int(sum(int(row["reward"]) == 0 for row in candidates)),
        "num_train_episodes": len(split["train_key_region_ids"]),
        "num_holdout_episodes": len(split["holdout_key_region_ids"]),
        "skipped": skipped,
        "candidates": candidates,
        "split": split,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "strict_td3_dataset_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "strict_td3_dataset_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidates[0].keys()) if candidates else ["source_shard_path"])
        writer.writeheader()
        writer.writerows(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--manifest", action="append", type=Path, dest="manifests")
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", choices=("audit", "sidecar", "extract-vla", "encode-vla", "all"), default="audit")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--encode-batch-size", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifests = tuple(args.manifests) if args.manifests else DEFAULT_MANIFESTS
    candidates, skipped = discover_eligible_shards(manifests, args.rollout_root)
    if not candidates:
        raise RuntimeError(f"No eligible shards. skipped={skipped}")
    split = stratified_split(candidates, holdout_ratio=args.holdout_ratio, seed=args.seed)
    write_audit(candidates, skipped, split, args.output_dir)
    if args.phase in {"sidecar", "all"}:
        build_sidecar_replay(candidates, split, args.output_dir)
    if args.phase in {"extract-vla", "all"}:
        extract_vla_token_blocks(candidates, args.output_dir, overwrite=args.overwrite)
    if args.phase in {"encode-vla", "all"}:
        encode_vla_replay(candidates, split, args.output_dir, overwrite=args.overwrite, batch_size=args.encode_batch_size)
    print(json.dumps({"output_dir": str(args.output_dir), "eligible_shards": len(candidates), "transitions": sum(row["rows"] for row in candidates)}, indent=2))


if __name__ == "__main__":
    main()
