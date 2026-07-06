#!/usr/bin/env python3
"""Rebuild online key-region replay into formal paper-subsampled-anchor replay.

The 2026-07-06 online replay shards contain valid action/reference/reward arrays,
but their saved z_rl/proprio state came from runtime action-cache blocks. This
script keeps the verified action-side arrays and rebuilds x_i and x_{i+C} from
the raw rollout frame anchors:

    x_i      = encode(frame_i)
    action_i = executed_actions[i : i + C]
    x_next   = encode(frame_{i + C})

The z_rl path matches the strict "B group" experiment: run the cam4 VLA forward,
take the low/right image token blocks from that same forward pass, then encode
those blocks with the lower+right 4-layer RLToken autoencoder.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import jax
import jax.numpy as jnp
import numpy as np

from scripts.compare_vla_same_forward_vs_sidecar_tokens import (
    DEFAULT_CAM4_CHECKPOINT,
    DEFAULT_CAM4_CONFIG,
    DEFAULT_SIDECAR_CHECKPOINT,
    DEFAULT_SIDECAR_CONFIG,
    _build_lower_right_prefix_from_blocks,
)
from scripts.reencode_clean_no_actor_z_rl import _VideoFrameReader, _load_qpos, load_manifest_from_npz


DEFAULT_SOURCE_REPLAY_ROOT = Path(
    "/home/eii/data/openpi0.5-rtc-reward-learning/replay/"
    "rlt_key_regions/twist_off_the_bottle_cap/2026-07-06"
)
DEFAULT_ROLLOUT_ROOT = Path(
    "/home/eii/data/openpi0.5-rtc-reward-learning/rollouts/key_regions/"
    "twist_off_the_bottle_cap/2026-07-06/rl"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/eii/data/openpi0.5-rtc-reward-learning/replay/"
    "paper_anchor_2048/twist_off_the_bottle_cap/2026-07-06"
)
DEFAULT_WORK_DIR = Path("local_rlt_reencoded/paper_anchor_today142_20260706")
DEFAULT_MANIFEST_DIR = Path("local_rlt_manifests/paper_anchor_today142_plus_original_20260706")
DEFAULT_DATASET_LABEL = "20260706_all"
PROMPT = "Twist off the bottle cap."
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


@dataclasses.dataclass(frozen=True)
class Candidate:
    key_region_id: str
    source_shard_path: Path
    rollout_dir: Path
    reward: int
    num_frames: int
    num_replay_transitions: int
    train_horizon: int
    chunk_stride: int
    action_max_abs_diff: float
    collection_group: str


COLLECTION_GROUP_ALIASES = {
    "all": None,
    "base142": "base142_legacy_unmarked",
    "actor93": "actor93_runtime_cache_block",
    "formal": "formal_paper_anchor",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def key_region_id_from_shard(path: Path, manifest: dict[str, Any]) -> str:
    raw = str(manifest.get("key_region_id") or path.stem.removeprefix("key_region_"))
    return raw.removeprefix("key_region_")


def infer_collection_group(manifest: dict[str, Any] | None) -> str:
    """Classify the 2026-07-06 online collection source before formal rebuild.

    The morning 142 shards are legacy/unmarked but numerically look like runtime
    cache blocks. The later 93 shards are explicitly marked as runtime cache
    blocks. Keeping them separate is important because the 93 were collected by
    an actor trained from the first, faulty representation.
    """

    manifest = dict(manifest or {})
    replay_state_grain = manifest.get("replay_state_grain")
    if replay_state_grain == "paper_subsampled_anchor":
        return "formal_paper_anchor"
    if manifest.get("requires_offline_reencode") is True or replay_state_grain == "runtime_action_cache_block":
        return "actor93_runtime_cache_block"
    if not replay_state_grain:
        return "base142_legacy_unmarked"
    return f"unsupported_{replay_state_grain}"


def filter_candidates_by_collection_group(candidates: list[Candidate], group: str) -> list[Candidate]:
    expected = COLLECTION_GROUP_ALIASES[group]
    if expected is None:
        return list(candidates)
    return [row for row in candidates if row.collection_group == expected]


def paper_anchor_manifest_name(dataset_label: str) -> str:
    return f"{dataset_label}_paper_anchor_manifest.jsonl"


def compute_anchor_starts(num_frames: int, train_horizon: int, chunk_stride: int) -> np.ndarray:
    if train_horizon <= 0:
        raise ValueError("train_horizon must be positive")
    if chunk_stride <= 0:
        raise ValueError("chunk_stride must be positive")
    last_start = int(num_frames) - (2 * int(train_horizon))
    if last_start < 0:
        raise ValueError(f"episode has {num_frames} frames, shorter than 2 * horizon {2 * train_horizon}")
    starts = list(range(0, last_start + 1, int(chunk_stride)))
    if starts and starts[-1] != last_start:
        starts.append(last_start)
    return np.asarray(starts, dtype=np.int64)


def build_action_windows(actions: np.ndarray, starts: np.ndarray, train_horizon: int) -> np.ndarray:
    if actions.ndim != 2:
        raise ValueError(f"actions must have shape [T, action_dim], got {actions.shape}")
    return np.stack([actions[int(start) : int(start) + train_horizon] for start in starts], axis=0).astype(np.float32)


def validate_action_windows(source_shard: Path, rollout_dir: Path, starts: np.ndarray, train_horizon: int) -> float:
    import h5py

    with h5py.File(rollout_dir / "episode.hdf5", "r") as file:
        actions = np.asarray(file["action"], dtype=np.float32)
    expected = build_action_windows(actions, starts, train_horizon)
    with np.load(source_shard, allow_pickle=False) as data:
        actual = np.asarray(data["action"], dtype=np.float32)
    if expected.shape != actual.shape:
        raise ValueError(f"{source_shard} action shape mismatch: expected {expected.shape}, got {actual.shape}")
    return float(np.max(np.abs(expected - actual))) if expected.size else 0.0


def discover_candidates(
    *,
    source_replay_root: Path,
    rollout_root: Path,
    limit: int | None,
    action_tolerance: float,
) -> tuple[list[Candidate], dict[str, int]]:
    shard_dir = source_replay_root / "shards"
    candidates: list[Candidate] = []
    skipped: dict[str, int] = {}
    for shard_path in sorted(shard_dir.glob("*.npz")):
        try:
            manifest = load_manifest_from_npz(shard_path)
            key_region_id = key_region_id_from_shard(shard_path, manifest)
            rollout_dir = rollout_root / f"key_region_{key_region_id}"
            if not (rollout_dir / "episode.hdf5").exists():
                raise FileNotFoundError(rollout_dir / "episode.hdf5")
            for camera in ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"):
                if not (rollout_dir / f"{camera}.mp4").exists():
                    raise FileNotFoundError(rollout_dir / f"{camera}.mp4")
            with np.load(shard_path, allow_pickle=False) as data:
                missing = [key for key in REPLAY_KEYS if key not in data]
                if missing:
                    raise KeyError(f"missing replay arrays {missing}")
                rows = int(data["action"].shape[0])
                train_horizon = int(data["action"].shape[1])
                if int(data["z_rl"].shape[-1]) != 2048:
                    raise ValueError(f"expected old z_dim=2048, got {data['z_rl'].shape[-1]}")
            qpos = _load_qpos(rollout_dir / "episode.hdf5")
            chunk_stride = int(manifest.get("chunk_stride") or manifest.get("train_chunk_stride") or 2)
            starts = compute_anchor_starts(len(qpos), train_horizon, chunk_stride)
            if len(starts) != rows:
                raise ValueError(f"computed {len(starts)} anchor rows, shard has {rows}")
            action_diff = validate_action_windows(shard_path, rollout_dir, starts, train_horizon)
            if action_diff > action_tolerance:
                raise ValueError(f"action window mismatch max_abs_diff={action_diff:.6g}")
            reward = int(float(manifest.get("reward", 0)) > 0.0)
            collection_group = infer_collection_group(manifest)
            candidates.append(
                Candidate(
                    key_region_id=key_region_id,
                    source_shard_path=shard_path.resolve(),
                    rollout_dir=rollout_dir.resolve(),
                    reward=reward,
                    num_frames=int(len(qpos)),
                    num_replay_transitions=rows,
                    train_horizon=train_horizon,
                    chunk_stride=chunk_stride,
                    action_max_abs_diff=action_diff,
                    collection_group=collection_group,
                )
            )
            if limit is not None and len(candidates) >= limit:
                break
        except Exception as exc:
            skipped[type(exc).__name__] = skipped.get(type(exc).__name__, 0) + 1
            logging.exception("skip %s: %s", shard_path, exc)
    return candidates, skipped


class VLAAnchorExtractor:
    def __init__(self, *, config_name: str, checkpoint: Path, prompt: str) -> None:
        from openpi.models import model as _model
        from openpi.policies import policy_config
        from openpi.policies.policy import _drop_language_from_prefix_hidden
        from openpi.training import config as train_config

        self._model_module = _model
        self._drop_language_from_prefix_hidden = _drop_language_from_prefix_hidden
        cfg = train_config.get_config(config_name)
        logging.info("loading VLA policy config=%s checkpoint=%s", config_name, checkpoint)
        self._policy = policy_config.create_trained_policy(cfg, checkpoint, default_prompt=prompt)
        self._prompt = prompt

    def extract(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        return self.extract_batch([obs])[0]

    def extract_batch(self, observations: list[dict[str, Any]]) -> list[dict[str, np.ndarray]]:
        if not observations:
            return []
        transformed = [self._policy._input_transform(jax.tree.map(lambda x: x, obs)) for obs in observations]  # noqa: SLF001
        batched = jax.tree.map(
            lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], axis=0),
            *transformed,
        )
        observation = self._model_module.Observation.from_dict(batched)
        prefix_hidden = self._policy._embed_prefix_hidden(observation)  # noqa: SLF001
        prefix_out, prefix_mask = self._drop_language_from_prefix_hidden(prefix_hidden, observation)
        prefix_out_np = np.asarray(jax.device_get(prefix_out), dtype=np.float32)
        prefix_mask_np = np.asarray(jax.device_get(prefix_mask), dtype=bool)
        slot_names = list(observation.images.keys())
        if "base_1_rgb" not in slot_names or "right_wrist_0_rgb" not in slot_names:
            raise RuntimeError(f"expected cam_low/base_1 and right wrist slots, got {slot_names}")
        tokens_per_slot = prefix_out_np.shape[1] // len(slot_names)
        outputs: list[dict[str, np.ndarray]] = []
        for batch_index in range(prefix_out_np.shape[0]):
            blocks: dict[str, np.ndarray] = {}
            for slot_idx, slot in enumerate(slot_names):
                start = slot_idx * tokens_per_slot
                end = start + tokens_per_slot
                blocks[slot] = prefix_out_np[batch_index, start:end][prefix_mask_np[batch_index, start:end]].astype(np.float32)
            outputs.append(
                {
                    "low_tokens": blocks["base_1_rgb"],
                    "right_tokens": blocks["right_wrist_0_rgb"],
                    "proprio": np.asarray(transformed[batch_index]["state"], dtype=np.float32),
                }
            )
        return outputs


def extract_candidate_token_blocks(
    *,
    row: Candidate,
    extractor: Any,
    out: Path,
    overwrite: bool,
    prompt: str,
    vla_batch_size: int,
    reader_factory: Any,
) -> None:
    if vla_batch_size <= 0:
        raise ValueError(f"vla_batch_size must be positive, got {vla_batch_size}")
    if out.exists() and not overwrite:
        logging.info("skip existing token blocks %s", out)
        return
    starts = compute_anchor_starts(row.num_frames, row.train_horizon, row.chunk_stride)
    next_frames = starts + row.train_horizon
    frame_order = sorted(set(int(frame) for frame in np.concatenate([starts, next_frames])))
    qpos = _load_qpos(row.rollout_dir / "episode.hdf5")
    reader = reader_factory(row.rollout_dir, convert_bgr_to_rgb=False)
    by_frame: dict[int, dict[str, np.ndarray]] = {}
    try:
        for batch_start in range(0, len(frame_order), vla_batch_size):
            batch_frames = frame_order[batch_start : batch_start + vla_batch_size]
            observations = [
                {
                    "images": reader.read_all(frame_index),
                    "state": np.asarray(qpos[frame_index], dtype=np.float32),
                    "prompt": prompt,
                }
                for frame_index in batch_frames
            ]
            for frame_index, features in zip(batch_frames, extractor.extract_batch(observations), strict=True):
                by_frame[int(frame_index)] = features
    finally:
        reader.close()
    np.savez_compressed(
        out,
        low_tokens=np.stack([by_frame[int(frame)]["low_tokens"] for frame in starts]).astype(np.float16),
        right_tokens=np.stack([by_frame[int(frame)]["right_tokens"] for frame in starts]).astype(np.float16),
        next_low_tokens=np.stack([by_frame[int(frame)]["low_tokens"] for frame in next_frames]).astype(np.float16),
        next_right_tokens=np.stack([by_frame[int(frame)]["right_tokens"] for frame in next_frames]).astype(np.float16),
        proprio=np.stack([by_frame[int(frame)]["proprio"] for frame in starts]).astype(np.float32),
        next_proprio=np.stack([by_frame[int(frame)]["proprio"] for frame in next_frames]).astype(np.float32),
        current_frames=starts.astype(np.int64),
        next_frames=next_frames.astype(np.int64),
        key_region_id=np.asarray(row.key_region_id),
        source_shard_path=np.asarray(str(row.source_shard_path)),
        rollout_dir=np.asarray(str(row.rollout_dir)),
    )
    logging.info(
        "extracted token blocks key_region=%s rows=%d unique_frames=%d vla_batch_size=%d",
        row.key_region_id,
        row.num_replay_transitions,
        len(frame_order),
        vla_batch_size,
    )

def extract_token_blocks(
    *,
    candidates: list[Candidate],
    work_dir: Path,
    overwrite: bool,
    prompt: str,
    vla_batch_size: int,
) -> None:
    token_dir = work_dir / "vla_token_blocks"
    token_dir.mkdir(parents=True, exist_ok=True)
    extractor = VLAAnchorExtractor(config_name=DEFAULT_CAM4_CONFIG, checkpoint=DEFAULT_CAM4_CHECKPOINT, prompt=prompt)
    for index, row in enumerate(candidates, start=1):
        out = token_dir / f"key_region_{row.key_region_id}.npz"
        existed = out.exists() and not overwrite
        extract_candidate_token_blocks(
            row=row,
            extractor=extractor,
            out=out,
            overwrite=overwrite,
            prompt=prompt,
            vla_batch_size=vla_batch_size,
            reader_factory=_VideoFrameReader,
        )
        status = "skipped" if existed else "extracted"
        logging.info(
            "%s token blocks %d/%d key_region=%s rows=%d",
            status,
            index,
            len(candidates),
            row.key_region_id,
            row.num_replay_transitions,
        )


def encode_blocks(autoencoder: Any, low: np.ndarray, right: np.ndarray, *, batch_size: int) -> np.ndarray:
    out: list[np.ndarray] = []
    for start in range(0, low.shape[0], batch_size):
        end = min(start + batch_size, low.shape[0])
        prefix, mask = _build_lower_right_prefix_from_blocks(low[start:end], right[start:end])
        z = autoencoder.encode(jax.lax.stop_gradient(jnp.asarray(prefix)), jnp.asarray(mask))
        out.append(np.asarray(jax.device_get(z), dtype=np.float32))
    return np.concatenate(out, axis=0)


def adjacent_exact_fraction(array: np.ndarray) -> float:
    if array.shape[0] <= 1:
        return 0.0
    flat = array.reshape(array.shape[0], -1)
    return float(np.mean(np.all(np.diff(flat, axis=0) == 0, axis=1)))


def adjacent_l2_stats(array: np.ndarray) -> dict[str, float]:
    if array.shape[0] <= 1:
        return {"min": 0.0, "median": 0.0, "mean": 0.0}
    flat = array.reshape(array.shape[0], -1)
    norms = np.linalg.norm(np.diff(flat, axis=0), axis=1)
    return {"min": float(np.min(norms)), "median": float(np.median(norms)), "mean": float(np.mean(norms))}


def write_rebuilt_shards(
    *,
    candidates: list[Candidate],
    output_root: Path,
    work_dir: Path,
    manifest_dir: Path,
    dataset_label: str,
    overwrite: bool,
    encode_batch_size: int,
    prompt: str,
) -> dict[str, Any]:
    from openpi.policies import policy_config
    from openpi.training import config as train_config

    shard_dir = output_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    token_dir = work_dir / "vla_token_blocks"
    cfg = train_config.get_config(DEFAULT_SIDECAR_CONFIG)
    logging.info("loading RLToken autoencoder config=%s checkpoint=%s", DEFAULT_SIDECAR_CONFIG, DEFAULT_SIDECAR_CHECKPOINT)
    policy = policy_config.create_trained_policy(cfg, DEFAULT_SIDECAR_CHECKPOINT, default_prompt=prompt)
    autoencoder = policy._model.rl_token_autoencoder  # noqa: SLF001

    manifest_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for index, row in enumerate(candidates, start=1):
        out = shard_dir / row.source_shard_path.name
        if out.exists() and not overwrite:
            logging.info("skip existing rebuilt shard %s", out)
        else:
            token_path = token_dir / f"key_region_{row.key_region_id}.npz"
            if not token_path.exists():
                raise FileNotFoundError(token_path)
            with np.load(token_path, allow_pickle=False) as token_data:
                low = np.asarray(token_data["low_tokens"], dtype=np.float32)
                right = np.asarray(token_data["right_tokens"], dtype=np.float32)
                next_low = np.asarray(token_data["next_low_tokens"], dtype=np.float32)
                next_right = np.asarray(token_data["next_right_tokens"], dtype=np.float32)
                proprio = np.asarray(token_data["proprio"], dtype=np.float32)
                next_proprio = np.asarray(token_data["next_proprio"], dtype=np.float32)
                current_frames = np.asarray(token_data["current_frames"], dtype=np.int64)
                next_frames = np.asarray(token_data["next_frames"], dtype=np.int64)
            z_rl = encode_blocks(autoencoder, low, right, batch_size=encode_batch_size)
            next_z_rl = encode_blocks(autoencoder, next_low, next_right, batch_size=encode_batch_size)
            with np.load(row.source_shard_path, allow_pickle=False) as source:
                arrays = {key: np.asarray(source[key]) for key in REPLAY_KEYS}
                manifest = load_manifest_from_npz(row.source_shard_path)
            if z_rl.shape[0] != arrays["action"].shape[0]:
                raise ValueError(f"{row.source_shard_path} z rows {z_rl.shape[0]} != action rows {arrays['action'].shape[0]}")
            previous_shapes = {key: list(value.shape) for key, value in arrays.items()}
            arrays["z_rl"] = z_rl.astype(np.float32)
            arrays["next_z_rl"] = next_z_rl.astype(np.float32)
            arrays["proprio"] = proprio.astype(np.float32)
            arrays["next_proprio"] = next_proprio.astype(np.float32)
            manifest.update(
                {
                    "key_region_id": row.key_region_id,
                    "reward": row.reward,
                    "z_rl_source": "vla_same_forward_low_right_tokens_then_lower_right_rl_token_encoder",
                    "z_rl_dim": int(z_rl.shape[-1]),
                    "proprio_source": "vla_policy_input_transform_at_anchor_frame",
                    "proprio_dim": int(proprio.shape[-1]),
                    "replay_state_grain": "paper_subsampled_anchor",
                    "requires_offline_reencode": False,
                    "formal_replay_state_grain": "paper_subsampled_anchor",
                    "formal_replay_ready": True,
                    "train_eligible": True,
                    "subsampled_transition_semantics": "x_i_action_i_to_i_plus_c_next_x_i_plus_c",
                    "source_runtime_cache_block_shard_path": str(row.source_shard_path),
                    "source_rollout_dir": str(row.rollout_dir),
                    "vla_base_config": DEFAULT_CAM4_CONFIG,
                    "vla_base_checkpoint": str(DEFAULT_CAM4_CHECKPOINT),
                    "rl_token_encoder_config": DEFAULT_SIDECAR_CONFIG,
                    "rl_token_encoder_checkpoint": str(DEFAULT_SIDECAR_CHECKPOINT),
                    "conversion_script": Path(__file__).name,
                    "conversion_cache_scope": "transition_anchor_frames",
                    "train_horizon": row.train_horizon,
                    "chunk_stride": row.chunk_stride,
                    "current_frames": [int(x) for x in current_frames],
                    "next_frames": [int(x) for x in next_frames],
                    "previous_replay_array_shapes": previous_shapes,
                    "replay_array_shapes": {key: list(value.shape) for key, value in arrays.items()},
                }
            )
            arrays["manifest"] = np.asarray(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
            tmp_path = out.with_suffix(out.suffix + ".tmp")
            with tmp_path.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            tmp_path.replace(out)
            logging.info("rebuilt paper-anchor shard %d/%d %s rows=%d", index, len(candidates), row.key_region_id, z_rl.shape[0])

        with np.load(out, allow_pickle=False) as rebuilt:
            z_rl = np.asarray(rebuilt["z_rl"], dtype=np.float32)
            proprio = np.asarray(rebuilt["proprio"], dtype=np.float32)
            x = np.concatenate([z_rl, proprio], axis=1)
        manifest_rows.append(
            {
                "shard_path": str(out.resolve()),
                "batch": f"{dataset_label}_paper_subsampled_anchor",
                "source_group": f"{row.collection_group}_rebuilt",
                "collection_group": row.collection_group,
                "key_region_id": row.key_region_id,
                "reward": row.reward,
                "num_transitions": row.num_replay_transitions,
                "z_dim": 2048,
                "replay_state_grain": "paper_subsampled_anchor",
            }
        )
        z_l2 = adjacent_l2_stats(z_rl)
        prop_l2 = adjacent_l2_stats(proprio)
        x_l2 = adjacent_l2_stats(x)
        audit_rows.append(
            {
                "key_region_id": row.key_region_id,
                "reward": row.reward,
                "collection_group": row.collection_group,
                "num_transitions": row.num_replay_transitions,
                "action_max_abs_diff": row.action_max_abs_diff,
                "z_adjacent_exact_fraction": adjacent_exact_fraction(z_rl),
                "proprio_adjacent_exact_fraction": adjacent_exact_fraction(proprio),
                "x_adjacent_exact_fraction": adjacent_exact_fraction(x),
                "z_adjacent_l2_min": z_l2["min"],
                "z_adjacent_l2_median": z_l2["median"],
                "proprio_adjacent_l2_min": prop_l2["min"],
                "proprio_adjacent_l2_median": prop_l2["median"],
                "x_adjacent_l2_min": x_l2["min"],
                "x_adjacent_l2_median": x_l2["median"],
            }
        )

    manifest_path = manifest_dir / paper_anchor_manifest_name(dataset_label)
    write_jsonl(manifest_path, manifest_rows)
    write_jsonl(output_root / "manifest.jsonl", manifest_rows)
    audit = {
        "output_root": str(output_root),
        "work_dir": str(work_dir),
        "manifest_path": str(manifest_path),
        "dataset_label": dataset_label,
        "num_shards": len(manifest_rows),
        "num_transitions": int(sum(row["num_transitions"] for row in manifest_rows)),
        "num_success_shards": int(sum(row["reward"] == 1 for row in manifest_rows)),
        "num_failure_shards": int(sum(row["reward"] == 0 for row in manifest_rows)),
        "collection_groups": {
            group: int(sum(row["collection_group"] == group for row in manifest_rows))
            for group in sorted({row["collection_group"] for row in manifest_rows})
        },
        "mean_x_adjacent_exact_fraction": float(np.mean([row["x_adjacent_exact_fraction"] for row in audit_rows])) if audit_rows else 0.0,
        "max_x_adjacent_exact_fraction": float(np.max([row["x_adjacent_exact_fraction"] for row in audit_rows])) if audit_rows else 0.0,
        "rows": audit_rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "paper_anchor_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_root / "paper_anchor_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0].keys()) if audit_rows else ["key_region_id"])
        writer.writeheader()
        writer.writerows(audit_rows)
    return audit


def write_discovery_audit(
    *,
    candidates: list[Candidate],
    skipped: dict[str, int],
    output_root: Path,
    work_dir: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_candidates": len(candidates),
        "num_transitions": int(sum(row.num_replay_transitions for row in candidates)),
        "num_success_shards": int(sum(row.reward == 1 for row in candidates)),
        "num_failure_shards": int(sum(row.reward == 0 for row in candidates)),
        "collection_groups": {
            group: {
                "num_shards": int(sum(row.collection_group == group for row in candidates)),
                "num_transitions": int(sum(row.num_replay_transitions for row in candidates if row.collection_group == group)),
                "num_success_shards": int(sum(row.collection_group == group and row.reward == 1 for row in candidates)),
                "num_failure_shards": int(sum(row.collection_group == group and row.reward == 0 for row in candidates)),
            }
            for group in sorted({row.collection_group for row in candidates})
        },
        "skipped": skipped,
        "candidates": [
            {
                "key_region_id": row.key_region_id,
                "source_shard_path": str(row.source_shard_path),
                "rollout_dir": str(row.rollout_dir),
                "reward": row.reward,
                "num_frames": row.num_frames,
                "num_replay_transitions": row.num_replay_transitions,
                "train_horizon": row.train_horizon,
                "chunk_stride": row.chunk_stride,
                "action_max_abs_diff": row.action_max_abs_diff,
                "collection_group": row.collection_group,
            }
            for row in candidates
        ],
    }
    (output_root / "paper_anchor_discovery_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def combine_with_original_train(
    *,
    manifest_dir: Path,
    today_manifest: Path,
    original_train_manifest: Path,
    dataset_label: str,
) -> Path:
    original_rows = read_jsonl(original_train_manifest)
    today_rows = read_jsonl(today_manifest)
    combined_path = manifest_dir / f"train_original_plus_{dataset_label}.jsonl"
    write_jsonl(combined_path, original_rows + today_rows)
    return combined_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-replay-root", type=Path, default=DEFAULT_SOURCE_REPLAY_ROOT)
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--dataset-label", default=DEFAULT_DATASET_LABEL)
    parser.add_argument("--collection-group", choices=tuple(COLLECTION_GROUP_ALIASES), default="all")
    parser.add_argument("--original-train-manifest", type=Path, default=Path("local_rlt_runs/strict_td3_z_ablation_20260704/replay/vla_token_z/train_manifest.jsonl"))
    parser.add_argument("--phase", choices=("audit", "extract", "encode", "combine", "all"), default="audit")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--action-tolerance", type=float, default=1e-6)
    parser.add_argument("--vla-batch-size", type=int, default=1)
    parser.add_argument("--encode-batch-size", type=int, default=1)
    parser.add_argument("--prompt", default=PROMPT)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.vla_batch_size <= 0:
        raise ValueError(f"--vla-batch-size must be positive, got {args.vla_batch_size}")
    if args.encode_batch_size <= 0:
        raise ValueError(f"--encode-batch-size must be positive, got {args.encode_batch_size}")
    candidates, skipped = discover_candidates(
        source_replay_root=args.source_replay_root,
        rollout_root=args.rollout_root,
        limit=args.limit,
        action_tolerance=args.action_tolerance,
    )
    if not candidates:
        raise RuntimeError(f"No candidates discovered. skipped={skipped}")
    candidates = filter_candidates_by_collection_group(candidates, args.collection_group)
    if not candidates:
        raise RuntimeError(f"No candidates left after --collection-group={args.collection_group}. skipped={skipped}")
    write_discovery_audit(candidates=candidates, skipped=skipped, output_root=args.output_root, work_dir=args.work_dir)
    logging.info(
        "discovered candidates=%d transitions=%d success=%d failure=%d skipped=%s",
        len(candidates),
        sum(row.num_replay_transitions for row in candidates),
        sum(row.reward == 1 for row in candidates),
        sum(row.reward == 0 for row in candidates),
        skipped,
    )
    if args.phase in {"extract", "all"}:
        extract_token_blocks(
            candidates=candidates,
            work_dir=args.work_dir,
            overwrite=args.overwrite,
            prompt=args.prompt,
            vla_batch_size=args.vla_batch_size,
        )
    if args.phase in {"encode", "all"}:
        audit = write_rebuilt_shards(
            candidates=candidates,
            output_root=args.output_root,
            work_dir=args.work_dir,
            manifest_dir=args.manifest_dir,
            dataset_label=args.dataset_label,
            overwrite=args.overwrite,
            encode_batch_size=args.encode_batch_size,
            prompt=args.prompt,
        )
        logging.info(
            "rebuilt shards=%d transitions=%d mean_x_repeat=%.6f max_x_repeat=%.6f",
            audit["num_shards"],
            audit["num_transitions"],
            audit["mean_x_adjacent_exact_fraction"],
            audit["max_x_adjacent_exact_fraction"],
        )
    if args.phase in {"combine", "all"}:
        combined = combine_with_original_train(
            manifest_dir=args.manifest_dir,
            today_manifest=args.manifest_dir / paper_anchor_manifest_name(args.dataset_label),
            original_train_manifest=args.original_train_manifest,
            dataset_label=args.dataset_label,
        )
        logging.info("wrote combined manifest %s", combined)


if __name__ == "__main__":
    main()
