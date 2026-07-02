from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path
from typing import Any

import numpy as np

from scripts.reencode_clean_no_actor_z_rl import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    ReencodeSummary,
    compute_replay_frame_indices,
    find_rollout_dir,
    load_manifest_from_npz,
    rewrite_shard_z_rl,
    _load_qpos,
    _shard_rows,
)


@dataclasses.dataclass(frozen=True)
class RepairArgs:
    input_root: Path
    rollout_root: Path
    output_root: Path
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    config_name: str = DEFAULT_CONFIG
    limit: int | None = None
    execute: bool = False
    overwrite: bool = False
    prompt: str = "Twist off the bottle cap."


class PolicyStateEncoder:
    """Rebuild the policy-input state/proprio without loading model weights."""

    def __init__(self, *, config_name: str, checkpoint_path: Path, prompt: str) -> None:
        import openpi.transforms as transforms
        from openpi.training import checkpoints
        from openpi.training import config as train_config

        cfg = train_config.get_config(config_name)
        data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
        if data_cfg.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = checkpoints.load_norm_stats(checkpoint_path / "assets", data_cfg.asset_id)
        self._transform = transforms.compose(
            [
                *data_cfg.data_transforms.inputs,
                transforms.Normalize(norm_stats, use_quantiles=data_cfg.use_quantile_norm),
                transforms.PadStatesAndActions(cfg.model.action_dim),
            ]
        )
        self._prompt = prompt
        self._dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)

    def encode_qpos(self, qpos: np.ndarray, frame_indices: np.ndarray) -> np.ndarray:
        states = []
        for frame_index in frame_indices:
            obs = {
                "images": {
                    "cam_low": self._dummy_image,
                    "cam_right_wrist": self._dummy_image,
                },
                "state": np.asarray(qpos[int(frame_index)], dtype=np.float32),
                "prompt": self._prompt,
            }
            states.append(np.asarray(self._transform(obs)["state"], dtype=np.float32))
        return np.stack(states, axis=0).astype(np.float32)


def discover_shards(input_root: Path) -> list[Path]:
    if not input_root.exists():
        return []
    return sorted(path for path in input_root.rglob("*.npz") if path.is_file())


def repair_shard(
    shard_path: Path,
    output_path: Path,
    *,
    rollout_root: Path,
    state_encoder: PolicyStateEncoder,
    checkpoint_path: Path,
    config_name: str,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    manifest = load_manifest_from_npz(shard_path)
    rollout_dir = find_rollout_dir(rollout_root, manifest)
    qpos = _load_qpos(rollout_dir / "episode.hdf5")
    current_frames, next_frames = compute_replay_frame_indices(
        manifest,
        clean_rows=_shard_rows(shard_path),
        episode_frames=len(qpos),
    )
    proprio = state_encoder.encode_qpos(qpos, current_frames)
    next_proprio = state_encoder.encode_qpos(qpos, next_frames)
    with np.load(shard_path, allow_pickle=False) as data:
        z_rl = np.asarray(data["z_rl"], dtype=np.float32)
        next_z_rl = np.asarray(data["next_z_rl"], dtype=np.float32)
    rewrite_shard_z_rl(
        shard_path,
        output_path,
        z_rl=z_rl,
        next_z_rl=next_z_rl,
        proprio=proprio,
        next_proprio=next_proprio,
        checkpoint_path=checkpoint_path,
        config_name=config_name,
    )


def repair_reencoded_proprio(args: RepairArgs) -> ReencodeSummary:
    shards = discover_shards(args.input_root)
    if args.limit is not None:
        shards = shards[: args.limit]
    if not args.execute:
        return ReencodeSummary(planned=len(shards), converted=0, skipped={}, output_root=args.output_root)

    state_encoder = PolicyStateEncoder(
        config_name=args.config_name,
        checkpoint_path=args.checkpoint_path,
        prompt=args.prompt,
    )
    converted = 0
    skipped: dict[str, int] = {}
    for index, shard_path in enumerate(shards, start=1):
        try:
            output_path = args.output_root / shard_path.relative_to(args.input_root)
            repair_shard(
                shard_path,
                output_path,
                rollout_root=args.rollout_root,
                state_encoder=state_encoder,
                checkpoint_path=args.checkpoint_path,
                config_name=args.config_name,
                overwrite=args.overwrite,
            )
            converted += 1
            logging.info("repaired %s/%s shard=%s output=%s", index, len(shards), shard_path, output_path)
        except Exception as exc:  # pragma: no cover - CLI diagnostics.
            key = type(exc).__name__
            skipped[key] = skipped.get(key, 0) + 1
            logging.exception("failed to repair %s: %s", shard_path, exc)
    return ReencodeSummary(
        planned=len(shards),
        converted=converted,
        skipped=dict(sorted(skipped.items())),
        output_root=args.output_root,
    )


def _parse_args() -> RepairArgs:
    parser = argparse.ArgumentParser(
        description="Repair re-encoded RLT replay by recomputing policy-input proprio/state for each re-encoded z_rl row."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prompt", default="Twist off the bottle cap.")
    ns = parser.parse_args()
    return RepairArgs(
        input_root=ns.input_root,
        rollout_root=ns.rollout_root,
        output_root=ns.output_root,
        checkpoint_path=ns.checkpoint_path,
        config_name=ns.config_name,
        limit=ns.limit,
        execute=ns.execute,
        overwrite=ns.overwrite,
        prompt=ns.prompt,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    summary = repair_reencoded_proprio(args)
    logging.info(
        "repair proprio summary planned=%s converted=%s skipped=%s output_root=%s execute=%s",
        summary.planned,
        summary.converted,
        summary.skipped,
        summary.output_root,
        args.execute,
    )
    if not args.execute:
        logging.info("dry-run only. Pass --execute to write repaired shards.")


if __name__ == "__main__":
    main()
