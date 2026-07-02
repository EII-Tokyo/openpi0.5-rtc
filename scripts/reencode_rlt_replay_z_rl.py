from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

from scripts.reencode_clean_no_actor_z_rl import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    ReencodeSummary,
    RLTokenPolicyEncoder,
    find_rollout_dir,
    load_manifest_from_npz,
    rewrite_shard_z_rl,
    _print_gpu_memory,
    _shard_rows,
)


DEFAULT_REPLAY_ROOT = Path("/home/eii/data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions")
DEFAULT_ROLLOUT_ROOT = Path("/home/eii/data/openpi0.5-rtc-reward-learning/rollouts/key_regions")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/eii/data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions_lower_right_z2048_4layer"
)


@dataclasses.dataclass(frozen=True)
class ReencodeReplayArgs:
    replay_root: Path = DEFAULT_REPLAY_ROOT
    rollout_root: Path = DEFAULT_ROLLOUT_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    config_name: str = DEFAULT_CONFIG
    limit: int | None = None
    execute: bool = False
    probe_only: bool = False
    overwrite: bool = False
    convert_bgr_to_rgb: bool = False
    prompt: str = "Twist off the bottle cap."
    require_camera: tuple[str, ...] = ("cam_low", "cam_right_wrist")


def discover_replay_shards(replay_root: Path) -> list[Path]:
    """Return RLT replay shards that can be rewritten with a new z_rl encoder."""
    shards: list[Path] = []
    for path in sorted(replay_root.rglob("*.npz")):
        try:
            load_manifest_from_npz(path)
            rows = _shard_rows(path)
        except Exception as exc:  # pragma: no cover - CLI diagnostics.
            logging.warning("skip unreadable replay shard %s: %s", path, exc)
            continue
        if rows <= 0:
            logging.warning("skip empty replay shard %s", path)
            continue
        shards.append(path)
    return shards


def reencode_rlt_replay(args: ReencodeReplayArgs) -> ReencodeSummary:
    shards = discover_replay_shards(args.replay_root)
    if args.limit is not None:
        shards = shards[: args.limit]
    if not args.execute:
        return ReencodeSummary(planned=len(shards), converted=0, skipped={}, output_root=args.output_root)

    encoder = RLTokenPolicyEncoder(
        config_name=args.config_name,
        checkpoint_path=args.checkpoint_path,
        prompt=args.prompt,
        convert_bgr_to_rgb=args.convert_bgr_to_rgb,
        require_camera=args.require_camera,
    )
    converted = 0
    skipped: dict[str, int] = {}
    for index, shard_path in enumerate(shards, start=1):
        try:
            output_path = args.output_root / shard_path.relative_to(args.replay_root)
            if output_path.exists() and not args.overwrite:
                skipped["output_exists"] = skipped.get("output_exists", 0) + 1
                continue
            manifest = load_manifest_from_npz(shard_path)
            rollout_dir = find_rollout_dir(args.rollout_root, manifest)
            current_z, current_proprio, next_z, next_proprio = encoder.encode_shard(
                rollout_dir,
                manifest,
                clean_rows=_shard_rows(shard_path),
            )
            rewrite_shard_z_rl(
                shard_path,
                output_path,
                z_rl=current_z,
                next_z_rl=next_z,
                proprio=current_proprio,
                next_proprio=next_proprio,
                checkpoint_path=args.checkpoint_path,
                config_name=args.config_name,
            )
            converted += 1
            logging.info("converted %s/%s shard=%s output=%s", index, len(shards), shard_path, output_path)
        except Exception as exc:  # pragma: no cover - CLI diagnostics.
            key = type(exc).__name__
            skipped[key] = skipped.get(key, 0) + 1
            logging.exception("failed to reencode %s: %s", shard_path, exc)
    return ReencodeSummary(
        planned=len(shards),
        converted=converted,
        skipped=dict(sorted(skipped.items())),
        output_root=args.output_root,
    )


def run_probe(args: ReencodeReplayArgs) -> None:
    shards = discover_replay_shards(args.replay_root)
    if not shards:
        raise RuntimeError(f"no replay shards found under {args.replay_root}")
    if args.limit is not None:
        shards = shards[: args.limit]
    shard_path = shards[0]
    manifest = load_manifest_from_npz(shard_path)
    rollout_dir = find_rollout_dir(args.rollout_root, manifest)
    logging.info("probe shard=%s", shard_path)
    logging.info("probe rollout=%s", rollout_dir)
    _print_gpu_memory("before-load")
    encoder = RLTokenPolicyEncoder(
        config_name=args.config_name,
        checkpoint_path=args.checkpoint_path,
        prompt=args.prompt,
        convert_bgr_to_rgb=args.convert_bgr_to_rgb,
        require_camera=args.require_camera,
    )
    _print_gpu_memory("after-load")
    z = encoder.probe_one(rollout_dir, manifest)
    _print_gpu_memory("after-one-infer")
    logging.info("probe z_rl shape=%s dtype=%s", z.shape, z.dtype)


def _parse_args() -> ReencodeReplayArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Re-encode arbitrary RLT replay shards with the lower+right 4-layer RL Token checkpoint. "
            "Writes to a separate output root and never overwrites unless --overwrite is passed."
        )
    )
    parser.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--convert-bgr-to-rgb", action="store_true")
    parser.add_argument("--prompt", default="Twist off the bottle cap.")
    parser.add_argument("--require-camera", action="append", default=["cam_low", "cam_right_wrist"])
    ns = parser.parse_args()
    return ReencodeReplayArgs(
        replay_root=ns.replay_root,
        rollout_root=ns.rollout_root,
        output_root=ns.output_root,
        checkpoint_path=ns.checkpoint_path,
        config_name=ns.config_name,
        limit=ns.limit,
        execute=ns.execute,
        probe_only=ns.probe_only,
        overwrite=ns.overwrite,
        convert_bgr_to_rgb=ns.convert_bgr_to_rgb,
        prompt=ns.prompt,
        require_camera=tuple(ns.require_camera),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    if args.probe_only:
        run_probe(args)
        return
    summary = reencode_rlt_replay(args)
    logging.info(
        "reencode replay summary planned=%s converted=%s skipped=%s output_root=%s execute=%s",
        summary.planned,
        summary.converted,
        summary.skipped,
        summary.output_root,
        args.execute,
    )
    if not args.execute:
        logging.info("dry-run only. Run --probe-only first, then use --execute.")


if __name__ == "__main__":
    main()
