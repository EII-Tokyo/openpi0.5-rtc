from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import numpy as np

from scripts.reencode_clean_no_actor_z_rl import DEFAULT_CHECKPOINT
from scripts.reencode_clean_no_actor_z_rl import DEFAULT_CONFIG
from scripts.reencode_clean_no_actor_z_rl import REPLAY_KEYS
from scripts.reencode_clean_no_actor_z_rl import ReencodeSummary
from scripts.reencode_clean_no_actor_z_rl import load_manifest_from_npz


@dataclasses.dataclass(frozen=True)
class SegmentRepairArgs:
    input_root: Path
    output_root: Path
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    config_name: str = DEFAULT_CONFIG
    limit: int | None = None
    execute: bool = False
    overwrite: bool = False


def change_rows(array: np.ndarray, *, tol: float = 1e-6) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.shape[0] <= 1:
        return np.asarray([], dtype=np.int64)
    diffs = np.max(np.abs(np.diff(values, axis=0)), axis=tuple(range(1, values.ndim)))
    return np.asarray(np.where(diffs > tol)[0] + 1, dtype=np.int64)


def align_to_reference_segments(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    starts = [0, *change_rows(reference).tolist(), int(values.shape[0])]
    aligned = np.array(values, copy=True)
    for start, end in zip(starts[:-1], starts[1:], strict=True):
        aligned[start:end] = values[start]
    return aligned


def discover_shards(input_root: Path) -> list[Path]:
    if not input_root.exists():
        return []
    return sorted(path for path in input_root.rglob("*.npz") if path.is_file())


def repair_shard_segments(
    input_path: Path,
    output_path: Path,
    *,
    checkpoint_path: Path,
    config_name: str,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    with np.load(input_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in REPLAY_KEYS if key in data}
    manifest = load_manifest_from_npz(input_path)
    arrays["z_rl"] = align_to_reference_segments(arrays["z_rl"], arrays["proprio"]).astype(np.float32)
    arrays["next_z_rl"] = align_to_reference_segments(arrays["next_z_rl"], arrays["next_proprio"]).astype(np.float32)
    manifest.update(
        {
            "z_rl_source": "rl_token_reencoded_aligned_to_proprio_segments",
            "z_rl_dim": int(arrays["z_rl"].shape[-1]),
            "rl_token_checkpoint_path": str(checkpoint_path),
            "rl_token_config_name": config_name,
            "z_rl_segment_alignment": "proprio_change_rows",
            "next_z_rl_segment_alignment": "next_proprio_change_rows",
        }
    )
    arrays["manifest"] = np.asarray(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def repair_reencoded_z_segments(args: SegmentRepairArgs) -> ReencodeSummary:
    shards = discover_shards(args.input_root)
    if args.limit is not None:
        shards = shards[: args.limit]
    if not args.execute:
        return ReencodeSummary(planned=len(shards), converted=0, skipped={}, output_root=args.output_root)

    converted = 0
    skipped: dict[str, int] = {}
    for index, shard_path in enumerate(shards, start=1):
        try:
            output_path = args.output_root / shard_path.relative_to(args.input_root)
            repair_shard_segments(
                shard_path,
                output_path,
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


def _parse_args() -> SegmentRepairArgs:
    parser = argparse.ArgumentParser(
        description="Align re-encoded 2048-dim z_rl arrays to the existing replay proprio chunk boundaries."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    ns = parser.parse_args()
    return SegmentRepairArgs(
        input_root=ns.input_root,
        output_root=ns.output_root,
        checkpoint_path=ns.checkpoint_path,
        config_name=ns.config_name,
        limit=ns.limit,
        execute=ns.execute,
        overwrite=ns.overwrite,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    summary = repair_reencoded_z_segments(args)
    logging.info(
        "repair z segments summary planned=%s converted=%s skipped=%s output_root=%s execute=%s",
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
