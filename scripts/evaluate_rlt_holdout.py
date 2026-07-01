from __future__ import annotations

import dataclasses
import pathlib

import tyro

from openpi.training import rlt_eval


@dataclasses.dataclass
class Args:
    checkpoint_dir: pathlib.Path
    replay_dir: pathlib.Path
    output_dir: pathlib.Path
    holdout_manifest_path: pathlib.Path | None = None
    recursive_scan: bool = False
    segment_db_path: pathlib.Path | None = None
    score_batch_size: int = 512


def main(args: Args) -> None:
    checkpoints = rlt_eval.discover_inference_checkpoints(args.checkpoint_dir)
    holdout_paths = rlt_eval.find_replay_shards(
        args.replay_dir,
        recursive=args.recursive_scan,
        segment_db_path=args.segment_db_path,
        manifest_path=args.holdout_manifest_path,
    )
    rlt_eval.evaluate_holdout_checkpoints(
        checkpoint_dirs=checkpoints,
        holdout_paths=holdout_paths,
        output_dir=args.output_dir,
        score_batch_size=args.score_batch_size,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
