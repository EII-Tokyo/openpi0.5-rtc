from __future__ import annotations

import argparse
import json
import pathlib

from openpi.training import rlt_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RLT critic checkpoints on a shard-level holdout split.")
    parser.add_argument("--checkpoint-dir", required=True, type=pathlib.Path)
    parser.add_argument("--replay-buffer", "--replay-dir", dest="replay_dir", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recursive-scan", action="store_true")
    parser.add_argument("--segment-db-path", type=pathlib.Path, default=None)
    parser.add_argument("--manifest-path", type=pathlib.Path, default=None)
    parser.add_argument("--holdout-manifest-path", type=pathlib.Path, default=None)
    parser.add_argument("--train-manifest-output", type=pathlib.Path, default=None)
    parser.add_argument("--holdout-manifest-output", type=pathlib.Path, default=None)
    parser.add_argument("--score-batch-size", type=int, default=512)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.holdout_manifest_path is None:
        shards = rlt_eval.find_replay_shards(
            args.replay_dir,
            recursive=args.recursive_scan,
            segment_db_path=args.segment_db_path,
            manifest_path=args.manifest_path,
        )
        split = rlt_eval.split_shards(shards, holdout_ratio=args.holdout_ratio, seed=args.seed)
        train_manifest = args.train_manifest_output or args.output_dir / "train_manifest.jsonl"
        holdout_manifest = args.holdout_manifest_output or args.output_dir / "holdout_manifest.jsonl"
        rlt_eval.write_manifest(split.train_paths, train_manifest)
        rlt_eval.write_manifest(split.holdout_paths, holdout_manifest)
        holdout_paths = list(split.holdout_paths)
        split_summary = {
            "num_total_shards": len(shards),
            "num_train_shards": len(split.train_paths),
            "num_holdout_shards": len(split.holdout_paths),
            "seed": args.seed,
            "holdout_ratio": args.holdout_ratio,
            "train_manifest": str(train_manifest),
            "holdout_manifest": str(holdout_manifest),
        }
    else:
        holdout_paths = rlt_eval.find_replay_shards(args.replay_dir, manifest_path=args.holdout_manifest_path)
        split_summary = {
            "num_total_shards": None,
            "num_train_shards": None,
            "num_holdout_shards": len(holdout_paths),
            "seed": args.seed,
            "holdout_ratio": None,
            "train_manifest": None,
            "holdout_manifest": str(args.holdout_manifest_path),
        }

    checkpoint_dirs = rlt_eval.discover_inference_checkpoints(args.checkpoint_dir)
    result = rlt_eval.evaluate_holdout_checkpoints(
        checkpoint_dirs=checkpoint_dirs,
        holdout_paths=holdout_paths,
        output_dir=args.output_dir,
        score_batch_size=args.score_batch_size,
    )
    split_summary["best_checkpoint"] = None if result.best_metric is None else result.best_metric["checkpoint_path"]
    (args.output_dir / "holdout_split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    print(json.dumps({"split": split_summary, "best": result.best_metric}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
