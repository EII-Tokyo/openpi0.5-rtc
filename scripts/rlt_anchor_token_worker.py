#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from openpi.training import rlt_anchor_token_cache


def candidate_from_job(payload: dict[str, Any]):
    from scripts import rebuild_online_rollout_paper_anchor_replay as rebuild

    source = payload.get("source_runtime_cache_block_shard_path")
    rollout = payload.get("rollout_dir")
    if not source:
        raise ValueError(f"job {payload.get('key_region_id')} has no source_runtime_cache_block_shard_path")
    if not rollout:
        raise ValueError(f"job {payload.get('key_region_id')} has no rollout_dir")
    key_region_id = str(payload.get("key_region_id") or Path(source).stem.removeprefix("key_region_"))
    return rebuild.Candidate(
        key_region_id=key_region_id,
        source_shard_path=Path(source).resolve(),
        rollout_dir=Path(rollout).resolve(),
        reward=int(float(payload.get("reward") or 0.0) > 0.0),
        num_frames=int(payload["num_frames"]),
        num_replay_transitions=int(payload["num_replay_transitions"]),
        train_horizon=int(payload["train_horizon"]),
        chunk_stride=int(payload["chunk_stride"]),
        action_max_abs_diff=0.0,
        collection_group="async_anchor_token_job",
    )


def encoded_cache_path(encoded_cache_root: Path, key_region_id: str) -> Path:
    return encoded_cache_root / f"{rlt_anchor_token_cache.safe_key_region_name(key_region_id)}.npz"


def append_manifest_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        stream.write("\n")


def assemble_ready_caches(
    *,
    job_root: Path,
    encoded_cache_root: Path,
    output_root: Path,
    manifest_path: Path,
    limit: int | None,
    overwrite: bool,
) -> dict[str, int]:
    summary = {"assembled": 0, "missing_cache": 0, "failed": 0}
    jobs = rlt_anchor_token_cache.list_jobs(job_root, status="pending")
    if limit is not None:
        jobs = jobs[:limit]
    for job in jobs:
        key_region_id = str(job.payload["key_region_id"])
        cache_path = encoded_cache_path(encoded_cache_root, key_region_id)
        if not cache_path.exists():
            summary["missing_cache"] += 1
            logging.info("cache not ready for key_region=%s path=%s", key_region_id, cache_path)
            continue
        try:
            result = rlt_anchor_token_cache.assemble_formal_replay_from_encoded_cache(
                job_path=job.path,
                encoded_cache_path=cache_path,
                output_root=output_root,
                overwrite=overwrite,
            )
            append_manifest_row(manifest_path, result.manifest)
            rlt_anchor_token_cache.move_job(
                job.path,
                "ready",
                extra={
                    "formal_shard_path": str(result.shard_path.resolve()),
                    "formal_manifest_path": str(manifest_path.resolve()),
                    "encoded_anchor_token_cache_path": str(cache_path.resolve()),
                },
            )
            summary["assembled"] += 1
            logging.info("assembled formal replay key_region=%s shard=%s", key_region_id, result.shard_path)
        except Exception as exc:
            summary["failed"] += 1
            rlt_anchor_token_cache.move_job(job.path, "failed", extra={"error": repr(exc)})
            logging.exception("failed assembling key_region=%s", key_region_id)
    return summary


def run_pending_jobs(
    *,
    job_root: Path,
    output_root: Path,
    work_dir: Path,
    manifest_dir: Path,
    dataset_label: str,
    prompt: str,
    limit: int | None,
    overwrite: bool,
    vla_batch_size: int,
    encode_batch_size: int,
) -> dict[str, int]:
    from scripts import rebuild_online_rollout_paper_anchor_replay as rebuild

    pending_jobs = rlt_anchor_token_cache.list_jobs(job_root, status="pending")
    if limit is not None:
        pending_jobs = pending_jobs[:limit]
    if not pending_jobs:
        return {"processed": 0, "failed": 0}

    running_jobs = [
        rlt_anchor_token_cache.move_job(job.path, "running", extra={"worker": Path(__file__).name}) for job in pending_jobs
    ]
    try:
        candidates = [candidate_from_job(job.payload) for job in running_jobs]
        rebuild.extract_token_blocks(
            candidates=candidates,
            work_dir=work_dir,
            overwrite=overwrite,
            prompt=prompt,
            vla_batch_size=vla_batch_size,
        )
        audit = rebuild.write_rebuilt_shards(
            candidates=candidates,
            output_root=output_root,
            work_dir=work_dir,
            manifest_dir=manifest_dir,
            dataset_label=dataset_label,
            overwrite=overwrite,
            encode_batch_size=encode_batch_size,
            prompt=prompt,
        )
        manifest_path = manifest_dir / rebuild.paper_anchor_manifest_name(dataset_label)
        for job in running_jobs:
            key_region_id = str(job.payload["key_region_id"])
            shard_path = output_root / "shards" / f"{rlt_anchor_token_cache.safe_key_region_name(key_region_id)}.npz"
            rlt_anchor_token_cache.move_job(
                job.path,
                "ready",
                extra={
                    "formal_shard_path": str(shard_path.resolve()),
                    "formal_manifest_path": str(manifest_path.resolve()),
                    "paper_anchor_audit_path": str((output_root / "paper_anchor_audit.json").resolve()),
                },
            )
        return {"processed": int(audit["num_shards"]), "failed": 0}
    except Exception as exc:
        for job in running_jobs:
            if job.path.exists():
                rlt_anchor_token_cache.move_job(job.path, "failed", extra={"error": repr(exc)})
        logging.exception("failed running pending async anchor-token jobs")
        return {"processed": 0, "failed": len(running_jobs)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process async RLT anchor-token jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    assemble = subparsers.add_parser("assemble-ready-caches", help="Assemble formal replay from pre-encoded anchor token caches.")
    assemble.add_argument("--job-root", type=Path, required=True)
    assemble.add_argument("--encoded-cache-root", type=Path, required=True)
    assemble.add_argument("--output-root", type=Path, required=True)
    assemble.add_argument("--manifest-path", type=Path, required=True)
    assemble.add_argument("--limit", type=int, default=None)
    assemble.add_argument("--overwrite", action="store_true")

    run = subparsers.add_parser("run-pending", help="Run VLA token extraction and formal replay rebuild for pending jobs.")
    run.add_argument("--job-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument("--manifest-dir", type=Path, required=True)
    run.add_argument("--dataset-label", required=True)
    run.add_argument("--prompt", default="Twist off the bottle cap.")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--vla-batch-size", type=int, default=1)
    run.add_argument("--encode-batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.command == "assemble-ready-caches":
        summary = assemble_ready_caches(
            job_root=args.job_root,
            encoded_cache_root=args.encoded_cache_root,
            output_root=args.output_root,
            manifest_path=args.manifest_path,
            limit=args.limit,
            overwrite=args.overwrite,
        )
    elif args.command == "run-pending":
        summary = run_pending_jobs(
            job_root=args.job_root,
            output_root=args.output_root,
            work_dir=args.work_dir,
            manifest_dir=args.manifest_dir,
            dataset_label=args.dataset_label,
            prompt=args.prompt,
            limit=args.limit,
            overwrite=args.overwrite,
            vla_batch_size=args.vla_batch_size,
            encode_batch_size=args.encode_batch_size,
        )
    else:
        raise AssertionError(args.command)
    logging.info("summary=%s", summary)


if __name__ == "__main__":
    main()
