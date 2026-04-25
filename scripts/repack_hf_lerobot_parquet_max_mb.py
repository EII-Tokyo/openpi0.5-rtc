#!/usr/bin/env python3
"""Download LeRobot-style HF dataset parquet shards, split each file to stay under a max size, upload as a new repo.

Preserves row order and global frame sequence when consuming files in sorted path order.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


def _shard_row_counts(n_rows: int, n_shards: int) -> list[int]:
    if n_shards <= 0:
        raise ValueError("n_shards must be positive")
    base, rem = divmod(n_rows, n_shards)
    return [base + (1 if i < rem else 0) for i in range(n_shards)]


def _split_one_parquet(
    src: Path,
    out_paths: list[Path],
    row_counts: list[int],
) -> None:
    if len(out_paths) != len(row_counts):
        raise ValueError("out_paths and row_counts length mismatch")
    pf = pq.ParquetFile(src)
    schema = pf.schema_arrow
    shard_idx = 0
    rows_needed = row_counts[shard_idx]
    rows_have = 0
    writer: pq.ParquetWriter | None = pq.ParquetWriter(str(out_paths[shard_idx]), schema)
    try:
        for batch in pf.iter_batches(batch_size=65536):
            offset = 0
            while offset < batch.num_rows:
                assert writer is not None
                take = min(batch.num_rows - offset, rows_needed - rows_have)
                writer.write_batch(batch.slice(offset, take))
                rows_have += take
                offset += take
                if rows_have >= rows_needed:
                    writer.close()
                    writer = None
                    shard_idx += 1
                    if shard_idx >= len(row_counts):
                        if offset < batch.num_rows:
                            raise RuntimeError("extra rows after last shard")
                        return
                    rows_needed = row_counts[shard_idx]
                    rows_have = 0
                    writer = pq.ParquetWriter(str(out_paths[shard_idx]), schema)
            if shard_idx >= len(row_counts):
                return
    finally:
        if writer is not None:
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-repo",
        default="lyl472324464/twist_subset_balanced_100k_448_multi_repo_300mb_train_action_true_process_all_bottles",
    )
    parser.add_argument(
        "--dest-repo",
        default="lyl472324464/twist_subset_balanced_100k_448_multi_repo_200mb_train_action_true_process_all_bottles",
    )
    parser.add_argument("--max-mb", type=float, default=200.0)
    parser.add_argument(
        "--files-per-commit",
        type=int,
        default=10,
        help="How many source parquet files to upload per Hub commit (HF free tier ~128 commits/hour).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN in the environment.")

    max_bytes = int(args.max_mb * 1024 * 1024)
    api = HfApi(token=token)

    data_files = sorted(
        f
        for f in api.list_repo_files(args.source_repo, repo_type="dataset")
        if f.startswith("data/chunk-") and f.endswith(".parquet")
    )
    if not data_files:
        raise SystemExit("No data parquet files found.")

    infos = api.get_paths_info(args.source_repo, data_files, repo_type="dataset")
    path_to_size = {i.path: i.size for i in infos}

    total_out = sum(max(1, math.ceil(path_to_size[p] / max_bytes)) for p in data_files)
    print(f"Source data files: {len(data_files)} -> estimated output shards: {total_out}")

    if args.dry_run:
        return

    api.create_repo(args.dest_repo, repo_type="dataset", exist_ok=True)

    all_files = api.list_repo_files(args.source_repo, repo_type="dataset")
    meta_copy = [
        f
        for f in all_files
        if not (f.startswith("data/chunk-") and f.endswith(".parquet"))
        and f != "README.md"
    ]
    tmp_meta = Path(tempfile.mkdtemp(prefix="hf_meta_"))
    try:
        for rel in meta_copy:
            local = hf_hub_download(
                args.source_repo,
                rel,
                repo_type="dataset",
                token=token,
            )
            dest = tmp_meta / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, dest)

        info_path = tmp_meta / "meta" / "info.json"
        if info_path.is_file():
            meta = json.loads(info_path.read_text())
            meta["data_files_size_in_mb"] = int(round(args.max_mb))
            info_path.write_text(json.dumps(meta, indent=4) + "\n")

        api.upload_folder(
            folder_path=str(tmp_meta),
            repo_id=args.dest_repo,
            repo_type="dataset",
            path_in_repo="",
            token=token,
        )
    finally:
        shutil.rmtree(tmp_meta, ignore_errors=True)

    if any(f == "README.md" for f in all_files):
        p = hf_hub_download(args.source_repo, "README.md", repo_type="dataset", token=token)
        extra = (
            "\n\n---\n\nRepacked from " + str(args.source_repo) + ": each data/chunk parquet shard is at most ~"
            + str(int(round(args.max_mb)))
            + " MB for easier Hub preview.\n"
        )
        text = Path(p).read_text() + extra
        api.upload_file(
            path_or_fileobj=text.encode(),
            path_in_repo="README.md",
            repo_id=args.dest_repo,
            repo_type="dataset",
            token=token,
        )

    global_idx = 0
    work = Path(tempfile.mkdtemp(prefix="repack_parquet_"))
    pending_ops: list[CommitOperationAdd] = []
    pending_locals: list[Path] = []
    try:
        for i, rel in enumerate(data_files):
            size = path_to_size[rel]
            local_src = hf_hub_download(
                args.source_repo,
                rel,
                repo_type="dataset",
                token=token,
            )
            src_path = Path(local_src)
            pf = pq.ParquetFile(src_path)
            n_rows = pf.metadata.num_rows
            raw_shards = max(1, math.ceil(size / max_bytes))
            if n_rows <= 0:
                n_shards = 1
                row_counts = [0]
            else:
                n_shards = min(raw_shards, n_rows)
                row_counts = _shard_row_counts(n_rows, n_shards)
            out_paths: list[Path] = []
            for _ in range(n_shards):
                chunk_dir = work / "data" / "chunk-000"
                chunk_dir.mkdir(parents=True, exist_ok=True)
                out_paths.append(chunk_dir / f"file-{global_idx:03d}.parquet")
                global_idx += 1
            _split_one_parquet(src_path, out_paths, row_counts)
            for op in out_paths:
                hub_path = str(Path("data") / "chunk-000" / op.name)
                pending_ops.append(
                    CommitOperationAdd(path_in_repo=hub_path, path_or_fileobj=str(op))
                )
                pending_locals.append(op)
            print(f"[{i + 1}/{len(data_files)}] {rel} ({size / 1e6:.1f} MB) -> {n_shards} shards", flush=True)
            src_path.unlink(missing_ok=True)
            is_last = i == len(data_files) - 1
            if (i + 1) % args.files_per_commit == 0 or is_last:
                if pending_ops:
                    api.create_commit(
                        repo_id=args.dest_repo,
                        repo_type="dataset",
                        operations=pending_ops,
                        commit_message="data parquet batch ending " + rel,
                        token=token,
                    )
                    for lp in pending_locals:
                        lp.unlink(missing_ok=True)
                    pending_ops = []
                    pending_locals = []
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("Done. Destination: https://huggingface.co/datasets/" + args.dest_repo)


if __name__ == "__main__":
    main()
