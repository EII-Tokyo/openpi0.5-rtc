#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
from PIL import Image

from qwen_high_level_utils import (
    DEFAULT_CLASS_SPEC_PATH,
    describe_task_mode,
    infer_task_mode_from_repo_id,
    load_class_map,
    row_class_id,
    system_prompt,
)

DEFAULT_CAMERA_COLUMNS = (
    "observation.images.cam_high",
    "observation.images.cam_low",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _repo_name(repo_id: str) -> str:
    return repo_id.rstrip("/").split("/")[-1].replace("-", "_")


def _to_pil(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _data_files(repo_id: str, max_parquet_files: int) -> str | list[str]:
    if max_parquet_files <= 0:
        return f"hf://datasets/{repo_id}/data/*/*.parquet"
    files = sorted(path for path in HfApi().list_repo_files(repo_id, repo_type="dataset") if path.startswith("data/") and path.endswith(".parquet"))
    selected = files[:max_parquet_files]
    if not selected:
        raise RuntimeError(f"No parquet files found for {repo_id}")
    return [f"hf://datasets/{repo_id}/{path}" for path in selected]


def _load_rows(repo_id: str, limit: int, seed: int, max_parquet_files: int):
    ds = load_dataset("parquet", data_files=_data_files(repo_id, max_parquet_files), split="train")
    if limit > 0 and limit < len(ds):
        rng = random.Random(seed)
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        ds = ds.select(indices[:limit])
    return ds


def _camera_name(column: str) -> str:
    return column.rsplit(".", 1)[-1]


def _user_prompt(camera_columns: tuple[str, ...]) -> str:
    camera_names = ", ".join(_camera_name(column) for column in camera_columns)
    return f"Use all {len(camera_columns)} images in this order: {camera_names}. Classify the current scene."


def _write_row_images(
    row: dict[str, Any],
    *,
    repo_slug: str,
    row_idx: int,
    image_dir: Path,
    camera_columns: tuple[str, ...],
) -> list[str] | None:
    image_paths: list[str] = []
    for column in camera_columns:
        if column not in row:
            return None
        camera = _camera_name(column)
        path = image_dir / repo_slug / f"{row_idx:08d}_{camera}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        _to_pil(row[column]).save(path, quality=95)
        image_paths.append(str(path.resolve()))
    return image_paths


def _export_repos(
    *,
    repo_ids: list[str],
    output_jsonl: Path,
    image_dir: Path,
    camera_columns: tuple[str, ...],
    limit_per_repo: int,
    max_parquet_files_per_repo: int,
    seed: int,
    class_map: dict[int, tuple[str, str]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for repo_id in repo_ids:
            ds = _load_rows(repo_id, limit=limit_per_repo, seed=seed, max_parquet_files=max_parquet_files_per_repo)
            repo_slug = _repo_name(repo_id)
            task_mode = infer_task_mode_from_repo_id(repo_id)
            prompt = system_prompt(class_map, task_mode)
            user_prompt = f"{_user_prompt(camera_columns)} Known task mode: {describe_task_mode(task_mode)}."
            written = 0
            skipped = 0
            for row_idx, row in enumerate(ds):
                row_dict = dict(row)
                class_id = row_class_id(row_dict, class_map)
                if class_id is None or class_id not in class_map:
                    skipped += 1
                    continue
                image_paths = _write_row_images(
                    row_dict,
                    repo_slug=repo_slug,
                    row_idx=row_idx,
                    image_dir=image_dir,
                    camera_columns=camera_columns,
                )
                if image_paths is None:
                    skipped += 1
                    continue
                example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": prompt,
                        },
                        {
                            "role": "user",
                            "content": f"{'<image>' * len(image_paths)}\n{user_prompt}",
                        },
                        {
                            "role": "assistant",
                            "content": str(class_id),
                        },
                    ],
                    "images": image_paths,
                }
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                written += 1
                if written % 1000 == 0:
                    print(f"[{repo_id}] written={written} skipped={skipped}", flush=True)
            counts[repo_id] = written
            print(f"[{repo_id}] done written={written} skipped={skipped}", flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-repo-id", action="append", required=True)
    parser.add_argument("--val-repo-id", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("qwen_twist_swift_data"))
    parser.add_argument("--limit-per-train-repo", type=int, default=0)
    parser.add_argument("--limit-per-val-repo", type=int, default=0)
    parser.add_argument("--max-train-parquet-files-per-repo", type=int, default=0)
    parser.add_argument("--max-val-parquet-files-per-repo", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--camera-columns", nargs="+", default=list(DEFAULT_CAMERA_COLUMNS))
    parser.add_argument("--class-spec", type=Path, default=DEFAULT_CLASS_SPEC_PATH)
    args = parser.parse_args()

    class_map = load_class_map(args.class_spec)
    camera_columns = tuple(args.camera_columns)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_counts = _export_repos(
        repo_ids=args.train_repo_id,
        output_jsonl=args.output_dir / "train.jsonl",
        image_dir=args.output_dir / "images" / "train",
        camera_columns=camera_columns,
        limit_per_repo=args.limit_per_train_repo,
        max_parquet_files_per_repo=args.max_train_parquet_files_per_repo,
        seed=args.seed,
        class_map=class_map,
    )
    val_counts = {}
    if args.val_repo_id:
        val_counts = _export_repos(
            repo_ids=args.val_repo_id,
            output_jsonl=args.output_dir / "val.jsonl",
            image_dir=args.output_dir / "images" / "val",
            camera_columns=camera_columns,
            limit_per_repo=args.limit_per_val_repo,
            max_parquet_files_per_repo=args.max_val_parquet_files_per_repo,
            seed=args.seed,
            class_map=class_map,
        )
    metadata = {
        "camera_columns": list(camera_columns),
        "class_spec": str(args.class_spec),
        "class_map": {str(cid): {"bottle_state": state, "subtask": subtask} for cid, (state, subtask) in class_map.items()},
        "train_counts": train_counts,
        "val_counts": val_counts,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
