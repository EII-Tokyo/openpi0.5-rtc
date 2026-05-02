import dataclasses
import os
import statistics
import time
from typing import Optional
import ast

import numpy as np
import torch
import tyro

import openpi.training.data_loader as data_loader
import run_rinse_fullft_jax


@dataclasses.dataclass
class Args:
    repo_id: str = "lyl472324464/2026-04-21_direction-lerobot-with-rinse"
    repo_ids: Optional[str] = None
    batch_size: int = 128
    num_workers: int = 4
    prefetch_factor: int = 2
    timed_batches: int = 8
    warmup_batches: int = 2
    video_memory_num_frames: int = 6
    video_memory_stride_seconds: float = 5.0
    assets_base_dir: str = "/home/eii/openpi0.5-rtc/assets"
    checkpoint_base_dir: str = "/home/eii/openpi0.5-rtc/checkpoints"


class TimedIndexDataset:
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        start = time.perf_counter()
        sample = self._dataset[index]
        elapsed = time.perf_counter() - start
        return {
            "idx": int(index),
            "elapsed_s": float(elapsed),
            "worker_pid": int(os.getpid()),
            "sample": sample,
        }


def _collate(items):
    idxs = [item["idx"] for item in items]
    elapsed = [item["elapsed_s"] for item in items]
    pids = [item["worker_pid"] for item in items]
    return {
        "idxs": idxs,
        "elapsed_s": elapsed,
        "worker_pids": pids,
    }


def main() -> None:
    args = tyro.cli(Args)
    cfg = run_rinse_fullft_jax.build_config(
        run_rinse_fullft_jax.Args(
            exp_name="profile_temporal_dataloader_batches",
            batch_size=args.batch_size,
            num_train_steps=1,
            num_workers=args.num_workers,
            checkpoint_base_dir=args.checkpoint_base_dir,
            assets_base_dir=args.assets_base_dir,
            wandb_enabled=False,
            overwrite=True,
            resume=False,
            fsdp_devices=1,
            video_memory_num_frames=args.video_memory_num_frames,
            video_memory_stride_seconds=args.video_memory_stride_seconds,
        )
    )
    repo_ids = [args.repo_id]
    if args.repo_ids:
        text = args.repo_ids.strip()
        if text.startswith("["):
            parsed = ast.literal_eval(text)
            if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
                raise ValueError("--repo-ids must be a Python/JSON-style list of strings")
            repo_ids = parsed
        else:
            repo_ids = [part.strip() for part in text.split(",") if part.strip()]
        if not repo_ids:
            raise ValueError("--repo-ids parsed to an empty list")
    data_config = dataclasses.replace(cfg.data, repo_ids=repo_ids).create(cfg.assets_dirs, cfg.model)
    base = data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    dataset = TimedIndexDataset(base)

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        multiprocessing_context=torch.multiprocessing.get_context("spawn"),
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor,
        collate_fn=_collate,
        worker_init_fn=data_loader._WorkerInitFn(cfg.seed),
        drop_last=True,
        generator=generator,
    )

    iterator = iter(loader)
    totals = []
    for batch_idx in range(args.warmup_batches + args.timed_batches):
        start = time.perf_counter()
        batch = next(iterator)
        batch_elapsed = time.perf_counter() - start
        totals.append(batch_elapsed)
        if batch_idx >= args.warmup_batches:
            print(
                "[batch]",
                {
                    "batch_idx": batch_idx,
                    "batch_elapsed_s": batch_elapsed,
                    "item_mean_s": statistics.mean(batch["elapsed_s"]),
                    "item_max_s": max(batch["elapsed_s"]),
                    "slowest_items": sorted(
                        [
                            {"idx": idx, "elapsed_s": elapsed, "worker_pid": pid}
                            for idx, elapsed, pid in zip(batch["idxs"], batch["elapsed_s"], batch["worker_pids"], strict=True)
                        ],
                        key=lambda x: x["elapsed_s"],
                        reverse=True,
                    )[:8],
                },
            )
    timed = totals[args.warmup_batches :]
    print(
        "[summary]",
        {
            "batch_times_s": timed,
            "mean_s": statistics.mean(timed),
            "median_s": statistics.median(timed),
            "min_s": min(timed),
            "max_s": max(timed),
        },
    )


if __name__ == "__main__":
    main()
