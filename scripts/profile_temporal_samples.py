import dataclasses
import json
import pathlib
import statistics
import time
from collections import Counter

import numpy as np
import tyro

import lerobot.datasets.lerobot_dataset as lerobot_dataset
import lerobot.datasets.video_utils as video_utils
import openpi.training.data_loader as data_loader
import run_rinse_fullft_jax


@dataclasses.dataclass
class Args:
    repo_id: str = "lyl472324464/2026-04-21_direction-lerobot-with-rinse"
    start_index: int = 0
    num_indices: int = 64
    repeats: int = 2
    video_memory_num_frames: int = 6
    video_memory_stride_seconds: float = 5.0
    action_horizon: int = 50
    assets_base_dir: str = "/home/eii/openpi0.5-rtc/assets"
    checkpoint_base_dir: str = "/home/eii/openpi0.5-rtc/checkpoints"
    output_json: str = "/home/eii/openpi0.5-rtc/tmp/temporal_profile.json"


def _build_base_dataset_and_temporal(args: Args):
    cfg = run_rinse_fullft_jax.build_config(
        run_rinse_fullft_jax.Args(
            exp_name="profile_temporal_samples",
            batch_size=128,
            num_train_steps=1,
            num_workers=0,
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
    data_config = dataclasses.replace(cfg.data, repo_ids=[args.repo_id]).create(cfg.assets_dirs, cfg.model)
    fps_meta = lerobot_dataset.LeRobotDatasetMetadata(args.repo_id, force_cache_sync=True)
    delta_timestamps = {
        key: [t / fps_meta.fps for t in range(cfg.model.action_horizon)] for key in data_config.action_sequence_keys
    }
    base_dataset = lerobot_dataset.LeRobotDataset(args.repo_id, delta_timestamps=delta_timestamps)
    trainable_mask = data_loader.IsForTrainingWrapper._build_trainable_mask(base_dataset)
    temporal = data_loader.TemporalFrameStackDataset(
        base_dataset,
        fps=float(fps_meta.fps),
        num_frames=args.video_memory_num_frames,
        stride_seconds=args.video_memory_stride_seconds,
        trainable_mask=trainable_mask,
    )
    return base_dataset, temporal, trainable_mask


def _history_indices(temporal: data_loader.TemporalFrameStackDataset, idx: int) -> list[int]:
    episode_index = int(temporal._episode_indices[idx])
    frame_index = int(temporal._frame_indices[idx])
    episode_start_idx = idx - frame_index
    out = []
    for offset in reversed(range(temporal._num_frames)):
        target_frame = max(frame_index - offset * temporal._stride_frames, 0)
        candidate_idx = episode_start_idx + target_frame
        if int(temporal._episode_indices[candidate_idx]) != episode_index:
            raise AssertionError(f"Episode boundary crossed for idx={idx}, candidate_idx={candidate_idx}")
        out.append(int(candidate_idx))
    return out


def _profile_index(base_dataset, temporal, idx: int):
    decode_records = []
    original_decode = video_utils.decode_video_frames_torchcodec

    def _timed_decode(video_path, timestamps, tolerance_s, *args, **kwargs):
        start = time.perf_counter()
        frames = original_decode(video_path, timestamps, tolerance_s, *args, **kwargs)
        decode_records.append(
            {
                "video_path": str(video_path),
                "timestamps": [float(ts) for ts in timestamps],
                "elapsed_s": time.perf_counter() - start,
            }
        )
        return frames

    video_utils.decode_video_frames_torchcodec = _timed_decode
    try:
        history = _history_indices(temporal, idx)
        frame_cache = {}
        fetch_records = []
        total_fetch_start = time.perf_counter()
        for hist_idx in history:
            if hist_idx in frame_cache:
                continue
            start = time.perf_counter()
            frame_cache[hist_idx] = base_dataset[hist_idx]
            fetch_records.append(
                {
                    "hist_idx": hist_idx,
                    "elapsed_s": time.perf_counter() - start,
                    "episode_index": int(temporal._episode_indices[hist_idx]),
                    "frame_index": int(temporal._frame_indices[hist_idx]),
                }
            )
        fetch_total = time.perf_counter() - total_fetch_start

        current = frame_cache[idx]
        stack_start = time.perf_counter()
        for key in current:
            if key.startswith(temporal.IMAGE_PREFIX):
                _ = np.stack([np.asarray(frame_cache[hist_idx][key]) for hist_idx in history], axis=0)
        stack_total = time.perf_counter() - stack_start

        return {
            "idx": idx,
            "episode_index": int(temporal._episode_indices[idx]),
            "frame_index": int(temporal._frame_indices[idx]),
            "history_indices": history,
            "unique_history_indices": len(frame_cache),
            "fetch_total_s": fetch_total,
            "stack_total_s": stack_total,
            "sample_total_s": fetch_total + stack_total,
            "fetch_records": fetch_records,
            "decode_records": decode_records,
        }
    finally:
        video_utils.decode_video_frames_torchcodec = original_decode


def main() -> None:
    args = tyro.cli(Args)
    base_dataset, temporal, trainable_mask = _build_base_dataset_and_temporal(args)
    indices = [idx for idx in range(args.start_index, args.start_index + args.num_indices) if trainable_mask[idx]]

    runs = []
    for repeat in range(args.repeats):
        results = [_profile_index(base_dataset, temporal, idx) for idx in indices]
        runs.append(results)

    per_idx = []
    for pos, idx in enumerate(indices):
        samples = [run[pos] for run in runs]
        totals = [sample["sample_total_s"] for sample in samples]
        per_idx.append(
            {
                "idx": idx,
                "episode_index": samples[0]["episode_index"],
                "frame_index": samples[0]["frame_index"],
                "history_indices": samples[0]["history_indices"],
                "totals": totals,
                "mean_total_s": statistics.mean(totals),
                "fetch_totals": [sample["fetch_total_s"] for sample in samples],
                "stack_totals": [sample["stack_total_s"] for sample in samples],
                "decode_count": len(samples[0]["decode_records"]),
                "slowest_fetches": sorted(samples[0]["fetch_records"], key=lambda x: x["elapsed_s"], reverse=True)[:3],
                "slowest_decodes": sorted(samples[0]["decode_records"], key=lambda x: x["elapsed_s"], reverse=True)[:6],
            }
        )

    top = sorted(per_idx, key=lambda x: x["mean_total_s"], reverse=True)
    chunk_counter = Counter()
    for item in top[:10]:
        for rec in item["slowest_decodes"]:
            chunk_counter[pathlib.Path(rec["video_path"]).name] += 1

    summary = {
        "config": dataclasses.asdict(args),
        "num_profiled_indices": len(indices),
        "overall_mean_s": statistics.mean(item["mean_total_s"] for item in per_idx),
        "overall_median_s": statistics.median(item["mean_total_s"] for item in per_idx),
        "top_slowest": top[:10],
        "top_chunk_filenames": chunk_counter.most_common(10),
    }

    out_path = pathlib.Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print("[profile] output_json", str(out_path))
    print("[profile] overall_mean_s", summary["overall_mean_s"])
    print("[profile] overall_median_s", summary["overall_median_s"])
    print("[profile] top_chunk_filenames", summary["top_chunk_filenames"])
    for item in summary["top_slowest"][:5]:
        print(
            "[profile] slow",
            {
                "idx": item["idx"],
                "episode_index": item["episode_index"],
                "frame_index": item["frame_index"],
                "mean_total_s": item["mean_total_s"],
                "totals": item["totals"],
                "slowest_fetches": item["slowest_fetches"],
                "slowest_decode_paths": [
                    (pathlib.Path(rec["video_path"]).name, rec["elapsed_s"]) for rec in item["slowest_decodes"][:4]
                ],
            },
        )


if __name__ == "__main__":
    main()
