import argparse
import dataclasses
from pathlib import Path
import time

import cv2
import numpy as np

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _to_uint8(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if np.issubdtype(x.dtype, np.floating):
        x = ((x + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    else:
        x = x.astype(np.uint8)
    return x


def _camera_key(camera: str) -> str:
    camera_alias = {
        "cam_high": "base_0_rgb",
        "cam_left_wrist": "left_wrist_0_rgb",
        "cam_right_wrist": "right_wrist_0_rgb",
    }
    return camera_alias.get(camera, camera)


def _write_frames(writer: cv2.VideoWriter, frames: np.ndarray) -> None:
    b, h, w, c = frames.shape
    if c != 3:
        raise ValueError(f"Expected RGB 3-channel images, got shape {frames.shape}")
    for i in range(b):
        rgb = _to_uint8(frames[i])
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def run_video_export(args: argparse.Namespace) -> None:
    cfg = _config.get_config(args.config)
    cfg = dataclasses.replace(cfg, num_workers=args.num_workers, batch_size=args.batch_size)
    dataloader = _data_loader.create_data_loader(cfg, shuffle=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    writer = None
    total_frames = 0
    cam_key = _camera_key(args.camera)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.num_batches:
            break

        load_end = time.time()
        obs = batch[0]  # dataloader returns (Observation, actions)
        if cam_key not in obs.images:
            raise KeyError(f"Camera '{args.camera}' resolved to '{cam_key}', available keys: {list(obs.images.keys())}")

        frames = np.asarray(obs.images[cam_key])  # [B, H, W, C]
        if writer is None:
            h, w = frames.shape[1], frames.shape[2]
            writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open writer: {args.output}")
        _write_frames(writer, frames)
        total_frames += int(frames.shape[0])

        print(
            f"[batch {batch_idx}] load={load_end - start_time:.3f}s "
            f"frames={frames.shape[0]} frame_shape={tuple(frames.shape[1:])}",
            flush=True,
        )
        start_time = time.time()

    if writer is not None:
        writer.release()
    print(f"saved merged video: {args.output} (total_frames={total_frames})", flush=True)


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def run_benchmark(args: argparse.Namespace) -> None:
    workers_list = _parse_int_list(args.num_workers_list)
    batch_list = _parse_int_list(args.batch_size_list)

    print("workers,batch_size,batches,samples,total_s,avg_batch_s,batches_per_s,samples_per_s")
    for nw in workers_list:
        for bs in batch_list:
            cfg = _config.get_config(args.config)
            cfg = dataclasses.replace(cfg, num_workers=nw, batch_size=bs)
            dl = _data_loader.create_data_loader(cfg, shuffle=False)

            # warmup first batch to avoid including startup effects unless explicitly requested
            it = iter(dl)
            if args.warmup_batches > 0:
                for _ in range(args.warmup_batches):
                    _ = next(it)

            t0 = time.time()
            n_batches = 0
            n_samples = 0
            per_batch = []
            for _ in range(args.benchmark_batches):
                t_batch0 = time.time()
                batch = next(it)
                obs = batch[0]
                # count samples from state batch dim
                n = int(np.asarray(obs.state).shape[0])
                n_samples += n
                n_batches += 1
                per_batch.append(time.time() - t_batch0)
            total = time.time() - t0
            avg_batch = float(np.mean(per_batch)) if per_batch else 0.0
            bps = n_batches / total if total > 0 else 0.0
            sps = n_samples / total if total > 0 else 0.0
            print(f"{nw},{bs},{n_batches},{n_samples},{total:.4f},{avg_batch:.4f},{bps:.2f},{sps:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Video export or dataloader throughput benchmark.")
    parser.add_argument("--mode", choices=["video", "benchmark"], default="video")
    parser.add_argument("--config", type=str, default="pi05_aloha_pen_uncap")

    # shared-ish
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)

    # video mode args
    parser.add_argument("--num-batches", type=int, default=3, help="Number of batches to export in video mode.")
    parser.add_argument("--camera", type=str, default="base_0_rgb", help="Camera key in Observation.images")
    parser.add_argument("--fps", type=float, default=50.0, help="Output video FPS")
    parser.add_argument("--output", type=Path, default=Path("batch_videos/merged_batches.mp4"))

    # benchmark mode args
    parser.add_argument("--num-workers-list", type=str, default="0,2,4")
    parser.add_argument("--batch-size-list", type=str, default="16,32,64")
    parser.add_argument("--benchmark-batches", type=int, default=50)
    parser.add_argument("--warmup-batches", type=int, default=2)

    args = parser.parse_args()

    if args.mode == "video":
        run_video_export(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
