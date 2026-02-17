#!/usr/bin/env python3
"""Measure PI05 GPU VRAM usage with one frame from an Aloha HDF5 episode."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

import h5py
import jax
import numpy as np
import torch
from PIL import Image, ImageFile

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


ImageFile.LOAD_TRUNCATED_IMAGES = True


def _bytes_to_rgb_image(buf: np.ndarray) -> np.ndarray:
    """Decode fixed-width jpeg bytes (possibly zero-padded) to HWC uint8 RGB."""
    img = Image.open(io.BytesIO(np.asarray(buf, dtype=np.uint8).tobytes())).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def load_aloha_frame(hdf5_path: Path, frame_idx: int) -> dict:
    with h5py.File(hdf5_path, "r") as f:
        qpos = np.asarray(f["observations/qpos"][frame_idx], dtype=np.float32)
        image_group = f["observations/images"]
        images = {cam: _bytes_to_rgb_image(image_group[cam][frame_idx]) for cam in image_group.keys()}

    return {
        "state": qpos,
        "images": images,
        "prompt": "measure vram",
    }


def mib(num_bytes: int) -> float:
    return num_bytes / (1024**2)


def query_pid_vram_mib(pid: int) -> int:
    """Read process GPU memory from nvidia-smi (MiB)."""
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
    except Exception:
        return 0
    total = 0
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            row_pid = int(parts[0])
            row_mem = int(parts[1])
        except ValueError:
            continue
        if row_pid == pid:
            total += row_mem
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=Path("/home/eii/learn/openpi0.5-rtc/episode_0.hdf5"),
        help="Path to episode hdf5 file.",
    )
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index to load.")
    parser.add_argument("--config", default="pi05_aloha", help="Train config name.")
    parser.add_argument(
        "--checkpoint",
        default="gs://openpi-assets/checkpoints/pi05_base",
        help="Checkpoint dir or gs path.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch CUDA device.")
    parser.add_argument("--runs", type=int, default=1, help="Number of inference runs.")
    args = parser.parse_args()

    if not args.hdf5.exists():
        raise FileNotFoundError(f"HDF5 not found: {args.hdf5}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot measure VRAM.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    obs = load_aloha_frame(args.hdf5, args.frame_idx)
    pid = os.getpid()
    vram_before = query_pid_vram_mib(pid)

    load_start = perf_counter()
    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config),
        args.checkpoint,
        pytorch_device=args.device,
    )
    torch.cuda.synchronize(device)
    load_ms = (perf_counter() - load_start) * 1000
    vram_after_load = query_pid_vram_mib(pid)

    model_alloc = torch.cuda.memory_allocated(device)
    model_reserved = torch.cuda.memory_reserved(device)
    model_peak = torch.cuda.max_memory_allocated(device)

    torch.cuda.reset_peak_memory_stats(device)
    infer_before = torch.cuda.memory_allocated(device)
    infer_start = perf_counter()
    result = None
    for _ in range(args.runs):
        result = policy.infer(obs, use_rtc=False)
    torch.cuda.synchronize(device)
    infer_ms = (perf_counter() - infer_start) * 1000 / args.runs
    vram_after_infer = query_pid_vram_mib(pid)
    infer_after = torch.cuda.memory_allocated(device)
    infer_peak = torch.cuda.max_memory_allocated(device)

    output = {
        "device": str(device),
        "jax_backend": jax.default_backend(),
        "hdf5": str(args.hdf5),
        "frame_idx": args.frame_idx,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "load_time_ms": round(load_ms, 2),
        "infer_avg_time_ms": round(infer_ms, 2),
        "model_memory": {
            "allocated_mib": round(mib(model_alloc), 2),
            "reserved_mib": round(mib(model_reserved), 2),
            "peak_allocated_during_load_mib": round(mib(model_peak), 2),
        },
        "inference_memory": {
            "allocated_before_mib": round(mib(infer_before), 2),
            "allocated_after_mib": round(mib(infer_after), 2),
            "peak_allocated_mib": round(mib(infer_peak), 2),
            "extra_peak_vs_before_mib": round(mib(max(infer_peak - infer_before, 0)), 2),
        },
        "nvidia_smi_memory": {
            "before_mib": vram_before,
            "after_load_mib": vram_after_load,
            "after_infer_mib": vram_after_infer,
            "load_delta_mib": max(vram_after_load - vram_before, 0),
            "infer_delta_mib": max(vram_after_infer - vram_after_load, 0),
            "total_delta_vs_before_mib": max(vram_after_infer - vram_before, 0),
        },
        "action_shape": list(np.asarray(result["actions"]).shape) if result is not None else None,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
