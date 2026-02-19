#!/usr/bin/env python3
"""Render a video with PI05 subtask text overlay for every frame in an HDF5 episode."""

from __future__ import annotations

import argparse
import gc
import io
from pathlib import Path

import cv2
import h5py
import jax
import numpy as np
from PIL import Image, ImageFile

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
import time


ImageFile.LOAD_TRUNCATED_IMAGES = True


def _decode_jpeg_bytes(buf: np.ndarray) -> np.ndarray:
    img = Image.open(io.BytesIO(np.asarray(buf, dtype=np.uint8).tobytes())).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _wrap_text(text: str, max_chars_per_line: int = 42) -> list[str]:
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    cur = words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) <= max_chars_per_line:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _draw_overlay(rgb: np.ndarray, subtask: str, frame_idx: int) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    lines = [f"frame: {frame_idx}", "sub task:"] + _wrap_text(subtask, max_chars_per_line=44)

    line_h = 28
    box_h = 16 + len(lines) * line_h
    box_w = min(w - 20, int(w * 0.95))
    overlay = bgr.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.50, bgr, 0.50, 0, bgr)

    y = 10 + 30
    for i, line in enumerate(lines):
        color = (0, 255, 255) if i <= 1 else (255, 255, 255)
        cv2.putText(bgr, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        y += line_h
    return bgr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, default=Path("episode_0.hdf5"))
    parser.add_argument("--output", type=Path, default=Path("episode_0_subtasks.mp4"))
    parser.add_argument("--prompt", type=str, default="twist off the bottle cap")
    parser.add_argument("--config", type=str, default="pi05_aloha_pen_uncap")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/20260219/1500/")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-text-token-id", type=int, default=240000)
    parser.add_argument("--decode-every", type=int, default=5, help="Run autoregressive subtask decoding every N frames.")
    parser.add_argument("--fps", type=float, default=50.0)
    args = parser.parse_args()

    def load_policy():
        return _policy_config.create_trained_policy(_config.get_config(args.config), args.checkpoint)

    policy = load_policy()

    with h5py.File(args.hdf5, "r") as f:
        qpos = f["observations/qpos"]
        cams = f["observations/images"]
        frame_count = int(qpos.shape[0])
        cam_keys = list(cams.keys())

        # Use first frame to initialize writer size.
        first_img = _decode_jpeg_bytes(cams["cam_high"][0])
        h, w = first_img.shape[:2]
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {args.output}")

        try:
            current_subtask = ""
            for i in range(frame_count):
                state = np.asarray(qpos[i], dtype=np.float32)
                images = {name: _decode_jpeg_bytes(cams[name][i]) for name in cam_keys}
                if i % args.decode_every == 0 or not current_subtask:
                    obs = {"state": state, "images": images, "prompt": args.prompt}
                    try:
                        time_start = time.time()
                        result = policy.infer_subtask(
                            obs,
                            temperature=args.temperature,
                            max_text_token_id=args.max_text_token_id,
                        )
                        time_end = time.time()
                        print(f"Time taken to infer subtask: {time_end - time_start} seconds", flush=True)
                        current_subtask = str(result["subtask_text"]).strip()
                    except Exception as e:
                        if "RESOURCE_EXHAUSTED" not in str(e) and "out of memory" not in str(e).lower():
                            raise
                        print(f"[warn] OOM at frame {i}, reloading policy and retrying...", flush=True)
                        del policy
                        gc.collect()
                        jax.clear_caches()
                        policy = load_policy()
                        result = policy.infer_subtask(
                            obs,
                            temperature=args.temperature,
                            max_text_token_id=args.max_text_token_id,
                        )
                        current_subtask = str(result["subtask_text"]).strip()

                frame_bgr = _draw_overlay(images["cam_high"], current_subtask, i)
                writer.write(frame_bgr)

                if (i + 1) % 20 == 0 or i + 1 == frame_count:
                    print(f"[{i + 1}/{frame_count}] subtask={current_subtask}", flush=True)
        finally:
            writer.release()

    print(f"Saved video: {args.output}")


if __name__ == "__main__":
    main()
