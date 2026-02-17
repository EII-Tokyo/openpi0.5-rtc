#!/usr/bin/env python3
"""Render one LeRobot episode with PI05 subtask overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import einops
import numpy as np
from lerobot.datasets import lerobot_dataset
import time

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def _to_uint8_hwc(img) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = einops.rearrange(arr, "c h w -> h w c")
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return arr


def _wrap_text(text: str, max_chars: int = 46) -> list[str]:
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    cur = words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _draw_overlay(rgb: np.ndarray, subtask: str, frame_idx: int, prompt: str) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lines = [f"frame: {frame_idx}", f"task: {prompt}"] + _wrap_text(f"sub task: {subtask}", max_chars=48)
    line_h = 28
    box_h = 16 + len(lines) * line_h
    overlay = bgr.copy()
    cv2.rectangle(overlay, (10, 10), (10 + int(bgr.shape[1] * 0.98), 10 + box_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.50, bgr, 0.50, 0, bgr)
    y = 38
    for i, line in enumerate(lines):
        color = (0, 255, 255) if i <= 1 else (255, 255, 255)
        cv2.putText(bgr, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        y += line_h
    return bgr


def _build_prompt(cleaned_text: str, state: np.ndarray) -> str:
    discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized_state))
    return f"Task: {cleaned_text}, State: {state_str}"


def _episode_bounds(dataset, episode_index: int) -> tuple[int, int]:
    start = None
    end = None
    for i in range(len(dataset)):
        ep = int(dataset[i]["episode_index"])
        if ep == episode_index and start is None:
            start = i
        elif ep != episode_index and start is not None:
            end = i
            break
    if start is None:
        raise ValueError(f"episode_index={episode_index} not found")
    if end is None:
        end = len(dataset)
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", type=str, default="lerobot/aloha_static_coffee")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="", help="If empty, use task text from dataset.")
    parser.add_argument("--config", type=str, default="pi05_aloha")
    parser.add_argument("--checkpoint", type=str, default="gs://openpi-assets/checkpoints/pi05_base")
    parser.add_argument("--decode-every", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-text-token-id", type=int, default=240000)
    parser.add_argument("--disable-max-text-token-id", action="store_true")
    parser.add_argument(
        "--debug-top-logits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print per-step top1/top2 logits and their gap during subtask decoding.",
    )
    parser.add_argument("--frame-limit", type=int, default=0, help="0 means full episode")
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    ds = lerobot_dataset.LeRobotDataset(args.repo_id)
    start, end = _episode_bounds(ds, args.episode_index)
    if args.frame_limit > 0:
        end = min(end, start + args.frame_limit)

    policy = _policy_config.create_trained_policy(_config.get_config(args.config), args.checkpoint)

    first = ds[start]
    first_img = _to_uint8_hwc(first["observation.images.cam_high"])
    h, w = first_img.shape[:2]
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer: {args.output}")

    prompt_text = args.prompt.strip()
    if not prompt_text:
        prompt_text = str(first.get("task", "")).strip()
    if not prompt_text:
        prompt_text = "manipulate coffee task"
    cleaned_text = prompt_text.replace("_", " ").replace("\n", " ").strip()

    current_subtask = ""
    current_prompt_for_overlay = ""
    total = end - start
    for idx, i in enumerate(range(start, end)):
        item = ds[i]
        images = {
            "cam_high": _to_uint8_hwc(item["observation.images.cam_high"]),
            "cam_left_wrist": _to_uint8_hwc(item["observation.images.cam_left_wrist"]),
            "cam_right_wrist": _to_uint8_hwc(item["observation.images.cam_right_wrist"]),
        }
        state = np.asarray(item["observation.state"], dtype=np.float32)

        if idx % args.decode_every == 0 or not current_subtask:
            full_prompt = _build_prompt(cleaned_text, state)
            current_prompt_for_overlay = full_prompt
            obs = {"state": state, "images": images, "prompt": cleaned_text}
            infer_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "debug_top_logits": args.debug_top_logits,
            }
            if not args.disable_max_text_token_id:
                infer_kwargs["max_text_token_id"] = args.max_text_token_id
            start_time = time.time()
            out = policy.infer_subtask(obs, **infer_kwargs)
            end_time = time.time()
            print(f"Time taken to infer subtask: {end_time - start_time} seconds", flush=True)
            current_subtask = str(out["subtask_text"]).strip()

        writer.write(_draw_overlay(images["cam_high"], current_subtask, idx, current_prompt_for_overlay))
        if (idx + 1) % 20 == 0 or idx + 1 == total:
            print(f"[{idx + 1}/{total}] subtask={current_subtask}", flush=True)

    writer.release()
    print(f"Saved video: {args.output}")


if __name__ == "__main__":
    main()
