#!/usr/bin/env python3
"""Export a dataloader-aligned video with prompt and is_for_training overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as _transforms


def _to_uint8_hwc(img) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return arr


def _wrap_text(text: str, max_chars: int = 56) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    cur = words[0]
    for word in words[1:]:
        if len(cur) + 1 + len(word) <= max_chars:
            cur += " " + word
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def _draw_overlay(rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    line_h = 24
    box_h = 16 + len(lines) * line_h
    overlay = bgr.copy()
    cv2.rectangle(overlay, (10, 10), (bgr.shape[1] - 10, 10 + box_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.45, bgr, 0.55, 0, bgr)
    y = 34
    for idx, line in enumerate(lines):
        color = (0, 255, 255) if idx < 3 else (255, 255, 255)
        cv2.putText(bgr, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)
        y += line_h
    return bgr


def _episode_bounds(dataset, episode_index: int) -> tuple[int, int]:
    ep_col = dataset.hf_dataset.data.column("episode_index").combine_chunks()
    ep_arr = ep_col.to_numpy(zero_copy_only=False)
    start = int(np.searchsorted(ep_arr, episode_index, side="left"))
    end = int(np.searchsorted(ep_arr, episode_index, side="right"))
    if start == end:
        raise ValueError(f"episode_index={episode_index} not found")
    return start, end


def _build_visual_transform(data_config, image_size: tuple[int, int]):
    image_height, image_width = image_size
    transforms = [
        _transforms.PromptFromLeRobotTask(),
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(data_config.norm_stats or {}, use_quantiles=data_config.use_quantile_norm),
        _transforms.ResizeImages(image_height, image_width),
    ]
    return _transforms.compose(transforms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="twist_off_the_bottle_cap_subtask_lora")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--frame-limit", type=int, default=0, help="0 means full episode")
    args = parser.parse_args()

    train_config = _config.get_config(args.config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = _data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)

    if not isinstance(dataset, _data_loader.IsForTrainingWrapper):
        raise TypeError("Expected create_torch_dataset() to return IsForTrainingWrapper")

    raw_dataset = dataset._dataset
    start, end = _episode_bounds(raw_dataset, args.episode_index)
    if args.frame_limit > 0:
        end = min(end, start + args.frame_limit)

    image_size = getattr(train_config.data, "image_size", (224, 224))
    visual_transform = _build_visual_transform(data_config, image_size)

    first_raw = raw_dataset[start]
    first_sample = visual_transform(first_raw)
    first_img = _to_uint8_hwc(first_sample["image"]["base_0_rgb"])
    h, w = first_img.shape[:2]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer: {args.output}")

    for global_idx in range(start, end):
        raw_item = raw_dataset[global_idx]
        sample = visual_transform(raw_item)
        rgb = _to_uint8_hwc(sample["image"]["base_0_rgb"])

        prompt = str(sample.get("prompt", ""))
        subtask = str(sample.get("subtask", ""))
        is_for_training = bool(dataset._trainable_mask[global_idx])
        lines = [
            f"config: {args.config}",
            f"global_idx: {global_idx}  episode_index: {int(raw_item['episode_index'])}",
            f"is_for_training: {is_for_training}  image_size: {rgb.shape[1]}x{rgb.shape[0]}",
        ]
        lines.extend(_wrap_text(f"prompt: {prompt}", max_chars=58))
        lines.extend(_wrap_text(f"subtask: {subtask}", max_chars=58))
        writer.write(_draw_overlay(rgb, lines))

    writer.release()
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
