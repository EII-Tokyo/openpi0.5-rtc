#!/usr/bin/env python3
"""Render DROID trajectory video with subtask text overlay using PI05."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.shared import download as _download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


def _wrap_text(text: str, max_chars_per_line: int = 52) -> list[str]:
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
    lines = [f"frame: {frame_idx}", "sub task:"] + _wrap_text(subtask, max_chars_per_line=54)

    line_h = 28
    box_h = 16 + len(lines) * line_h
    box_w = min(w - 20, int(w * 0.96))
    overlay = bgr.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.50, bgr, 0.50, 0, bgr)

    y = 10 + 30
    for i, line in enumerate(lines):
        color = (0, 255, 255) if i <= 1 else (255, 255, 255)
        cv2.putText(bgr, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        y += line_h
    return bgr


def _build_prompt(cleaned_text: str, state: np.ndarray, action_mode: str, rng: np.random.Generator) -> str:
    # Keep the same prompt formatting convention as Pi05 tokenizer state-conditioning.
    discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized_state))

    if "[bad action]" in cleaned_text:
        cleaned_text = cleaned_text.replace("[bad action] ", "")
        return f"Task: {cleaned_text}, State: {state_str}"

    if action_mode == "bad":
        return f"Task: {cleaned_text}, State: {state_str}"
    if action_mode == "good":
        return f"Task: {cleaned_text}, State: {state_str}"
    if action_mode == "action":
        return f"Task: {cleaned_text}, State: {state_str}"

    # random mode: 20% good action, otherwise action.
    _ = rng.random()
    return f"Task: {cleaned_text}, State: {state_str}"


def _find_metadata_file(episode_dir: Path) -> Path:
    files = sorted(episode_dir.glob("metadata_*.json"))
    if not files:
        raise FileNotFoundError(f"No metadata_*.json found in {episode_dir}")
    return files[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="clean the bedroom")
    parser.add_argument("--config", type=str, default="pi05_droid")
    parser.add_argument("--checkpoint", type=str, default="gs://openpi-assets/checkpoints/pi05_base")
    parser.add_argument(
        "--norm-stats-checkpoint",
        type=str,
        default="gs://openpi-assets/checkpoints/pi05_droid",
        help="Checkpoint directory used only to load droid norm_stats.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-text-token-id", type=int, default=240000)
    parser.add_argument("--decode-every", type=int, default=5)
    parser.add_argument("--action-mode", choices=("random", "action", "good", "bad"), default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-ext2", action="store_true")
    parser.add_argument("--fps", type=float, default=0.0, help="0 means read from source video metadata.")
    parser.add_argument("--frame-limit", type=int, default=0, help="0 means full trajectory.")
    args = parser.parse_args()

    episode_dir = args.episode_dir.resolve()
    trajectory_path = episode_dir / "trajectory.h5"
    meta_path = _find_metadata_file(episode_dir)
    meta = json.loads(meta_path.read_text())

    ext_serial = meta["ext2_cam_serial"] if args.use_ext2 else meta["ext1_cam_serial"]
    wrist_serial = meta["wrist_cam_serial"]
    ext_mp4 = episode_dir / "recordings" / "MP4" / f"{ext_serial}.mp4"
    wrist_mp4 = episode_dir / "recordings" / "MP4" / f"{wrist_serial}.mp4"
    if not ext_mp4.exists() or not wrist_mp4.exists():
        raise FileNotFoundError(f"Missing mp4s: {ext_mp4}, {wrist_mp4}")

    train_cfg = _config.get_config(args.config)
    # We already inject discretized state into prompt text manually; disable automatic state-token injection.
    train_cfg = dataclasses.replace(train_cfg, model=dataclasses.replace(train_cfg.model, discrete_state_input=False))
    norm_assets_dir = _download.maybe_download(args.norm_stats_checkpoint)
    norm_stats = _checkpoints.load_norm_stats(Path(norm_assets_dir) / "assets", "droid")
    policy = _policy_config.create_trained_policy(train_cfg, args.checkpoint, norm_stats=norm_stats)

    rng = np.random.default_rng(args.seed)
    cleaned_text = args.prompt.strip().replace("_", " ").replace("\n", " ")

    with h5py.File(trajectory_path, "r") as f:
        joint = np.asarray(f["observation/robot_state/joint_positions"], dtype=np.float32)
        grip = np.asarray(f["observation/robot_state/gripper_position"], dtype=np.float32)

    cap_ext = cv2.VideoCapture(str(ext_mp4))
    cap_wrist = cv2.VideoCapture(str(wrist_mp4))
    if not cap_ext.isOpened() or not cap_wrist.isOpened():
        raise RuntimeError("Failed to open source mp4 files.")

    src_fps = cap_ext.get(cv2.CAP_PROP_FPS)
    out_fps = args.fps if args.fps > 0 else (src_fps if src_fps > 0 else 15.0)
    width = int(cap_ext.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_ext.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video: {args.output}")

    total = int(min(len(joint), len(grip)))
    if args.frame_limit > 0:
        total = min(total, args.frame_limit)

    current_subtask = ""
    try:
        for i in range(total):
            ok_ext, frame_ext_bgr = cap_ext.read()
            ok_wrist, frame_wrist_bgr = cap_wrist.read()
            if not ok_ext or not ok_wrist:
                print(f"[warn] video ended early at frame {i}")
                break

            frame_ext = cv2.cvtColor(frame_ext_bgr, cv2.COLOR_BGR2RGB)
            frame_wrist = cv2.cvtColor(frame_wrist_bgr, cv2.COLOR_BGR2RGB)
            state = np.concatenate([joint[i], np.asarray([grip[i]], dtype=np.float32)], axis=0)

            if i % args.decode_every == 0 or not current_subtask:
                full_prompt = _build_prompt(cleaned_text, state, args.action_mode, rng)
                obs = {
                    "observation/exterior_image_1_left": frame_ext,
                    "observation/wrist_image_left": frame_wrist,
                    "observation/joint_position": joint[i],
                    "observation/gripper_position": np.asarray([grip[i]], dtype=np.float32),
                    "prompt": full_prompt,
                }
                result = policy.infer_subtask(
                    obs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    max_text_token_id=args.max_text_token_id,
                )
                current_subtask = str(result["subtask_text"]).strip()

            writer.write(_draw_overlay(frame_ext, current_subtask, i))
            if (i + 1) % 20 == 0 or i + 1 == total:
                print(f"[{i + 1}/{total}] subtask={current_subtask}", flush=True)
    finally:
        cap_ext.release()
        cap_wrist.release()
        writer.release()

    print(f"Saved video: {args.output}")


if __name__ == "__main__":
    main()
