#!/usr/bin/env python3
"""LeRobot Hub dataset → subsampled frames → PI05 ``infer_subtask_batch`` → quad MP4 (H.264).

One video per ``episode_index``, same 2×2 layout + caption as ``eval_subtask_accuracy_train.py``.
Uses each row's ``task`` for the policy unless ``--prompt-override`` is set.

Example:
  uv run python scripts/lerobot_subtask_quad_video.py \\
    --repo-id lyl472324464/2026-03-12-one-have-cap-direction \\
    --output-dir ./one_have_cap_videos
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tyro
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

from openpi.evaluation import subtask_obs_utils as _obs
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def _load_eval_quad_builder():
    path = Path(__file__).resolve().parent / "eval_subtask_accuracy_train.py"
    spec = importlib.util.spec_from_file_location("eval_subtask_accuracy_train", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod._build_quad_preview_image


_BUILD_QUAD = _load_eval_quad_builder()

_QUAD_MODEL_KEYS: tuple[str, ...] = (
    "cam_high",
    "cam_low",
    "cam_left_wrist",
    "cam_right_wrist",
)


def _reencode_mp4_h264(path: Path) -> None:
    import subprocess

    tmp = path.with_name(path.name + ".h264.tmp.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(tmp),
            ],
            check=True,
        )
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _dummy_obs_for_layout() -> dict:
    return {f"observation.images.{k}": np.zeros((3, 256, 256), dtype=np.uint8) for k in _QUAD_MODEL_KEYS}


def _episode_bounds(dataset: LeRobotDataset, episode_index: int) -> tuple[int, int]:
    ep_col = dataset.hf_dataset.data.column("episode_index").combine_chunks()
    ep_arr = ep_col.to_numpy(zero_copy_only=False)
    if ep_arr.size > 1 and np.all(ep_arr[1:] >= ep_arr[:-1]):
        start = int(np.searchsorted(ep_arr, episode_index, side="left"))
        end = int(np.searchsorted(ep_arr, episode_index, side="right"))
        if start == end:
            raise ValueError(f"episode_index={episode_index} not found")
        return start, end
    match = np.flatnonzero(ep_arr == episode_index)
    if match.size == 0:
        raise ValueError(f"episode_index={episode_index} not found")
    return int(match[0]), int(match[-1] + 1)


def _unique_episode_indices(dataset: LeRobotDataset) -> list[int]:
    col = dataset.hf_dataset.data.column("episode_index").combine_chunks()
    arr = col.to_numpy(zero_copy_only=False)
    return sorted(np.unique(arr).astype(int).tolist())


def _row_to_infer_obs(
    row: dict,
    *,
    action_horizon: int,
    prompt_override: str | None,
) -> dict:
    d = {k: v for k, v in row.items()}
    if prompt_override is not None:
        d["task"] = prompt_override
    return _obs.lerobot_row_to_subtask_infer_obs(d, action_horizon=action_horizon, pop_subtask=True)


@dataclass
class Args:
    repo_id: str = "lyl472324464/2026-03-12-one-have-cap-direction"
    """LeRobot Hub ``repo_id``."""

    output_dir: Path = Path("lerobot_subtask_videos")
    """Output directory (one ``episode_{idx}_subtask_overlay.mp4`` per episode)."""

    revision: str = "main"
    """Dataset revision on the Hub."""

    config_name: str = "twist_and_static_mixture_full_finetune"
    checkpoint: Path = Path(
        "checkpoints/twist_and_static_mixture_full_finetune/"
        "twist_and_static_mixture_full_finetune_vast_20260405_100600/39999"
    )

    source_fps: float = 50.0
    target_fps: float = 5.0
    """Temporal subsample: about ``round(source_fps / target_fps)`` dataset rows per step."""

    batch_size: int = 8
    temperature: float = 0.0

    prompt_override: str | None = None
    """If set, replaces every row's ``task`` before inference (otherwise use dataset ``task``)."""

    episode_indices: tuple[int, ...] | None = None
    """If set, only these ``episode_index`` values; otherwise all episodes in the table."""

    skip_ffmpeg_h264: bool = False


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    out_root = args.output_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    step = max(1, int(round(args.source_fps / args.target_fps)))
    logging.info(
        "Opening %r (revision=%r) | subsample every %d row(s) (~%.2f Hz → %.2f Hz)",
        args.repo_id,
        args.revision,
        step,
        args.source_fps,
        args.target_fps,
    )

    ds = LeRobotDataset(
        args.repo_id,
        revision=args.revision,
        force_cache_sync=False,
        download_videos=False,
        delta_timestamps=None,
    )

    if args.episode_indices is not None:
        episode_list = list(args.episode_indices)
    else:
        episode_list = _unique_episode_indices(ds)
    if not episode_list:
        raise SystemExit("No episodes found in dataset.")

    logging.info("Episodes to process: %d — %s", len(episode_list), episode_list)

    train_cfg = _config.get_config(args.config_name)
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    logging.info("Loading policy from %s", args.checkpoint)
    policy = _policy_config.create_trained_policy(
        train_cfg,
        args.checkpoint,
        repack_transforms=data_config.repack_transforms,
    )
    ah = train_cfg.model.action_horizon
    infer_bs = max(1, args.batch_size)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dummy = _dummy_obs_for_layout()

    for ep_idx in episode_list:
        try:
            start, end = _episode_bounds(ds, ep_idx)
        except ValueError as e:
            logging.warning("Skip episode_index=%s: %s", ep_idx, e)
            continue

        sampled = list(range(start, end, step))
        if not sampled:
            logging.warning("episode_index=%s: empty after subsample, skip.", ep_idx)
            continue

        logging.info(
            "episode_index=%s: dataset rows [%d, %d), %d subsampled frame(s) — infer …",
            ep_idx,
            start,
            end,
            len(sampled),
        )

        texts_ep: list[str] = []
        for s in range(0, len(sampled), infer_bs):
            chunk_i = sampled[s : s + infer_bs]
            obs_list = [
                _row_to_infer_obs(
                    dict(ds[i].items()),
                    action_horizon=ah,
                    prompt_override=args.prompt_override,
                )
                for i in chunk_i
            ]
            outs = policy.infer_subtask_batch(
                obs_list,
                batch_size=infer_bs,
                temperature=args.temperature,
            )
            if len(outs) != len(chunk_i):
                raise RuntimeError(f"infer_subtask_batch: expected {len(chunk_i)} outputs, got {len(outs)}")
            for out in outs:
                texts_ep.append(str(out.get("subtask_text") or ""))

        captions_ep: list[str] = []
        for global_i, subtask_text in zip(sampled, texts_ep, strict=True):
            raw = subtask_text[:500]
            frame_in_ep = global_i - start
            captions_ep.append(
                "\n".join(
                    [
                        f"repo_id={args.repo_id}",
                        f"episode_index={ep_idx}",
                        f"dataset_index={global_i}",
                        f"frame_in_episode={frame_in_ep}",
                        f"subtask_text={subtask_text}",
                        f"raw_subtask_text={raw}",
                    ]
                )
            )

        max_w, max_h = 0, 0
        for cap in captions_ep:
            sz = _BUILD_QUAD(dummy, caption_text=cap).size
            max_w = max(max_w, sz[0])
            max_h = max(max_h, sz[1])

        safe_repo = args.repo_id.replace("/", "__")
        out_path = out_root / f"{safe_repo}__ep{ep_idx:03d}_subtask_overlay.mp4"
        writer = cv2.VideoWriter(str(out_path), fourcc, float(args.target_fps), (max_w, max_h))
        if not writer.isOpened():
            raise SystemExit(f"Failed to open VideoWriter for {out_path}")

        try:
            for global_i, cap in zip(sampled, captions_ep, strict=True):
                obs = _row_to_infer_obs(
                    dict(ds[global_i].items()),
                    action_horizon=ah,
                    prompt_override=args.prompt_override,
                )
                quad = _BUILD_QUAD(obs, caption_text=cap)
                canvas = Image.new("RGB", (max_w, max_h), (255, 255, 255))
                canvas.paste(quad, (0, 0))
                bgr = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
                writer.write(bgr)
        finally:
            writer.release()

        if not args.skip_ffmpeg_h264:
            logging.info("Re-encoding %s to H.264 …", out_path.name)
            _reencode_mp4_h264(out_path)

        logging.info("Wrote %s (%d frames, %dx%d)", out_path, len(sampled), max_w, max_h)


if __name__ == "__main__":
    try:
        main(tyro.cli(Args))
    except KeyboardInterrupt:
        sys.exit(130)
