#!/usr/bin/env python3
"""Evaluate nine-class subtask accuracy on one LeRobot repo with fixed temporal subsampling + per-episode H.264.

- Subsample ``sample_hz`` frames per wall-clock second within each episode (stride ≈ round(fps / sample_hz)).
- Accuracy: only rows whose GT (bottle_state, subtask) matches one of the nine canonical pairs (same as training eval).
- Writes one libx264 MP4 per episode under ``--video-dir`` (requires ``ffmpeg`` on PATH).

Example:
  uv run scripts/eval_subtask_single_repo_5fps_episode_videos.py \\
    --config-name twist_and_static_mixture_lora \\
    --checkpoint checkpoints/twist_and_static_mixture_lora/twist_static_mix_lora_eval1k_20260407/4000 \\
    --repo-id lyl472324464/2026-03-12-one-have-cap-direction \\
    --sample-hz 5 \\
    --video-dir ./cap_direction_5fps_episode_h264 \\
    --batch-size 8
"""

from __future__ import annotations

import logging
import pickle
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from PIL import Image, ImageDraw, ImageFont

from openpi.evaluation import subtask_obs_utils as _obs
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training import subtask_eval as _subtask_eval
from openpi_client import subtask_parsing as _subtask_parsing

_QUAD_CAM_KEYS: tuple[str, ...] = (
    "cam_high",
    "cam_low",
    "cam_left_wrist",
    "cam_right_wrist",
)


def _array_to_rgb_uint8_hwc(arr: np.ndarray) -> np.ndarray | None:
    x = np.asarray(arr)
    if x.ndim != 3:
        return None
    if x.dtype in (np.float32, np.float64):
        x = np.clip(x, 0.0, 1.0)
        x = (x * 255.0).round().astype(np.uint8)
    else:
        x = x.astype(np.uint8, copy=False)
    c0, c1, c2 = x.shape[0], x.shape[1], x.shape[2]
    if c0 in (1, 3) and c0 <= c1 and c0 <= c2:
        x = np.transpose(x, (1, 2, 0))
    elif c2 not in (1, 3):
        return None
    if x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)
    return x


def _try_eval_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_pixel_width(draw: ImageDraw.ImageDraw, s: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), s, font=font)
    return max(1, bbox[2] - bbox[0])


def _break_long_unspaced(s: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    out: list[str] = []
    while s:
        if _text_pixel_width(draw, s, font) <= max_width:
            out.append(s)
            break
        lo, hi = 1, len(s)
        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if _text_pixel_width(draw, s[:mid], font) <= max_width:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        take = max(1, best)
        out.append(s[:take])
        s = s[take:]
    return out


def _wrap_paragraph_to_lines(para: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    para = para.rstrip()
    if not para:
        return [""]
    if " " not in para and "\t" not in para:
        return _break_long_unspaced(para, draw, font, max_width)
    words = para.split()
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = " ".join(cur + [w])
        if _text_pixel_width(draw, trial, font) <= max_width or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def _caption_to_draw_lines(caption: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    all_lines: list[str] = []
    for para in (caption or "").split("\n"):
        all_lines.extend(_wrap_paragraph_to_lines(para, draw, font, max_width))
    return all_lines if all_lines else [""]


def _build_quad_preview_image(
    obs: dict,
    *,
    caption_text: str,
    cell_size: int = 256,
    header_pad: int = 12,
    line_gap: int = 3,
) -> Image.Image:
    tiles: list[Image.Image] = []
    for name in _QUAD_CAM_KEYS:
        key = f"observation.images.{name}"
        raw = obs.get(key)
        if raw is None or isinstance(raw, dict):
            tile = Image.new("RGB", (cell_size, cell_size), (32, 32, 32))
        else:
            hwc = _array_to_rgb_uint8_hwc(np.asarray(raw))
            if hwc is None:
                tile = Image.new("RGB", (cell_size, cell_size), (32, 32, 32))
            else:
                tile = Image.fromarray(hwc).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
        tiles.append(tile)

    grid_w, grid_h = 2 * cell_size, 2 * cell_size
    grid = Image.new("RGB", (grid_w, grid_h))
    grid.paste(tiles[0], (0, 0))
    grid.paste(tiles[1], (cell_size, 0))
    grid.paste(tiles[2], (0, cell_size))
    grid.paste(tiles[3], (cell_size, cell_size))

    font = _try_eval_font(15)
    margin = header_pad
    max_text_w = grid_w - 2 * margin

    probe = ImageDraw.Draw(Image.new("RGB", (grid_w, 400)))
    draw_lines = _caption_to_draw_lines(caption_text, probe, font, max_text_w)
    line_h = max(
        probe.textbbox((0, 0), "Ay", font=font)[3] - probe.textbbox((0, 0), "Ay", font=font)[1],
        1,
    )
    header_h = margin + len(draw_lines) * (line_h + line_gap) + margin

    header = Image.new("RGB", (grid_w, header_h), (255, 255, 255))
    draw = ImageDraw.Draw(header)
    y = margin
    for ln in draw_lines:
        draw.text((margin, y), ln, fill=(0, 0, 0), font=font)
        y += line_h + line_gap

    out = Image.new("RGB", (grid_w, header_h + grid_h), (255, 255, 255))
    out.paste(header, (0, 0))
    out.paste(grid, (0, header_h))
    return out


def _write_h264_mp4(frames_rgb: list[np.ndarray], out_path: Path, *, fps: float) -> None:
    """Encode RGB uint8 frames to H.264 (yuv420p) via ffmpeg."""
    if not frames_rgb:
        raise ValueError("no frames")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH; required for libx264 export")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_rgb[0].shape[:2]
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    broken = False
    try:
        for fr in frames_rgb:
            if fr.shape[0] != h or fr.shape[1] != w or fr.shape[2] != 3:
                raise ValueError(f"frame shape mismatch: {fr.shape} vs ({h},{w},3)")
            try:
                proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
            except BrokenPipeError:
                broken = True
                break
    finally:
        proc.stdin.close()
    rc = proc.wait()
    err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    if broken or rc != 0:
        raise RuntimeError(
            f"ffmpeg failed (broken_pipe={broken}, rc={rc}). Last stderr:\n{err[-6000:]}"
        )


@dataclass
class Args:
    config_name: str = "twist_and_static_mixture_lora"
    checkpoint: Path = Path(
        "checkpoints/twist_and_static_mixture_lora/twist_static_mix_lora_eval1k_20260407/4000"
    )
    repo_id: str = "lyl472324464/2026-03-12-one-have-cap-direction"
    sample_hz: float = 5.0
    """Target number of frames per wall-clock second within each episode."""

    temperature: float = 0.0
    batch_size: int = 8
    video_dir: Path = Path("./subtask_eval_episode_h264")
    preds_pickle: Path | None = None
    """If set and file exists: skip model load + inference, load preds from this pickle. If set and file missing: infer then save before encoding videos."""


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    if args.sample_hz <= 0:
        raise SystemExit("--sample-hz must be positive")

    train_cfg = _config.get_config(args.config_name)
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    canonical = _subtask_eval.resolve_canonical_pairs(train_cfg)
    n_classes = len(canonical)

    meta = LeRobotDatasetMetadata(args.repo_id, revision="main", force_cache_sync=False)
    fps = float(meta.fps)
    stride = max(1, int(round(fps / args.sample_hz)))
    logging.info("repo=%r fps=%.4f sample_hz=%g -> stride=%d (~%.3f Hz)", args.repo_id, fps, args.sample_hz, stride, fps / stride)

    ds = LeRobotDataset(
        args.repo_id,
        revision="main",
        force_cache_sync=False,
        download_videos=False,
        delta_timestamps=None,
    )
    n = len(ds)
    ep_col = np.asarray(ds.hf_dataset["episode_index"], dtype=np.int64)

    # episode_index -> sorted global row indices
    episodes = sorted(int(x) for x in np.unique(ep_col))
    ep_to_indices: dict[int, list[int]] = {}
    for ep in episodes:
        gidx = np.where(ep_col == ep)[0]
        # parquet order should match time order; sort defensively
        gidx = np.sort(gidx.astype(np.int64))
        local = np.arange(len(gidx), dtype=np.int64)
        sel = local[::stride]
        ep_to_indices[ep] = [int(gidx[i]) for i in sel]

    all_indices: list[int] = []
    for ep in episodes:
        all_indices.extend(ep_to_indices[ep])
    logging.info("Episodes: %d | subsampled rows: %d (of %d)", len(episodes), len(all_indices), n)

    preds: dict[int, tuple[str, str, str]]
    if args.preds_pickle is not None and args.preds_pickle.exists():
        logging.info("Loading cached preds from %s", args.preds_pickle)
        preds = pickle.loads(args.preds_pickle.read_bytes())
        need = set(all_indices)
        have = set(preds.keys())
        if need != have:
            raise SystemExit(
                f"Preds pickle keys mismatch need={len(need)} have={len(have)} "
                f"(missing {len(need - have)}, extra {len(have - need)}). Remove the pickle or re-run without it."
            )
    else:
        logging.info("Loading policy from %s", args.checkpoint)
        policy = _policy_config.create_trained_policy(
            train_cfg,
            args.checkpoint,
            repack_transforms=data_config.repack_transforms,
        )

        infer_bs = max(1, args.batch_size)
        load_chunk = max(infer_bs * 8, 256)
        preds = {}

        t0 = time.perf_counter()
        for start in range(0, len(all_indices), load_chunk):
            chunk_idx = all_indices[start : start + load_chunk]
            packed: list[tuple[dict, int]] = []
            for gidx in chunk_idx:
                row = ds[int(gidx)]
                obs = _obs.lerobot_row_to_subtask_infer_obs(
                    {k: v for k, v in row.items()},
                    action_horizon=train_cfg.model.action_horizon,
                    pop_subtask=True,
                )
                packed.append((obs, int(gidx)))
            infer_out = policy.infer_subtask_batch([p[0] for p in packed], batch_size=infer_bs, temperature=args.temperature)
            if len(infer_out) != len(packed):
                raise RuntimeError(f"infer batch size mismatch: {len(infer_out)} vs {len(packed)}")

            def _norm(v: object) -> str:
                if v is None:
                    return ""
                if isinstance(v, str):
                    return v.strip()
                return str(v).strip()

            for out, (_, gidx) in zip(infer_out, packed, strict=True):
                raw = str(out.get("subtask_text") or "")
                fields = _subtask_parsing.parse_structured_fields(raw)
                pred_bs = _norm(fields.get("bottle_state"))
                pred_st = _norm(fields.get("subtask"))
                preds[gidx] = (pred_bs, pred_st, raw)

        logging.info("Inference done in %.1fs (%d rows)", time.perf_counter() - t0, len(all_indices))
        if args.preds_pickle is not None:
            args.preds_pickle.parent.mkdir(parents=True, exist_ok=True)
            args.preds_pickle.write_bytes(pickle.dumps(preds))
            logging.info("Wrote preds cache %s", args.preds_pickle)

    sub_only = ds.hf_dataset.select_columns(["subtask"])

    by_class_total: dict[int, int] = {i: 0 for i in range(n_classes)}
    by_class_correct: dict[int, int] = {i: 0 for i in range(n_classes)}
    correct = 0
    total_scored = 0

    for gidx in all_indices:
        raw_cell = _obs.subtask_cell_to_str(sub_only[int(gidx)]["subtask"])
        parsed = _obs.parse_json_bottle_state_subtask(raw_cell)
        if parsed is None:
            continue
        gt_bs, gt_st = parsed
        cid = _subtask_eval.class_id_for_pair(gt_bs, gt_st, canonical)
        if cid is None:
            continue
        total_scored += 1
        by_class_total[cid] += 1
        pred_bs, pred_st, _ = preds[gidx]
        ok = pred_bs == gt_bs.strip() and pred_st == gt_st.strip()
        if ok:
            correct += 1
            by_class_correct[cid] += 1

    acc = correct / total_scored if total_scored else 0.0
    print("\n========== Single-repo subtask eval (nine classes, temporal subsample) ==========")
    print(f"Repo: {args.repo_id}")
    print(f"Config: {args.config_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Sample ~{args.sample_hz} Hz (stride={stride}, dataset fps={fps})")
    print(f"Scored frames (canonical GT): {total_scored}  |  Correct: {correct}  |  Accuracy: {100.0 * acc:.2f}%")
    for c in range(n_classes):
        bs, st = canonical[c]
        tot = by_class_total[c]
        cc = by_class_correct[c]
        a = cc / tot if tot else 0.0
        print(f"  class_{c} ({bs!r} -> {st!r}): {cc}/{tot} = {100.0 * a:.2f}%")
    print("================================================================================\n")

    # --- per-episode H.264 (quad + caption) ---
    args.video_dir.mkdir(parents=True, exist_ok=True)
    video_fps = float(fps / stride)
    logging.info("Writing episode videos to %s at %.4f fps", args.video_dir.resolve(), video_fps)

    for ep in episodes:
        frames_pil: list[Image.Image] = []
        for gidx in ep_to_indices[ep]:
            row = ds[int(gidx)]
            raw_cell = _obs.subtask_cell_to_str(sub_only[int(gidx)]["subtask"])
            parsed = _obs.parse_json_bottle_state_subtask(raw_cell)
            gt_bs, gt_st = parsed if parsed else ("", "")
            pred_bs, pred_st, raw = preds[gidx]
            summary_gt = f"{gt_bs} | {gt_st}" if parsed else "(non-json or missing)"
            summary_pred = f"{pred_bs} | {pred_st}"
            obs = _obs.lerobot_row_to_subtask_infer_obs(
                {k: v for k, v in row.items()},
                action_horizon=train_cfg.model.action_horizon,
                pop_subtask=True,
            )
            caption_lines = [
                f"repo_id={args.repo_id}",
                f"episode={ep}",
                f"frame_idx={gidx}",
                f"gt={summary_gt}",
                f"pred={summary_pred}",
                f"raw_subtask_text={raw[:800]}",
            ]
            frames_pil.append(_build_quad_preview_image(obs, caption_text="\n".join(caption_lines)))

        max_h = max(im.size[1] for im in frames_pil)
        w0 = frames_pil[0].size[0]
        # libx264 + yuv420p requires even width and height
        w0 = (w0 + 1) // 2 * 2
        max_h = (max_h + 1) // 2 * 2
        frames_rgb: list[np.ndarray] = []
        for im in frames_pil:
            w, h = im.size
            if w != frames_pil[0].size[0]:
                raise RuntimeError(f"Inconsistent quad width within episode {ep}: {w} vs {frames_pil[0].size[0]}")
            canvas = Image.new("RGB", (w0, max_h), (255, 255, 255))
            canvas.paste(im, (0, 0))
            frames_rgb.append(np.asarray(canvas, dtype=np.uint8))

        safe_repo = args.repo_id.replace("/", "__")
        out_mp4 = args.video_dir / f"{safe_repo}__ep_{ep:05d}.mp4"
        _write_h264_mp4(frames_rgb, out_mp4, fps=video_fps)
        logging.info("Wrote %s (%d frames)", out_mp4, len(frames_rgb))

    logging.info("Done.")


if __name__ == "__main__":
    main(tyro.cli(Args))
