#!/usr/bin/env python3
"""Evaluate high-level subtask accuracy on LeRobot data (same transforms as TrainConfig).

Sampling: scans every twist repo and every frame to build per-class pools, then draws
``max_per_class`` frames per class uniformly at random (so early repos do not dominate).

Default: nine (bottle_state, subtask) classes matching runtime / training holdout.
Legacy mode: filter by subtask string only (e.g. two-class twist smoke test).

Example:
  uv run scripts/eval_subtask_accuracy_train.py \\
    --config-name twist_and_static_mixture_full_finetune \\
    --checkpoint checkpoints/.../39999 \\
    --max-per-class 300 \\
    --batch-size 8 \\
    --save-wrong-predictions-dir /tmp/wrong \\
    --save-correct-predictions-dir /tmp/correct \\
    --save-correct-fraction 0.1
"""

from __future__ import annotations

import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image, ImageDraw, ImageFont

from openpi.evaluation import subtask_obs_utils as _obs

# Fixed order for 2×2 quad (matches Aloha repack camera names).
_QUAD_CAM_KEYS: tuple[str, ...] = (
    "cam_high",
    "cam_low",
    "cam_left_wrist",
    "cam_right_wrist",
)
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training import subtask_eval as _subtask_eval
from openpi_client import subtask_parsing as _subtask_parsing


def _repo_has_subtask_column(
    repo_id: str,
    *,
    probe_index: int,
    probe_total: int,
) -> bool:
    """Open LeRobot once to see if the HF table has a ``subtask`` column (can hang on network/cache)."""
    logging.info(
        "Subtask probe %d/%d: opening %r (Hub may print Fetching…; first open per repo can take minutes)",
        probe_index,
        probe_total,
        repo_id,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    t0 = time.perf_counter()
    ds = LeRobotDataset(
        repo_id,
        revision="main",
        force_cache_sync=False,
        download_videos=False,
        delta_timestamps=None,
    )
    ok = "subtask" in ds.hf_dataset.column_names
    logging.info(
        "Subtask probe %d/%d: %r finished in %.1fs, has_subtask_column=%s",
        probe_index,
        probe_total,
        repo_id,
        time.perf_counter() - t0,
        ok,
    )
    sys.stdout.flush()
    return ok


def _open_lerobot(repo_id: str) -> LeRobotDataset:
    return LeRobotDataset(
        repo_id,
        revision="main",
        force_cache_sync=False,
        download_videos=False,
        delta_timestamps=None,
    )


def _subsample_pool_indices(rng: np.random.Generator, pool_len: int, cap: int) -> np.ndarray:
    """Uniform subset without replacement; cap <= 0 means use all."""
    if pool_len <= 0:
        return np.array([], dtype=np.int64)
    if cap <= 0 or pool_len <= cap:
        return np.arange(pool_len, dtype=np.int64)
    return rng.permutation(pool_len)[:cap]


def _build_nine_class_pools(
    twist_repos: list[str],
    canonical: tuple[tuple[str, str], ...],
) -> dict[int, list[tuple[str, int, str, str]]]:
    """Scan every repo and every frame; pool by canonical class id (metadata only)."""
    n_classes = len(canonical)
    pools: dict[int, list[tuple[str, int, str, str]]] = {i: [] for i in range(n_classes)}
    for ri, repo_id in enumerate(twist_repos):
        logging.info("Nine-class pool scan: repo %d/%d %r", ri + 1, len(twist_repos), repo_id)
        ds = _open_lerobot(repo_id)
        n = len(ds)
        sub_only = ds.hf_dataset.select_columns(["subtask"])
        for idx in range(n):
            if idx > 0 and idx % 50_000 == 0:
                logging.info("  %r: scanned %d / %d rows …", repo_id, idx, n)
            parsed = _obs.parse_json_bottle_state_subtask(
                _obs.subtask_cell_to_str(sub_only[int(idx)]["subtask"])
            )
            if parsed is None:
                continue
            bs, st = parsed
            cid = _subtask_eval.class_id_for_pair(bs, st, canonical)
            if cid is None:
                continue
            pools[cid].append((repo_id, int(idx), bs, st))
    return pools


def _build_legacy_subtask_pools(
    twist_repos: list[str],
    targets: frozenset[str],
) -> dict[str, list[tuple[str, int]]]:
    pools: dict[str, list[tuple[str, int]]] = {t: [] for t in targets}
    for ri, repo_id in enumerate(twist_repos):
        logging.info(
            "Legacy pool scan: repo %d/%d %r",
            ri + 1,
            len(twist_repos),
            repo_id,
        )
        ds = _open_lerobot(repo_id)
        n = len(ds)
        sub_only = ds.hf_dataset.select_columns(["subtask"])
        for idx in range(n):
            if idx > 0 and idx % 50_000 == 0:
                logging.info("  %r: scanned %d / %d rows …", repo_id, idx, n)
            parsed = _obs.parse_json_bottle_state_subtask(
                _obs.subtask_cell_to_str(sub_only[int(idx)]["subtask"])
            )
            gt = parsed[1] if parsed else ""
            if gt not in targets:
                continue
            pools[gt].append((repo_id, int(idx)))
    return pools


def _safe_filename_part(s: str, max_len: int = 96) -> str:
    s = s.replace("\n", " ").strip()[:max_len]
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s) or "na"


def _array_to_rgb_uint8_hwc(arr: np.ndarray) -> np.ndarray | None:
    """LeRobot / policy raw obs: CHW or HWC, uint8 or float in [0,1]. Returns HWC RGB uint8 or None."""
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


def _extract_raw_subtask_from_extra(extra_meta_lines: list[str]) -> str:
    for line in extra_meta_lines:
        if line.startswith("raw_subtask_text="):
            return line.split("=", 1)[1].strip()
    return ""


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
    """2×2 camera grid + white strip with full caption (same fields as sidecar .txt)."""
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


def _save_eval_obs_bundle(
    out_dir: Path,
    counter: list[int],
    *,
    kind: str,
    repo_id: str,
    frame_idx: int,
    summary_gt: str,
    summary_pred: str,
    obs: dict,
    extra_meta_lines: list[str],
    model_raw_text: str = "",
) -> None:
    """Write one txt + one quad PNG (2×2 cams + white header with model text)."""
    counter[0] += 1
    seq = counter[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{kind}_{seq:05d}"
    repo_safe = _safe_filename_part(repo_id.replace("/", "__"), max_len=120)
    lines = [
        f"repo_id={repo_id}",
        f"frame_idx={frame_idx}",
        f"gt={summary_gt}",
        f"pred={summary_pred}",
        *extra_meta_lines,
    ]
    (out_dir / f"{prefix}__{repo_safe}__f{frame_idx}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    raw_from_extra = _extract_raw_subtask_from_extra(extra_meta_lines)
    full_raw = (model_raw_text or raw_from_extra).strip()
    caption_lines: list[str] = [
        f"repo_id={repo_id}",
        f"frame_idx={frame_idx}",
        f"gt={summary_gt}",
        f"pred={summary_pred}",
    ]
    for el in extra_meta_lines:
        if el.startswith("raw_subtask_text="):
            caption_lines.append(f"raw_subtask_text={full_raw}")
        else:
            caption_lines.append(el)
    if not any(x.startswith("raw_subtask_text=") for x in caption_lines):
        caption_lines.append(f"raw_subtask_text={full_raw}")
    caption_text = "\n".join(caption_lines)
    quad = _build_quad_preview_image(obs, caption_text=caption_text)
    quad_path = out_dir / f"{prefix}__{repo_safe}__f{frame_idx}__quad.png"
    quad.save(quad_path)


@dataclass
class Args:
    config_name: str = "twist_and_static_mixture_full_finetune"
    checkpoint: Path = Path(
        "checkpoints/twist_and_static_mixture_full_finetune/"
        "twist_and_static_mixture_full_finetune_vast_20260405_100600/39999"
    )
    max_per_class: int = 400
    """Max frames per class. Use 0 for no cap (very slow)."""

    seed: int = 0
    temperature: float = 0.0
    batch_size: int = 8
    legacy_subtask_only: bool = False
    """If true, evaluate only JSON `subtask` against --legacy-targets (ignores bottle_state)."""
    legacy_targets: tuple[str, ...] = (
        "Rotate so opening faces right",
        "Pick up with left hand",
    )
    """Used only when --legacy-subtask-only."""

    save_wrong_predictions_dir: Path | None = None
    """If set, save camera images + small txt for each incorrect prediction (legacy and nine-class)."""

    save_correct_predictions_dir: Path | None = None
    """If set, randomly save this fraction of *correct* predictions (same image layout as wrong)."""

    save_correct_fraction: float = 0.1
    """Used only when ``save_correct_predictions_dir`` is set (e.g. 0.1 ≈ one tenth)."""


def main(args: Args) -> None:
    # JAX/openpi may configure logging on import; force so probe / pool progress is visible.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    train_cfg = _config.get_config(args.config_name)
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    repo_ids = data_config.repo_ids
    if not repo_ids:
        raise SystemExit("Train config has no repo_ids")

    logging.info("Will probe %d repo_id(s) from TrainConfig for a ``subtask`` column.", len(repo_ids))
    twist_repos: list[str] = []
    for i, r in enumerate(repo_ids):
        if _repo_has_subtask_column(r, probe_index=i + 1, probe_total=len(repo_ids)):
            twist_repos.append(r)
    logging.info("Repos with subtask column: %d / %d", len(twist_repos), len(repo_ids))

    logging.info(
        "=== Subtask probes done. Next: load checkpoint + JIT (often silent for several minutes; not more Fetching) ==="
    )
    sys.stdout.flush()
    logging.info("Loading policy from %s", args.checkpoint)
    policy = _policy_config.create_trained_policy(
        train_cfg,
        args.checkpoint,
        repack_transforms=data_config.repack_transforms,
    )

    rng = np.random.default_rng(args.seed)
    per_class_cap = args.max_per_class
    unlimited = per_class_cap <= 0

    if args.legacy_subtask_only:
        _run_legacy_subtask_only(args, train_cfg, twist_repos, policy, rng, unlimited, per_class_cap)
        return

    canonical = _subtask_eval.resolve_canonical_pairs(train_cfg)
    n_classes = len(canonical)
    logging.info("Scanning all twist repos to build per-class pools (full pass) …")
    pools = _build_nine_class_pools(twist_repos, canonical)
    pool_sizes = {c: len(pools[c]) for c in range(n_classes)}
    logging.info("Pool sizes by class_id (before cap): %s", pool_sizes)

    collected: list[tuple[str, int, int, str, str]] = []
    counts: dict[int, int] = {i: 0 for i in range(n_classes)}
    for c in range(n_classes):
        pool = pools[c]
        if not pool:
            logging.warning("class_id %d: empty pool", c)
            continue
        sel = _subsample_pool_indices(rng, len(pool), 0 if unlimited else per_class_cap)
        for j in sel:
            repo_id, idx, bs, st = pool[int(j)]
            collected.append((repo_id, int(idx), c, bs, st))
            counts[c] += 1

    order = rng.permutation(len(collected))
    collected = [collected[int(i)] for i in order]

    logging.info("Collected samples by class_id: %s (total %d)", counts, len(collected))
    if not collected:
        raise SystemExit("No matching frames for nine canonical classes; check repos / labels.")

    t0 = time.perf_counter()
    ds_cache: dict[str, LeRobotDataset] = {}
    infer_bs = max(1, args.batch_size)
    # Load rows in chunks so we never hold all frames' images in RAM at once.
    load_chunk = max(infer_bs * 8, 256)
    n_total = len(collected)
    logging.info(
        "Running infer_subtask_batch in chunks of %d rows, infer batch_size=%d (%d observations)",
        load_chunk,
        infer_bs,
        n_total,
    )

    by_class_correct: dict[int, int] = {i: 0 for i in range(n_classes)}
    by_class_total: dict[int, int] = {i: 0 for i in range(n_classes)}
    correct = 0
    processed = 0
    wrong_counter: list[int] = [0]
    wrong_dir = args.save_wrong_predictions_dir
    correct_counter: list[int] = [0]
    correct_dir = args.save_correct_predictions_dir
    correct_frac = (
        float(min(1.0, max(0.0, args.save_correct_fraction))) if correct_dir is not None else 0.0
    )

    def _norm(v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return v.strip()
        return str(v).strip()

    for start in range(0, n_total, load_chunk):
        sl = collected[start : start + load_chunk]
        packed: list[tuple[dict, str, int, str, str, int]] = []
        for repo_id, idx, cid, gt_bs, gt_st in sl:
            if repo_id not in ds_cache:
                ds_cache[repo_id] = _open_lerobot(repo_id)
            ds = ds_cache[repo_id]
            row = ds[idx]
            obs = _obs.lerobot_row_to_subtask_infer_obs(
                {k: v for k, v in row.items()},
                action_horizon=train_cfg.model.action_horizon,
                pop_subtask=True,
            )
            packed.append((obs, repo_id, idx, gt_bs, gt_st, cid))
        infer_out = policy.infer_subtask_batch(
            [p[0] for p in packed],
            batch_size=infer_bs,
            temperature=args.temperature,
        )
        if len(infer_out) != len(packed):
            raise RuntimeError(f"Chunk: expected {len(packed)} outputs, got {len(infer_out)}")
        for out, (obs, repo_id, idx, gt_bs, gt_st, cid) in zip(infer_out, packed, strict=True):
            fields = _subtask_parsing.parse_structured_fields(str(out.get("subtask_text") or ""))
            pred_bs = _norm(fields.get("bottle_state"))
            pred_st = _norm(fields.get("subtask"))
            ok = pred_bs == gt_bs.strip() and pred_st == gt_st.strip()
            by_class_total[cid] += 1
            if ok:
                correct += 1
                by_class_correct[cid] += 1
                if correct_dir is not None and correct_frac > 0.0 and rng.random() < correct_frac:
                    raw = str(out.get("subtask_text") or "")
                    _save_eval_obs_bundle(
                        correct_dir,
                        correct_counter,
                        kind="correct",
                        repo_id=repo_id,
                        frame_idx=int(idx),
                        summary_gt=f"{gt_bs.strip()} | {gt_st.strip()}",
                        summary_pred=f"{pred_bs} | {pred_st}",
                        obs=obs,
                        extra_meta_lines=[
                            f"class_id={cid}",
                            f"raw_subtask_text={raw[:500]}",
                        ],
                        model_raw_text=raw,
                    )
            elif wrong_dir is not None:
                raw = str(out.get("subtask_text") or "")
                _save_eval_obs_bundle(
                    wrong_dir,
                    wrong_counter,
                    kind="wrong",
                    repo_id=repo_id,
                    frame_idx=int(idx),
                    summary_gt=f"{gt_bs.strip()} | {gt_st.strip()}",
                    summary_pred=f"{pred_bs} | {pred_st}",
                    obs=obs,
                    extra_meta_lines=[
                        f"class_id={cid}",
                        f"raw_subtask_text={raw[:500]}",
                    ],
                    model_raw_text=raw,
                )
        processed += len(sl)
        if processed % 2000 == 0 or processed >= n_total:
            logging.info("Processed %d / %d (%.1f s)", processed, n_total, time.perf_counter() - t0)

    elapsed = time.perf_counter() - t0
    n_eval = sum(by_class_total.values())
    acc = correct / n_eval if n_eval else 0.0
    print("\n========== Subtask eval: nine (bottle_state, subtask) classes ==========")
    print(f"Config: {args.config_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {n_eval}  |  Correct: {correct}  |  Accuracy: {100.0 * acc:.2f}%")
    print(f"Batch size: {args.batch_size}")
    print(f"Wall time: {elapsed:.1f}s ({elapsed / max(n_eval, 1) * 1000:.0f} ms / sample)")
    for c in range(n_classes):
        bs, st = canonical[c]
        tot = by_class_total[c]
        cc = by_class_correct[c]
        a = cc / tot if tot else 0.0
        print(f"  class_{c} ({bs!r} -> {st!r}): {cc}/{tot} = {100.0 * a:.2f}%")
    if wrong_dir is not None:
        print(f"Wrong predictions: {wrong_counter[0]} sample(s) -> {wrong_dir.resolve()}")
    if correct_dir is not None and correct_frac > 0.0:
        print(
            f"Correct sample dump (~{100.0 * correct_frac:.0f}% random): "
            f"{correct_counter[0]} sample(s) -> {correct_dir.resolve()}"
        )
    print("==========================================================================\n")


def _run_legacy_subtask_only(
    args: Args,
    train_cfg: _config.TrainConfig,
    twist_repos: list[str],
    policy: object,
    rng: np.random.Generator,
    unlimited: bool,
    per_class_cap: int,
) -> None:
    targets = frozenset(args.legacy_targets)
    logging.info("Scanning all twist repos for legacy targets (full pass) …")
    pools = _build_legacy_subtask_pools(twist_repos, targets)
    for t in sorted(targets):
        logging.info("Legacy pool %r: %d frames", t, len(pools[t]))

    collected: list[tuple[str, int, str]] = []
    counts: dict[str, int] = {t: 0 for t in targets}
    for t in sorted(targets):
        pool = pools[t]
        if not pool:
            logging.warning("Legacy target %r: empty pool", t)
            continue
        sel = _subsample_pool_indices(rng, len(pool), 0 if unlimited else per_class_cap)
        for j in sel:
            repo_id, idx = pool[int(j)]
            collected.append((repo_id, int(idx), t))
            counts[t] += 1

    order = rng.permutation(len(collected))
    collected = [collected[int(i)] for i in order]

    logging.info("Legacy collected: %s (total %d)", counts, len(collected))
    if not collected:
        raise SystemExit("No matching frames for legacy targets.")

    t0 = time.perf_counter()
    ds_cache: dict[str, LeRobotDataset] = {}
    infer_bs = max(1, args.batch_size)
    load_chunk = max(infer_bs * 8, 256)
    n_total = len(collected)
    logging.info(
        "Legacy infer in chunks of %d rows, infer batch_size=%d (%d observations)",
        load_chunk,
        infer_bs,
        n_total,
    )

    correct = 0
    by_class_correct: dict[str, int] = {t: 0 for t in targets}
    by_class_total: dict[str, int] = {t: 0 for t in targets}
    processed = 0
    wrong_counter: list[int] = [0]
    wrong_dir = args.save_wrong_predictions_dir
    correct_counter: list[int] = [0]
    correct_dir = args.save_correct_predictions_dir
    correct_frac = (
        float(min(1.0, max(0.0, args.save_correct_fraction))) if correct_dir is not None else 0.0
    )

    for start in range(0, n_total, load_chunk):
        sl = collected[start : start + load_chunk]
        packed: list[tuple[dict, str, int, str]] = []
        for repo_id, idx, gt in sl:
            if repo_id not in ds_cache:
                ds_cache[repo_id] = _open_lerobot(repo_id)
            row = ds_cache[repo_id][idx]
            obs = _obs.lerobot_row_to_subtask_infer_obs(
                {k: v for k, v in row.items()},
                action_horizon=train_cfg.model.action_horizon,
                pop_subtask=True,
            )
            packed.append((obs, repo_id, idx, gt))

        infer_out = policy.infer_subtask_batch(
            [p[0] for p in packed],
            batch_size=infer_bs,
            temperature=args.temperature,
        )
        if len(infer_out) != len(packed):
            raise RuntimeError(f"Chunk: expected {len(packed)} outputs, got {len(infer_out)}")
        for out, (obs, repo_id, idx, gt) in zip(infer_out, packed, strict=True):
            parsed = _subtask_parsing.parse_structured_fields(str(out.get("subtask_text") or ""))
            pred_st = parsed.get("subtask")
            pred = pred_st.strip() if isinstance(pred_st, str) else ""
            ok = pred == gt
            by_class_total[gt] += 1
            if ok:
                correct += 1
                by_class_correct[gt] += 1
                if correct_dir is not None and correct_frac > 0.0 and rng.random() < correct_frac:
                    raw = str(out.get("subtask_text") or "")
                    _save_eval_obs_bundle(
                        correct_dir,
                        correct_counter,
                        kind="correct",
                        repo_id=repo_id,
                        frame_idx=int(idx),
                        summary_gt=gt,
                        summary_pred=pred or "(empty)",
                        obs=obs,
                        extra_meta_lines=[f"raw_subtask_text={raw[:500]}"],
                        model_raw_text=raw,
                    )
            elif wrong_dir is not None:
                raw = str(out.get("subtask_text") or "")
                _save_eval_obs_bundle(
                    wrong_dir,
                    wrong_counter,
                    kind="wrong",
                    repo_id=repo_id,
                    frame_idx=int(idx),
                    summary_gt=gt,
                    summary_pred=pred or "(empty)",
                    obs=obs,
                    extra_meta_lines=[f"raw_subtask_text={raw[:500]}"],
                    model_raw_text=raw,
                )
        processed += len(sl)
        if processed % 2000 == 0 or processed >= n_total:
            logging.info(
                "Legacy infer %d / %d (%.1f s)",
                processed,
                n_total,
                time.perf_counter() - t0,
            )

    elapsed = time.perf_counter() - t0
    n_eval = sum(by_class_total.values())
    acc = correct / n_eval if n_eval else 0.0
    print("\n========== Legacy: JSON subtask field only ==========")
    print(f"Targets: {sorted(targets)}")
    print(f"Overall: {correct}/{n_eval} = {100.0 * acc:.2f}%  (wall {elapsed:.1f}s)")
    for t in sorted(targets):
        tot = by_class_total[t]
        cc = by_class_correct[t]
        a = cc / tot if tot else 0.0
        print(f"  {t!r}: {cc}/{tot} = {100.0 * a:.2f}%")
    if wrong_dir is not None:
        print(f"Wrong predictions: {wrong_counter[0]} sample(s) -> {wrong_dir.resolve()}")
    if correct_dir is not None and correct_frac > 0.0:
        print(
            f"Correct sample dump (~{100.0 * correct_frac:.0f}% random): "
            f"{correct_counter[0]} sample(s) -> {correct_dir.resolve()}"
        )
    print()


if __name__ == "__main__":
    try:
        main(tyro.cli(Args))
    except KeyboardInterrupt:
        sys.exit(130)
