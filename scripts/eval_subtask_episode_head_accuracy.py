#!/usr/bin/env python3
"""Per-episode *first N* frames (full dataset FPS): subtask accuracy vs GT JSON.

Uses the same (bottle_state, subtask) string equality as training eval for canonical nine classes.
Also reports accuracy restricted to GT class_0 (rotate pair), which matters for episode starts.

Example (first 10 frames / ep):
  uv run scripts/eval_subtask_episode_head_accuracy.py \\
    --repo-id lyl472324464/2026-03-12-one-have-cap-direction \\
    --head-frames 10

Example (frames 20..39 within each episode, i.e. 0-based slice [20, 40)):
  uv run scripts/eval_subtask_episode_head_accuracy.py \\
    --repo-id lyl472324464/2026-03-12-one-have-cap-direction \\
    --ep-slice-start 20 --ep-slice-end-exclusive 40
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from openpi.evaluation import subtask_obs_utils as _obs
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training import subtask_eval as _subtask_eval
from openpi_client import subtask_parsing as _subtask_parsing


def _norm(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    return str(v).strip()


@dataclass
class Args:
    config_name: str = "twist_and_static_mixture_lora"
    checkpoint: Path = Path(
        "checkpoints/twist_and_static_mixture_lora/twist_static_mix_lora_eval1k_20260407/4000"
    )
    repo_id: str = "lyl472324464/2026-03-12-one-have-cap-direction"
    ep_slice_start: int = 0
    """0-based start index within each episode (rows sorted by time / global index)."""

    ep_slice_end_exclusive: int | None = None
    """If set, take rows ep_slice_start <= local_idx < ep_slice_end_exclusive (``head_frames`` ignored)."""

    head_frames: int = 10
    """If ``ep_slice_end_exclusive`` is None: take this many rows starting at ``ep_slice_start``."""

    temperature: float = 0.0
    batch_size: int = 8

    image_size: tuple[int, int] | None = None
    """Override ``data.image_size`` and ``model.image_resolution`` (H W). Use 224 224 for full-finetune ckpts trained at 224 when config now uses 448."""


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    if args.ep_slice_start < 0:
        raise SystemExit("--ep-slice-start must be >= 0")
    if args.ep_slice_end_exclusive is not None:
        if args.ep_slice_end_exclusive <= args.ep_slice_start:
            raise SystemExit("--ep-slice-end-exclusive must be greater than --ep-slice-start")
    elif args.head_frames <= 0:
        raise SystemExit("--head-frames must be positive when --ep-slice-end-exclusive is not set")

    train_cfg = _config.get_config(args.config_name)
    if args.image_size is not None:
        h, w = int(args.image_size[0]), int(args.image_size[1])
        if h <= 0 or w <= 0:
            raise SystemExit("--image-size heights/widths must be positive")
        new_data = dataclasses.replace(train_cfg.data, image_size=(h, w))
        new_model = dataclasses.replace(train_cfg.model, image_resolution=(h, w))
        train_cfg = dataclasses.replace(train_cfg, data=new_data, model=new_model)
        logging.info("Override image_size / image_resolution -> (%d, %d)", h, w)

    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    canonical = _subtask_eval.resolve_canonical_pairs(train_cfg)
    n_classes = len(canonical)

    ds = LeRobotDataset(
        args.repo_id,
        revision="main",
        force_cache_sync=False,
        download_videos=False,
        delta_timestamps=None,
    )
    ep_col = np.asarray(ds.hf_dataset["episode_index"], dtype=np.int64)
    episodes = sorted(int(x) for x in np.unique(ep_col))

    all_indices: list[int] = []
    for ep in episodes:
        gidx = np.sort(np.where(ep_col == ep)[0].astype(np.int64))
        if args.ep_slice_end_exclusive is not None:
            take = gidx[args.ep_slice_start : args.ep_slice_end_exclusive]
        else:
            take = gidx[args.ep_slice_start : args.ep_slice_start + args.head_frames]
        all_indices.extend(int(x) for x in take)

    if args.ep_slice_end_exclusive is not None:
        per = args.ep_slice_end_exclusive - args.ep_slice_start
        slice_desc = f"ep slice [{args.ep_slice_start}, {args.ep_slice_end_exclusive}) = {per} rows/ep"
    else:
        per = args.head_frames
        slice_desc = f"ep slice [{args.ep_slice_start}, {args.ep_slice_start + args.head_frames}) = {per} rows/ep"

    logging.info(
        "repo=%r episodes=%d | %s -> %d rows (dataset len=%d)",
        args.repo_id,
        len(episodes),
        slice_desc,
        len(all_indices),
        len(ds),
    )

    logging.info("Loading policy from %s", args.checkpoint)
    policy = _policy_config.create_trained_policy(
        train_cfg,
        args.checkpoint,
        repack_transforms=data_config.repack_transforms,
    )

    infer_bs = max(1, args.batch_size)
    load_chunk = max(infer_bs * 8, 256)
    preds: dict[int, tuple[str, str, str]] = {}

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
        for out, (_, gidx) in zip(infer_out, packed, strict=True):
            raw = str(out.get("subtask_text") or "")
            fields = _subtask_parsing.parse_structured_fields(raw)
            preds[gidx] = (_norm(fields.get("bottle_state")), _norm(fields.get("subtask")), raw[:400])

    logging.info("Inference done in %.1fs (%d rows)", time.perf_counter() - t0, len(all_indices))

    sub_only = ds.hf_dataset.select_columns(["subtask"])

    by_class_total: dict[int, int] = {i: 0 for i in range(n_classes)}
    by_class_correct: dict[int, int] = {i: 0 for i in range(n_classes)}
    correct = 0
    total_scored = 0

    rotate_correct = 0
    rotate_total = 0

    bad_json = 0
    non_canonical = 0

    pred_pair_all: Counter[tuple[str, str]] = Counter()
    pred_pair_rotate_gt: Counter[tuple[str, str]] = Counter()
    pred_class_all: Counter[str] = Counter()
    pred_class_rotate_gt: Counter[str] = Counter()

    for gidx in all_indices:
        raw_cell = _obs.subtask_cell_to_str(sub_only[int(gidx)]["subtask"])
        parsed = _obs.parse_json_bottle_state_subtask(raw_cell)
        if parsed is None:
            bad_json += 1
            continue
        gt_bs, gt_st = parsed
        cid = _subtask_eval.class_id_for_pair(gt_bs, gt_st, canonical)
        if cid is None:
            non_canonical += 1
            continue

        total_scored += 1
        by_class_total[cid] += 1
        pred_bs, pred_st, _raw = preds[gidx]
        pair = (pred_bs, pred_st)
        pred_pair_all[pair] += 1
        pid = _subtask_eval.class_id_for_pair(pred_bs, pred_st, canonical)
        pred_lbl = f"class_{pid}" if pid is not None else ("empty" if not pred_bs and not pred_st else "non_canonical")
        pred_class_all[pred_lbl] += 1

        ok = pred_bs == gt_bs.strip() and pred_st == gt_st.strip()
        if ok:
            correct += 1
            by_class_correct[cid] += 1

        if cid == 0:
            rotate_total += 1
            pred_pair_rotate_gt[pair] += 1
            pred_class_rotate_gt[pred_lbl] += 1
            if ok:
                rotate_correct += 1

    print("\n========== Episode slice subtask accuracy (full dataset FPS, per-episode time order) ==========")
    print(f"Repo: {args.repo_id}")
    print(f"Config: {args.config_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Per episode: {slice_desc}")
    print(f"Total rows: {len(all_indices)}")
    print(f"GT not parseable as JSON bottle_state+subtask: {bad_json}")
    print(f"GT parseable but not in nine canonical pairs: {non_canonical}")
    acc = correct / total_scored if total_scored else 0.0
    print(f"Nine-class eval rows: {total_scored}  |  Correct: {correct}  |  Accuracy: {100.0 * acc:.2f}%")
    r_acc = rotate_correct / rotate_total if rotate_total else 0.0
    print(
        f"GT class_0 only (rotate pair): {rotate_correct}/{rotate_total} = {100.0 * r_acc:.2f}% "
        f"({canonical[0][0]!r} -> {canonical[0][1]!r})"
    )
    print("Per-class (within slice rows, canonical GT only):")
    for c in range(n_classes):
        bs, st = canonical[c]
        tot = by_class_total[c]
        cc = by_class_correct[c]
        a = cc / tot if tot else 0.0
        print(f"  class_{c}: {cc}/{tot} = {100.0 * a:.2f}%  ({bs!r} -> {st!r})")

    def _print_pred_hist(title: str, pc: Counter[tuple[str, str]], limit: int) -> None:
        print(title)
        for (pbs, pst), n in pc.most_common(limit):
            if not pbs and not pst:
                print(f"  {n:4d}  (empty bottle_state and subtask after parse)")
            else:
                print(f"  {n:4d}  {pbs!r}  |  {pst!r}")

    _print_pred_hist("Predicted pair counts (all slice rows, canonical GT only):", pred_pair_all, 25)
    _print_pred_hist("Predicted pair counts (GT class_0 rotate only):", pred_pair_rotate_gt, 25)

    print("Predicted coarse bucket (parsed -> which canonical class or other):")
    for lbl, n in pred_class_all.most_common():
        print(f"  {n:4d}  {lbl}")
    print("Same buckets, GT class_0 only:")
    for lbl, n in pred_class_rotate_gt.most_common():
        print(f"  {n:4d}  {lbl}")

    print("================================================================================\n")


if __name__ == "__main__":
    main(tyro.cli(Args))
