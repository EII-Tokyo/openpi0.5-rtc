#!/usr/bin/env python3
"""Evaluate one or more checkpoints on a fixed twist nine-class manifest."""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

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
    config_name: str = "twist_only_lora_triplet_10k"
    manifest: Path = Path("assets/twist_nine_class_eval_manifest.json")
    checkpoints: tuple[Path, ...] = ()
    checkpoint_parent: Path | None = None
    batch_size: int = 8
    temperature: float = 0.0
    image_size: tuple[int, int] | None = None
    output_json: Path | None = None
    semantic_normalize: bool = False


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_checkpoints(args: Args) -> list[Path]:
    ckpts = list(args.checkpoints)
    if args.checkpoint_parent is not None:
        for p in sorted(args.checkpoint_parent.iterdir()):
            if p.is_dir() and p.name.isdigit():
                ckpts.append(p)
    # preserve order, dedupe
    out: list[Path] = []
    seen: set[Path] = set()
    for p in ckpts:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    if not out:
        raise SystemExit("No checkpoints specified.")
    return out


def _canon_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").strip().lower()).strip()


def _semantic_normalize_pair(raw_text: str) -> tuple[str, str] | None:
    text = _canon_text(raw_text)
    if not text:
        return None

    # Return/home
    if (
        "return to the initial pose" in text
        or "return to initial pose" in text
        or "return to the starting position" in text
        or "return to the start position" in text
        or "go back to the initial pose" in text
        or "move back to the initial pose" in text
        or "return to home" in text
    ):
        return ("No bottle on table", "Return to initial pose")

    # Rotate
    if (
        "rotate" in text
        or "turn the bottle so the opening faces right" in text
        or "opening faces right" in text and "pick up" not in text
    ):
        return ("Bottle on table, opening faces left", "Rotate so opening faces right")

    # Pick up bottle
    if (
        "pick up with left hand" in text
        or "take the bottle with the left hand" in text
        or "take the bottle with left hand" in text
        or "take bottle with left hand" in text
        or "use left hand to pick up bottle" in text
        or "take hold of the bottle with left hand" in text
    ):
        return ("Bottle on table, opening faces right", "Pick up with left hand")

    # Unscrew cap
    if (
        "unscrew cap" in text
        or "unscrew the cap" in text
        or "take off the cap by twisting" in text
        or "open the bottle by unscrewing cap" in text
        or "use left hand to unscrew the cap" in text
        or "take the cap off" in text
    ):
        return ("Bottle in left hand and capped", "Unscrew cap")

    # Bottle + cap into bins
    if (
        ("bottle" in text and "left trash bin" in text and "cap" in text and "right trash bin" in text)
        or "throw bottle left and cap right" in text
        or "put each item in its matching trash bin" in text
        or "move the bottle left and cap right into trash" in text
    ):
        return ("Bottle in left hand, cap removed, and cap in right hand", "Bottle to left trash bin, cap to right trash bin")

    # Bottle only into left bin
    if (
        "bottle to left trash bin" in text
        or ("bottle" in text and "left trash bin" in text and "cap" not in text)
        or "put the bottle in the left trash bin" in text
    ):
        return ("Bottle in left hand, cap removed, and cap not in right hand", "Bottle to left trash bin")

    # Stuck bottle rescue
    if (
        "use right hand to remove and place into left trash bin" in text
        or ("right hand" in text and "remove" in text and "left trash bin" in text)
    ):
        return ("Bottle stuck in left hand", "Use right hand to remove and place into left trash bin")

    # Cap pickup
    if (
        "pick up cap and place into right trash bin" in text
        or ("cap" in text and "right trash bin" in text and "pick up" in text)
    ):
        return ("Cap on table", "Pick up cap and place into right trash bin")

    return None


def _prediction_pair(raw_text: str, *, semantic_normalize: bool) -> tuple[str, str]:
    fields = _subtask_parsing.parse_structured_fields(str(raw_text or ""))
    pred_bs = _norm(fields.get("bottle_state"))
    pred_st = _norm(fields.get("subtask"))
    if pred_bs and pred_st:
        return pred_bs, pred_st
    if semantic_normalize:
        normalized = _semantic_normalize_pair(str(raw_text or ""))
        if normalized is not None:
            return normalized
    return pred_bs, pred_st


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    manifest = _load_manifest(args.manifest)
    rows = manifest["rows"]
    train_cfg = _config.get_config(args.config_name)
    if args.image_size is not None:
        h, w = int(args.image_size[0]), int(args.image_size[1])
        train_cfg = dataclasses.replace(
            train_cfg,
            data=dataclasses.replace(train_cfg.data, image_size=(h, w)),
            model=dataclasses.replace(train_cfg.model, image_resolution=(h, w)),
        )
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    canonical = _subtask_eval.resolve_canonical_pairs(train_cfg)
    ckpts = _collect_checkpoints(args)

    ds_cache: dict[str, LeRobotDataset] = {}

    all_results: list[dict] = []
    for ckpt in ckpts:
        logging.info("Evaluating checkpoint: %s", ckpt)
        policy = _policy_config.create_trained_policy(
            train_cfg,
            ckpt,
            repack_transforms=data_cfg.repack_transforms,
        )
        t0 = time.perf_counter()

        total = 0
        correct = 0
        per_class_total = Counter()
        per_class_correct = Counter()
        pred_class_counts = Counter()

        load_chunk = max(256, args.batch_size * 8)
        for start in range(0, len(rows), load_chunk):
            row_chunk = rows[start : start + load_chunk]
            obs_chunk: list[dict] = []
            for row in row_chunk:
                repo_id = row["repo_id"]
                if repo_id not in ds_cache:
                    ds_cache[repo_id] = LeRobotDataset(
                        repo_id,
                        revision="main",
                        force_cache_sync=False,
                        download_videos=False,
                        delta_timestamps=None,
                    )
                sample = ds_cache[repo_id][int(row["frame_index"])]
                obs = _obs.lerobot_row_to_subtask_infer_obs(
                    {k: v for k, v in sample.items()},
                    action_horizon=train_cfg.model.action_horizon,
                    pop_subtask=True,
                )
                obs_chunk.append(obs)
            outs = policy.infer_subtask_batch(
                obs_chunk,
                batch_size=max(1, args.batch_size),
                temperature=args.temperature,
            )
            if len(outs) != len(row_chunk):
                raise RuntimeError(f"Expected {len(row_chunk)} predictions, got {len(outs)}")

            for out, row in zip(outs, row_chunk, strict=True):
                pred_bs, pred_st = _prediction_pair(
                    str(out.get("subtask_text") or ""),
                    semantic_normalize=args.semantic_normalize,
                )
                cid = int(row["class_id"])
                total += 1
                per_class_total[cid] += 1
                pid = _subtask_eval.class_id_for_pair(pred_bs, pred_st, canonical)
                pred_class_counts[
                    "empty" if pid is None and not pred_bs and not pred_st else f"class_{pid}" if pid is not None else "non_canonical"
                ] += 1
                if pred_bs == row["bottle_state"] and pred_st == row["subtask"]:
                    correct += 1
                    per_class_correct[cid] += 1
            logging.info(
                "  %s: processed %d/%d rows",
                ckpt.name,
                min(start + len(row_chunk), len(rows)),
                len(rows),
            )

        elapsed = time.perf_counter() - t0
        result = {
            "checkpoint": str(ckpt),
            "total": total,
            "correct": correct,
            "accuracy": (correct / total) if total else 0.0,
            "elapsed_sec": elapsed,
            "ms_per_sample": (elapsed * 1000.0 / total) if total else 0.0,
            "per_class": {
                str(c): {
                    "bottle_state": canonical[c][0],
                    "subtask": canonical[c][1],
                    "correct": int(per_class_correct[c]),
                    "total": int(per_class_total[c]),
                    "accuracy": (per_class_correct[c] / per_class_total[c]) if per_class_total[c] else 0.0,
                }
                for c in range(len(canonical))
            },
            "predicted_buckets": dict(pred_class_counts),
        }
        all_results.append(result)
        print(
            f"{ckpt}: accuracy={100.0 * result['accuracy']:.2f}% "
            f"({correct}/{total}), {result['ms_per_sample']:.1f} ms/sample"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "manifest": str(args.manifest),
                    "config_name": args.config_name,
                    "semantic_normalize": args.semantic_normalize,
                    "results": all_results,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved results to: {args.output_json}")


if __name__ == "__main__":
    main(tyro.cli(Args))
