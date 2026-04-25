#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import difflib
import json
import os
import re
import random
import statistics
import time
from pathlib import Path

from datasets import load_dataset

from openpi.evaluation import subtask_obs_utils as _obs
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi_client import subtask_parsing as _subtask_parsing


_TEMPLATES_PATH = Path(__file__).resolve().parents[1] / "assets" / "short_language_templates.json"


def _normalize_text(value: object) -> str:
    return str(value or "").strip()


def _normalize_free_text(value: object) -> str:
    s = _normalize_text(value).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _load_templates() -> dict[tuple[str, str], list[str]]:
    if not _TEMPLATES_PATH.exists():
        return {}
    raw = json.loads(_TEMPLATES_PATH.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], list[str]] = {}
    for key, values in raw.items():
        if "|||" not in key:
            continue
        bs, st = key.split("|||", 1)
        phrases = [str(v) for v in values if isinstance(v, str)]
        out[(bs.strip(), st.strip())] = phrases
    return out


def _best_fuzzy_class_id(
    pred_text: str,
    *,
    pair_to_class: dict[tuple[str, str], int],
    templates: dict[tuple[str, str], list[str]],
) -> int:
    norm_pred = _normalize_free_text(pred_text)
    if not norm_pred:
        return -1

    best_score = -1.0
    best_class_id = -1
    pred_tokens = set(norm_pred.split())

    for pair, cid in pair_to_class.items():
        bs, st = pair
        candidates = [st, f"{bs} {st}", *templates.get(pair, [])]
        pair_best = 0.0
        for candidate in candidates:
            norm_cand = _normalize_free_text(candidate)
            if not norm_cand:
                continue
            cand_tokens = set(norm_cand.split())
            token_overlap = len(pred_tokens & cand_tokens) / max(len(cand_tokens), 1)
            seq_score = difflib.SequenceMatcher(None, norm_pred, norm_cand).ratio()
            pair_best = max(pair_best, 0.7 * token_overlap + 0.3 * seq_score)
        if pair_best > best_score:
            best_score = pair_best
            best_class_id = cid
    return best_class_id if best_score >= 0.25 else -1


def _load_rows(repo_id: str, limit: int | None):
    ds = load_dataset("parquet", data_files=f"hf://datasets/{repo_id}/data/*/*.parquet", split="train")
    if limit is not None and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def _infer_subtask_batch(policy, obs_list, *, batch_size: int, show_infer_prints: bool):
    if show_infer_prints:
        return policy.infer_subtask_batch(obs_list, batch_size=batch_size, temperature=0.0)
    with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
        return policy.infer_subtask_batch(obs_list, batch_size=batch_size, temperature=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--config", default="twist_only_lora_triplet_100k")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prompt-override", default="Process all bottles")
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--show-infer-prints", action="store_true")
    args = parser.parse_args()

    train_cfg = _config.get_config(args.config)
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    policy = _policy_config.create_trained_policy(
        train_cfg,
        args.checkpoint,
        repack_transforms=data_cfg.repack_transforms,
    )
    action_horizon = train_cfg.model.action_horizon

    ds = _load_rows(args.repo_id, args.limit if args.limit > 0 else None)
    if args.sample_size > 0 and args.sample_size < len(ds):
        rng = random.Random(args.sample_seed)
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        ds = ds.select(indices[: args.sample_size])

    prompt_override = args.prompt_override
    if args.prompt_file is not None:
        prompts = json.loads(args.prompt_file.read_text(encoding="utf-8"))
        prompt_override = str(prompts[args.prompt_index]).strip()

    templates = _load_templates()
    pair_to_class: dict[tuple[str, str], int] = {}
    label_columns = [c for c in ("bottle_state", "subtask", "class_id") if c in ds.column_names]
    label_ds = ds.select_columns(label_columns)
    for row in label_ds:
        bs = _normalize_text(row.get("bottle_state"))
        st = _normalize_text(row.get("subtask"))
        cid = int(row["class_id"]) if "class_id" in row else -1
        if bs and st and cid >= 0:
            pair_to_class.setdefault((bs, st), cid)

    total = 0
    exact_correct = 0
    class_correct = 0
    fuzzy_class_correct = 0
    latencies: list[float] = []
    examples: list[dict[str, object]] = []

    batch_size = max(1, args.batch_size)
    warmed_up = False
    for start in range(0, len(ds), batch_size):
        batch_rows = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
        obs_list = []
        gt_pairs: list[tuple[str, str]] = []
        gt_class_ids: list[int] = []
        for row in batch_rows:
            row_dict = dict(row)
            row_dict["task"] = prompt_override
            obs_list.append(_obs.lerobot_row_to_subtask_infer_obs(row_dict, action_horizon=action_horizon, pop_subtask=True))
            gt_pairs.append((_normalize_text(row.get("bottle_state")), _normalize_text(row.get("subtask"))))
            gt_class_ids.append(int(row["class_id"]) if "class_id" in row else -1)

        if not warmed_up:
            _infer_subtask_batch(policy, obs_list, batch_size=batch_size, show_infer_prints=args.show_infer_prints)
            warmed_up = True

        t0 = time.perf_counter()
        outs = _infer_subtask_batch(policy, obs_list, batch_size=batch_size, show_infer_prints=args.show_infer_prints)
        per_sample_latency = (time.perf_counter() - t0) / len(batch_rows)
        latencies.extend([per_sample_latency] * len(batch_rows))

        for row, out, gt_pair, gt_class_id in zip(batch_rows, outs, gt_pairs, gt_class_ids, strict=True):
            pred_text = _normalize_text(out.get("subtask_text"))
            parsed = _subtask_parsing.parse_structured_fields(pred_text)
            pred_pair = (_normalize_text(parsed.get("bottle_state")), _normalize_text(parsed.get("subtask")))
            pred_class_id = pair_to_class.get(pred_pair, -1)
            fuzzy_pred_class_id = pred_class_id if pred_class_id >= 0 else _best_fuzzy_class_id(
                pred_text,
                pair_to_class=pair_to_class,
                templates=templates,
            )
            exact = pred_pair == gt_pair
            class_ok = (pred_class_id == gt_class_id) if gt_class_id >= 0 else exact
            fuzzy_class_ok = (fuzzy_pred_class_id == gt_class_id) if gt_class_id >= 0 else exact
            total += 1
            exact_correct += int(exact)
            class_correct += int(class_ok)
            fuzzy_class_correct += int(fuzzy_class_ok)
            if len(examples) < 10:
                examples.append(
                    {
                        "index": int(row.get("index", total - 1)),
                        "gt_bottle_state": gt_pair[0],
                        "gt_subtask": gt_pair[1],
                        "gt_class_id": gt_class_id,
                        "pred_text": pred_text,
                        "pred_bottle_state": pred_pair[0],
                        "pred_subtask": pred_pair[1],
                        "pred_class_id": pred_class_id,
                        "fuzzy_pred_class_id": fuzzy_pred_class_id,
                        "exact_correct": exact,
                        "class_correct": class_ok,
                        "fuzzy_class_correct": fuzzy_class_ok,
                    }
                )

        if total % (batch_size * 10) == 0 or total == len(ds):
            print(
                f"processed {total}/{len(ds)} exact_acc={exact_correct / total:.4f} "
                f"class_acc={class_correct / total:.4f} fuzzy_class_acc={fuzzy_class_correct / total:.4f} "
                f"avg_latency={sum(latencies) / len(latencies):.4f}s",
                flush=True,
            )

    result = {
        "repo_id": args.repo_id,
        "checkpoint": str(args.checkpoint),
        "config": args.config,
        "num_samples": total,
        "exact_accuracy": exact_correct / max(total, 1),
        "class_accuracy": class_correct / max(total, 1),
        "fuzzy_class_accuracy": fuzzy_class_correct / max(total, 1),
        "avg_latency_sec": sum(latencies) / max(len(latencies), 1),
        "median_latency_sec": statistics.median(latencies) if latencies else None,
        "batch_size": batch_size,
        "prompt_override": prompt_override,
        "sample_size": len(ds),
        "sample_seed": args.sample_seed,
        "examples": examples,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
