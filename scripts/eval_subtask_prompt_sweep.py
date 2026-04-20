#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import statistics
import time
from pathlib import Path

from datasets import load_dataset

from openpi.evaluation import subtask_obs_utils as _obs
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


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


def _load_rows(repo_id: str, sample_size: int, sample_seed: int):
    ds = load_dataset("parquet", data_files=f"hf://datasets/{repo_id}/data/*/*.parquet", split="train")
    if sample_size > 0 and sample_size < len(ds):
        rng = random.Random(sample_seed)
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        ds = ds.select(indices[:sample_size])
    return ds


def _pair_to_class(ds) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for row in ds:
        bs = _normalize_text(row.get("bottle_state"))
        st = _normalize_text(row.get("subtask"))
        cid = int(row["class_id"]) if "class_id" in row else -1
        if bs and st and cid >= 0:
            out.setdefault((bs, st), cid)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-ids", nargs="+", required=True)
    parser.add_argument("--config", default="twist_only_lora_triplet_100k")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    prompts = json.loads(args.prompt_file.read_text(encoding="utf-8"))[: args.num_prompts]
    train_cfg = _config.get_config(args.config)
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    policy = _policy_config.create_trained_policy(
        train_cfg,
        args.checkpoint,
        repack_transforms=data_cfg.repack_transforms,
    )
    action_horizon = train_cfg.model.action_horizon
    templates = _load_templates()

    summaries = []
    for repo_id in args.repo_ids:
        ds = _load_rows(repo_id, args.sample_size, args.sample_seed)
        pair_to_class = _pair_to_class(ds)
        repo_results = []
        for prompt_i, prompt in enumerate(prompts):
            total = 0
            exact_correct = 0
            class_correct = 0
            fuzzy_class_correct = 0
            latencies: list[float] = []
            warmed_up = False
            for start in range(0, len(ds), args.batch_size):
                batch_rows = [ds[i] for i in range(start, min(start + args.batch_size, len(ds)))]
                obs_list = []
                gt_pairs = []
                gt_class_ids = []
                for row in batch_rows:
                    row_dict = dict(row)
                    row_dict["task"] = prompt
                    obs_list.append(
                        _obs.lerobot_row_to_subtask_infer_obs(
                            row_dict,
                            action_horizon=action_horizon,
                            pop_subtask=True,
                        )
                    )
                    gt_pairs.append((_normalize_text(row.get("bottle_state")), _normalize_text(row.get("subtask"))))
                    gt_class_ids.append(int(row["class_id"]) if "class_id" in row else -1)

                if not warmed_up:
                    policy.infer_subtask_batch(obs_list, batch_size=args.batch_size, temperature=0.0)
                    warmed_up = True

                t0 = time.perf_counter()
                outs = policy.infer_subtask_batch(obs_list, batch_size=args.batch_size, temperature=0.0)
                per_sample_latency = (time.perf_counter() - t0) / len(batch_rows)
                latencies.extend([per_sample_latency] * len(batch_rows))

                for out, gt_pair, gt_class_id in zip(outs, gt_pairs, gt_class_ids, strict=True):
                    pred_text = _normalize_text(out.get("subtask_text"))
                    pred_class_id = -1
                    fuzzy_pred_class_id = _best_fuzzy_class_id(
                        pred_text,
                        pair_to_class=pair_to_class,
                        templates=templates,
                    )
                    exact = False
                    class_ok = pred_class_id == gt_class_id if gt_class_id >= 0 else exact
                    fuzzy_class_ok = fuzzy_pred_class_id == gt_class_id if gt_class_id >= 0 else exact
                    total += 1
                    exact_correct += int(exact)
                    class_correct += int(class_ok)
                    fuzzy_class_correct += int(fuzzy_class_ok)

            result = {
                "prompt_index": prompt_i,
                "prompt": prompt,
                "num_samples": total,
                "exact_accuracy": exact_correct / max(total, 1),
                "class_accuracy": class_correct / max(total, 1),
                "fuzzy_class_accuracy": fuzzy_class_correct / max(total, 1),
                "avg_latency_sec": sum(latencies) / max(len(latencies), 1),
                "median_latency_sec": statistics.median(latencies) if latencies else None,
            }
            repo_results.append(result)
            print(
                f"[{repo_id}] prompt {prompt_i+1}/{len(prompts)} "
                f"fuzzy_acc={result['fuzzy_class_accuracy']:.4f} avg_latency={result['avg_latency_sec']:.4f}s",
                flush=True,
            )

        summaries.append(
            {
                "repo_id": repo_id,
                "sample_size": len(ds),
                "num_prompts": len(prompts),
                "avg_exact_accuracy": statistics.mean(r["exact_accuracy"] for r in repo_results),
                "avg_class_accuracy": statistics.mean(r["class_accuracy"] for r in repo_results),
                "avg_fuzzy_class_accuracy": statistics.mean(r["fuzzy_class_accuracy"] for r in repo_results),
                "avg_latency_sec": statistics.mean(r["avg_latency_sec"] for r in repo_results),
                "median_latency_sec": statistics.median(r["avg_latency_sec"] for r in repo_results),
                "results": repo_results,
            }
        )

    payload = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "batch_size": args.batch_size,
        "num_prompts": len(prompts),
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
        "repo_summaries": summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
