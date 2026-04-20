#!/usr/bin/env python3
"""Measure Pi05 PaligemmaTokenizer pre-pad lengths (no truncation warnings).

Uses the same rules as openpi.models.tokenizer.PaligemmaTokenizer.tokenize:
- train_action=False: prompt = task text + BOS; subtask stream = VQA answer + EOS
- train_action=True: prompt = Task + state; subtask stream = subtask_text + EOS + suffix + FAST(action)
"""
from __future__ import annotations

import argparse
import os
import random
import warnings

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from openpi.models.tokenizer import (
    PaligemmaTokenizer,
    get_good_bad_action_label,
    get_subtask_text,
    get_vqa_answer_text,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def prompt_len_train_action(tok: PaligemmaTokenizer, prompt: str, state: np.ndarray) -> int:
    cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
    st = np.asarray(state, dtype=np.float64).reshape(-1)
    discretized_state = np.digitize(st, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized_state))
    prompt_prefix = f"Task: {cleaned_text}, State: {state_str}, "
    return len(tok._tokenizer.encode(prompt_prefix, add_bos=True))


def prompt_len_vqa(tok: PaligemmaTokenizer, prompt: str) -> int:
    cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
    return len(tok._tokenizer.encode(cleaned_text, add_bos=True))


def subtask_len_vqa(tok: PaligemmaTokenizer, subtask) -> int:
    answer_text = get_vqa_answer_text(subtask)
    if answer_text is None:
        return 0
    answer_tokens = tok._tokenizer.encode(
        answer_text.strip().replace("_", " ").replace("\n", " "), add_bos=False
    )
    return len(answer_tokens) + 1


def subtask_len_train_action(tok: PaligemmaTokenizer, subtask, actions: np.ndarray | None) -> int:
    label = get_good_bad_action_label(subtask)
    if label == "bad action":
        action_suffix = " Give a bad action: "
    elif label == "good action":
        action_suffix = " Give a good action: "
    else:
        action_suffix = " Action: "
    subtask_text = get_subtask_text(subtask)
    subtask_only_tokens: list[int] = []
    if subtask_text is not None:
        cleaned_subtask = subtask_text.strip().replace("_", " ").replace("\n", " ")
        if cleaned_subtask:
            subtask_only_tokens = tok._tokenizer.encode(cleaned_subtask, add_bos=False)
    fast_action_tokens: list[int] = []
    if actions is not None:
        actions_np = np.asarray(actions)
        if actions_np.ndim == 2:
            fast_action_tokens_raw = tok._get_fast_tokenizer()(actions_np[None])[0]
            fast_action_tokens = tok._act_tokens_to_paligemma_tokens(fast_action_tokens_raw).tolist()
    action_suffix_tokens = tok._tokenizer.encode(action_suffix, add_bos=False)
    eos_token = [tok.eos_token_id] if subtask_only_tokens else []
    full = subtask_only_tokens + eos_token + action_suffix_tokens + fast_action_tokens
    return len(full)


def fast_action_only_len(tok: PaligemmaTokenizer, actions: np.ndarray) -> int:
    actions_np = np.asarray(actions)
    if actions_np.ndim != 2:
        return 0
    raw = tok._get_fast_tokenizer()(actions_np[None])[0]
    return len(tok._act_tokens_to_paligemma_tokens(raw).tolist())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repos", nargs="+", required=True)
    p.add_argument("--samples-per-repo", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fast-random-trials", type=int, default=2000)
    args = p.parse_args()

    rng = random.Random(args.seed)
    tok = PaligemmaTokenizer(max_len=65536, subtask_max_len=65536, fast_tokenizer_path="physical-intelligence/fast")

    max_prompt_ta = max_prompt_vqa = 0
    max_sub_ta = max_sub_vqa = 0
    max_fast = 0

    for repo in args.repos:
        meta = LeRobotDatasetMetadata(repo, revision="main", force_cache_sync=False)
        delta = {"action": [t / meta.fps for t in range(50)]}
        ds = LeRobotDataset(repo, revision="main", force_cache_sync=False, delta_timestamps=delta)
        n = len(ds)
        k = min(args.samples_per_repo, n)
        idxs = [rng.randrange(n) for _ in range(k)]
        for i in idxs:
            row = ds[i]
            prompt = row["task"]
            st = np.asarray(row["observation.state"], dtype=np.float64)
            sub = row["subtask"]
            if hasattr(sub, "item"):
                try:
                    sub = sub.item()
                except Exception:
                    pass
            if isinstance(sub, bytes):
                sub = sub.decode("utf-8")
            act = np.asarray(row["action"], dtype=np.float32)
            ta = row.get("train_action", True)
            if hasattr(ta, "item"):
                ta = bool(np.asarray(ta).reshape(-1)[0])
            else:
                ta = bool(ta)
            if ta:
                max_prompt_ta = max(max_prompt_ta, prompt_len_train_action(tok, str(prompt), st))
                max_sub_ta = max(max_sub_ta, subtask_len_train_action(tok, sub, act))
                max_fast = max(max_fast, fast_action_only_len(tok, act))
            else:
                max_prompt_vqa = max(max_prompt_vqa, prompt_len_vqa(tok, str(prompt)))
                max_sub_vqa = max(max_sub_vqa, subtask_len_vqa(tok, sub))

        print(
            f"{repo} samples={k} max_prompt_train_action={max_prompt_ta} max_prompt_vqa={max_prompt_vqa} "
            f"max_subtask_train_action={max_sub_ta} max_subtask_vqa={max_sub_vqa} max_fast_only={max_fast}"
        )

    # Random continuous actions in [-1,1] to stress FAST length (dataset may not cover extremes).
    for _ in range(args.fast_random_trials):
        a = rng.uniform(-1, 1, size=(50, 14)).astype(np.float32)
        max_fast = max(max_fast, fast_action_only_len(tok, a))

    print(
        "GLOBAL max_prompt_train_action",
        max_prompt_ta,
        "max_prompt_vqa",
        max_prompt_vqa,
        "=> recommend max_token_len >=",
        max(max_prompt_ta, max_prompt_vqa),
    )
    print(
        "GLOBAL max_subtask_train_action",
        max_sub_ta,
        "max_subtask_vqa",
        max_sub_vqa,
        "max_fast_only",
        max_fast,
        "=> recommend subtask_max_token_len >=",
        max(max_sub_ta, max_sub_vqa),
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*Token length.*")
    main()
