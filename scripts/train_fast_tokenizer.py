#!/usr/bin/env python3
"""Train a FAST action tokenizer from local/HF datasets.

Reference workflow (official model card):
https://huggingface.co/physical-intelligence/fast
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoProcessor


def _load_seq_from_hdf5(path: Path, action_key: str, state_key: str) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as f:
        if action_key not in f:
            raise KeyError(f"action_key='{action_key}' not found in hdf5: {path}")
        if state_key not in f:
            raise KeyError(f"state_key='{state_key}' not found in hdf5: {path}")
        actions = np.asarray(f[action_key], dtype=np.float32)
        state = np.asarray(f[state_key], dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected hdf5 actions [T,D], got {actions.shape}")
    if state.ndim != 2:
        raise ValueError(f"Expected hdf5 state [T,D], got {state.shape}")
    if actions.shape[0] != state.shape[0]:
        raise ValueError(f"T mismatch actions={actions.shape} state={state.shape}")
    return actions, state


def _load_from_lerobot_with_state(repo_id: str, action_key: str, state_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from lerobot.datasets import lerobot_dataset

    ds = lerobot_dataset.LeRobotDataset(repo_id)
    hf = ds.hf_dataset
    if action_key not in hf.column_names:
        raise KeyError(f"action_key='{action_key}' not in dataset columns: {hf.column_names}")
    if state_key not in hf.column_names:
        raise KeyError(f"state_key='{state_key}' not in dataset columns: {hf.column_names}")
    if "episode_index" not in hf.column_names:
        raise KeyError("episode_index column not found in LeRobot dataset")
    actions = np.asarray(hf[action_key], dtype=np.float32)
    states = np.asarray(hf[state_key], dtype=np.float32)
    episode_index = np.asarray(hf["episode_index"], dtype=np.int64)
    if actions.ndim != 2 or states.ndim != 2:
        raise ValueError(f"Expected [N,D] for action/state, got action={actions.shape}, state={states.shape}")
    if len(actions) != len(states) or len(actions) != len(episode_index):
        raise ValueError("Mismatched lengths among action/state/episode_index")
    return actions, states, episode_index


def _load_from_lerobot_preprocessed(
    repo_id: str,
    *,
    action_key: str,
    state_key: str,
    chunk_len: int,
    stride: int,
    adapt_to_pi: bool,
    use_quantiles: bool,
    norm_stats: dict,
) -> np.ndarray:
    actions, states, episode_index = _load_from_lerobot_with_state(repo_id, action_key, state_key)
    change = np.flatnonzero(np.diff(episode_index) != 0) + 1
    starts = np.concatenate([np.array([0]), change])
    ends = np.concatenate([change, np.array([len(episode_index)])])
    all_chunks: list[np.ndarray] = []
    for s, e in zip(starts, ends, strict=True):
        chunks = _preprocess_aloha_actions_for_fast(
            actions[s:e],
            states[s:e],
            chunk_len=chunk_len,
            stride=stride,
            adapt_to_pi=adapt_to_pi,
            use_quantiles=use_quantiles,
            norm_stats=norm_stats,
        )
        all_chunks.append(chunks)
    if not all_chunks:
        raise ValueError(f"No preprocessed chunks produced from {repo_id}")
    return np.concatenate(all_chunks, axis=0)


def _load_from_lerobot_multi_preprocessed(
    repo_ids: list[str],
    *,
    action_key: str,
    state_key: str,
    chunk_len: int,
    stride: int,
    adapt_to_pi: bool,
    use_quantiles: bool,
    norm_stats: dict,
) -> np.ndarray:
    all_chunks: list[np.ndarray] = []
    for rid in repo_ids:
        chunks = _load_from_lerobot_preprocessed(
            rid,
            action_key=action_key,
            state_key=state_key,
            chunk_len=chunk_len,
            stride=stride,
            adapt_to_pi=adapt_to_pi,
            use_quantiles=use_quantiles,
            norm_stats=norm_stats,
        )
        print(f"[{rid}] chunks={chunks.shape[0]} (preprocessed)")
        all_chunks.append(chunks)
    if not all_chunks:
        raise ValueError("No repo ids provided")
    return np.concatenate(all_chunks, axis=0)


def _preprocess_aloha_actions_for_fast(
    actions_seq: np.ndarray,
    state_seq: np.ndarray,
    *,
    chunk_len: int,
    stride: int,
    adapt_to_pi: bool,
    use_quantiles: bool,
    norm_stats: dict,
) -> np.ndarray:
    """Match OpenPI Aloha training preprocessing before FAST fit.

    Equivalent chain:
    AlohaInputs(adapt_to_pi=True) -> DeltaActions(mask=make_bool_mask(6,-1,6,-1)) -> Normalize(...)
    """
    from openpi import transforms as _transforms
    from openpi.policies import aloha_policy

    state_seq = aloha_policy._decode_state(state_seq, adapt_to_pi=adapt_to_pi)
    actions_seq = aloha_policy._encode_actions_inv(actions_seq, adapt_to_pi=adapt_to_pi)

    chunks: list[np.ndarray] = []
    chunk_states: list[np.ndarray] = []
    t = actions_seq.shape[0]
    for s in range(0, t, stride):
        chunks.append(actions_seq[s : s + chunk_len].copy())
        chunk_states.append(state_seq[s].copy())
    if not chunks:
        raise ValueError(f"No chunks from sequence with T={t}, chunk_len={chunk_len}, stride={stride}")

    actions = []
    for c in chunks:
        if c.shape[0] < chunk_len:
            pad = np.repeat(c[-1:, :], chunk_len - c.shape[0], axis=0)
            c = np.concatenate([c, pad], axis=0)
        actions.append(c)
    actions = np.stack(actions, axis=0)
    states = np.stack(chunk_states, axis=0)

    data = {"actions": actions, "state": states}
    data = _transforms.DeltaActions(_transforms.make_bool_mask(6, -1, 6, -1))(data)
    data = _transforms.Normalize(norm_stats=norm_stats, use_quantiles=use_quantiles, strict=False)(data)
    return np.asarray(data["actions"], dtype=np.float32)


def _maybe_subsample(chunks: np.ndarray, max_chunks: int, seed: int) -> np.ndarray:
    if max_chunks <= 0 or len(chunks) <= max_chunks:
        return chunks
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(chunks), size=max_chunks, replace=False)
    return chunks[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-ids", type=str, nargs="+", default=None, help="Multiple LeRobot dataset repo ids")
    parser.add_argument("--hdf5", type=Path, default=None, help="HDF5 file containing action array")

    parser.add_argument("--action-key", type=str, default="action", help="Action key/column/path")
    parser.add_argument("--lerobot-state-key", type=str, default="observation.state", help="State key for LeRobot preprocess")
    parser.add_argument("--hdf5-state-key", type=str, default="observations/qpos", help="State key for HDF5 preprocess")
    parser.add_argument("--chunk-len", type=int, default=50, help="Action chunk length (timesteps)")
    parser.add_argument("--stride", type=int, default=1, help="Sliding-window stride")
    parser.add_argument("--max-chunks", type=int, default=0, help="If >0, randomly subsample this many chunks")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional OpenPI train config name (e.g. pi05_aloha_pen_uncap) to auto-load norm stats.",
    )
    parser.add_argument("--adapt-to-pi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--norm-stats-json",
        type=Path,
        default=None,
        help="Optional explicit norm_stats.json path. Overrides --config auto-load.",
    )
    parser.add_argument(
        "--use-quantile-norm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If unset, infer from --config when provided; otherwise defaults to True.",
    )

    parser.add_argument("--base-tokenizer", type=str, default="physical-intelligence/fast")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to save trained tokenizer")
    parser.add_argument("--push-to-hub", type=str, default="", help="Optional repo id to push, e.g. user/my_fast")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.chunk_len <= 0:
        raise ValueError("--chunk-len must be > 0")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")
    if not args.repo_ids and args.hdf5 is None:
        raise ValueError("One of --repo-ids/--hdf5 must be provided")
    if args.hdf5 is not None and args.repo_ids:
        raise ValueError("--hdf5 cannot be used together with --repo-ids")

    # Always use OpenPI Aloha preprocessing for repo/hdf5 sources.
    norm_stats = None
    use_quantile_norm = args.use_quantile_norm if args.use_quantile_norm is not None else True
    from openpi.shared import normalize as _normalize
    from openpi.training import config as _config

    norm_stats_path = args.norm_stats_json
    if norm_stats_path is None and args.config:
        cfg = _config.get_config(args.config)
        data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
        if data_cfg.norm_stats is None:
            raise ValueError(f"norm stats missing for config={args.config}")
        norm_stats = data_cfg.norm_stats
        if args.use_quantile_norm is None:
            use_quantile_norm = data_cfg.use_quantile_norm
    else:
        if norm_stats_path is None:
            norm_stats_path = Path("assets/trossen/norm_stats.json")
        if not norm_stats_path.exists():
            raise FileNotFoundError(f"norm stats not found: {norm_stats_path}")
        norm_stats = _normalize.load(norm_stats_path)

    if args.repo_ids:
        chunks = _load_from_lerobot_multi_preprocessed(
            args.repo_ids,
            action_key=args.action_key,
            state_key=args.lerobot_state_key,
            chunk_len=args.chunk_len,
            stride=args.stride,
            adapt_to_pi=args.adapt_to_pi,
            use_quantiles=use_quantile_norm,
            norm_stats=norm_stats,
        )
    elif args.hdf5:
        actions_seq, state_seq = _load_seq_from_hdf5(args.hdf5, args.action_key, args.hdf5_state_key)
        chunks = _preprocess_aloha_actions_for_fast(
            actions_seq,
            state_seq,
            chunk_len=args.chunk_len,
            stride=args.stride,
            adapt_to_pi=args.adapt_to_pi,
            use_quantiles=use_quantile_norm,
            norm_stats=norm_stats,
        )
    else:
        raise ValueError("One of --repo-id/--repo-ids/--hdf5 must be provided")

    print(f"Original chunks: shape={chunks.shape}, dtype={chunks.dtype}, mean={chunks.mean():.9f}")
    chunks = _maybe_subsample(chunks, args.max_chunks, args.seed)
    print(f"Loaded chunks: shape={chunks.shape}, dtype={chunks.dtype}, mean={chunks.mean():.9f}")
    print(
        "Value range: "
        f"min={chunks.min():.6f}, max={chunks.max():.6f}, "
        f"p1={np.percentile(chunks, 1):.6f}, p99={np.percentile(chunks, 99):.6f}"
    )

    tokenizer = AutoProcessor.from_pretrained(args.base_tokenizer, trust_remote_code=args.trust_remote_code)
    tokenizer = tokenizer.fit(chunks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved tokenizer to: {args.output_dir}")

    if args.push_to_hub:
        tokenizer.push_to_hub(args.push_to_hub)
        print(f"Pushed tokenizer to hub: {args.push_to_hub}")


if __name__ == "__main__":
    main()
