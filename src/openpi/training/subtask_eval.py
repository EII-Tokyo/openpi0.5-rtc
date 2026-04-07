"""Stratified holdout and periodic subtask (bottle_state, subtask) accuracy during JAX training."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pyarrow.parquet as pq

from openpi.evaluation import subtask_obs_utils as _obs
from openpi.policies import policy as _policy
from openpi.training import config as _config
from openpi_client.runtime import low_level_subtask_defaults as _lld
from openpi_client import subtask_parsing as _subtask_parsing


def default_canonical_pairs() -> tuple[tuple[str, str], ...]:
    """Nine (bottle_state, subtask) classes; single source matches runtime defaults."""
    return tuple((a, b) for a, b in _lld.DEFAULT_STATE_SUBTASK_PAIRS)


def resolve_canonical_pairs(config: _config.TrainConfig) -> tuple[tuple[str, str], ...]:
    raw = getattr(config, "subtask_eval_canonical_pairs", None)
    if raw is not None and len(raw) > 0:
        return tuple((str(a).strip(), str(b).strip()) for a, b in raw)
    return default_canonical_pairs()


def class_id_for_pair(
    bottle_state: str, subtask: str, canonical: tuple[tuple[str, str], ...]
) -> int | None:
    t = (bottle_state.strip(), subtask.strip())
    for i, pair in enumerate(canonical):
        if pair == t:
            return i
    return None


def _frame_row_at_index(outer_torch_dataset: Any, index: int) -> dict[str, Any]:
    """Row dict for global `index` (same indexing as training dataset before transform)."""
    from openpi.training.data_loader import IsForTrainingWrapper, TemporalFrameStackDataset

    if isinstance(outer_torch_dataset, IsForTrainingWrapper):
        return outer_torch_dataset._dataset[index]
    if isinstance(outer_torch_dataset, TemporalFrameStackDataset):
        return outer_torch_dataset._dataset[index]
    return outer_torch_dataset[index]


def _trainable_mask_for_outer(outer_torch_dataset: Any) -> np.ndarray:
    from openpi.training.data_loader import IsForTrainingWrapper, TemporalFrameStackDataset

    if isinstance(outer_torch_dataset, IsForTrainingWrapper):
        return IsForTrainingWrapper._build_trainable_mask(outer_torch_dataset._dataset)
    if isinstance(outer_torch_dataset, TemporalFrameStackDataset):
        m = outer_torch_dataset._trainable_mask
        if m is not None:
            return np.asarray(m, dtype=bool)
        return np.ones(len(outer_torch_dataset), dtype=bool)
    return np.ones(len(outer_torch_dataset), dtype=bool)


def _val_count_for_class(n_c: int, frac: float) -> int:
    """Holdout count per class; singletons stay in train only."""
    if n_c <= 0:
        return 0
    if n_c == 1:
        return 0
    return max(1, int(np.ceil(frac * n_c)))


def _full_scan_progress_interval(n: int) -> int:
    """Log at most ~20 times per pass; at least every 50k frames."""
    return max(50_000, min(500_000, max(n // 20, 1)))


def _unwrap_lerobot_for_parquet(outer_torch_dataset: Any) -> Any:
    """Strip training wrappers down to LeRobotDataset / MultiLeRobotDataset."""
    from openpi.training.data_loader import IsForTrainingWrapper, TemporalFrameStackDataset

    d = outer_torch_dataset
    for _ in range(8):
        if isinstance(d, IsForTrainingWrapper):
            d = d._dataset
        elif isinstance(d, TemporalFrameStackDataset):
            d = d._dataset
        else:
            break
    return d


def _read_lerobot_parquet_subtask_strings(ds: Any) -> dict[int, str] | None:
    """Read local `data/**/*.parquet` index -> subtask string for one LeRobotDataset."""
    root = getattr(ds, "root", None)
    if root is None:
        return None
    root_path = Path(root)
    if not root_path.is_dir():
        return None
    paths = sorted(root_path.glob("data/**/*.parquet"))
    if not paths:
        return None
    out: dict[int, str] = {}
    saw_subtask_column = False
    for path in paths:
        try:
            pf = pq.ParquetFile(path)
        except Exception:
            return None
        names = pf.schema.names
        if "index" not in names:
            continue
        has_sub = "subtask" in names
        if has_sub:
            saw_subtask_column = True
        cols = ["index"] + (["subtask"] if has_sub else [])
        try:
            table = pq.read_table(path, columns=cols)
        except Exception:
            return None
        idx_list = table["index"].to_pylist()
        if has_sub:
            sub_list = table["subtask"].to_pylist()
        else:
            sub_list = [""] * len(idx_list)
        if len(sub_list) != len(idx_list):
            return None
        for ix, v in zip(idx_list, sub_list, strict=True):
            out[int(ix)] = _obs.subtask_cell_to_str(v)

    expected = int(len(ds))
    if len(out) < max(1, int(expected * 0.95)):
        logging.warning(
            "subtask_eval parquet: repo %s indexed_rows=%d expected~%d; fast path disabled",
            getattr(ds, "repo_id", root_path.name),
            len(out),
            expected,
        )
        return None
    if not saw_subtask_column:
        logging.info(
            "subtask_eval parquet: repo %s has no subtask column in parquet; using empty labels for fast path",
            getattr(ds, "repo_id", root_path.name),
        )
    return out


def try_build_parquet_subtask_label_map(outer_torch_dataset: Any) -> dict[int, str] | None:
    """Global frame index -> subtask cell string via Parquet only (no __getitem__)."""
    import lerobot.datasets.lerobot_dataset as lerobot_dataset

    t0 = time.perf_counter()
    inner = _unwrap_lerobot_for_parquet(outer_torch_dataset)

    if isinstance(inner, lerobot_dataset.MultiLeRobotDataset):
        n_outer = len(outer_torch_dataset)
        offset = 0
        full: dict[int, str] = {}
        for sub in inner._datasets:
            part = _read_lerobot_parquet_subtask_strings(sub)
            if part is None:
                logging.info(
                    "subtask_eval parquet: fast path disabled (sub-repo %s)",
                    getattr(sub, "repo_id", "?"),
                )
                return None
            for li, s in part.items():
                full[offset + int(li)] = s
            offset += int(len(sub))
        if offset != n_outer:
            logging.warning(
                "subtask_eval parquet: length mismatch outer_len=%d parquet_span=%d; fast path disabled",
                n_outer,
                offset,
            )
            return None
        logging.info(
            "subtask_eval parquet: built global label map len=%d from %d repos in %.2fs",
            len(full),
            len(inner._datasets),
            time.perf_counter() - t0,
        )
        return full

    if isinstance(inner, lerobot_dataset.LeRobotDataset):
        part = _read_lerobot_parquet_subtask_strings(inner)
        if part is None:
            logging.info(
                "subtask_eval parquet: fast path disabled for single repo %s",
                getattr(inner, "repo_id", "?"),
            )
            return None
        logging.info(
            "subtask_eval parquet: built label map len=%d in %.2fs",
            len(part),
            time.perf_counter() - t0,
        )
        return part

    logging.info(
        "subtask_eval parquet: fast path not applicable (dataset type=%s)",
        type(inner).__name__,
    )
    return None


@dataclass(frozen=True)
class SubtaskEvalSplit:
    train_indices: np.ndarray  # int64, sorted
    val_indices: np.ndarray  # int64, sorted
    canonical_pairs: tuple[tuple[str, str], ...]
    per_class_total: dict[int, int]
    per_class_val: dict[int, int]
    unknown_label_indices: int
    """Frames with subtask JSON that did not match any canonical pair."""

    @property
    def num_classes(self) -> int:
        return len(self.canonical_pairs)


def compute_subtask_eval_split(
    *,
    outer_torch_dataset: Any,
    canonical_pairs: tuple[tuple[str, str], ...],
    holdout_fraction: float,
    seed: int,
    label_map: dict[int, str] | None = None,
) -> SubtaskEvalSplit:
    """Stratified val split on frames that map to a canonical class; trainable indices only."""
    t_all = time.perf_counter()
    n = len(outer_torch_dataset)
    logging.info(
        "subtask_eval.split: START compute_subtask_eval_split n_frames=%d num_canonical_classes=%d holdout_fraction=%g%s",
        n,
        len(canonical_pairs),
        holdout_fraction,
        f" (parquet label_map len={len(label_map)})" if label_map is not None else "",
    )
    t0 = time.perf_counter()
    mask = _trainable_mask_for_outer(outer_torch_dataset)
    logging.info(
        "subtask_eval.split: trainable mask ready in %.2fs (mask_len=%d trainable_count=%d)",
        time.perf_counter() - t0,
        len(mask),
        int(np.count_nonzero(mask)),
    )
    rng = np.random.default_rng(seed)

    by_class: dict[int, list[int]] = {i: [] for i in range(len(canonical_pairs))}
    unknown = 0

    step = _full_scan_progress_interval(n)
    t_scan = time.perf_counter()
    last_log_i = 0
    for i in range(n):
        if label_map is None and i > 0 and i % step == 0:
            now = time.perf_counter()
            dt = now - t_scan
            rate = (i - last_log_i) / dt if dt > 0 else 0.0
            logging.info(
                "subtask_eval.split: scan pass 1/1 i=%d/%d (%.2f%%) elapsed=%.1fs ~%.0f rows/s",
                i,
                n,
                100.0 * i / max(n, 1),
                now - t_all,
                rate,
            )
            t_scan = now
            last_log_i = i
        if not bool(mask[i]):
            continue
        if label_map is not None:
            cell = label_map.get(i, "")
        else:
            row = _frame_row_at_index(outer_torch_dataset, i)
            cell = row.get("subtask")
        parsed = _obs.parse_json_bottle_state_subtask(_obs.subtask_cell_to_str(cell))
        if parsed is None:
            continue
        bs, st = parsed
        cid = class_id_for_pair(bs, st, canonical_pairs)
        if cid is None:
            unknown += 1
            continue
        by_class[cid].append(i)

    logging.info(
        "subtask_eval.split: full index scan finished in %.2fs unknown_labels=%d",
        time.perf_counter() - t_all,
        unknown,
    )

    per_class_total = {c: len(by_class[c]) for c in by_class}
    val_lists: list[int] = []
    per_class_val: dict[int, int] = {}

    for c, idxs in by_class.items():
        if not idxs:
            per_class_val[c] = 0
            continue
        order = rng.permutation(len(idxs))
        shuffled = [idxs[j] for j in order]
        k = _val_count_for_class(len(shuffled), holdout_fraction)
        take = shuffled[:k]
        val_lists.extend(take)
        per_class_val[c] = len(take)

    val_set = set(val_lists)
    t_tr = time.perf_counter()
    logging.info(
        "subtask_eval.split: building train_indices / val_indices (n=%d |val_set|=%d)",
        n,
        len(val_set),
    )
    train_indices = np.array([i for i in range(n) if i not in val_set], dtype=np.int64)
    val_indices = np.array(sorted(val_set), dtype=np.int64)
    logging.info(
        "subtask_eval.split: index arrays done in %.2fs train_size=%d val_size=%d TOTAL=%.2fs",
        time.perf_counter() - t_tr,
        int(train_indices.size),
        int(val_indices.size),
        time.perf_counter() - t_all,
    )

    return SubtaskEvalSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        canonical_pairs=canonical_pairs,
        per_class_total=per_class_total,
        per_class_val=per_class_val,
        unknown_label_indices=unknown,
    )


def save_split_json(path: Path, split: SubtaskEvalSplit, *, seed: int, holdout_fraction: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "canonical_pairs": [{"bottle_state": a, "subtask": b} for a, b in split.canonical_pairs],
        "val_indices": split.val_indices.tolist(),
        "train_indices_count": int(split.train_indices.size),
        "per_class_total": {str(k): v for k, v in split.per_class_total.items()},
        "per_class_val": {str(k): v for k, v in split.per_class_val.items()},
        "unknown_label_indices": split.unknown_label_indices,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def subsample_val_indices(
    val_indices: np.ndarray,
    *,
    index_to_class: dict[int, int],
    max_per_class: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_per_class is None or max_per_class <= 0:
        return val_indices
    by_c: dict[int, list[int]] = {}
    for i in val_indices.tolist():
        c = index_to_class.get(i)
        if c is None:
            continue
        by_c.setdefault(c, []).append(i)
    out: list[int] = []
    for c, idxs in by_c.items():
        if len(idxs) <= max_per_class:
            out.extend(idxs)
        else:
            pick = rng.choice(len(idxs), size=max_per_class, replace=False)
            out.extend(idxs[j] for j in pick)
    return np.array(sorted(out), dtype=np.int64)


def build_index_to_class_map(
    outer_torch_dataset: Any,
    canonical_pairs: tuple[tuple[str, str], ...],
    *,
    label_map: dict[int, str] | None = None,
) -> dict[int, int]:
    t_all = time.perf_counter()
    n = len(outer_torch_dataset)
    if label_map is not None:
        logging.info(
            "subtask_eval.map: START build_index_to_class_map n_frames=%d (parquet label_map, no __getitem__)",
            n,
        )
    else:
        logging.info("subtask_eval.map: START build_index_to_class_map n_frames=%d", n)
    t0 = time.perf_counter()
    mask = _trainable_mask_for_outer(outer_torch_dataset)
    logging.info(
        "subtask_eval.map: trainable mask in %.2fs (same as split if uncached)",
        time.perf_counter() - t0,
    )
    m: dict[int, int] = {}
    step = _full_scan_progress_interval(n)
    t_scan = time.perf_counter()
    last_log_i = 0
    for i in range(n):
        if label_map is None and i > 0 and i % step == 0:
            now = time.perf_counter()
            dt = now - t_scan
            rate = (i - last_log_i) / dt if dt > 0 else 0.0
            logging.info(
                "subtask_eval.map: scan i=%d/%d (%.2f%%) elapsed=%.1fs ~%.0f rows/s",
                i,
                n,
                100.0 * i / max(n, 1),
                now - t_all,
                rate,
            )
            t_scan = now
            last_log_i = i
        if not bool(mask[i]):
            continue
        if label_map is not None:
            cell = label_map.get(i, "")
        else:
            row = _frame_row_at_index(outer_torch_dataset, i)
            cell = row.get("subtask")
        parsed = _obs.parse_json_bottle_state_subtask(_obs.subtask_cell_to_str(cell))
        if parsed is None:
            continue
        cid = class_id_for_pair(parsed[0], parsed[1], canonical_pairs)
        if cid is not None:
            m[i] = cid
    if label_map is not None:
        logging.info(
            "subtask_eval.map: DONE build_index_to_class_map size=%d in %.2fs (parquet-backed pass)",
            len(m),
            time.perf_counter() - t_all,
        )
    else:
        logging.info(
            "subtask_eval.map: DONE build_index_to_class_map size=%d in %.2fs (second full pass over dataset)",
            len(m),
            time.perf_counter() - t_all,
        )
    return m


def run_subtask_val_eval(
    *,
    policy: _policy.Policy,
    outer_torch_dataset: Any,
    val_indices: np.ndarray,
    canonical_pairs: tuple[tuple[str, str], ...],
    index_to_class: dict[int, int],
    action_horizon: int,
    infer_batch_size: int,
    temperature: float = 0.0,
) -> tuple[dict[str, float], dict[int, tuple[int, int]]]:
    """Returns (wandb_flat_metrics, per_class (correct, total)).

    Processes validation frames in chunks of ``infer_batch_size`` so we never hold tens of
    thousands of full image observations in host RAM (which can trigger OOM-kill with no traceback).
    """
    bs = max(1, infer_batch_size)
    per_class_correct: dict[int, int] = {i: 0 for i in range(len(canonical_pairs))}
    per_class_total: dict[int, int] = {i: 0 for i in range(len(canonical_pairs))}
    overall_correct = 0

    def _norm(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return v.strip()
        return str(v).strip()

    def _accumulate_chunk(obs_chunk: list[dict[str, Any]], meta_chunk: list[tuple[str, str, int]]) -> None:
        nonlocal overall_correct
        if not obs_chunk:
            return
        infer_out = policy.infer_subtask_batch(
            obs_chunk,
            batch_size=bs,
            temperature=temperature,
        )
        if len(infer_out) != len(meta_chunk):
            raise RuntimeError(
                f"infer_subtask_batch: expected {len(meta_chunk)} outputs, got {len(infer_out)}"
            )
        for out, (gt_bs, gt_st, row_idx) in zip(infer_out, meta_chunk, strict=True):
            fields = _subtask_parsing.parse_structured_fields(str(out.get("subtask_text") or ""))
            pred_bs = _norm(fields.get("bottle_state"))
            pred_st = _norm(fields.get("subtask"))
            ok = pred_bs == gt_bs.strip() and pred_st == gt_st.strip()
            cid = index_to_class.get(row_idx)
            if cid is None:
                continue
            per_class_total[cid] += 1
            if ok:
                per_class_correct[cid] += 1
                overall_correct += 1

    pending_obs: list[dict[str, Any]] = []
    pending_meta: list[tuple[str, str, int]] = []
    for idx in val_indices.tolist():
        row = _frame_row_at_index(outer_torch_dataset, idx)
        cell = row.get("subtask")
        parsed = _obs.parse_json_bottle_state_subtask(_obs.subtask_cell_to_str(cell))
        if parsed is None:
            continue
        gt_bs, gt_st = parsed
        obs = _obs.lerobot_row_to_subtask_infer_obs(row, action_horizon=action_horizon, pop_subtask=True)
        pending_obs.append(obs)
        pending_meta.append((gt_bs, gt_st, idx))
        if len(pending_obs) >= bs:
            _accumulate_chunk(pending_obs, pending_meta)
            pending_obs.clear()
            pending_meta.clear()
    _accumulate_chunk(pending_obs, pending_meta)

    if sum(per_class_total.values()) == 0:
        return {}, {i: (0, 0) for i in range(len(canonical_pairs))}

    n_eval = sum(per_class_total.values())
    metrics: dict[str, float] = {}
    metrics["subtask_val/accuracy"] = float(overall_correct / n_eval) if n_eval else 0.0
    for c in range(len(canonical_pairs)):
        tot = per_class_total[c]
        cor = per_class_correct[c]
        metrics[f"subtask_val/acc_class_{c}"] = float(cor / tot) if tot else 0.0
        metrics[f"subtask_val/total_class_{c}"] = float(tot)
    metrics["subtask_val/total"] = float(n_eval)
    metrics["subtask_val/correct"] = float(overall_correct)

    per_class_tuples = {c: (per_class_correct[c], per_class_total[c]) for c in range(len(canonical_pairs))}
    return metrics, per_class_tuples


def log_split_summary(split: SubtaskEvalSplit) -> None:
    logging.info(
        "Subtask eval holdout: trainable canonical frames by class: %s",
        split.per_class_total,
    )
    logging.info("Subtask eval holdout: val counts by class: %s", split.per_class_val)
    logging.info(
        "Subtask eval holdout: %d train indices, %d val indices, %d unknown/mismatch labels",
        split.train_indices.size,
        split.val_indices.size,
        split.unknown_label_indices,
    )


def eval_policy_from_train_state(
    config: _config.TrainConfig,
    train_state: Any,
    *,
    policy_rng: jax.Array,
) -> _policy.Policy:
    """Build a JAX `Policy` with current train weights (same transforms as `create_trained_policy`)."""
    import flax.nnx as nnx
    from openpi.policies import policy_config as _policy_config

    model = nnx.merge(train_state.model_def, train_state.params)
    data_config = config.data.create(config.assets_dirs, config.model)
    return _policy_config.policy_from_model_and_data_config(
        config,
        model,
        rng=policy_rng,
        repack_transforms=data_config.repack_transforms,
    )
