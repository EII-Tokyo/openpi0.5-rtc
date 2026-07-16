from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def split_episode_ids(episode_ids: Sequence[str]) -> dict[str, list[str]]:
    ids = list(episode_ids)
    if len(ids) < 3:
        raise ValueError("need at least three episodes for identification/validation/heldout split")
    n = len(ids)
    n_ident = max(1, int(round(n * 0.6)))
    n_val = max(1, int(round(n * 0.2)))
    if n_ident + n_val >= n:
        n_ident = max(1, n - 2)
        n_val = 1
    return {
        "identification": ids[:n_ident],
        "validation": ids[n_ident : n_ident + n_val],
        "heldout": ids[n_ident + n_val :],
    }


def rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(pred, dtype=np.float64) - np.asarray(ref, dtype=np.float64)))))


def mae(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(ref, dtype=np.float64))))


def model_m0(action: np.ndarray, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    samples = min(action.shape[0], qpos.shape[0])
    return action[:samples], qpos[:samples]


def model_m1(action: np.ndarray, qpos: np.ndarray, *, delay: int) -> tuple[np.ndarray, np.ndarray]:
    if delay < 0:
        raise ValueError("delay must be non-negative")
    samples = min(action.shape[0], qpos.shape[0] - delay)
    return action[:samples], qpos[delay : delay + samples]


def rollout_first_order(action: np.ndarray, qpos: np.ndarray, *, delay: int, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    samples = min(action.shape[0], qpos.shape[0] - 1)
    if samples <= delay:
        raise ValueError("episode too short for delay")
    q_hat = np.asarray(qpos[0], dtype=np.float64).copy()
    preds = []
    refs = []
    for t in range(samples):
        src = max(0, t - delay)
        q_hat = q_hat + alpha * (action[src] - q_hat)
        preds.append(q_hat.copy())
        refs.append(qpos[t + 1].copy())
    return np.asarray(preds), np.asarray(refs)


def rollout_first_order_velocity_clip(
    action: np.ndarray,
    qpos: np.ndarray,
    *,
    delay: int,
    alpha: np.ndarray,
    velocity_limit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    samples = min(action.shape[0], qpos.shape[0] - 1)
    if samples <= delay:
        raise ValueError("episode too short for delay")
    q_hat = np.asarray(qpos[0], dtype=np.float64).copy()
    preds = []
    refs = []
    for t in range(samples):
        src = max(0, t - delay)
        step = alpha * (action[src] - q_hat)
        step = np.clip(step, -velocity_limit, velocity_limit)
        q_hat = q_hat + step
        preds.append(q_hat.copy())
        refs.append(qpos[t + 1].copy())
    return np.asarray(preds), np.asarray(refs)


def _concat_model_errors(episodes: Sequence[dict[str, np.ndarray]], model_fn) -> tuple[np.ndarray, np.ndarray]:
    preds = []
    refs = []
    for episode in episodes:
        pred, ref = model_fn(episode["action"], episode["qpos"])
        preds.append(pred)
        refs.append(ref)
    return np.concatenate(preds, axis=0), np.concatenate(refs, axis=0)


def fit_first_order_alpha(episodes: Sequence[dict[str, np.ndarray]], *, delay: int) -> np.ndarray:
    alphas = np.linspace(0.01, 1.0, 100)
    joint_count = episodes[0]["action"].shape[1]
    best = np.zeros(joint_count, dtype=np.float64)
    for joint in range(joint_count):
        best_rmse = float("inf")
        for alpha in alphas:
            errors = []
            for episode in episodes:
                pred, ref = rollout_first_order(
                    episode["action"][:, [joint]], episode["qpos"][:, [joint]], delay=delay, alpha=np.asarray([alpha])
                )
                errors.append(pred - ref)
            score = float(np.sqrt(np.mean(np.square(np.concatenate(errors, axis=0)))))
            if score < best_rmse:
                best_rmse = score
                best[joint] = alpha
    return best


def estimate_velocity_limit(episodes: Sequence[dict[str, np.ndarray]], percentile: float = 95.0) -> np.ndarray:
    diffs = []
    for episode in episodes:
        diffs.append(np.abs(np.diff(episode["qpos"], axis=0)))
    values = np.concatenate(diffs, axis=0)
    return np.maximum(np.percentile(values, percentile, axis=0), 1e-6)


def evaluate_offline_models(
    episodes_by_id: dict[str, dict[str, np.ndarray]],
    splits: dict[str, list[str]],
    *,
    common_delay: int,
) -> dict[str, Any]:
    ident = [episodes_by_id[episode_id] for episode_id in splits["identification"]]
    alpha = fit_first_order_alpha(ident, delay=common_delay)
    velocity_limit = estimate_velocity_limit(ident)

    def evaluate(split_name: str) -> dict[str, Any]:
        episodes = [episodes_by_id[episode_id] for episode_id in splits[split_name]]
        out: dict[str, Any] = {}
        for model_name, model_fn in {
            "M0": lambda a, q: model_m0(a, q),
            "M1": lambda a, q: model_m1(a, q, delay=common_delay),
            "M2": lambda a, q: rollout_first_order(a, q, delay=common_delay, alpha=alpha),
            "M3": lambda a, q: rollout_first_order_velocity_clip(
                a, q, delay=common_delay, alpha=alpha, velocity_limit=velocity_limit
            ),
        }.items():
            pred, ref = _concat_model_errors(episodes, lambda a_q, q_q, fn=model_fn: fn(a_q, q_q))  # type: ignore[arg-type]
            out[model_name] = {"rmse": rmse(pred, ref), "mae": mae(pred, ref)}
        return out

    return {
        "common_delay": int(common_delay),
        "alpha": alpha.tolist(),
        "velocity_limit_per_step": velocity_limit.tolist(),
        "splits": splits,
        "identification": evaluate("identification"),
        "validation": evaluate("validation"),
        "heldout": evaluate("heldout"),
    }

