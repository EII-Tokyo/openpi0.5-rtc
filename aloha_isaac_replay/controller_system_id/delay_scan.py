from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from aloha_isaac_replay.controller_system_id.continuous_joints import nearest_equivalent_sequence


def _metrics(error: np.ndarray, pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    if error.size == 0:
        raise ValueError("cannot compute metrics on empty arrays")
    if pred.size > 1 and float(np.std(pred)) > 1e-12 and float(np.std(ref)) > 1e-12:
        corr = float(np.corrcoef(pred.reshape(-1), ref.reshape(-1))[0, 1])
    else:
        corr = 0.0
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs": float(np.max(np.abs(error))),
        "bias": float(np.mean(error)),
        "correlation": corr,
    }


def scan_action_qpos_delays(
    action_targets: np.ndarray,
    qpos_observed: np.ndarray,
    *,
    max_delay: int,
    joint_names: Sequence[str],
) -> dict[str, Any]:
    actions = np.asarray(action_targets, dtype=np.float64)
    qpos = np.asarray(qpos_observed, dtype=np.float64)
    if actions.ndim != 2 or qpos.ndim != 2:
        raise ValueError("action_targets and qpos_observed must be 2D")
    if actions.shape[1] != qpos.shape[1] or actions.shape[1] != len(joint_names):
        raise ValueError(
            f"shape/name mismatch: actions={actions.shape}, qpos={qpos.shape}, joint_names={len(joint_names)}"
        )
    if max_delay < 0:
        raise ValueError("max_delay must be non-negative")

    rows = []
    per_joint: dict[str, dict[str, Any]] = {name: {"rows": []} for name in joint_names}
    for delay in range(max_delay + 1):
        samples = min(actions.shape[0], qpos.shape[0] - delay)
        if samples <= 0:
            continue
        raw_pred = actions[:samples]
        ref = qpos[delay : delay + samples]
        nearest_pred, wrap_counts = nearest_equivalent_sequence(raw_pred, ref, joint_names)
        raw_error = raw_pred - ref
        nearest_error = nearest_pred - ref
        aggregate = {
            "delay": delay,
            "samples": int(samples),
            **_metrics(nearest_error, nearest_pred, ref),
            "raw_rmse": float(np.sqrt(np.mean(np.square(raw_error)))),
            "raw_mae": float(np.mean(np.abs(raw_error))),
            "raw_max_abs": float(np.max(np.abs(raw_error))),
            "wrap_events": int(sum(wrap_counts.values())),
        }
        rows.append(aggregate)
        for idx, name in enumerate(joint_names):
            joint_raw_error = raw_error[:, idx]
            joint_nearest_error = nearest_error[:, idx]
            joint_row = {
                "delay": delay,
                "samples": int(samples),
                **_metrics(joint_nearest_error, nearest_pred[:, idx], ref[:, idx]),
                "raw_rmse": float(np.sqrt(np.mean(np.square(joint_raw_error)))),
                "raw_mae": float(np.mean(np.abs(joint_raw_error))),
                "raw_max_abs": float(np.max(np.abs(joint_raw_error))),
                "real_qpos_min": float(np.min(ref[:, idx])),
                "real_qpos_max": float(np.max(ref[:, idx])),
                "target_min": float(np.min(raw_pred[:, idx])),
                "target_max": float(np.max(raw_pred[:, idx])),
                "nearest_target_min": float(np.min(nearest_pred[:, idx])),
                "nearest_target_max": float(np.max(nearest_pred[:, idx])),
                "wrap_events": int(wrap_counts.get(name, 0)),
            }
            per_joint[name]["rows"].append(joint_row)

    if not rows:
        raise ValueError("delay scan produced no rows")
    best = min(rows, key=lambda row: row["rmse"])
    result_per_joint = {}
    for name, payload in per_joint.items():
        best_joint = min(payload["rows"], key=lambda row: row["rmse"])
        result_per_joint[name] = {"best_delay": int(best_joint["delay"]), "best": best_joint, "rows": payload["rows"]}
    return {
        "range": [0, int(max_delay)],
        "aggregate": {"best_delay": int(best["delay"]), "best": best, "rows": rows},
        "per_joint": result_per_joint,
    }

