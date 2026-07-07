#!/usr/bin/env python3
"""Audit whether runtime-cached RLT tokens match formal paper-anchor tokens."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class AuditPaths:
    runtime_npz: Path
    formal_npz: Path
    h5_path: Path


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / (norms + 1e-12)


def _cosine_rowwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = min(len(a), len(b))
    if n <= 0:
        return np.asarray([], dtype=np.float32)
    return np.sum(_normalize_rows(a[:n]) * _normalize_rows(b[:n]), axis=-1)


def _best_cosine(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(a) == 0 or len(b) == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.int64)
    sim = _normalize_rows(a) @ _normalize_rows(b).T
    indices = np.argmax(sim, axis=1)
    return sim[np.arange(sim.shape[0]), indices], indices.astype(np.int64)


def _metric_stats(prefix: str, values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {
            f"{prefix}_min": None,
            f"{prefix}_mean": None,
            f"{prefix}_max": None,
            f"{prefix}_p05": None,
        }
    return {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_p05": float(np.quantile(values, 0.05)),
    }


def _unique_rows_rounded(values: np.ndarray, decimals: int = 6) -> int:
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] == 0:
        return 0
    return int(np.unique(np.round(values, decimals=decimals), axis=0).shape[0])


def _load_manifest(npz: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "manifest" not in npz.files:
        return {}
    raw = npz["manifest"]
    try:
        text = raw.item() if raw.shape == () else raw.tolist()
    except Exception:
        text = str(raw)
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    if not isinstance(text, str):
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _load_z_from_npz(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=True) as data:
        if "z_rl" not in data.files:
            raise KeyError(f"{path} does not contain z_rl")
        return np.asarray(data["z_rl"], dtype=np.float32), _load_manifest(data)


def _load_cached_z_from_h5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as root:
        if "rlt/cached_z_rl" not in root:
            raise KeyError(f"{path} does not contain rlt/cached_z_rl")
        return np.asarray(root["rlt/cached_z_rl"][:], dtype=np.float32)


def audit_one_key_region(
    *,
    key_region_id: str,
    runtime_npz: Path,
    formal_npz: Path,
    h5_path: Path,
) -> dict[str, Any]:
    runtime_z, runtime_manifest = _load_z_from_npz(runtime_npz)
    formal_z, formal_manifest = _load_z_from_npz(formal_npz)
    cached_z = _load_cached_z_from_h5(h5_path)

    formal_vs_runtime = _cosine_rowwise(formal_z, runtime_z)
    runtime_vs_cached_best, runtime_best_indices = _best_cosine(runtime_z, cached_z)
    formal_vs_cached_best, formal_best_indices = _best_cosine(formal_z, cached_z)

    row: dict[str, Any] = {
        "key_region_id": key_region_id,
        "runtime_npz": str(runtime_npz),
        "formal_npz": str(formal_npz),
        "h5_path": str(h5_path),
        "reward": formal_manifest.get("reward", runtime_manifest.get("reward")),
        "runtime_rows": int(runtime_z.shape[0]),
        "formal_rows": int(formal_z.shape[0]),
        "cached_rows": int(cached_z.shape[0]),
        "z_dim": int(formal_z.shape[1]) if formal_z.ndim == 2 else None,
        "runtime_unique_rows_rounded6": _unique_rows_rounded(runtime_z),
        "formal_unique_rows_rounded6": _unique_rows_rounded(formal_z),
        "cached_unique_rows_rounded6": _unique_rows_rounded(cached_z),
        "runtime_vs_cached_best_unique_indices": int(len(set(runtime_best_indices.tolist()))),
        "formal_vs_cached_best_unique_indices": int(len(set(formal_best_indices.tolist()))),
        "formal_replay_ready": formal_manifest.get("formal_replay_ready"),
        "formal_z_rl_source": formal_manifest.get("z_rl_source"),
        "runtime_replay_status": runtime_manifest.get("replay_status"),
    }
    row.update(_metric_stats("formal_vs_runtime_rowwise_cos", formal_vs_runtime))
    row.update(_metric_stats("runtime_vs_cached_best_cos", runtime_vs_cached_best))
    row.update(_metric_stats("formal_vs_cached_best_cos", formal_vs_cached_best))
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"sample_count": len(rows)}
    if not rows:
        summary.update(
            {
                "is_runtime_cached_equivalent_to_formal": False,
                "severity": "blocked",
                "reason": "no comparable rows",
            }
        )
        return summary

    numeric_fields = [
        "formal_vs_runtime_rowwise_cos_mean",
        "formal_vs_runtime_rowwise_cos_min",
        "formal_vs_cached_best_cos_mean",
        "formal_vs_cached_best_cos_min",
        "runtime_vs_cached_best_cos_mean",
    ]
    for field in numeric_fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        if values:
            summary[f"{field}_overall_mean"] = float(np.mean(values))
            summary[f"{field}_overall_min"] = float(np.min(values))

    runtime_unique_ratio = [
        float(row["runtime_unique_rows_rounded6"]) / max(float(row["runtime_rows"]), 1.0)
        for row in rows
        if row.get("runtime_rows") is not None
    ]
    formal_unique_ratio = [
        float(row["formal_unique_rows_rounded6"]) / max(float(row["formal_rows"]), 1.0)
        for row in rows
        if row.get("formal_rows") is not None
    ]
    summary["runtime_unique_row_ratio_mean"] = float(np.mean(runtime_unique_ratio)) if runtime_unique_ratio else None
    summary["formal_unique_row_ratio_mean"] = float(np.mean(formal_unique_ratio)) if formal_unique_ratio else None

    best_mean = float(summary.get("formal_vs_cached_best_cos_mean_overall_mean", 0.0) or 0.0)
    rowwise_mean = float(summary.get("formal_vs_runtime_rowwise_cos_mean_overall_mean", 0.0) or 0.0)
    equivalent = best_mean >= 0.999 and rowwise_mean >= 0.999
    summary["is_runtime_cached_equivalent_to_formal"] = bool(equivalent)
    if equivalent:
        summary["severity"] = "low"
        summary["reason"] = "runtime cached z and formal z are numerically equivalent under cosine thresholds"
    elif best_mean < 0.95 or rowwise_mean < 0.95:
        summary["severity"] = "high"
        summary["reason"] = "cosine similarity is far below numerical-noise range; likely different token semantics or input path"
    else:
        summary["severity"] = "medium"
        summary["reason"] = "tokens are close but not equivalent under strict threshold"
    return summary


def _key_id_from_npz(path: Path) -> str:
    name = path.stem
    return name.removeprefix("key_region_")


def discover_paths(runtime_replay_root: Path, formal_replay_root: Path, rollout_root: Path) -> dict[str, AuditPaths]:
    runtime_by_id = {_key_id_from_npz(path): path for path in runtime_replay_root.rglob("key_region_*.npz")}
    formal_by_id = {_key_id_from_npz(path): path for path in formal_replay_root.rglob("key_region_*.npz")}
    h5_by_id = {
        path.parent.name.removeprefix("key_region_"): path
        for path in rollout_root.rglob("key_region_*/episode.hdf5")
    }
    common = sorted(set(runtime_by_id) & set(formal_by_id) & set(h5_by_id))
    return {
        key: AuditPaths(runtime_npz=runtime_by_id[key], formal_npz=formal_by_id[key], h5_path=h5_by_id[key])
        for key in common
    }


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "runtime_vs_formal_z_consistency.csv"
    json_path = output_dir / "runtime_vs_formal_z_consistency.json"
    report_path = output_dir / "runtime_vs_formal_z_consistency_report.md"

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False))
    report_path.write_text(_render_markdown(rows, summary, csv_path=csv_path, json_path=json_path))


def _render_markdown(rows: list[dict[str, Any]], summary: dict[str, Any], *, csv_path: Path, json_path: Path) -> str:
    lines = [
        "# Runtime cached z 与 formal same-forward z 一致性审计",
        "",
        "## 结论",
        "",
        f"- 样本数：{summary.get('sample_count')}",
        f"- 是否等价：{summary.get('is_runtime_cached_equivalent_to_formal')}",
        f"- 严重程度：{summary.get('severity')}",
        f"- 原因：{summary.get('reason')}",
        "",
        "## 核心指标",
        "",
        f"- formal vs runtime shard rowwise cosine mean：{summary.get('formal_vs_runtime_rowwise_cos_mean_overall_mean')}",
        f"- formal vs runtime shard rowwise cosine min：{summary.get('formal_vs_runtime_rowwise_cos_min_overall_min')}",
        f"- formal vs HDF5 cached best cosine mean：{summary.get('formal_vs_cached_best_cos_mean_overall_mean')}",
        f"- formal vs HDF5 cached best cosine min：{summary.get('formal_vs_cached_best_cos_min_overall_min')}",
        f"- runtime unique row ratio mean：{summary.get('runtime_unique_row_ratio_mean')}",
        f"- formal unique row ratio mean：{summary.get('formal_unique_row_ratio_mean')}",
        "",
        "## 解释",
        "",
        "如果两种 token 只是 GPU 或浮点细节不同，cosine 应该接近 0.999 甚至更高。",
        "如果 cosine 只有 0.85-0.95，通常说明输入图像、预处理、forward 路径、token 层或采样 grain 不一致。",
        "",
        "runtime unique row ratio 低，说明旧 runtime replay 中很多 transition 共用同一个 cached token，",
        "这符合 `runtime_action_cache_block` 粒度，不符合正式训练需要的 `paper_subsampled_anchor` 粒度。",
        "",
        "## 明细文件",
        "",
        f"- CSV：`{csv_path}`",
        f"- JSON：`{json_path}`",
        "",
        "## 样本明细",
        "",
        "| key_region_id | reward | rows runtime/formal/cached | runtime unique | formal unique | formal-runtime mean | formal-cached best mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:50]:
        lines.append(
            "| {key_region_id} | {reward} | {runtime_rows}/{formal_rows}/{cached_rows} | "
            "{runtime_unique_rows_rounded6} | {formal_unique_rows_rounded6} | "
            "{formal_vs_runtime_rowwise_cos_mean:.6f} | {formal_vs_cached_best_cos_mean:.6f} |".format(**row)
        )
    if len(rows) > 50:
        lines.append(f"| ... | ... | ... | ... | ... | ... | ... |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-replay-root", type=Path, required=True)
    parser.add_argument("--formal-replay-root", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    discovered = discover_paths(args.runtime_replay_root, args.formal_replay_root, args.rollout_root)
    items = list(discovered.items())
    if args.limit and args.limit > 0:
        items = items[: args.limit]
    rows = [
        audit_one_key_region(
            key_region_id=key,
            runtime_npz=paths.runtime_npz,
            formal_npz=paths.formal_npz,
            h5_path=paths.h5_path,
        )
        for key, paths in items
    ]
    summary = summarize_rows(rows)
    write_outputs(rows, summary, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
