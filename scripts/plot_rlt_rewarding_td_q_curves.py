#!/usr/bin/env python3
"""Plot per-trajectory rewarding/reference value, TD target, and Q curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openpi.training import rlt_eval
from openpi.training import rlt_trainable_manifest


def _episode_label(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        done = np.asarray(data["done"], dtype=bool)
        reward_seq = np.asarray(data["reward_seq"], dtype=np.float32)
    terminal = float(reward_seq[done].sum()) if np.any(done) else float(reward_seq.sum())
    return int(terminal > 0.0)


def _num_rows(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        return int(data["z_rl"].shape[0])


def _select_shards(paths: list[Path], *, per_class: int) -> list[Path]:
    by_label = {0: [], 1: []}
    for path in paths:
        by_label[_episode_label(path)].append(path)
    selected: list[Path] = []
    for label in (1, 0):
        ranked = sorted(by_label[label], key=lambda path: (-_num_rows(path), path.name))
        selected.extend(ranked[:per_class])
    return selected


def _write_manifest(path: Path, shards: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"shard_path": str(shard)}, sort_keys=True) + "\n" for shard in shards), encoding="utf-8")


def _resolve_actor_dir(path: Path) -> Path:
    if path.name == "LATEST":
        return Path(path.read_text(encoding="utf-8").strip())
    if path.is_dir() and (path / "LATEST").exists():
        return Path((path / "LATEST").read_text(encoding="utf-8").strip())
    if path.is_dir():
        return path
    raise FileNotFoundError(path)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def _plot_one_episode(rows: list[dict[str, Any]], out_path: Path, *, title: str) -> None:
    rows = sorted(rows, key=lambda row: int(row["transition_index"]))
    progress = np.asarray([float(row["progress"]) for row in rows], dtype=np.float32)
    predicted_q = np.asarray([float(row["predicted_q"]) for row in rows], dtype=np.float32)
    actor_q = np.asarray([float(row["actor_q"]) for row in rows], dtype=np.float32)
    target_q = np.asarray([float(row["target_q"]) for row in rows], dtype=np.float32)
    reference_value = np.asarray([float(row["reference_value"]) for row in rows], dtype=np.float32)
    actor_delta = np.asarray([float(row["actor_delta_norm"]) for row in rows], dtype=np.float32)

    fig, axis = plt.subplots(figsize=(13.5, 6.6))
    axis.plot(progress, predicted_q, color="#159947", linewidth=2.8, label="Q(data action)")
    if np.any(np.isfinite(actor_q)):
        axis.plot(progress, actor_q, color="#2563eb", linewidth=2.8, label="Q(actor action)")
    axis.plot(progress, target_q, color="#64748b", linestyle="--", linewidth=2.4, label="TD target")
    axis.plot(progress, reference_value, color="#dc2626", linestyle=":", linewidth=2.2, label="rewarding/reference value")
    axis.axhline(0.0, color="#94a3b8", linewidth=1.0, alpha=0.6)
    axis.set_xlabel("trajectory progress")
    axis.set_ylabel("Q / target / reference value")
    axis.grid(True, alpha=0.24)
    axis.set_title(title)

    delta_axis = axis.twinx()
    if np.any(np.isfinite(actor_delta)):
        delta_axis.plot(progress, actor_delta, color="#c084fc", linewidth=2.0, alpha=0.78, label="actor delta norm")
    delta_axis.set_ylabel("actor delta norm")

    handles, labels = axis.get_legend_handles_labels()
    delta_handles, delta_labels = delta_axis.get_legend_handles_labels()
    axis.legend(handles + delta_handles, labels + delta_labels, loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_audit(audit_json: Path, out_path: Path) -> None:
    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    rows = audit.get("rows", [])
    if not rows:
        return
    values = np.asarray([float(row["x_adjacent_exact_fraction"]) for row in rows], dtype=np.float32)
    fig, axis = plt.subplots(figsize=(9.5, 4.8))
    axis.hist(values, bins=30, color="#0f766e", alpha=0.82)
    axis.axvline(float(np.mean(values)), color="#dc2626", linestyle="--", linewidth=2.0, label=f"mean={float(np.mean(values)):.4f}")
    axis.set_xlabel("adjacent exact repeat fraction of x=(z_rl, proprio)")
    axis.set_ylabel("number of trajectories")
    axis.set_title("Paper-anchor replay state repeat audit")
    axis.grid(True, alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_html(output_dir: Path, image_paths: list[Path], summary: dict[str, Any]) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>RLT rewarding TD Q curves</title>",
        "<style>body{font-family:Inter,Arial,sans-serif;margin:32px;background:#f8fafc;color:#111827}"
        "img{max-width:100%;display:block;margin:18px 0 34px;border:1px solid #d1d5db;background:white}"
        "code{background:#e5e7eb;padding:2px 5px;border-radius:4px}pre{background:#111827;color:#f9fafb;padding:16px;overflow:auto}</style>",
        "</head><body><h1>RLT rewarding + TD + Q curves</h1>",
        "<pre>" + json.dumps(summary, ensure_ascii=False, indent=2) + "</pre>",
    ]
    for image in image_paths:
        parts.append(f"<h2>{image.stem}</h2><img src='{image.name}' alt='{image.stem}'>")
    parts.append("</body></html>")
    (output_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-dir", required=True, type=Path)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--score-batch-size", type=int, default=512)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    actor_dir = _resolve_actor_dir(args.actor_dir)
    config, critic, actor, metadata = rlt_eval.load_inference_modules(actor_dir)
    all_paths = rlt_trainable_manifest.read_manifest_paths(args.manifest_path)
    selected = _select_shards(all_paths, per_class=args.per_class)
    selected_manifest = args.output_dir / "selected_curve_shards.jsonl"
    _write_manifest(selected_manifest, selected)

    arrays, rows, skipped = rlt_eval.load_holdout_arrays(selected, config=config)
    if arrays is None:
        raise RuntimeError(f"No selected shards could be loaded. skipped={skipped}")
    arrays = rlt_eval.apply_z_rl_normalization(arrays, metadata)
    scored = rlt_eval.score_holdout_rows(
        arrays,
        rows,
        critic=critic,
        actor=actor,
        config=config,
        score_batch_size=args.score_batch_size,
    )
    _write_rows(args.output_dir / "selected_curve_rows.csv", scored)

    image_paths: list[Path] = []
    if args.audit_json is not None and args.audit_json.exists():
        audit_path = args.output_dir / "paper_anchor_state_repeat_audit.png"
        _plot_audit(args.audit_json, audit_path)
        image_paths.append(audit_path)

    by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in scored:
        by_episode.setdefault(str(row["episode_id"]), []).append(row)
    for episode_id, episode_rows in sorted(by_episode.items()):
        label = int(episode_rows[0]["label"])
        out_path = args.output_dir / f"{'success' if label else 'failure'}_{episode_id}_td_q.png"
        _plot_one_episode(
            episode_rows,
            out_path,
            title=f"{'SUCCESS' if label else 'FAILURE'} sample: {episode_id}",
        )
        image_paths.append(out_path)

    summary = {
        "actor_dir": str(actor_dir),
        "actor_step": metadata.get("step"),
        "manifest_path": str(args.manifest_path),
        "selected_manifest": str(selected_manifest),
        "num_selected_shards": len(selected),
        "num_selected_transitions": len(scored),
        "success_shards": len({row["episode_id"] for row in scored if int(row["label"]) == 1}),
        "failure_shards": len({row["episode_id"] for row in scored if int(row["label"]) == 0}),
        "skipped": skipped,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_html(args.output_dir, image_paths, summary)
    print(json.dumps({"html": str(args.output_dir / "index.html"), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
