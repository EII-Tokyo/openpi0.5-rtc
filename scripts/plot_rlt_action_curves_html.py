#!/usr/bin/env python3
"""Plot action curves for selected successful and failed RLT replay trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openpi.models import rlt
from openpi.training import rlt_eval
from openpi.training import rlt_trainable_manifest


def _resolve_actor_dir(path: Path) -> Path:
    if path.name == "LATEST":
        return Path(path.read_text(encoding="utf-8").strip())
    if path.is_dir() and (path / "LATEST").exists():
        return Path((path / "LATEST").read_text(encoding="utf-8").strip())
    if path.is_dir():
        return path
    raise FileNotFoundError(path)


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
        selected.extend(sorted(by_label[label], key=lambda p: (-_num_rows(p), p.name))[:per_class])
    return selected


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def _actor_actions(
    arrays: dict[str, np.ndarray],
    *,
    actor: rlt.RLTActor | None,
    batch_size: int,
) -> np.ndarray:
    if actor is None:
        return np.asarray(arrays["reference_action"], dtype=np.float32)

    @jax.jit
    def act_batch(z_rl, proprio, reference_action):
        x = rlt.make_state(z_rl, proprio)
        return actor(x, reference_action, sample=False)

    outputs: list[np.ndarray] = []
    total = int(arrays["z_rl"].shape[0])
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        result = act_batch(
            jnp.asarray(arrays["z_rl"][start:end]),
            jnp.asarray(arrays["proprio"][start:end]),
            jnp.asarray(arrays["reference_action"][start:end]),
        )
        outputs.append(np.asarray(jax.device_get(result), dtype=np.float32))
    return np.concatenate(outputs, axis=0)


def _plot_episode(
    *,
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    actor_action: np.ndarray,
    output_path: Path,
) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: int(row["transition_index"]))
    indices = np.asarray([int(row["row_index"]) for row in rows], dtype=np.int64)
    progress = np.asarray([float(row["progress"]) for row in rows], dtype=np.float32)
    label = int(rows[0]["label"])
    episode_id = str(rows[0]["episode_id"])

    data_action = np.asarray(arrays["action"][indices], dtype=np.float32)
    reference_action = np.asarray(arrays["reference_action"][indices], dtype=np.float32)
    actor = np.asarray(actor_action[indices], dtype=np.float32)

    data_ref = data_action - reference_action
    actor_ref = actor - reference_action
    actor_data = actor - data_action
    data_ref_norm = np.linalg.norm(data_ref.reshape(data_ref.shape[0], -1), axis=1)
    actor_ref_norm = np.linalg.norm(actor_ref.reshape(actor_ref.shape[0], -1), axis=1)
    actor_data_norm = np.linalg.norm(actor_data.reshape(actor_data.shape[0], -1), axis=1)
    chunk_mean_actor_ref = np.mean(actor_ref, axis=1)
    chunk_mean_data_ref = np.mean(data_ref, axis=1)
    vmax = float(max(np.max(np.abs(chunk_mean_actor_ref)), np.max(np.abs(chunk_mean_data_ref)), 1e-6))

    fig, axes = plt.subplots(3, 1, figsize=(13.8, 10.5), gridspec_kw={"height_ratios": [1.1, 1.0, 1.0]})
    axes[0].plot(progress, data_ref_norm, color="#2563eb", linewidth=2.2, label="executed - reference")
    axes[0].plot(progress, actor_ref_norm, color="#dc2626", linewidth=2.2, label="actor - reference")
    axes[0].plot(progress, actor_data_norm, color="#16a34a", linewidth=2.0, label="actor - executed")
    axes[0].set_title(f"{'SUCCESS' if label else 'FAILURE'} action curves: {episode_id}")
    axes[0].set_xlabel("trajectory progress")
    axes[0].set_ylabel("chunk L2 norm")
    axes[0].grid(True, alpha=0.24)
    axes[0].legend(loc="upper left")

    image0 = axes[1].imshow(
        chunk_mean_data_ref.T,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        extent=[float(progress[0]), float(progress[-1]), -0.5, chunk_mean_data_ref.shape[1] - 0.5],
    )
    axes[1].set_title("executed - reference, averaged across action horizon")
    axes[1].set_ylabel("action dimension")
    fig.colorbar(image0, ax=axes[1], fraction=0.02, pad=0.01)

    image1 = axes[2].imshow(
        chunk_mean_actor_ref.T,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        extent=[float(progress[0]), float(progress[-1]), -0.5, chunk_mean_actor_ref.shape[1] - 0.5],
    )
    axes[2].set_title("actor - reference, averaged across action horizon")
    axes[2].set_xlabel("trajectory progress")
    axes[2].set_ylabel("action dimension")
    fig.colorbar(image1, ax=axes[2], fraction=0.02, pad=0.01)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)

    return {
        "episode_id": episode_id,
        "label": label,
        "num_transitions": len(rows),
        "data_ref_norm_mean": float(np.mean(data_ref_norm)),
        "actor_ref_norm_mean": float(np.mean(actor_ref_norm)),
        "actor_data_norm_mean": float(np.mean(actor_data_norm)),
        "data_ref_norm_max": float(np.max(data_ref_norm)),
        "actor_ref_norm_max": float(np.max(actor_ref_norm)),
        "image": output_path.name,
    }


def _write_html(output_dir: Path, summary: dict[str, Any], episode_summaries: list[dict[str, Any]]) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>RLT action curves</title>",
        "<style>body{font-family:Inter,Arial,sans-serif;margin:30px;background:#f8fafc;color:#111827}"
        "img{max-width:100%;display:block;margin:14px 0 34px;border:1px solid #d1d5db;background:white}"
        "pre{background:#111827;color:#f9fafb;padding:16px;overflow:auto}"
        "table{border-collapse:collapse;background:white}td,th{border:1px solid #d1d5db;padding:6px 9px}</style>",
        "</head><body><h1>RLT success/failure action curves</h1>",
        "<pre>" + json.dumps(summary, ensure_ascii=False, indent=2) + "</pre>",
        "<table><thead><tr><th>episode</th><th>label</th><th>rows</th><th>actor-ref mean</th><th>actor-ref max</th></tr></thead><tbody>",
    ]
    for row in episode_summaries:
        label = "success" if int(row["label"]) else "failure"
        parts.append(
            "<tr>"
            f"<td>{row['episode_id']}</td><td>{label}</td><td>{row['num_transitions']}</td>"
            f"<td>{row['actor_ref_norm_mean']:.6f}</td><td>{row['actor_ref_norm_max']:.6f}</td>"
            "</tr>"
        )
    parts.append("</tbody></table>")
    for row in episode_summaries:
        label = "SUCCESS" if int(row["label"]) else "FAILURE"
        parts.append(f"<h2>{label}: {row['episode_id']}</h2><img src='{row['image']}' alt='{row['episode_id']}'>")
    parts.append("</body></html>")
    (output_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-dir", required=True, type=Path)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--score-batch-size", type=int, default=512)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    actor_dir = _resolve_actor_dir(args.actor_dir)
    config, _critic, actor, metadata = rlt_eval.load_inference_modules(actor_dir)
    all_paths = rlt_trainable_manifest.read_manifest_paths(args.manifest_path)
    selected = _select_shards(all_paths, per_class=args.per_class)
    arrays, rows, skipped = rlt_eval.load_holdout_arrays(selected, config=config)
    if arrays is None:
        raise RuntimeError(f"No selected shards could be loaded. skipped={skipped}")
    arrays = rlt_eval.apply_z_rl_normalization(arrays, metadata)
    actor_action = _actor_actions(arrays, actor=actor, batch_size=args.score_batch_size)

    by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(str(row["episode_id"]), []).append(row)

    episode_summaries: list[dict[str, Any]] = []
    for episode_id, episode_rows in sorted(by_episode.items()):
        label = int(episode_rows[0]["label"])
        image_path = args.output_dir / f"{'success' if label else 'failure'}_{episode_id}_actions.png"
        episode_summaries.append(
            _plot_episode(rows=episode_rows, arrays=arrays, actor_action=actor_action, output_path=image_path)
        )

    summary = {
        "actor_dir": str(actor_dir),
        "actor_step": metadata.get("step"),
        "manifest_path": str(args.manifest_path),
        "num_selected_shards": len(selected),
        "num_selected_transitions": len(rows),
        "success_shards": sum(1 for row in episode_summaries if int(row["label"]) == 1),
        "failure_shards": sum(1 for row in episode_summaries if int(row["label"]) == 0),
        "skipped": skipped,
    }
    _write_rows(args.output_dir / "episode_action_summary.csv", episode_summaries)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_html(args.output_dir, summary, episode_summaries)
    print(json.dumps({"html": str(args.output_dir / "index.html"), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
