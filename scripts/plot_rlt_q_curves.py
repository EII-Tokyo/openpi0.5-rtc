from __future__ import annotations

import argparse
import functools
import json
import os
import pickle
import textwrap
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
try:
    import seaborn as sns
except ImportError:  # Keep this analysis script usable in the lightweight robot env.
    sns = None

from openpi.rlt import actor_critic
from openpi.rlt import replay


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "executed": "#5477C4",
    "actor": "#CC6F47",
    "target": "#7A828F",
}


def _load_config(path: Path) -> actor_critic.RLTActorCriticConfig:
    data = json.loads(path.read_text())
    allowed = actor_critic.RLTActorCriticConfig.__dataclass_fields__.keys()
    return actor_critic.RLTActorCriticConfig(**{key: data[key] for key in allowed if key in data})


def _episode_label(dataset: replay.ReplayDataset, episode_id: int) -> str:
    mask = dataset.split_episode_id == episode_id
    done = np.asarray(dataset.data["done"])[mask].astype(bool)
    reward = np.asarray(dataset.data["reward"])[mask]
    terminal_reward = reward[done]
    if terminal_reward.size and float(terminal_reward[-1]) > 0.5:
        return "success"
    return "failure"


def _split_indices_by_episode(dataset: replay.ReplayDataset, *, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    episode_ids = np.unique(dataset.split_episode_id)
    rng = np.random.default_rng(seed)
    shuffled = episode_ids.copy()
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    if val_count >= len(shuffled):
        val_count = max(1, len(shuffled) - 1)
    val_episodes = set(int(x) for x in shuffled[:val_count])
    split_episode_id = np.asarray(dataset.split_episode_id)
    val_mask = np.asarray([int(ep) in val_episodes for ep in split_episode_id], dtype=bool)
    all_indices = np.arange(len(dataset), dtype=np.int64)
    return all_indices[~val_mask], all_indices[val_mask]


def _episode_indices(dataset: replay.ReplayDataset, split_indices: np.ndarray, label: str) -> np.ndarray:
    split_episode_id = np.asarray(dataset.split_episode_id)
    for episode_id in np.unique(split_episode_id[split_indices]):
        if _episode_label(dataset, int(episode_id)) == label:
            return np.where(split_episode_id == episode_id)[0]
    raise ValueError(f"No {label} episode found in selected split.")


def _batch_from_indices(dataset: replay.ReplayDataset, indices: np.ndarray) -> replay.ReplayBatch:
    return replay.ReplayBatch(**{key: jnp.asarray(value[indices]) for key, value in dataset.data.items()})


@functools.partial(jax.jit, static_argnames=("config",))
def _score_batch(params, batch: replay.ReplayBatch, config: actor_critic.RLTActorCriticConfig):
    actor_action = actor_critic.actor_apply(
        params["actor"],
        batch.rlt_token,
        batch.state,
        batch.reference_action_chunk,
        config,
    )
    executed_q1 = actor_critic.critic_apply(
        params["critic1"], batch.rlt_token, batch.state, batch.executed_action_chunk, config
    )
    executed_q2 = actor_critic.critic_apply(
        params["critic2"], batch.rlt_token, batch.state, batch.executed_action_chunk, config
    )
    actor_q1 = actor_critic.critic_apply(params["critic1"], batch.rlt_token, batch.state, actor_action, config)
    actor_q2 = actor_critic.critic_apply(params["critic2"], batch.rlt_token, batch.state, actor_action, config)
    actor_mae = jnp.mean(
        jnp.abs(actor_action - batch.executed_action_chunk[:, : config.rlt_chunk_horizon, : config.action_dim]),
        axis=(1, 2),
    )
    return {
        "executed_q": jnp.minimum(executed_q1, executed_q2),
        "actor_q": jnp.minimum(actor_q1, actor_q2),
        "actor_minus_executed_q": jnp.minimum(actor_q1, actor_q2) - jnp.minimum(executed_q1, executed_q2),
        "actor_mae": actor_mae,
    }


def _theme() -> None:
    rc = {
        "figure.facecolor": TOKENS["surface"],
        "figure.edgecolor": "none",
        "savefig.facecolor": TOKENS["surface"],
        "savefig.edgecolor": "none",
        "axes.facecolor": TOKENS["panel"],
        "axes.edgecolor": TOKENS["axis"],
        "axes.labelcolor": TOKENS["ink"],
        "grid.color": TOKENS["grid"],
        "grid.linewidth": 0.8,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
    }
    if sns is not None:
        sns.set_theme(style="whitegrid", rc=rc)
    else:
        plt.rcParams.update(rc)


def _add_header(fig, ax, title: str, subtitle: str) -> None:
    title = textwrap.fill(title, width=80, break_long_words=False)
    subtitle = textwrap.fill(subtitle, width=120, break_long_words=False)
    ax.set_title("")
    fig.subplots_adjust(top=0.84)
    left = ax.get_position().x0
    fig.text(left, 0.975, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, 0.925, subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot(df: np.ndarray, *, title: str, subtitle: str, path: Path) -> None:
    _theme()
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
    ax.plot(df["step_index"], df["executed_q"], label="Q(executed action)", color=COLORS["executed"], linewidth=1.4)
    ax.plot(df["step_index"], df["actor_q"], label="Q(actor action)", color=COLORS["actor"], linewidth=1.4)
    ax.axhline(0.0, color=COLORS["target"], linewidth=1.0, linestyle=":")
    ax.set_xlabel("Replay step index")
    ax.set_ylabel("Critic Q, min(Q1, Q2)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2, borderaxespad=0)
    _add_header(fig, ax, title, subtitle)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_csv(df: np.ndarray, path: Path) -> None:
    header = ",".join(df.dtype.names)
    np.savetxt(path, df, delimiter=",", header=header, comments="", fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f", "%.8f", "%d"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    replay_dir = Path(args.replay_dir)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = replay.ReplayDataset(replay_dir)
    _, val_indices = _split_indices_by_episode(dataset, val_fraction=args.val_fraction, seed=args.seed)
    config = _load_config(checkpoint / "config.json")
    with (checkpoint / "params.pkl").open("rb") as f:
        params = pickle.load(f)

    summary = {}
    for label in ("failure", "success"):
        indices = _episode_indices(dataset, val_indices, label)
        episode_id = int(dataset.split_episode_id[indices[0]])
        batch = _batch_from_indices(dataset, indices)
        scores = jax.tree.map(np.asarray, _score_batch(params, batch, config))
        step_index = np.asarray(dataset.data["step_index"])[indices].astype(np.int32)
        reward = np.asarray(dataset.data["reward"])[indices].astype(np.float32)
        done = np.asarray(dataset.data["done"])[indices].astype(np.int32)
        df = np.zeros(
            indices.shape[0],
            dtype=[
                ("step_index", "i4"),
                ("executed_q", "f4"),
                ("actor_q", "f4"),
                ("actor_minus_executed_q", "f4"),
                ("actor_mae", "f4"),
                ("reward", "f4"),
                ("done", "i4"),
            ],
        )
        df["step_index"] = step_index
        for key in ("executed_q", "actor_q", "actor_minus_executed_q", "actor_mae"):
            df[key] = scores[key]
        df["reward"] = reward
        df["done"] = done
        csv_path = output_dir / f"val_{label}_episode_{episode_id:03d}_q_curve.csv"
        png_path = output_dir / f"val_{label}_episode_{episode_id:03d}_q_curve.png"
        _write_csv(df, csv_path)
        _plot(
            df,
            title=f"RLT critic Q curve for one validation {label} episode",
            subtitle=(
                f"Episode split id {episode_id}; {indices.shape[0]} stride-2 transitions; "
                "lines compare min(Q1,Q2) for recorded executed chunks and fresh actor-generated chunks."
            ),
            path=png_path,
        )
        summary[label] = {
            "episode_id": episode_id,
            "source_file": dataset.source_files[episode_id],
            "transitions": int(indices.shape[0]),
            "csv": str(csv_path),
            "png": str(png_path),
            "executed_q_mean": float(df["executed_q"].mean()),
            "actor_q_mean": float(df["actor_q"].mean()),
            "actor_mae_mean": float(df["actor_mae"].mean()),
            "terminal_reward": float(df["reward"][-1]),
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
