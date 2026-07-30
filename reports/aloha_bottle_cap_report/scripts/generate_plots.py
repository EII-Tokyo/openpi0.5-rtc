#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
PLOT = ART / "plot_data"
FIG = ROOT / "figures"
PLOT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

ds = json.loads((ART / "dataset_statistics.json").read_text())
ckpt = json.loads((ART / "checkpoint_metadata.json").read_text())

plt.rcParams.update({"figure.dpi": 160, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})

lengths = np.asarray(ds["raw_dataset"]["episode_trimmed_lengths"])
with (PLOT / "episode_lengths.csv").open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["episode_index", "trimmed_frames"])
    w.writerows(enumerate(lengths.tolist()))
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.hist(lengths, bins=35, color="#2b6cb0", edgecolor="white")
ax.axvline(np.median(lengths), color="#c53030", linestyle="--", label=f"median={np.median(lengths):.0f}")
ax.set(xlabel="Trimmed trajectory length (frames)", ylabel="Episodes", title="Dataset trajectory-length distribution")
ax.legend(); fig.tight_layout(); fig.savefig(FIG / "dataset_length_distribution.pdf"); plt.close(fig)

labels = ["success label", "failure label"]
counts = [ds["raw_dataset"]["success_by_positive_terminal_reward"], ds["raw_dataset"]["failure_by_zero_reward"]]
with (PLOT / "label_counts.csv").open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["label", "count"]); w.writerows(zip(labels, counts))
fig, ax = plt.subplots(figsize=(5.2, 3.6))
bars = ax.bar(labels, counts, color=["#2f855a", "#c53030"])
ax.bar_label(bars); ax.set(ylabel="Episodes", title="Manual terminal-reward labels (not task success rate)")
fig.tight_layout(); fig.savefig(FIG / "manual_label_counts.pdf"); plt.close(fig)

metric_re = re.compile(r"(\w+)=(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")
round_eval = []
fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.0), sharex=True)
for rnd in range(28, 33):
    rows = []
    for line in (ART / f"train_round{rnd}.log").read_text(errors="replace").splitlines():
        if line.startswith("step="):
            vals = {k: float(v) for k, v in metric_re.findall(line)}
            if "critic_loss" in vals:
                rows.append(vals)
        if line.startswith("eval step="):
            vals = {k: float(v) for k, v in metric_re.findall(line)}
            vals["round"] = rnd
            round_eval.append(vals)
    with (PLOT / f"train_round{rnd}.csv").open("w", newline="") as f:
        keys = sorted({k for row in rows for k in row})
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    steps = [r["step"] for r in rows]
    critic = [r["critic_loss"] for r in rows]
    axes[0].plot(steps, critic, alpha=0.75, linewidth=0.9, label=f"round {rnd}")
    actor_rows = [r for r in rows if "actor_loss" in r]
    axes[1].plot([r["step"] for r in actor_rows], [r["actor_loss"] for r in actor_rows], alpha=0.75, linewidth=0.9)
axes[0].set(ylabel="critic loss", title="Round 28--32 training traces"); axes[0].legend(ncol=5, fontsize=7)
axes[1].set(xlabel="local step", ylabel="actor loss")
fig.tight_layout(); fig.savefig(FIG / "training_losses_round28_32.pdf"); plt.close(fig)

with (PLOT / "round_eval.csv").open("w", newline="") as f:
    keys = ["round", "step", "train_actor_mae", "val_actor_mae", "val_critic_loss", "val_executed_q", "val_actor_q"]
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for row in round_eval:
        w.writerow({k: row.get(k, "") for k in keys})
fig, ax1 = plt.subplots(figsize=(7.2, 3.8))
x = [r["round"] for r in round_eval]
ax1.plot(x, [r["val_actor_mae"] for r in round_eval], "o-", label="validation actor MAE", color="#2b6cb0")
ax1.set(xlabel="Online training round", ylabel="Actor MAE", xticks=x)
ax2 = ax1.twinx()
ax2.plot(x, [r["val_critic_loss"] for r in round_eval], "s--", label="validation critic loss", color="#c53030")
ax2.set_ylabel("Critic loss")
lines = ax1.lines + ax2.lines
ax1.legend(lines, [l.get_label() for l in lines], loc="best")
ax1.set_title("Saved-checkpoint offline validation comparison")
fig.tight_layout(); fig.savefig(FIG / "checkpoint_validation_comparison.pdf"); plt.close(fig)

roots = ckpt["parameters_by_root"]
with (PLOT / "parameter_counts.csv").open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["module", "parameters"]); w.writerows(roots.items())
fig, ax = plt.subplots(figsize=(5.5, 3.5))
bars = ax.bar(roots.keys(), [v / 1e6 for v in roots.values()], color=["#805ad5", "#3182ce", "#319795"])
ax.bar_label(bars, fmt="%.2f M"); ax.set(ylabel="Million parameters", title="Round-32 actor/critic parameter allocation")
fig.tight_layout(); fig.savefig(FIG / "checkpoint_parameter_counts.pdf"); plt.close(fig)

print(f"generated_figures=5 generated_plot_csv={len(list(PLOT.glob('*.csv')))}")
