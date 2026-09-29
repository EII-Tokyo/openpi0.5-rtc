#!/usr/bin/env python3
"""Offline hidden-feature and residual-block ablation probe for a trained Pi0 policy."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import pathlib
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.data import datasets as data_datasets
from openpi.data import transforms
from openpi.models import gemma
from openpi.models import model as model_lib
from openpi.shared import nnx_utils
from openpi.training import config as training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--denoising-steps", type=int, default=10)
    return parser.parse_args()


def host_stats(value: np.ndarray, previous: np.ndarray | None) -> dict[str, float | None]:
    x = np.asarray(value, dtype=np.float32)
    flat = x.reshape(-1)
    result: dict[str, float | None] = {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "rms": float(np.sqrt(np.mean(np.square(flat)))),
        "abs_mean": float(np.mean(np.abs(flat))),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "zero_fraction": float(np.mean(np.abs(flat) < 1e-6)),
        "l2_per_token": float(np.mean(np.linalg.norm(x, axis=-1))),
        "cosine_to_input": None,
        "block_influence": None,
        "relative_change": None,
    }
    if previous is not None:
        prev = np.asarray(previous, dtype=np.float32).reshape(-1)
        denom = float(np.linalg.norm(prev) * np.linalg.norm(flat))
        cosine = float(np.dot(prev, flat) / max(denom, 1e-12))
        result["cosine_to_input"] = cosine
        result["block_influence"] = 1.0 - cosine
        result["relative_change"] = float(np.linalg.norm(flat - prev) / max(np.linalg.norm(prev), 1e-12))
    return result


def add_stack_stats(rows: list[dict], component: str, stack: np.ndarray, initial: np.ndarray | None) -> None:
    previous = initial
    for layer, value in enumerate(stack):
        rows.append({"component": component, "layer": layer, **host_stats(value, previous)})
        previous = value


def action_metrics(candidate: np.ndarray, baseline: np.ndarray, active_dims: int) -> dict[str, float]:
    cand = np.asarray(candidate, dtype=np.float32)[..., :active_dims]
    base = np.asarray(baseline, dtype=np.float32)[..., :active_dims]
    delta = cand - base
    cand_flat = cand.reshape(-1)
    base_flat = base.reshape(-1)
    cosine = float(
        np.dot(cand_flat, base_flat)
        / max(np.linalg.norm(cand_flat) * np.linalg.norm(base_flat), 1e-12)
    )
    return {
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "mae": float(np.mean(np.abs(delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "first_step_rmse": float(np.sqrt(np.mean(np.square(delta[:, 0])))),
        "cosine": cosine,
        "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(base), 1e-12)),
    }


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = training_config.get_config(args.config)
    checkpoint = args.checkpoint.resolve()

    # Use one cached training repository and one deterministic random frame.
    repo_id = config.data.repo_ids[0]
    checkpoint_assets = transforms.AssetsConfig(
        assets_dir=str(checkpoint / "assets"),
        asset_id=config.data.transform_pipeline.assets.asset_id,
    )
    data_config = dataclasses.replace(
        config.data,
        repo_ids=[repo_id],
        transform_pipeline=dataclasses.replace(
            config.data.transform_pipeline,
            assets=checkpoint_assets,
        ),
    )
    dataset = data_datasets.create_torch_dataset(data_config, config.model.action_horizon)
    dataset = data_datasets.transform_dataset(dataset, data_config)
    rng_np = np.random.default_rng(args.seed)
    sample_index = int(rng_np.integers(0, len(dataset)))
    sample = dataset[sample_index]
    batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], sample)
    observation = model_lib.Observation.from_dict(batch)

    print(f"Loading checkpoint {checkpoint}", flush=True)
    params = model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16)
    model = config.model.load(params)
    depth = gemma.get_config(config.model.paligemma_variant).depth
    experts = 2
    no_mask = jnp.zeros((depth, experts), dtype=jnp.bool_)
    noise = jax.random.normal(
        jax.random.key(args.seed),
        (1, model.action_horizon, model.action_dim),
    )

    capture = nnx_utils.module_jit(model.compute_layer_features)
    sample_actions = nnx_utils.module_jit(model.sample_action_chunk)

    print("Compiling and capturing hidden features...", flush=True)
    features = capture(observation, noise, timestep=1.0, layer_mask=no_mask)
    features = jax.device_get(features)

    feature_rows: list[dict] = []
    for camera, stack in features["vision"].items():
        add_stack_stats(feature_rows, f"siglip:{camera}", stack, None)
    add_stack_stats(feature_rows, "vlm", features["vlm_layers"], features["vlm_input"])
    add_stack_stats(feature_rows, "action_expert", features["action_layers"], features["action_input"])
    write_csv(args.output_dir / "hidden_feature_stats.csv", feature_rows)

    print("Compiling baseline action sampling...", flush=True)
    baseline = sample_actions(
        jax.random.key(args.seed),
        observation,
        denoising_steps=args.denoising_steps,
        noise=noise,
        layer_mask=no_mask,
    )
    baseline = np.asarray(jax.device_get(baseline))
    np.save(args.output_dir / "baseline_actions.npy", baseline)

    ablations: list[tuple[str, list[int], list[int]]] = []
    for layer in range(depth):
        ablations.append((f"vlm_layer_{layer:02d}", [layer], []))
        ablations.append((f"action_layer_{layer:02d}", [], [layer]))
    thirds = [list(range(0, depth // 3)), list(range(depth // 3, 2 * depth // 3)), list(range(2 * depth // 3, depth))]
    for name, layers in zip(("early", "middle", "late"), thirds, strict=True):
        ablations.extend(
            [
                (f"vlm_{name}", layers, []),
                (f"action_{name}", [], layers),
                (f"both_{name}", layers, layers),
            ]
        )
    ablations.extend([("vlm_all", list(range(depth)), []), ("action_all", [], list(range(depth)))])

    ablation_rows = []
    started = time.monotonic()
    for index, (name, vlm_layers, action_layers) in enumerate(ablations, start=1):
        mask = np.zeros((depth, experts), dtype=bool)
        mask[vlm_layers, 0] = True
        mask[action_layers, 1] = True
        candidate = sample_actions(
            jax.random.key(args.seed),
            observation,
            denoising_steps=args.denoising_steps,
            noise=noise,
            layer_mask=jnp.asarray(mask),
        )
        candidate = np.asarray(jax.device_get(candidate))
        ablation_rows.append(
            {
                "name": name,
                "vlm_layers": ";".join(map(str, vlm_layers)),
                "action_layers": ";".join(map(str, action_layers)),
                **action_metrics(candidate, baseline, min(14, model.action_dim)),
            }
        )
        print(f"[{index}/{len(ablations)}] {name}: rmse={ablation_rows[-1]['rmse']:.6f}", flush=True)
    write_csv(args.output_dir / "layer_ablation_action_deltas.csv", ablation_rows)

    summary = {
        "config": args.config,
        "checkpoint": str(checkpoint),
        "dataset_repo": repo_id,
        "sample_index": sample_index,
        "seed": args.seed,
        "denoising_steps": args.denoising_steps,
        "transformer_depth": depth,
        "vision_depth": int(next(iter(features["vision"].values())).shape[0]),
        "active_action_dims": min(14, model.action_dim),
        "elapsed_seconds": time.monotonic() - started,
        "most_sensitive_single_layers": sorted(
            [row for row in ablation_rows if "_layer_" in row["name"]],
            key=lambda row: row["rmse"],
            reverse=True,
        )[:10],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
        for expert, label in (("vlm", "VLM"), ("action", "Action expert")):
            rows = [row for row in ablation_rows if row["name"].startswith(f"{expert}_layer_")]
            axes[0].plot(range(depth), [row["rmse"] for row in rows], marker="o", label=label)
        axes[0].set(title="Action change from skipping one residual block", xlabel="Layer", ylabel="RMSE (normalized action)")
        axes[0].grid(alpha=0.25)
        axes[0].legend()
        for component, label in (("vlm", "VLM"), ("action_expert", "Action expert")):
            rows = [row for row in feature_rows if row["component"] == component]
            axes[1].plot(range(depth), [row["block_influence"] for row in rows], marker="o", label=label)
        axes[1].set(title="Block influence = 1 - cosine(input, output)", xlabel="Layer", ylabel="Block influence")
        axes[1].grid(alpha=0.25)
        axes[1].legend()
        fig.savefig(args.output_dir / "layer_diagnostics.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"Plot skipped: {exc}", flush=True)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
