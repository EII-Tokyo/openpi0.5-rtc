#!/usr/bin/env python3
"""Evaluate paired VLM/action suffix-layer ablations over many training frames."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import itertools
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
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--feature-samples", type=int, default=32)
    parser.add_argument("--max-suffix-layers", type=int, default=6)
    parser.add_argument("--mode", choices=("suffix", "single", "pairs"), default="suffix")
    parser.add_argument("--candidate-layers", default="13,7,16,10,4")
    parser.add_argument("--denoising-steps", type=int, default=10)
    return parser.parse_args()


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


def hidden_metrics(value: np.ndarray, previous: np.ndarray) -> dict[str, float]:
    x = np.asarray(value, dtype=np.float32)
    prev = np.asarray(previous, dtype=np.float32)
    flat = x.reshape(-1)
    prev_flat = prev.reshape(-1)
    denom = float(np.linalg.norm(prev_flat) * np.linalg.norm(flat))
    cosine = float(np.dot(prev_flat, flat) / max(denom, 1e-12))
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "rms": float(np.sqrt(np.mean(np.square(flat)))),
        "block_influence": 1.0 - cosine,
        "relative_change": float(np.linalg.norm(flat - prev_flat) / max(np.linalg.norm(prev_flat), 1e-12)),
    }


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for group_key, group_rows in sorted(grouped.items()):
        record = dict(zip(keys, group_key, strict=True))
        record["count"] = len(group_rows)
        for metric in metrics:
            values = np.asarray([row[metric] for row in group_rows], dtype=np.float64)
            record.update(
                {
                    f"{metric}_mean": float(values.mean()),
                    f"{metric}_std": float(values.std()),
                    f"{metric}_median": float(np.median(values)),
                    f"{metric}_p10": float(np.quantile(values, 0.10)),
                    f"{metric}_p90": float(np.quantile(values, 0.90)),
                    f"{metric}_p95": float(np.quantile(values, 0.95)),
                    f"{metric}_max": float(values.max()),
                }
            )
        output.append(record)
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = training_config.get_config(args.config)
    checkpoint = args.checkpoint.resolve()
    repo_id = config.data.repo_ids[0]
    checkpoint_assets = transforms.AssetsConfig(
        assets_dir=str(checkpoint / "assets"),
        asset_id=config.data.transform_pipeline.assets.asset_id,
    )
    data_config = dataclasses.replace(
        config.data,
        repo_ids=[repo_id],
        transform_pipeline=dataclasses.replace(config.data.transform_pipeline, assets=checkpoint_assets),
    )
    dataset = data_datasets.create_torch_dataset(data_config, config.model.action_horizon)
    dataset = data_datasets.transform_dataset(dataset, data_config)
    rng_np = np.random.default_rng(args.seed)
    sample_count = min(args.samples, len(dataset))
    sample_indices = rng_np.choice(len(dataset), size=sample_count, replace=False).tolist()

    print(f"Loading checkpoint {checkpoint}", flush=True)
    params = model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16)
    model = config.model.load(params)
    depth = gemma.get_config(config.model.paligemma_variant).depth
    max_suffix_layers = min(args.max_suffix_layers, depth)
    active_dims = min(14, model.action_dim)
    no_mask = jnp.zeros((depth, 2), dtype=jnp.bool_)
    sample_actions = nnx_utils.module_jit(model.sample_action_chunk)
    capture = nnx_utils.module_jit(model.compute_layer_features)

    per_sample_rows: list[dict] = []
    feature_rows: list[dict] = []
    started = time.monotonic()
    for sample_number, sample_index in enumerate(sample_indices):
        sample = dataset[sample_index]
        batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], sample)
        observation = model_lib.Observation.from_dict(batch)
        noise_seed = args.seed + sample_number
        noise = jax.random.normal(
            jax.random.key(noise_seed),
            (1, model.action_horizon, model.action_dim),
        )
        baseline = sample_actions(
            jax.random.key(noise_seed),
            observation,
            denoising_steps=args.denoising_steps,
            noise=noise,
            layer_mask=no_mask,
        )
        baseline = np.asarray(jax.device_get(baseline))

        if sample_number < args.feature_samples:
            features = jax.device_get(capture(observation, noise, timestep=1.0, layer_mask=no_mask))
            for component, stack_name, input_name in (
                ("vlm", "vlm_layers", "vlm_input"),
                ("action_expert", "action_layers", "action_input"),
            ):
                previous = features[input_name]
                for layer, value in enumerate(features[stack_name]):
                    feature_rows.append(
                        {
                            "sample_number": sample_number,
                            "sample_index": sample_index,
                            "component": component,
                            "layer": layer,
                            **hidden_metrics(value, previous),
                        }
                    )
                    previous = value

        if args.mode == "suffix":
            experiments = [
                (f"{scope}_suffix_{suffix_layers}", scope, suffix_layers, depth - suffix_layers, list(range(depth - suffix_layers, depth)), mask_columns)
                for suffix_layers in range(1, max_suffix_layers + 1)
                for scope, mask_columns in (("vlm", (0,)), ("action", (1,)), ("both", (0, 1)))
            ]
        elif args.mode == "single":
            experiments = [
                (f"paired_layer_{layer:02d}", "both", 1, layer, [layer], (0, 1))
                for layer in range(depth)
            ]
        else:
            candidate_layers = [int(value) for value in args.candidate_layers.split(",")]
            experiments = [
                (
                    "paired_layers_" + "_".join(f"{layer:02d}" for layer in layers),
                    "both",
                    len(layers),
                    min(layers),
                    list(layers),
                    (0, 1),
                )
                for layers in itertools.combinations(candidate_layers, 2)
            ]

        for name, scope, suffix_layers, first_layer, layers, mask_columns in experiments:
                mask = np.zeros((depth, 2), dtype=bool)
                for column in mask_columns:
                    mask[layers, column] = True
                candidate = sample_actions(
                    jax.random.key(noise_seed),
                    observation,
                    denoising_steps=args.denoising_steps,
                    noise=noise,
                    layer_mask=jnp.asarray(mask),
                )
                candidate = np.asarray(jax.device_get(candidate))
                per_sample_rows.append(
                    {
                        "name": name,
                        "sample_number": sample_number,
                        "sample_index": sample_index,
                        "noise_seed": noise_seed,
                        "scope": scope,
                        "suffix_layers": suffix_layers,
                        "first_layer": first_layer,
                        **action_metrics(candidate, baseline, active_dims),
                    }
                )
        if (sample_number + 1) % 8 == 0 or sample_number + 1 == sample_count:
            print(
                f"[{sample_number + 1}/{sample_count}] elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    write_csv(args.output_dir / "paired_suffix_per_sample.csv", per_sample_rows)
    action_summary = aggregate(
        per_sample_rows,
        ("name", "scope", "suffix_layers", "first_layer"),
        ("rmse", "mae", "first_step_rmse", "cosine", "relative_l2"),
    )
    write_csv(args.output_dir / "paired_suffix_summary.csv", action_summary)

    if feature_rows:
        feature_summary = aggregate(
            feature_rows,
            ("component", "layer"),
            ("mean", "std", "rms", "block_influence", "relative_change"),
        )
        write_csv(args.output_dir / "hidden_feature_summary_32_samples.csv", feature_summary)
        write_csv(args.output_dir / "hidden_feature_per_sample_32_samples.csv", feature_rows)

    metadata = {
        "config": args.config,
        "checkpoint": str(checkpoint),
        "dataset_repo": repo_id,
        "sample_count": sample_count,
        "feature_sample_count": min(args.feature_samples, sample_count),
        "sample_indices": sample_indices,
        "seed": args.seed,
        "denoising_steps": args.denoising_steps,
        "transformer_depth": depth,
        "active_action_dims": active_dims,
        "mode": args.mode,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
