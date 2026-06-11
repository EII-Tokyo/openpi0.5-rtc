from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

import openpi.models.model as _model
from openpi.data import dataloaders as _data_loader
from openpi.data import transforms as _transforms
from openpi.rlt import token_model
from openpi.shared import nnx_utils
from openpi.training import config as _config

DEFAULT_BASE_CONFIG = "eii_rinse_11repo_cam4_fullft"
DEFAULT_BASE_CHECKPOINT = "/home/eii/openpi0.5-rtc/checkpoints/eii_rinse_11repo_cam4_fullft/rinse_11repo_insertx5_fullft_bs256_nw64_fsdp8_20260513/9000"


def _save(path: Path, params, config: token_model.RLTTokenConfig) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "params.pkl").open("wb") as f:
        pickle.dump(jax.tree.map(np.asarray, params), f)
    (path / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2) + "\n")


def _synthetic_batch(rng, batch_size: int, seq_len: int, input_dim: int):
    emb_rng, mask_rng = jax.random.split(rng)
    embeddings = jax.random.normal(emb_rng, (batch_size, seq_len, input_dim), dtype=jnp.float32)
    mask = jax.random.bernoulli(mask_rng, 0.9, (batch_size, seq_len))
    return embeddings, mask


def _train_config_for_real_data(args: argparse.Namespace) -> _config.TrainConfig:
    train_config = _config.get_config(args.base_config)
    checkpoint_assets = _transforms.AssetsConfig(
        assets_dir=str(Path(args.base_checkpoint) / "assets"),
        asset_id=train_config.data.transform_pipeline.assets.asset_id,
    )
    data_config = dataclasses.replace(
        train_config.data,
        transform_pipeline=dataclasses.replace(
            train_config.data.transform_pipeline,
            assets=checkpoint_assets,
        ),
    )
    return dataclasses.replace(
        train_config,
        data=data_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )


def _load_frozen_base_model(train_config: _config.TrainConfig, checkpoint_dir: str):
    logging.info("Loading frozen base VLA from %s", checkpoint_dir)
    model = train_config.model.load(_model.restore_params(Path(checkpoint_dir) / "params", dtype=jnp.bfloat16))
    model.eval()
    return model


def _init_wandb(args: argparse.Namespace, *, train_config: _config.TrainConfig | None) -> None:
    if args.no_wandb:
        wandb.init(mode="disabled")
        return

    run_name = args.wandb_run_name or f"rlt_token_{args.base_config}_{int(time.time())}"
    config: dict[str, Any] = vars(args).copy()
    if train_config is not None:
        pipeline = train_config.data.transform_pipeline
        config.update(
            {
                "repo_ids": train_config.data.repo_ids,
                "model_image_resolution": train_config.model.image_resolution,
                "action_horizon": train_config.model.action_horizon,
                "transform_pipeline": {
                    "include_low": pipeline.include_low,
                    "include_subtask": pipeline.include_subtask,
                    "image_resolution": pipeline.image_resolution,
                    "max_token_len": pipeline.max_token_len,
                    "discrete_state_input": pipeline.discrete_state_input,
                    "assets_dir": pipeline.assets.assets_dir,
                    "asset_id": pipeline.assets.asset_id,
                    "use_quantile_norm": pipeline.use_quantile_norm,
                    "video_memory_num_frames": pipeline.video_memory_num_frames,
                    "video_memory_stride_seconds": pipeline.video_memory_stride_seconds,
                    "adapt_to_pi": pipeline.adapt_to_pi,
                    "use_delta_joint_actions": pipeline.use_delta_joint_actions,
                    "action_dim": pipeline.action_dim,
                },
            }
        )
    wandb.init(project=args.wandb_project, name=run_name, config=config)


def _make_token_step(tx: optax.GradientTransformation, config: token_model.RLTTokenConfig):
    @jax.jit
    def step(params, opt_state, embeddings, mask):
        def loss_fn(p):
            return token_model.reconstruction_loss(p, embeddings, mask, config)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, metrics

    return step


def _next_real_embeddings(data_iter, encode_rlt_state, rng, train_config: _config.TrainConfig, *, augment: bool):
    wait_start = time.monotonic()
    observation, _actions = next(data_iter)
    data_wait_s = time.monotonic() - wait_start

    rng, preprocess_rng = jax.random.split(rng)
    observation = _transforms.AlohaTransformPipeline.preprocess_observation(
        preprocess_rng if augment else None,
        observation,
        train=augment,
        image_resolution=train_config.model.image_resolution,
    )
    encode_start = time.monotonic()
    rlt_state = encode_rlt_state(observation)
    embeddings = rlt_state["embeddings"]
    mask = rlt_state["mask"]
    jax.block_until_ready(embeddings)
    encode_s = time.monotonic() - encode_start
    return rng, embeddings, mask, data_wait_s, encode_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/openpi-rlt-token")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--input-dim", type=int, default=2048)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic embeddings for smoke tests only.")
    parser.add_argument("--augment", action="store_true", help="Apply train-time image augmentation before VLA encoding.")
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--wandb-project", default="openpi-rlt")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rng = jax.random.key(args.seed)

    train_config = None
    data_iter = None
    encode_rlt_state = None
    if not args.synthetic:
        train_config = _train_config_for_real_data(args)
        data_loader = _data_loader.create_data_loader(train_config, shuffle=True)
        data_iter = iter(data_loader)
        base_model = _load_frozen_base_model(train_config, args.base_checkpoint)
        encode_rlt_state = nnx_utils.module_jit(base_model.encode_rlt_state)
        logging.info(
            "Using real data: config=%s batch_size=%d num_workers=%d repos=%d",
            args.base_config,
            args.batch_size,
            args.num_workers,
            len(train_config.data.repo_ids),
        )

    _init_wandb(args, train_config=train_config)

    params = None
    opt_state = None
    token_step = None
    token_config = None
    tx = optax.adam(args.lr)

    for step_idx in range(args.max_steps):
        if args.synthetic:
            rng, batch_rng = jax.random.split(rng)
            embeddings, mask = _synthetic_batch(batch_rng, args.batch_size, args.seq_len, args.input_dim)
            data_wait_s = 0.0
            encode_s = 0.0
        else:
            assert data_iter is not None
            assert encode_rlt_state is not None
            assert train_config is not None
            rng, embeddings, mask, data_wait_s, encode_s = _next_real_embeddings(
                data_iter,
                encode_rlt_state,
                rng,
                train_config,
                augment=args.augment,
            )
            args.input_dim = int(embeddings.shape[-1])
            args.seq_len = int(embeddings.shape[1])

        if params is None:
            token_config = token_model.RLTTokenConfig(
                input_dim=int(embeddings.shape[-1]),
                token_dim=args.token_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
            )
            rng, init_rng = jax.random.split(rng)
            params = token_model.init_token_params(init_rng, token_config)
            opt_state = tx.init(params)
            token_step = _make_token_step(tx, token_config)
            logging.info("Initialized RLT token model: %s", token_config)

        train_start = time.monotonic()
        params, opt_state, loss, metrics = token_step(params, opt_state, embeddings, mask)
        jax.block_until_ready(loss)
        train_s = time.monotonic() - train_start

        log_data = {
            "rlt_token/loss": float(loss),
            "rlt_token/token_norm": float(metrics["token_norm"]),
            "rlt_token/data_wait_s": data_wait_s,
            "rlt_token/encode_s": encode_s,
            "rlt_token/train_s": train_s,
            "rlt_token/batch_size": int(embeddings.shape[0]),
            "rlt_token/seq_len": int(embeddings.shape[1]),
        }
        wandb.log(log_data, step=step_idx)
        print(
            "step={step} loss={loss:.6f} token_norm={token_norm:.6f} "
            "data_wait_s={data_wait_s:.3f} encode_s={encode_s:.3f} train_s={train_s:.3f}".format(
                step=step_idx,
                loss=log_data["rlt_token/loss"],
                token_norm=log_data["rlt_token/token_norm"],
                data_wait_s=data_wait_s,
                encode_s=encode_s,
                train_s=train_s,
            ),
            flush=True,
        )

    assert params is not None
    assert token_config is not None
    _save(Path(args.output_dir) / "rlt_token", params, token_config)
    wandb.finish()
    print("saved_token={}".format(Path(args.output_dir) / "rlt_token"))


if __name__ == "__main__":
    main()
