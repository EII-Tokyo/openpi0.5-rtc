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

import openpi.models.gemma as _gemma
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


def _init_wandb(args: argparse.Namespace, *, train_config: _config.TrainConfig) -> None:
    if args.no_wandb:
        wandb.init(mode="disabled")
        return

    run_name = args.wandb_run_name or f"rlt_token_{args.base_config}_{int(time.time())}"
    pipeline = train_config.data.transform_pipeline
    config: dict[str, Any] = vars(args).copy()
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


def _make_train_step(
    tx: optax.GradientTransformation,
    config: token_model.RLTTokenConfig,
    encode_rlt_state,
    train_config: _config.TrainConfig,
    *,
    augment: bool,
    debug_shapes: bool,
):
    @jax.jit
    def step(params, opt_state, observation, rng):
        rng, preprocess_rng = jax.random.split(rng)
        observation = _transforms.AlohaTransformPipeline.preprocess_observation(
            preprocess_rng if augment else None,
            observation,
            train=augment,
            image_resolution=train_config.model.image_resolution,
        )
        rlt_state = encode_rlt_state(observation)
        embeddings = rlt_state["embeddings"]
        mask = rlt_state["mask"]

        if debug_shapes:
            jax.debug.print(
                "train_step shapes: embeddings={emb} mask={mask_shape} state={state} valid_tokens={valid}",
                emb=embeddings.shape,
                mask_shape=mask.shape,
                state=rlt_state["state"].shape,
                valid=jnp.sum(mask, axis=1),
            )

        def loss_fn(p):
            return token_model.reconstruction_loss(p, embeddings, mask, config)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return rng, params, opt_state, loss, metrics

    return step


def _print_batch_shapes(observation, actions, train_config: _config.TrainConfig) -> None:
    image_token_per_image = (train_config.model.image_resolution[0] // 16) * (train_config.model.image_resolution[1] // 16)
    print("batch observation shapes:", flush=True)
    for key, image in observation.images.items():
        print(f"  image[{key}]={image.shape} image_mask={observation.image_masks[key].shape}", flush=True)
    print(f"  tokenized_prompt={observation.tokenized_prompt.shape}", flush=True)
    print(f"  tokenized_prompt_mask={observation.tokenized_prompt_mask.shape}", flush=True)
    print(f"  prompt_valid_lengths={np.asarray(jnp.sum(observation.tokenized_prompt_mask, axis=1))}", flush=True)
    print(f"  state={observation.state.shape}", flush=True)
    print(f"  actions={actions.shape}", flush=True)
    print(f"  image_token_per_image={image_token_per_image}", flush=True)
    print(f"  image_token_total={len(observation.images) * image_token_per_image}", flush=True)
    print(f"  prefix_max_len={len(observation.images) * image_token_per_image + observation.tokenized_prompt.shape[1]}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/openpi-rlt-token")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--token-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--augment", action="store_true", help="Apply train-time image augmentation before VLA encoding.")
    parser.add_argument("--debug-shapes", action="store_true", help="Print observation and RLT tensor shapes.")
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--wandb-project", default="openpi-rlt")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rng = jax.random.key(args.seed)

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

    paligemma_width = _gemma.get_config(train_config.model.paligemma_variant).width
    token_config = token_model.RLTTokenConfig(
        input_dim=paligemma_width,
        token_dim=args.token_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    rng, init_rng = jax.random.split(rng)
    params = token_model.init_token_params(init_rng, token_config)
    tx = optax.adam(args.lr)
    opt_state = tx.init(params)
    train_step = _make_train_step(
        tx,
        token_config,
        encode_rlt_state,
        train_config,
        augment=args.augment,
        debug_shapes=args.debug_shapes,
    )
    logging.info("Initialized RLT token model: %s", token_config)

    for step_idx in range(args.max_steps):
        wait_start = time.monotonic()
        observation, actions = next(data_iter)
        data_wait_s = time.monotonic() - wait_start

        if args.debug_shapes and step_idx == 0:
            _print_batch_shapes(observation, actions, train_config)

        train_start = time.monotonic()
        rng, params, opt_state, loss, metrics = train_step(params, opt_state, observation, rng)
        jax.block_until_ready(loss)
        train_s = time.monotonic() - train_start

        log_data = {
            "rlt_token/loss": float(loss),
            "rlt_token/token_norm": float(metrics["token_norm"]),
            "rlt_token/valid_token_count": float(metrics["valid_token_count"]),
            "rlt_token/padding_output_abs_mean": float(metrics["padding_output_abs_mean"]),
            "rlt_token/data_wait_s": data_wait_s,
            "rlt_token/train_s": train_s,
            "rlt_token/batch_size": args.batch_size,
        }
        wandb.log(log_data, step=step_idx)
        print(
            "step={step} loss={loss:.6f} token_norm={token_norm:.6f} "
            "valid_tokens={valid_tokens:.0f} padding_abs={padding_abs:.6f} "
            "data_wait_s={data_wait_s:.3f} train_s={train_s:.3f}".format(
                step=step_idx,
                loss=log_data["rlt_token/loss"],
                token_norm=log_data["rlt_token/token_norm"],
                valid_tokens=log_data["rlt_token/valid_token_count"],
                padding_abs=log_data["rlt_token/padding_output_abs_mean"],
                data_wait_s=data_wait_s,
                train_s=train_s,
            ),
            flush=True,
        )

    _save(Path(args.output_dir) / "rlt_token", params, token_config)
    wandb.finish()
    print("saved_token={}".format(Path(args.output_dir) / "rlt_token"))


if __name__ == "__main__":
    main()
