from __future__ import annotations

import argparse
import concurrent.futures as futures
import dataclasses
import functools
import json
import logging
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb

import openpi.models.gemma as _gemma
import openpi.models.model as _model
from openpi.data import dataloaders as _data_loader
from openpi.data import transforms as _transforms
from openpi.rlt import token_model
from openpi.shared import nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer

DEFAULT_BASE_CONFIG = "eii_rinse_11repo_cam4_fullft"
DEFAULT_BASE_CHECKPOINT = "/home/eii/openpi0.5-rtc/checkpoints/eii_rinse_11repo_cam4_fullft/rinse_11repo_insertx5_fullft_bs256_nw64_fsdp8_20260513/9000"
DEFAULT_NUM_TRAIN_STEPS = 10_000
DEFAULT_WARMUP_STEPS = 2_000


def _save_checkpoint(
    output_dir: Path,
    step: int,
    params,
    opt_state,
    ema_params,
    config: token_model.RLTTokenConfig,
    *,
    train_config: _config.TrainConfig,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_params = ema_params if ema_params is not None else params
    train_state = {
        "step": np.asarray(step, dtype=np.int64),
        "params": params,
        "ema_params": ema_params,
        "opt_state": opt_state,
    }
    manager = ocp.CheckpointManager(
        output_dir,
        item_handlers={
            "params": ocp.PyTreeCheckpointHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(max_to_keep=None, create=True),
    )
    manager.save(
        step,
        {
            "params": {"params": inference_params},
            "train_state": train_state,
        },
    )
    manager.wait_until_finished()

    step_dir = output_dir / str(step)
    (step_dir / "rlt_token_config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2) + "\n")
    training_metadata = {
        "optimizer": dataclasses.asdict(train_config.optimizer),
        "lr_schedule": dataclasses.asdict(train_config.lr_schedule),
        "ema_decay": train_config.ema_decay,
    }
    (step_dir / "training.json").write_text(json.dumps(training_metadata, indent=2) + "\n")
    return step_dir


def _train_config_for_real_data(args: argparse.Namespace) -> _config.TrainConfig:
    train_config = _config.get_config(args.base_config)
    lr_schedule = dataclasses.replace(
        train_config.lr_schedule,
        warmup_steps=args.warmup_steps,
        decay_steps=max(args.max_steps, args.warmup_steps + 1),
    )
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
        lr_schedule=lr_schedule,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.max_steps,
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
    lr_schedule: optax.Schedule,
    config: token_model.RLTTokenConfig,
    encode_rlt_state,
    train_config: _config.TrainConfig,
    *,
    augment: bool,
    debug_shapes: bool,
):
    @functools.partial(jax.jit, donate_argnums=(0, 1, 2))
    def token_update(params, opt_state, ema_params, embeddings, mask, step):
        def loss_fn(p):
            return token_model.reconstruction_loss(p, embeddings, mask, config)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if train_config.ema_decay is not None:
            ema_params = jax.tree.map(
                lambda old, new: train_config.ema_decay * old + (1 - train_config.ema_decay) * new,
                ema_params,
                params,
            )
        metrics = {
            **metrics,
            "grad_norm": optax.global_norm(grads),
            "learning_rate": lr_schedule(step),
        }
        return params, opt_state, ema_params, loss, metrics

    def step(params, opt_state, ema_params, step_idx, observation, rng):
        rng, preprocess_rng = jax.random.split(rng)

        preprocess_start = time.monotonic()
        observation = _transforms.AlohaTransformPipeline.preprocess_observation(
            preprocess_rng if augment else None,
            observation,
            train=augment,
            image_resolution=train_config.model.image_resolution,
        )
        preprocess_s = time.monotonic() - preprocess_start

        encode_start = time.monotonic()
        rlt_state = encode_rlt_state(observation)
        embeddings = rlt_state["embeddings"]
        mask = rlt_state["mask"]
        jax.block_until_ready(embeddings)
        encode_s = time.monotonic() - encode_start

        if debug_shapes:
            print(
                "train_step shapes: "
                f"embeddings={embeddings.shape} mask={mask.shape} state={rlt_state['state'].shape} "
                f"valid_tokens={np.asarray(jnp.sum(mask, axis=1))}",
                flush=True,
            )

        update_start = time.monotonic()
        params, opt_state, ema_params, loss, metrics = token_update(
            params,
            opt_state,
            ema_params,
            embeddings,
            mask,
            step_idx,
        )
        jax.block_until_ready(loss)
        update_s = time.monotonic() - update_start
        timings = {"preprocess_s": preprocess_s, "encode_s": encode_s, "update_s": update_s}
        return rng, params, opt_state, ema_params, loss, metrics, timings

    return step


def _print_batch_shapes(observation, actions, train_config: _config.TrainConfig) -> None:
    # Pi0 uses SigLIP So400m/14, so 224x224 images produce a 16x16 token grid.
    siglip_patch_size = 14
    image_token_per_image = (train_config.model.image_resolution[0] // siglip_patch_size) * (
        train_config.model.image_resolution[1] // siglip_patch_size
    )
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


def _timed_next(data_iter):
    start = time.monotonic()
    batch = next(data_iter)
    return batch, time.monotonic() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/openpi-rlt-token")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_NUM_TRAIN_STEPS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--token-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None, help="Override TrainConfig lr schedule peak_lr.")
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--ema-decay", type=float, default=None, help="Override TrainConfig EMA decay.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--augment", action="store_true", help="Apply train-time image augmentation before VLA encoding.")
    parser.add_argument("--debug-shapes", action="store_true", help="Print observation and RLT tensor shapes.")
    parser.add_argument("--dump-masks", default=None, help="Save first-batch RLT encoder/decoder masks to this .npz path.")
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--wandb-project", default="openpi-rlt")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rng = jax.random.key(args.seed)

    train_config = _train_config_for_real_data(args)
    if args.lr is not None:
        lr_schedule = dataclasses.replace(train_config.lr_schedule, peak_lr=args.lr)
        train_config = dataclasses.replace(train_config, lr_schedule=lr_schedule)
    if args.ema_decay is not None:
        train_config = dataclasses.replace(train_config, ema_decay=args.ema_decay)
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
    lr_schedule = train_config.lr_schedule.create()
    tx = _optimizer.create_optimizer(train_config.optimizer, train_config.lr_schedule, weight_decay_mask=None)
    opt_state = tx.init(params)
    ema_params = None if train_config.ema_decay is None else jax.tree.map(lambda x: jnp.array(x, copy=True), params)
    train_step = _make_train_step(
        tx,
        lr_schedule,
        token_config,
        encode_rlt_state,
        train_config,
        augment=args.augment,
        debug_shapes=args.debug_shapes,
    )
    logging.info(
        "Initialized RLT token model: %s optimizer=%s lr_schedule=%s ema_decay=%s",
        token_config,
        train_config.optimizer,
        train_config.lr_schedule,
        train_config.ema_decay,
    )

    prefetch_executor = futures.ThreadPoolExecutor(max_workers=1)
    next_batch_future = prefetch_executor.submit(_timed_next, data_iter)
    try:
        for step_idx in range(args.max_steps):
            wait_start = time.monotonic()
            (observation, actions), data_load_s = next_batch_future.result()
            data_wait_s = time.monotonic() - wait_start
            if step_idx != args.max_steps - 1:
                next_batch_future = prefetch_executor.submit(_timed_next, data_iter)

            if args.debug_shapes and step_idx == 0:
                _print_batch_shapes(observation, actions, train_config)

            if args.dump_masks and step_idx == 0:
                dump_rng = None
                dump_observation = _transforms.AlohaTransformPipeline.preprocess_observation(
                    dump_rng,
                    observation,
                    train=False,
                    image_resolution=train_config.model.image_resolution,
                )
                dump_rlt_state = encode_rlt_state(dump_observation)
                debug_masks = token_model.make_debug_masks(dump_rlt_state["mask"])
                dump_path = Path(args.dump_masks)
                dump_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    dump_path,
                    **{key: np.asarray(value) for key, value in debug_masks.items()},
                )
                print(f"dumped_masks={dump_path}", flush=True)

            train_start = time.monotonic()
            rng, params, opt_state, ema_params, loss, metrics, step_timings = train_step(
                params,
                opt_state,
                ema_params,
                jnp.asarray(step_idx, dtype=jnp.int32),
                observation,
                rng,
            )
            train_s = time.monotonic() - train_start

            log_data = {
                "rlt_token/loss": float(loss),
                "rlt_token/token_norm": float(metrics["token_norm"]),
                "rlt_token/grad_norm": float(metrics["grad_norm"]),
                "rlt_token/learning_rate": float(metrics["learning_rate"]),
                "rlt_token/valid_token_count": float(metrics["valid_token_count"]),
                "rlt_token/padding_output_abs_mean": float(metrics["padding_output_abs_mean"]),
                "rlt_token/data_wait_s": data_wait_s,
                "rlt_token/data_load_s": data_load_s,
                "rlt_token/preprocess_s": step_timings["preprocess_s"],
                "rlt_token/encode_s": step_timings["encode_s"],
                "rlt_token/update_s": step_timings["update_s"],
                "rlt_token/train_s": train_s,
                "rlt_token/batch_size": args.batch_size,
            }
            wandb.log(log_data, step=step_idx)
            print(
                "step={step} loss={loss:.6f} lr={lr:.3e} grad_norm={grad_norm:.6f} token_norm={token_norm:.6f} "
                "valid_tokens={valid_tokens:.0f} padding_abs={padding_abs:.6f} "
                "data_wait_s={data_wait_s:.3f} data_load_s={data_load_s:.3f} preprocess_s={preprocess_s:.3f} "
                "encode_s={encode_s:.3f} update_s={update_s:.3f} train_s={train_s:.3f}".format(
                    step=step_idx,
                    loss=log_data["rlt_token/loss"],
                    lr=log_data["rlt_token/learning_rate"],
                    grad_norm=log_data["rlt_token/grad_norm"],
                    token_norm=log_data["rlt_token/token_norm"],
                    valid_tokens=log_data["rlt_token/valid_token_count"],
                    padding_abs=log_data["rlt_token/padding_output_abs_mean"],
                    data_wait_s=data_wait_s,
                    data_load_s=data_load_s,
                    preprocess_s=log_data["rlt_token/preprocess_s"],
                    encode_s=log_data["rlt_token/encode_s"],
                    update_s=log_data["rlt_token/update_s"],
                    train_s=train_s,
                ),
                flush=True,
            )
    finally:
        prefetch_executor.shutdown(wait=False, cancel_futures=True)

    checkpoint_dir = _save_checkpoint(
        Path(args.output_dir),
        args.max_steps - 1,
        params,
        opt_state,
        ema_params,
        token_config,
        train_config=train_config,
    )
    wandb.finish()
    print("saved_token={}".format(checkpoint_dir))


if __name__ == "__main__":
    main()
