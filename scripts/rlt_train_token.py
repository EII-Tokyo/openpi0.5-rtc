from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.rlt import token_model
from openpi.data import transforms as _transforms
from openpi.serving import policy_config as _policy_config
from openpi.training import config as _config

DEFAULT_BASE_CONFIG = "eii_rinse_11repo_cam4_fullft"
DEFAULT_BASE_CHECKPOINT = "/home/eii/openpi0.5-rtc/checkpoints/eii_rinse_11repo_cam4_fullft/rinse_11repo_insertx5_fullft_bs256_nw64_fsdp8_20260513/9000"


def _save(path: Path, params, config) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "params.pkl").open("wb") as f:
        pickle.dump(jax.tree.map(np.asarray, params), f)
    (path / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2) + "\n")


def _synthetic_batch(rng, batch_size: int, seq_len: int, input_dim: int):
    emb_rng, mask_rng = jax.random.split(rng)
    embeddings = jax.random.normal(emb_rng, (batch_size, seq_len, input_dim), dtype=jnp.float32)
    mask = jax.random.bernoulli(mask_rng, 0.9, (batch_size, seq_len))
    return embeddings, mask


def _real_dummy_batch(base_config: str, base_checkpoint: str, denoising_steps: int):
    train_config = _config.get_config(base_config)
    policy = _policy_config.create_trained_policy(
        train_config,
        base_checkpoint,
        sample_kwargs={"denoising_steps": denoising_steps},
    )
    obs = _transforms.make_aloha_example()
    output = policy.infer(obs, chunking_mode="sync", return_rlt_state=True)
    embeddings = jnp.asarray(output["rlt_embeddings"])[None, ...]
    mask = jnp.asarray(output["rlt_mask"])[None, ...]
    return embeddings, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/openpi-rlt-token")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--input-dim", type=int, default=2048)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-real-base", action="store_true")
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--denoising-steps", type=int, default=2)
    args = parser.parse_args()

    rng = jax.random.key(args.seed)
    if args.use_real_base:
        embeddings, mask = _real_dummy_batch(args.base_config, args.base_checkpoint, args.denoising_steps)
        args.input_dim = int(embeddings.shape[-1])
        args.seq_len = int(embeddings.shape[1])
    config = token_model.RLTTokenConfig(input_dim=args.input_dim, token_dim=args.token_dim, hidden_dim=args.hidden_dim)
    rng, init_rng = jax.random.split(rng)
    params = token_model.init_token_params(init_rng, config)
    tx = optax.adam(args.lr)
    opt_state = tx.init(params)

    @jax.jit
    def step(params, opt_state, embeddings, mask):
        (loss, metrics), grads = jax.value_and_grad(token_model.reconstruction_loss, has_aux=True)(params, embeddings, mask)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, metrics

    for step_idx in range(args.max_steps):
        if not args.use_real_base:
            rng, batch_rng = jax.random.split(rng)
            embeddings, mask = _synthetic_batch(batch_rng, args.batch_size, args.seq_len, args.input_dim)
        params, opt_state, loss, metrics = step(params, opt_state, embeddings, mask)
        print(f"step={step_idx} loss={float(loss):.6f} token_norm={float(metrics['token_norm']):.6f}")

    _save(Path(args.output_dir) / "rlt_token", params, config)
    print(f"saved_token={Path(args.output_dir) / 'rlt_token'}")


if __name__ == "__main__":
    main()
