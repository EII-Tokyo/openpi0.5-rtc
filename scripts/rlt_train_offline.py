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

from openpi.rlt import actor_critic
from openpi.rlt import replay


def _save(path: Path, params, config) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "params.pkl").open("wb") as f:
        pickle.dump(jax.tree.map(np.asarray, params), f)
    (path / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2) + "\n")


def _synthetic_tokens(rng, batch_size: int, token_dim: int):
    return jax.random.normal(rng, (batch_size, token_dim), dtype=jnp.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", default="/tmp/openpi-rlt-replay")
    parser.add_argument("--output-dir", default="/tmp/openpi-rlt-actor")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--create-synthetic-replay", action="store_true")
    args = parser.parse_args()

    replay_dir = Path(args.replay_dir)
    if args.create_synthetic_replay or not any(replay_dir.glob("*.npz")):
        replay.create_synthetic_replay(replay_dir, num_samples=max(1024, args.batch_size * 4), seed=args.seed)
    dataset = replay.ReplayDataset(replay_dir)
    config = actor_critic.RLTActorCriticConfig()
    rng = jax.random.key(args.seed)
    np_rng = np.random.default_rng(args.seed)
    rng, init_rng = jax.random.split(rng)
    params = actor_critic.init_actor_critic_params(init_rng, config)
    target_params = jax.tree.map(lambda x: x.copy(), params)
    actor_tx = optax.adam(config.actor_lr)
    critic_tx = optax.adam(config.critic_lr)
    critic_opt = critic_tx.init(params)

    @jax.jit
    def critic_step(params, target_params, opt_state, batch, token, next_token, rng):
        (loss, metrics), grads = jax.value_and_grad(actor_critic.critic_loss, has_aux=True)(params, target_params, batch, token, next_token, config, rng)
        updates, opt_state = critic_tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, metrics

    actor_opt = actor_tx.init(params["actor"])

    @jax.jit
    def actor_step(params, target_params, opt_state, batch, token, rng):
        def loss_for_actor(actor_params):
            actor_only_params = dict(params)
            actor_only_params["actor"] = actor_params
            return actor_critic.actor_loss(actor_only_params, token, batch, config, rng)

        (loss, metrics), grads = jax.value_and_grad(loss_for_actor, has_aux=True)(params["actor"])
        updates, opt_state = actor_tx.update(grads, opt_state, params["actor"])
        new_actor = optax.apply_updates(params["actor"], updates)
        params = dict(params)
        params["actor"] = new_actor
        target_params = actor_critic.soft_update(params, target_params, config.tau)
        return params, target_params, opt_state, loss, metrics

    for step_idx in range(args.max_steps):
        batch = dataset.sample(np_rng, args.batch_size)
        rng, token_rng, next_token_rng, update_rng = jax.random.split(rng, 4)
        token = _synthetic_tokens(token_rng, args.batch_size, config.token_dim)
        next_token = _synthetic_tokens(next_token_rng, args.batch_size, config.token_dim)
        params, critic_opt, critic_loss, critic_metrics = critic_step(params, target_params, critic_opt, batch, token, next_token, update_rng)
        if step_idx % 2 == 0:
            rng, actor_rng = jax.random.split(rng)
            params, target_params, actor_opt, actor_loss, actor_metrics = actor_step(params, target_params, actor_opt, batch, token, actor_rng)
            print(f"step={step_idx} critic_loss={float(critic_loss):.6f} actor_loss={float(actor_loss):.6f} bc_loss={float(actor_metrics['bc_loss']):.6f}")
        else:
            print(f"step={step_idx} critic_loss={float(critic_loss):.6f} q1={float(critic_metrics['q1']):.6f}")

    out = Path(args.output_dir) / "rlt_actor_critic"
    _save(out, params, config)
    print(f"saved_actor_critic={out}")


if __name__ == "__main__":
    main()
