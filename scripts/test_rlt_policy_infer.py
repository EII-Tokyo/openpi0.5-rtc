#!/usr/bin/env python3
"""Smoke-test RLTPolicy.infer without loading a real VLA checkpoint."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from openpi.rlt import actor_critic
from openpi.rlt import policy as rlt_policy
from openpi.rlt import token_model


class FakeBasePolicy:
    def __init__(self, *, horizon: int, action_dim: int, state_dim: int, embed_len: int, embed_dim: int):
        self._horizon = horizon
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._embed_len = embed_len
        self._embed_dim = embed_dim
        self._calls = 0

    @property
    def metadata(self) -> dict:
        return {"name": "fake_base_policy"}

    def infer(self, obs: dict, **kwargs) -> dict:
        self._calls += 1
        base = np.arange(self._horizon * self._action_dim, dtype=np.float32).reshape(self._horizon, self._action_dim)
        model_actions = base / 1000.0 + self._calls
        model_state = np.arange(self._state_dim, dtype=np.float32) / 100.0
        embeddings = np.arange(self._embed_len * self._embed_dim, dtype=np.float32).reshape(
            self._embed_len, self._embed_dim
        )
        embeddings = embeddings / np.maximum(float(embeddings.size), 1.0)
        mask = np.ones((self._embed_len,), dtype=np.bool_)
        if self._embed_len > 2:
            mask[-1] = False
        return {
            "actions": model_actions + 10.0,
            "state": model_state + 10.0,
            "model_actions": model_actions,
            "model_state": model_state,
            "rlt_embeddings": embeddings,
            "rlt_mask": mask,
            "policy_timing": {"infer_ms": 1.0},
        }

    def transform_model_outputs(self, *, state: np.ndarray, actions: np.ndarray) -> dict:
        return {
            "actions": actions + 10.0,
            "state": state + 10.0,
            "model_actions": actions,
            "model_state": state,
        }

    def reset(self) -> None:
        self._calls = 0


def _print_summary(name: str, out: dict) -> None:
    print(f"\n{name}")
    for key in (
        "actions",
        "state",
        "model_actions",
        "model_state",
        "rlt_token",
        "rlt_reference_action_chunk",
        "rlt_policy_action_chunk",
        "rlt_chunk_q_min",
        "rlt_vla_chunk_q_min",
        "rlt_actor_chunk_q_min",
    ):
        if key not in out:
            continue
        value = out[key]
        arr = np.asarray(value)
        if arr.ndim == 0:
            print(f"  {key}: {arr.item()}")
        else:
            print(f"  {key}: shape={arr.shape} dtype={arr.dtype} first={arr.reshape(-1)[0]:.6f}")
    print(f"  rlt_actor_enabled: {out.get('rlt_actor_enabled')}")
    print(f"  policy_timing: {out.get('policy_timing')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--state-dim", type=int, default=14)
    parser.add_argument("--embed-len", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    args = parser.parse_args()

    token_config = token_model.RLTTokenConfig(
        input_dim=args.embed_dim,
        num_layers=1,
        num_heads=4,
        hidden_dim=args.hidden_dim,
    )
    actor_config = actor_critic.RLTActorCriticConfig(
        token_dim=args.embed_dim,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        action_horizon=args.horizon,
        rlt_chunk_horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        actor_hidden_layers=1,
        critic_hidden_layers=1,
    )

    rng = jax.random.key(0)
    token_rng, actor_rng = jax.random.split(rng)
    token_params = token_model.init_token_params(token_rng, token_config)
    actor_params = actor_critic.init_actor_critic_params(actor_rng, actor_config)
    base_policy = FakeBasePolicy(
        horizon=args.horizon,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        embed_len=args.embed_len,
        embed_dim=args.embed_dim,
    )
    policy = rlt_policy.RLTPolicy(
        base_policy,
        token_params=token_params,
        token_config=token_config,
        actor_params=actor_params,
        actor_config=actor_config,
        enabled=True,
    )

    obs = {"state": np.zeros((args.state_dim,), dtype=np.float32)}
    vla_out = policy.infer(obs, chunking_mode="inference_time", rlt_actor_enabled=False)
    actor_out = policy.infer(obs, chunking_mode="inference_time", rlt_actor_enabled=True)
    _print_summary("VLA/reference path", vla_out)
    _print_summary("Actor path", actor_out)

    assert vla_out["actions"].shape == (args.horizon, args.action_dim)
    assert vla_out["model_actions"].shape == (args.horizon, args.action_dim)
    assert vla_out["rlt_token"].shape == (args.embed_dim,)
    assert vla_out["rlt_reference_action_chunk"].shape == (args.horizon, args.action_dim)
    assert vla_out["rlt_policy_action_chunk"].shape == (args.horizon, args.action_dim)
    assert actor_out["actions"].shape == (args.horizon, args.action_dim)
    assert actor_out["model_actions"].shape == (args.horizon, args.action_dim)
    assert actor_out["rlt_policy_action_chunk"].shape == (args.horizon, args.action_dim)
    assert not np.allclose(actor_out["model_actions"], actor_out["rlt_reference_action_chunk"])
    print("\nRLTPolicy.infer smoke test passed.")


if __name__ == "__main__":
    main()
