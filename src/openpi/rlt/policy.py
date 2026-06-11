from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.rlt import actor_critic
from openpi.rlt import token_model
from openpi.serving import base_policy as _base_policy


class RLTPolicy(_base_policy.BasePolicy):
    def __init__(self, base_policy: _base_policy.BasePolicy, *, token_params=None, actor_params=None, token_config: token_model.RLTTokenConfig | None = None, actor_config: actor_critic.RLTActorCriticConfig | None = None, enabled: bool = True):
        self._base_policy = base_policy
        self._token_params = token_params
        self._actor_params = actor_params
        self._token_config = token_config
        self._actor_config = actor_config
        self._enabled = enabled and token_params is not None and actor_params is not None and token_config is not None and actor_config is not None

    @override
    def infer(self, obs: dict, **kwargs) -> dict:
        start = time.monotonic()
        base = self._base_policy.infer(obs, chunking_mode="sync", return_rlt_state=True)
        if not self._enabled:
            base.setdefault("policy_timing", {})["rlt_enabled"] = False
            return base
        embeddings = jnp.asarray(base.get("rlt_embeddings"))
        if embeddings.ndim == 2:
            embeddings = embeddings[None, ...]
        mask = jnp.asarray(base.get("rlt_mask"))
        if mask.ndim == 1:
            mask = mask[None, ...]
        state = jnp.asarray(base["state"])[None, ...] if np.asarray(base["state"]).ndim == 1 else jnp.asarray(base["state"])
        reference = jnp.asarray(base["origin_actions"])[None, ...] if np.asarray(base["origin_actions"]).ndim == 2 else jnp.asarray(base["origin_actions"])
        token = token_model.encode(self._token_params, embeddings, mask)
        refined = actor_critic.actor_apply(self._actor_params, token, state, reference, self._actor_config)
        full_actions = jnp.asarray(reference).at[:, : self._actor_config.rlt_chunk_horizon, : self._actor_config.action_dim].set(refined)
        out = dict(base)
        out["actions"] = np.asarray(full_actions[0])
        out["origin_actions"] = np.asarray(reference[0])
        out.setdefault("policy_timing", {})["rlt_ms"] = (time.monotonic() - start) * 1000.0
        out["policy_timing"]["rlt_enabled"] = True
        return out

    @override
    def reset(self) -> None:
        self._base_policy.reset()
