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
    def __init__(
        self,
        base_policy: _base_policy.BasePolicy,
        *,
        token_params,
        token_config: token_model.RLTTokenConfig,
        actor_params=None,
        actor_config: actor_critic.RLTActorCriticConfig | None = None,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ):
        if token_params is None or token_config is None:
            raise ValueError("RLTPolicy requires token_params and token_config.")
        self._base_policy = base_policy
        self._token_params = token_params
        self._actor_params = actor_params
        self._token_config = token_config
        self._actor_config = actor_config
        base_metadata = getattr(base_policy, "metadata", {})
        self._metadata = {**base_metadata, **(metadata or {})}
        self._actor_enabled = (
            enabled
            and actor_params is not None
            and actor_config is not None
        )
        token_config = self._token_config

        @jax.jit
        def encode_rlt_token(token_params, embeddings, mask):
            return token_model.encode(token_params, embeddings, mask, token_config)

        self._encode_rlt_token = encode_rlt_token

        self._apply_actor = None
        if self._actor_enabled:
            actor_config = self._actor_config

            @jax.jit
            def apply_actor(actor_params, rlt_token, state, reference):
                actor_network_params = actor_params["actor"] if isinstance(actor_params, dict) and "actor" in actor_params else actor_params
                refined = actor_critic.actor_apply(actor_network_params, rlt_token, state, reference, actor_config)
                full_actions = jnp.asarray(reference).at[
                    :, : actor_config.rlt_chunk_horizon, : actor_config.action_dim
                ].set(refined)
                q1 = actor_critic.critic_apply(actor_params["critic1"], rlt_token, state, full_actions, actor_config)
                q2 = actor_critic.critic_apply(actor_params["critic2"], rlt_token, state, full_actions, actor_config)
                return full_actions, q1, q2

            self._apply_actor = apply_actor

            @jax.jit
            def score_action_chunk(actor_params, rlt_token, state, action_chunk):
                q1 = actor_critic.critic_apply(actor_params["critic1"], rlt_token, state, action_chunk, actor_config)
                q2 = actor_critic.critic_apply(actor_params["critic2"], rlt_token, state, action_chunk, actor_config)
                return q1, q2

            self._score_action_chunk = score_action_chunk

    @property
    def actor_available(self) -> bool:
        return self._actor_enabled

    @staticmethod
    def _attach_replay_chunk(
        out: dict,
        *,
        rlt_token: jax.Array,
        embeddings: jax.Array,
        mask: jax.Array,
        state: jax.Array,
        reference: jax.Array,
        policy_actions: jax.Array,
        actor_enabled: bool,
        q1: jax.Array | None = None,
        q2: jax.Array | None = None,
        vla_q1: jax.Array | None = None,
        vla_q2: jax.Array | None = None,
        actor_q1: jax.Array | None = None,
        actor_q2: jax.Array | None = None,
    ) -> None:
        out["rlt_token"] = np.asarray(rlt_token[0])
        out["rlt_embeddings"] = np.asarray(embeddings[0], dtype=np.float32)
        out["rlt_mask"] = np.asarray(mask[0], dtype=np.bool_)
        out["rlt_state"] = np.asarray(state[0])
        out["rlt_state_is_normalized"] = True
        out["rlt_state_normalization"] = "policy_input_transform"
        out["rlt_reference_action_chunk"] = np.asarray(reference[0])
        out["rlt_policy_action_chunk"] = np.asarray(policy_actions[0])
        out["rlt_actor_enabled"] = actor_enabled
        if q1 is not None and q2 is not None:
            q1_value = float(np.asarray(q1[0]))
            q2_value = float(np.asarray(q2[0]))
            out["rlt_chunk_q1"] = q1_value
            out["rlt_chunk_q2"] = q2_value
            out["rlt_chunk_q_min"] = min(q1_value, q2_value)
        if vla_q1 is not None and vla_q2 is not None:
            q1_value = float(np.asarray(vla_q1[0]))
            q2_value = float(np.asarray(vla_q2[0]))
            out["rlt_vla_chunk_q1"] = q1_value
            out["rlt_vla_chunk_q2"] = q2_value
            out["rlt_vla_chunk_q_min"] = min(q1_value, q2_value)
        if actor_q1 is not None and actor_q2 is not None:
            q1_value = float(np.asarray(actor_q1[0]))
            q2_value = float(np.asarray(actor_q2[0]))
            out["rlt_actor_chunk_q1"] = q1_value
            out["rlt_actor_chunk_q2"] = q2_value
            out["rlt_actor_chunk_q_min"] = min(q1_value, q2_value)

    @override
    def infer(self, obs: dict, **kwargs) -> dict:
        start = time.monotonic()
        actor_requested = bool(kwargs.get("rlt_actor_enabled", False))
        chunking_mode = kwargs.get("chunking_mode", "inference_time")
        if chunking_mode != "inference_time":
            raise ValueError("RLTPolicy only supports chunking_mode='inference_time' for robot runtime.")
        base_kwargs = {
            "chunking_mode": "inference_time",
            "return_rlt_state": True,
        }
        if "prev_action" in kwargs:
            base_kwargs["prev_action"] = kwargs["prev_action"]
        if "noise" in kwargs:
            base_kwargs["noise"] = kwargs["noise"]
        base = self._base_policy.infer(obs, **base_kwargs)
        model_state = base.get("model_state", base["state"])
        state = jnp.asarray(model_state)[None, ...] if np.asarray(model_state).ndim == 1 else jnp.asarray(model_state)
        model_actions = base["model_actions"]
        reference = jnp.asarray(model_actions)[None, ...] if np.asarray(model_actions).ndim == 2 else jnp.asarray(model_actions)
        embeddings = jnp.asarray(base["rlt_embeddings"])
        if embeddings.ndim == 2:
            embeddings = embeddings[None, ...]
        mask = jnp.asarray(base["rlt_mask"])
        if mask.ndim == 1:
            mask = mask[None, ...]
        rlt_token = self._encode_rlt_token(self._token_params, embeddings, mask)
        base.pop("rlt_embeddings", None)
        base.pop("rlt_mask", None)
        vla_q1 = vla_q2 = actor_q1 = actor_q2 = None
        actor_actions = None
        if self._actor_enabled:
            vla_q1, vla_q2 = self._score_action_chunk(self._actor_params, rlt_token, state, reference)
            actor_actions, actor_q1, actor_q2 = self._apply_actor(self._actor_params, rlt_token, state, reference)

        if not actor_requested or not self._actor_enabled:
            timing = base.setdefault("policy_timing", {})
            timing["rlt_ms"] = (time.monotonic() - start) * 1000.0
            timing["rlt_enabled"] = False
            timing["rlt_actor_requested"] = actor_requested
            timing["rlt_actor_available"] = self._actor_enabled
            self._attach_replay_chunk(
                base,
                rlt_token=rlt_token,
                embeddings=embeddings,
                mask=mask,
                state=state,
                reference=reference,
                policy_actions=reference,
                actor_enabled=False,
                q1=vla_q1,
                q2=vla_q2,
                vla_q1=vla_q1,
                vla_q2=vla_q2,
                actor_q1=actor_q1,
                actor_q2=actor_q2,
            )
            return base
        full_actions = actor_actions
        if not hasattr(self._base_policy, "transform_model_outputs"):
            raise NotImplementedError("RLT actor inference requires base policy output transforms.")
        out = self._base_policy.transform_model_outputs(
            state=np.asarray(model_state),
            actions=np.asarray(full_actions[0]),
        )
        self._attach_replay_chunk(
            out,
            rlt_token=rlt_token,
            embeddings=embeddings,
            mask=mask,
            state=state,
            reference=reference,
            policy_actions=full_actions,
            actor_enabled=True,
            q1=actor_q1,
            q2=actor_q2,
            vla_q1=vla_q1,
            vla_q2=vla_q2,
            actor_q1=actor_q1,
            actor_q2=actor_q2,
        )
        out.setdefault("policy_timing", {})["rlt_ms"] = (time.monotonic() - start) * 1000.0
        out["policy_timing"]["rlt_enabled"] = True
        out["policy_timing"]["rlt_actor_requested"] = actor_requested
        out["policy_timing"]["rlt_actor_available"] = self._actor_enabled
        return out

    @override
    def reset(self) -> None:
        self._base_policy.reset()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
