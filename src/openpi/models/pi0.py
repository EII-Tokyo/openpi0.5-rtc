import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def apply_block_attention_mask(
    attn_mask: at.Bool[at.Array, "b q k"],
    query_block_mask: at.Bool[at.Array, "b q"],
    key_block_mask: at.Bool[at.Array, "b k"],
) -> at.Bool[at.Array, "b q k"]:
    """Forbid attention edges where query_block_mask and key_block_mask are both true."""
    blocked = jnp.logical_and(query_block_mask[:, :, None], key_block_mask[:, None, :])
    return jnp.logical_and(attn_mask, jnp.logical_not(blocked))


def put_along_last_axis(arr, indices, values):
    """Like np.put_along_axis(..., axis=-1), since jax is missing it."""
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim)
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot)
    put_values = jnp.einsum("...i,...in->...n", values, onehot)
    return jnp.where(put_mask, put_values, arr)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.subtask_loss_weight = config.subtask_loss_weight
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        if obs.tokenized_subtask is not None:
            tokenized_subtask = self.PaliGemma.llm(obs.tokenized_subtask, method="embed")
            tokens.append(tokenized_subtask)
            input_mask.append(obs.tokenized_subtask_mask)
            # subtask tokens are autoregressive, but can attend to prompt/image prefix.
            ar_mask += [True] * tokenized_subtask.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _prefix_fast_token_mask(
        self, observation: _model.Observation, prefix_len: int
    ) -> at.Bool[at.Array, "b p"]:
        """Mask for FAST action tokens that live inside tokenized_subtask span within prefix."""
        batch_size = observation.state.shape[0]
        out = jnp.zeros((batch_size, prefix_len), dtype=jnp.bool_)
        if (
            observation.tokenized_subtask is None
            or observation.tokenized_subtask_mask is None
            or observation.tokenized_subtask_fast_mask is None
        ):
            return out

        prompt_len = observation.tokenized_prompt.shape[1] if observation.tokenized_prompt is not None else 0
        subtask_len = int(observation.tokenized_subtask.shape[1])
        image_len = prefix_len - prompt_len - subtask_len
        if image_len < 0:
            return out

        subtask_fast_mask = jnp.logical_and(
            observation.tokenized_subtask_mask.astype(jnp.bool_),
            observation.tokenized_subtask_fast_mask.astype(jnp.bool_),
        )
        subtask_start = image_len + prompt_len
        out = out.at[:, subtask_start : subtask_start + subtask_len].set(subtask_fast_mask)
        return out

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        total_loss, _, _ = self.compute_loss_with_metrics(rng, observation, actions, train=train)
        return total_loss

    def compute_loss_with_metrics(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, "*b ah"], at.Float[at.Array, "*b 1"]]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        # Knowledge-insulation mask extension:
        # forbid FAST language tokens <-> flow suffix tokens attention in both directions.
        prefix_fast_mask = self._prefix_fast_token_mask(observation, prefix_mask.shape[1])
        zeros_prefix = jnp.zeros_like(prefix_mask)
        full_fast_mask = jnp.concatenate([prefix_fast_mask, jnp.zeros_like(suffix_mask)], axis=1)
        full_flow_mask = jnp.concatenate([zeros_prefix, suffix_mask], axis=1)
        attn_mask = apply_block_attention_mask(attn_mask, full_flow_mask, full_fast_mask)  # flow q -> fast k
        attn_mask = apply_block_attention_mask(attn_mask, full_fast_mask, full_flow_mask)  # fast q -> flow k
        # Block-aware positions:
        # suffix(flow) positions should not be shifted by prefix FAST token count.
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        prefix_nonfast_mask = jnp.logical_and(prefix_mask, jnp.logical_not(prefix_fast_mask))
        suffix_positions = jnp.sum(prefix_nonfast_mask, axis=1, keepdims=True) + jnp.cumsum(suffix_mask, axis=1) - 1
        positions = jnp.concatenate([prefix_positions, suffix_positions], axis=1)
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if self.subtask_loss_weight <= 0.0:
            subtask_ar_loss = jnp.zeros((flow_loss.shape[0], 1), dtype=flow_loss.dtype)
            return flow_loss, flow_loss, subtask_ar_loss
        if (
            observation.tokenized_subtask is None
            or observation.tokenized_subtask_mask is None
            or observation.tokenized_subtask_loss_mask is None
            or observation.tokenized_prompt_mask is None
        ):
            subtask_ar_loss = jnp.zeros((flow_loss.shape[0], 1), dtype=flow_loss.dtype)
            return flow_loss, flow_loss, subtask_ar_loss
        assert prefix_out is not None
        target_tokens = observation.tokenized_subtask.astype(jnp.int32)
        target_mask = observation.tokenized_subtask_mask
        loss_mask = jnp.logical_and(target_mask, observation.tokenized_subtask_loss_mask).astype(jnp.float32)
        subtask_len = target_tokens.shape[1]
        subtask_start = prefix_out.shape[1] - subtask_len
        # Subtask token i is predicted from hidden state at (subtask_start - 1 + i).
        idx = (subtask_start - 1) + jnp.arange(subtask_len, dtype=jnp.int32)[None, :]
        idx = jnp.broadcast_to(idx, (prefix_out.shape[0], subtask_len))
        pre_logits = jnp.take_along_axis(prefix_out, idx[..., None], axis=1)
        embed_table = self.PaliGemma.llm.embedder["input_embedding"].value
        logits = jnp.einsum("bld,vd->blv", pre_logits, embed_table, preferred_element_type=jnp.float32)
        logp = jax.nn.log_softmax(logits, axis=-1)
        targets = jax.nn.one_hot(target_tokens, embed_table.shape[0])
        token_logp = jnp.sum(targets * logp, axis=-1)
        denom = jnp.clip(jnp.sum(loss_mask, axis=-1), a_min=1.0)
        subtask_ar_loss = -jnp.sum(token_logp * loss_mask, axis=-1)
        subtask_ar_loss = (subtask_ar_loss / denom)[:, None]
        total_loss = flow_loss + self.subtask_loss_weight * subtask_ar_loss
        return total_loss, flow_loss, subtask_ar_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    @at.typecheck
    def sample_subtask_tokens(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        temperature: float = 0.0,
        eos_token_id: int | None = None,
        max_text_token_id: int = 240000,
        debug_top_logits: bool = False,
    ) -> at.Int[at.Array, "b _t"]:
        """Autoregressively decode language tokens from image+prompt prefix."""
        observation = _model.preprocess_observation(None, observation, train=False)

        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            raise ValueError("tokenized_prompt and tokenized_prompt_mask are required for subtask decoding.")
        if observation.tokenized_subtask is None:
            raise ValueError("tokenized_subtask is required for subtask decoding.")
        # jax.debug.print("observation.tokenized_prompt={tokenized_prompt}", tokenized_prompt=observation.tokenized_prompt)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        max_steps = int(observation.tokenized_subtask.shape[1])
        prefix_len = prefix_tokens.shape[1]
        subtask_len = int(observation.tokenized_subtask.shape[1])
        subtask_start = prefix_len - subtask_len

        embed_table = self.PaliGemma.llm.embedder["input_embedding"].value
        vocab_size = embed_table.shape[0]
        # Last 128 ids are reserved/special in the project tokenization stack.
        special_token_mask = jnp.arange(vocab_size) >= (vocab_size - 128)
        non_text_mask = jnp.arange(vocab_size) > max_text_token_id

        # Prefix forward pass to get initial next-token logits context.
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        (prefix_out, _), _ = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions
        )
        assert prefix_out is not None
        batch_size = prefix_out.shape[0]
        last_valid_idx = jnp.max(
            jnp.where(prefix_mask, jnp.arange(prefix_len)[None, :], -1),
            axis=1,
        )
        last_hidden = prefix_out[jnp.arange(batch_size), last_valid_idx, :]

        def step(carry):
            rng, last_hidden, output_tokens, finished, step_i = carry
            logits = jnp.einsum("bd,vd->bv", last_hidden, embed_table, preferred_element_type=jnp.float32)
            # top_vals, top_idx = jax.lax.top_k(logits[0], 100)
            # jax.debug.print("subtask step {s} top10 logits idx={i} val={v}", s=step_i, i=top_idx, v=top_vals)
            logits = jnp.where(special_token_mask[None, :], -jnp.inf, logits)
            logits = jnp.where(non_text_mask[None, :], -jnp.inf, logits)
            logits = logits.at[:, 0].set(-jnp.inf)  # never emit pad token
            if debug_top_logits:
                top_vals, top_idx = jax.lax.top_k(logits[0], 2)
                jax.debug.print(
                    "subtask step {s}: top1={i1}({v1}) top2={i2}({v2}) gap={g}",
                    s=step_i,
                    i1=top_idx[0],
                    v1=top_vals[0],
                    i2=top_idx[1],
                    v2=top_vals[1],
                    g=top_vals[0] - top_vals[1],
                )
            if temperature <= 0.0:
                next_token = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            else:
                rng, sample_rng = jax.random.split(rng)
                next_token = jax.random.categorical(sample_rng, logits / temperature, axis=-1).astype(jnp.int32)

            if eos_token_id is not None:
                next_token = jnp.where(finished, jnp.asarray(eos_token_id, dtype=jnp.int32), next_token)
                finished = jnp.logical_or(finished, next_token == jnp.asarray(eos_token_id, dtype=jnp.int32))
            output_tokens = put_along_last_axis(
                output_tokens,
                jnp.broadcast_to(step_i, (next_token.shape[0], 1)),
                next_token[:, None],
            )
            # Fill subtask slots inside the existing prefix token block.
            generated_valid = jnp.arange(max_steps)[None, :] <= step_i
            subtask_tokens = jnp.where(generated_valid, output_tokens, 0)
            subtask_embeddings = self.PaliGemma.llm(subtask_tokens, method="embed")
            full_prefix_tokens = prefix_tokens.at[:, subtask_start:, :].set(subtask_embeddings)

            full_prefix_mask = prefix_mask.at[:, subtask_start:].set(generated_valid)
            full_attn_mask = make_attn_mask(full_prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(full_prefix_mask, axis=-1) - 1
            (full_out, _), _ = self.PaliGemma.llm([full_prefix_tokens, None], mask=full_attn_mask, positions=positions)
            assert full_out is not None
            last_hidden = full_out[:, subtask_start + step_i, :]

            return rng, last_hidden, output_tokens, finished, step_i + 1

        def cond(carry):
            _, _, _, finished, step_i = carry
            if eos_token_id is None:
                return step_i < max_steps
            return (~jnp.all(finished)) & (step_i < max_steps)

        init_tokens = jnp.zeros((batch_size, max_steps), dtype=jnp.int32)
        init_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)
        _, _, output_tokens, _, _ = jax.lax.while_loop(
            cond,
            step,
            (rng, last_hidden, init_tokens, init_finished, 0),
        )
        return output_tokens
        
    def guided_inference(
        self,
        rng: at.KeyArrayLike,
        prev_action: _model.Actions,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,       
        s: int = 25,
        d: int = 10,
        beta: float = 8.0,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        
        # get prev_action from s-th step to the end, and then pad s steps with zeros
        prev_action_slice = prev_action[:, s:, :]  # get prev_action from s-th step to the end
        # jax.debug.print("prev_action_slice shape: {prev_action_slice_shape}", prev_action_slice_shape=prev_action_slice.shape)
        # create s steps with zeros
        zero_actions = jnp.zeros((batch_size, s, self.action_dim))
        # concatenate prev_action_slice and zero_actions
        prev_action_slice = jnp.concatenate([prev_action_slice, zero_actions], axis=1)

        def make_W(d: int, s: int) -> jnp.ndarray:
            """
            generate the weight vector W ∈ ℝ^H
            parameters
            ----
            H : int  # sequence length
            d : int  # "deterministic region" threshold
            s : int  # "truncated" window length
            return
            ----
            W : jnp.ndarray, shape (H,)
            """
            H = self.action_horizon
            i = jnp.arange(H)           # 0,1,2,...,H-1

            # three-segment condition
            cond_1 = i < d
            cond_2 = (i >= d) & (i < H - s)
            cond_3 = i >= H - s         # actually can be else

            # segment (1): all 1
            w1 = jnp.ones_like(i, dtype=float)

            # segment (2): exponential decay
            c_i = (H - s - i) / (H - s - d + 1)
            w2  = jnp.exp(c_i) - 1
            w2  = c_i * w2 / (jnp.e - 1)      # (e^{c_i} - 1) / (e - 1)

            # segment (3): all 0
            w3 = jnp.zeros_like(i, dtype=float)

            # concatenate three segments
            W = jnp.where(cond_1, w1,
                jnp.where(cond_2, w2, w3)
            )

            D = jnp.diag(W)

            D_batch = jnp.stack([D] * 1, axis=0)
            return D_batch

        # create W
        diag_W = make_W(d, s)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def func_a_1_prime(x_t, time):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache, adarms_cond=[None, adarms_cond]
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t - time * v_t, v_t

        def step(carry):
            x_t, time = carry
            (a_1_prime, v_t), f_vjp = jax.vjp(func_a_1_prime, x_t, time)

            e = prev_action_slice - a_1_prime
            e = jnp.matmul(diag_W, e)
            #Compute vector-Jacobian product
            grad_a_1_prime_x_t = f_vjp((e, jnp.zeros_like(v_t)))
            # jax.debug.print("grad_a_1_prime_x_t 0 shape: {grad_a_1_prime_x_t_shape}", grad_a_1_prime_x_t_shape=grad_a_1_prime_x_t[0].shape)
            # jax.debug.print("grad_a_1_prime_x_t 1 shape: {grad_a_1_prime_x_t_shape}", grad_a_1_prime_x_t_shape=grad_a_1_prime_x_t[1].shape)
            r_t = time * time / (time * time + (1 - time) * (1 - time))

            a_2_prime = x_t + dt * (v_t - jax.lax.min(beta, time / ((1 - time) * r_t * r_t + 1e-6)) * grad_a_1_prime_x_t[0])
            
            return a_2_prime, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        
        # 在guided_inference最后保存前14个关节的平均值，每次调用都保存一次
        def save_guided_inference_result(x_0):
            import numpy as np
            import time as time_module
            # 计算前14个关节的平均值
            first_14_joints = x_0[:, :, :14]
            joint_averages = np.mean(first_14_joints, axis=1)
            # 创建时间戳
            timestamp = time_module.strftime("%Y%m%d_%H%M%S")
            # 保存为numpy格式
            # np.save(f'guided_inference_joint_averages_{timestamp}.npy', joint_averages)
            # 保存为文本格式便于查看
            np.savetxt(f'guided_inference_joint_averages_{timestamp}.txt', first_14_joints[0], fmt='%.6f')
            # 保存详细信息
            # with open(f'guided_inference_info_{timestamp}.txt', 'w') as f:
            #     f.write(f"Timestamp: {timestamp}\n")
            #     f.write(f"Joint averages shape: {joint_averages.shape}\n")
            #     f.write(f"Full action shape: {x_0.shape}\n")
            #     f.write(f"Joint averages:\n{joint_averages}\n")
            return x_0
        # 使用jax.pure_callback来执行文件保存操作
        # x_0 = jax.pure_callback(save_guided_inference_result, jax.ShapeDtypeStruct(x_0.shape, x_0.dtype), x_0)
        return x_0
