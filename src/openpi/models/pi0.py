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


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, "..."], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "..."]:
    """Computes sine-cosine positional embedding vectors for scalar or tokenwise positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    scale = 1.0 / period * 2 * jnp.pi
    if pos.ndim == 1:
        sinusoid_input = jnp.einsum("i,j->ij", pos, scale, precision=jax.lax.Precision.HIGHEST)
    else:
        sinusoid_input = pos[..., None] * scale
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.image_resolution = config.image_resolution
        self.training_time_rtc = config.training_time_rtc
        self.rtc_max_delay = config.rtc_max_delay
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=True,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])
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
        self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
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
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def encode_rlt_state(
        self, obs: _model.Observation
    ) -> dict[str, at.Array]:
        """Return frozen VLA prefix representations for RLT heads.

        The returned embeddings are the PaliGemma prefix hidden states after image/text
        fusion. They are stop-gradient by construction so RLT training can consume them
        without updating the base VLA.
        """
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(obs)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        return {
            "embeddings": jax.lax.stop_gradient(prefix_out.astype(jnp.float32)),
            "mask": jax.lax.stop_gradient(prefix_mask),
            "state": jax.lax.stop_gradient(obs.state.astype(jnp.float32)),
        }

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, "..."]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "..."] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        action_expert_tokens = action_tokens
        adarms_cond = time_emb
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        batch_shape = actions.shape[:-2]
        _, noise_rng, time_rng = jax.random.split(rng, 3)
        delay_rng = jax.random.fold_in(rng, 0)

        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        u_t = noise - actions

        if self.training_time_rtc and train:
            max_delay = jnp.minimum(self.rtc_max_delay, self.action_horizon - 1)
            delay = jax.random.randint(delay_rng, batch_shape, minval=0, maxval=max_delay + 1)
            rtc_prefix_mask = jnp.arange(self.action_horizon) < delay[..., None]
            token_time = jnp.broadcast_to(time[..., None], actions.shape[:-1])
            # This repo uses t=1 as noise and t=0 as clean action. Setting prefix time to 0
            # makes the prefix exactly clean without a second explicit replacement.
            token_time = jnp.where(rtc_prefix_mask, 0.0, token_time)
            x_t = token_time[..., None] * noise + (1 - token_time[..., None]) * actions
            timestep = token_time
        else:
            rtc_prefix_mask = None
            time_expanded = time[..., None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            timestep = time

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, timestep)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if rtc_prefix_mask is not None:
            loss = jnp.where(rtc_prefix_mask, 0.0, loss)
        return loss

    @override
    def sample_action_chunk(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        denoising_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / denoising_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
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
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def sample_action_chunk_with_rlt_context(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        denoising_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> dict[str, at.Array]:
        """Sample actions and return RLT prefix context with a single prefix pass."""
        dt = -1.0 / denoising_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_step_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_step_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return {
            "actions": x_0,
            "rlt_embeddings": jax.lax.stop_gradient(prefix_out.astype(jnp.float32)),
            "rlt_mask": jax.lax.stop_gradient(prefix_mask),
            "rlt_state": jax.lax.stop_gradient(observation.state.astype(jnp.float32)),
        }

    def sample_action_chunk_with_training_time_rtc(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        action_prefix: _model.Actions,
        handoff_delay_steps: int | at.Int[at.Array, ""],
        denoising_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        # Hard-prefix sampling used by training-time RTC. The prefix is already in model action space.
        dt = -1.0 / denoising_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        action_prefix = jnp.asarray(action_prefix)
        prefix_mask = (jnp.arange(self.action_horizon) < handoff_delay_steps)[None, :, None]

        def preserve_prefix(actions):
            return jnp.where(prefix_mask, action_prefix, actions)

        prefix_tokens, prefix_mask_tokens, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask_tokens, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask_tokens, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            token_time = jnp.broadcast_to(time, (batch_size, self.action_horizon))
            token_time = jnp.where(prefix_mask[..., 0], 0.0, token_time)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, token_time)
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask_tokens, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask_tokens, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return preserve_prefix(x_t + dt * v_t), time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (preserve_prefix(noise), 1.0))
        return x_0

    def _sample_action_chunk_with_inference_time_rtc_and_context(
        self,
        rng: at.KeyArrayLike,
        prev_action_chunk: _model.Actions,
        observation: _model.Observation,
        *,
        denoising_steps: int | at.Int[at.Array, ""] = 10,
        replan_start_step: int = 25,
        handoff_delay_steps: int = 10,
        guidance_scale: float = 8.0,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, at.Array, at.Array]:
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / denoising_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        guidance_indices = jnp.arange(self.action_horizon) + replan_start_step
        guidance_mask = guidance_indices < self.action_horizon
        guidance_indices = jnp.minimum(guidance_indices, self.action_horizon - 1)
        prev_action_guidance = jnp.take(prev_action_chunk, guidance_indices, axis=1)
        prev_action_guidance = jnp.where(guidance_mask[None, :, None], prev_action_guidance, 0.0)

        def make_W(handoff_delay_steps: int, replan_start_step: int) -> jnp.ndarray:
            """Generate the inference-time RTC weighting matrix."""
            H = self.action_horizon
            i = jnp.arange(H)
            cond_1 = i < handoff_delay_steps
            cond_2 = (i >= handoff_delay_steps) & (i < H - replan_start_step)
            w1 = jnp.ones_like(i, dtype=float)
            c_i = (H - replan_start_step - i) / (H - replan_start_step - handoff_delay_steps + 1)
            w2 = jnp.exp(c_i) - 1
            w2 = c_i * w2 / (jnp.e - 1)
            w3 = jnp.zeros_like(i, dtype=float)
            W = jnp.where(cond_1, w1, jnp.where(cond_2, w2, w3))
            D = jnp.diag(W)
            return jnp.stack([D] * 1, axis=0)

        diag_W = make_W(handoff_delay_steps, replan_start_step)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def func_a_1_prime(x_t, time):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
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
            return x_t - time * v_t, v_t

        def step(carry):
            x_t, time = carry
            (a_1_prime, v_t), f_vjp = jax.vjp(func_a_1_prime, x_t, time)
            e = prev_action_guidance - a_1_prime
            e = jnp.matmul(diag_W, e)
            grad_a_1_prime_x_t = f_vjp((e, jnp.zeros_like(v_t)))
            r_t = time * time / (time * time + (1 - time) * (1 - time))
            a_2_prime = x_t + dt * (
                v_t
                - jax.lax.min(guidance_scale, time / ((1 - time) * r_t * r_t + 1e-6)) * grad_a_1_prime_x_t[0]
            )
            return a_2_prime, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0, prefix_out, prefix_mask

    def sample_action_chunk_with_inference_time_rtc(
        self,
        rng: at.KeyArrayLike,
        prev_action_chunk: _model.Actions,
        observation: _model.Observation,
        *,
        denoising_steps: int | at.Int[at.Array, ""] = 10,
        replan_start_step: int = 25,
        handoff_delay_steps: int = 10,
        guidance_scale: float = 8.0,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        x_0, _, _ = self._sample_action_chunk_with_inference_time_rtc_and_context(
            rng,
            prev_action_chunk,
            observation,
            denoising_steps=denoising_steps,
            replan_start_step=replan_start_step,
            handoff_delay_steps=handoff_delay_steps,
            guidance_scale=guidance_scale,
            noise=noise,
        )
        return x_0

    def sample_action_chunk_with_inference_time_rtc_context(
        self,
        rng: at.KeyArrayLike,
        prev_action_chunk: _model.Actions,
        observation: _model.Observation,
        *,
        denoising_steps: int | at.Int[at.Array, ""] = 10,
        replan_start_step: int = 25,
        handoff_delay_steps: int = 10,
        guidance_scale: float = 8.0,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> dict[str, at.Array]:
        x_0, prefix_out, prefix_mask = self._sample_action_chunk_with_inference_time_rtc_and_context(
            rng,
            prev_action_chunk,
            observation,
            denoising_steps=denoising_steps,
            replan_start_step=replan_start_step,
            handoff_delay_steps=handoff_delay_steps,
            guidance_scale=guidance_scale,
            noise=noise,
        )
        return {
            "actions": x_0,
            "rlt_embeddings": jax.lax.stop_gradient(prefix_out.astype(jnp.float32)),
            "rlt_mask": jax.lax.stop_gradient(prefix_mask),
            "rlt_state": jax.lax.stop_gradient(observation.state.astype(jnp.float32)),
        }
