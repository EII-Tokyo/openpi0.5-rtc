"""One JIT call per microbatch; Python owns the accumulation loop."""
import dataclasses
import time

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from openpi.data import transforms
from openpi.shared import nnx_utils


def initialize_accumulator(config, state):
    return (
        jnp.array(0.0),
        jax.tree.map(jnp.zeros_like, state.params.filter(config.trainable_filter)),
    )


def accumulate_microbatch(config, rng, state, batch, micro_index, loss_sum, grads_sum):
    """One microbatch only. State/optimizer/EMA are read-only here."""
    model = nnx.merge(state.model_def, state.params)
    model.train()
    train_rng = jax.random.fold_in(rng, state.step)
    micro_rng = (
        train_rng if config.gradient_accumulation_steps == 1
        else jax.random.fold_in(train_rng, micro_index)
    )

    def loss_fn(model, rng, observation, actions):
        observation = transforms.AlohaTransformPipeline.preprocess_observation(
            jax.random.split(rng, 3)[0], observation, train=True,
            image_resolution=config.model.image_resolution,
        )
        return jnp.mean(model.compute_loss(rng, observation, actions, train=True))

    loss, grads = nnx.value_and_grad(
        loss_fn, argnums=nnx.DiffState(0, config.trainable_filter)
    )(model, micro_rng, *batch)
    count = config.gradient_accumulation_steps
    # Same order and normalization as the original lax.scan body.
    return (
        loss_sum + loss / count,
        jax.tree.map(lambda total, g: total + g / count, grads_sum, grads),
    )


def apply_accumulated_update(config, state, loss, grads):
    """Exactly one optimizer update, step increment and EMA update per GA group."""
    model = nnx.merge(state.model_def, state.params)
    model.train()
    params = state.params.filter(config.trainable_filter)
    updates, opt_state = state.tx.update(grads, state.opt_state, params)
    nnx.update(model, optax.apply_updates(params, updates))
    new_params = nnx.state(model)
    new_state = dataclasses.replace(
        state, step=state.step + 1, params=new_params, opt_state=opt_state,
    )
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params, new_params,
            ),
        )
    kernels = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    return new_state, {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernels),
        "gradient_accumulation_steps": jnp.asarray(config.gradient_accumulation_steps),
        "effective_batch_size": jnp.asarray(config.batch_size * config.gradient_accumulation_steps),
    }


class StreamingAccumulator:
    """Three compiled kernels; Python owns the microbatch loop.

    Parameter and accumulator layouts match the original FSDP train state.
    Only disposable accumulators are donated during accumulation. Parameters
    must remain valid for *all* microbatches; whole-state donation is delayed
    until the optimizer kernel. No per-microbatch device_get/barrier is used.
    """

    def __init__(self, config, state_sharding, data_sharding, replicated_sharding):
        if config.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        self.count = config.gradient_accumulation_steps
        grads_sharding = state_sharding.params.filter(config.trainable_filter)
        self.initialize = jax.jit(
            lambda state: initialize_accumulator(config, state),
            in_shardings=(state_sharding,),
            out_shardings=(replicated_sharding, grads_sharding),
        )
        self.accumulate = jax.jit(
            lambda rng, state, batch, i, loss, grads:
                accumulate_microbatch(config, rng, state, batch, i, loss, grads),
            in_shardings=(replicated_sharding, state_sharding, data_sharding,
                          replicated_sharding, replicated_sharding, grads_sharding),
            out_shardings=(replicated_sharding, grads_sharding),
            donate_argnums=(4, 5),
        )
        self.update = jax.jit(
            lambda state, loss, grads: apply_accumulated_update(config, state, loss, grads),
            in_shardings=(state_sharding, replicated_sharding, grads_sharding),
            out_shardings=(state_sharding, replicated_sharding),
            donate_argnums=(0,),
        )

    def run(self, rng, state, first_batch, next_batch, first_batch_wait=0.0):
        """next_batch() returns (batch, retrieval_seconds); same iterator/order.

        Raw retrieval wait may overlap device execution: it must NOT be added
        to streaming_update_wall_time to estimate throughput.
        """
        started = time.perf_counter()
        loss, grads = self.initialize(state)
        retrieval = first_batch_wait
        batch = first_batch
        for i in range(self.count):
            if i:
                batch, elapsed = next_batch()
                retrieval += elapsed
            loss, grads = self.accumulate(
                rng, state, batch, jnp.asarray(i, dtype=jnp.int32), loss, grads,
            )
        state, info = self.update(state, loss, grads)
        dispatched = time.perf_counter() - started
        block_started = time.perf_counter()
        jax.block_until_ready((state, info))
        return state, info, {
            "data_wait_time": retrieval,
            "streaming_dispatch_wall_time": dispatched,
            "train_block_time": time.perf_counter() - block_started,
            "streaming_update_wall_time": time.perf_counter() - started,
        }
