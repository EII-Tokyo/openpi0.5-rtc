"""Benchmark: is make_W overhead real inside JIT?

Compares two versions of a simplified guided_inference step:
  1. make_W is built inside the function (current code)
  2. make_W is precomputed and passed in

We measure wall-clock time after JIT compilation (i.e., excluding first-call tracing).
"""

import time
import jax
import jax.numpy as jnp

H = 50   # action_horizon
d = 10
s = 25
batch_size = 1
action_dim = 24
N_ITERS = 200


def make_W(d: int, s: int, H: int) -> jnp.ndarray:
    i = jnp.arange(H)
    cond_1 = i < d
    cond_2 = (i >= d) & (i < H - s)
    w1 = jnp.ones_like(i, dtype=float)
    c_i = (H - s - i) / (H - s - d + 1)
    w2 = c_i * (jnp.exp(c_i) - 1) / (jnp.e - 1)
    w3 = jnp.zeros_like(i, dtype=float)
    W = jnp.where(cond_1, w1, jnp.where(cond_2, w2, w3))
    return jnp.diag(W)[None]  # (1, H, H)


# --- Version 1: make_W inside the jitted function ---
@jax.jit
def step_inline(x_t, prev_action_slice):
    diag_W = make_W(d, s, H)
    e = prev_action_slice - x_t
    e = jnp.matmul(diag_W, e)
    return e


# --- Version 2: precomputed W passed as a static arg ---
precomputed_W = make_W(d, s, H)

@jax.jit
def step_precomputed(x_t, prev_action_slice, diag_W):
    e = prev_action_slice - x_t
    e = jnp.matmul(diag_W, e)
    return e


def benchmark(name, fn, args, n=N_ITERS):
    # Warmup: first call triggers JIT compilation
    result = fn(*args)
    result.block_until_ready()

    # Timed runs
    start = time.perf_counter()
    for _ in range(n):
        result = fn(*args)
    result.block_until_ready()
    elapsed = time.perf_counter() - start

    avg_us = elapsed / n * 1e6
    print(f"{name:30s}  {n} calls in {elapsed:.4f}s  =>  {avg_us:.1f} µs/call")


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    x_t = jax.random.normal(key, (batch_size, H, action_dim))
    prev = jax.random.normal(key, (batch_size, H, action_dim))

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(f"H={H}, d={d}, s={s}, batch={batch_size}, action_dim={action_dim}")
    print(f"Iterations: {N_ITERS}\n")

    benchmark("make_W inline (current)", step_inline, (x_t, prev))
    benchmark("make_W precomputed",     step_precomputed, (x_t, prev, precomputed_W))

    # Also compare XLA HLO size to see if the compiled graphs differ
    print("\n--- Compiled HLO analysis ---")
    hlo_inline = jax.jit(step_inline).lower(x_t, prev).compile()
    hlo_precomp = jax.jit(step_precomputed).lower(x_t, prev, precomputed_W).compile()
    cost_inline = hlo_inline.cost_analysis()
    cost_precomp = hlo_precomp.cost_analysis()
    print(f"Inline cost:      {cost_inline}")
    print(f"Precomputed cost: {cost_precomp}")
    # Compare HLO text sizes as a proxy for graph complexity
    hlo_text_inline = hlo_inline.as_text()
    hlo_text_precomp = hlo_precomp.as_text()
    print(f"Inline HLO size:      {len(hlo_text_inline)} chars")
    print(f"Precomputed HLO size: {len(hlo_text_precomp)} chars")
