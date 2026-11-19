"""cuDNN SDPA for explicitly enabled vision inference.

Keeps Flax's query pre-scaling. Fused reductions/intermediate rounding still
change BF16 outputs; this is not a bitwise-equivalent backend.
"""

import jax
import jax.numpy as jnp


def dot_product_attention(
    query,
    key,
    value,
    *,
    bias=None,
    mask=None,
    dropout_rng=None,
    dropout_rate=0.0,
    broadcast_dropout=True,
    deterministic=False,
    dtype=None,
    precision=None,
    force_fp32_for_softmax=False,
    module=None,
):
    """Flax-compatible inference attention, with explicit mask forwarding.

    There is no sequence-length cutoff. Unsupported CUDA/cuDNN combinations
    raise from JAX instead of silently selecting a different backend.
    """
    if not deterministic and dropout_rate:
        raise ValueError("cuDNN vision attention is inference-only; disable dropout")
    if module is not None:
        raise ValueError("cuDNN attention cannot expose probability intermediates; use native attention")
    if query.dtype not in (jnp.bfloat16, jnp.float16):
        raise ValueError("cuDNN vision attention requires BF16 or FP16 inputs")
    scaled_query = query / jnp.sqrt(query.shape[-1]).astype(query.dtype)
    return jax.nn.dot_product_attention(
        scaled_query,
        key,
        value,
        bias=bias,
        mask=mask,
        scale=1.0,
        implementation="cudnn",
    )
