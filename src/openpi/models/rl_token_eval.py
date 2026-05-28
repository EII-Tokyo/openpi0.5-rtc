import dataclasses

import jax.numpy as jnp

from openpi.models import rl_token
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class RLTokenAblationMetrics:
    real_loss: at.Float[at.Array, ""]
    shuffled_loss: at.Float[at.Array, ""]
    zero_loss: at.Float[at.Array, ""]
    shuffled_over_real: at.Float[at.Array, ""]
    zero_over_real: at.Float[at.Array, ""]
    real_vs_shuffled_gap: at.Float[at.Array, ""]
    real_vs_zero_gap: at.Float[at.Array, ""]
    z_rl_cosine_mean: at.Float[at.Array, ""]
    z_rl_cosine_std: at.Float[at.Array, ""]

    def as_dict(self) -> dict[str, at.Array]:
        return dataclasses.asdict(self)


def compute_reconstruction_ablations(
    autoencoder: rl_token.RLTokenAutoencoder,
    h_vla: at.Float[at.Array, "b n d"],
    prefix_mask: at.Bool[at.Array, "b n"],
) -> RLTokenAblationMetrics:
    """Compare reconstruction with real, batch-shuffled, and zero RL tokens.

    A useful trained RL Token network should show:
      real_loss < shuffled_loss << zero_loss

    `shuffled_loss` checks whether z_rl carries sample-specific information.
    `zero_loss` checks whether the decoder can reconstruct by teacher forcing and
    position alone without a meaningful compressed token.
    """

    z_real = autoencoder.encode(h_vla, prefix_mask)
    z_shuffled = _batch_shuffle(z_real)
    z_zero = jnp.zeros_like(z_real)

    real_loss = jnp.mean(autoencoder.compute_loss_with_z(h_vla, prefix_mask, z_real))
    shuffled_loss = jnp.mean(autoencoder.compute_loss_with_z(h_vla, prefix_mask, z_shuffled))
    zero_loss = jnp.mean(autoencoder.compute_loss_with_z(h_vla, prefix_mask, z_zero))
    eps = jnp.asarray(1e-8, dtype=real_loss.dtype)

    cosine_values = _off_diagonal_cosines(z_real)
    return RLTokenAblationMetrics(
        real_loss=real_loss,
        shuffled_loss=shuffled_loss,
        zero_loss=zero_loss,
        shuffled_over_real=shuffled_loss / jnp.maximum(real_loss, eps),
        zero_over_real=zero_loss / jnp.maximum(real_loss, eps),
        real_vs_shuffled_gap=shuffled_loss - real_loss,
        real_vs_zero_gap=zero_loss - real_loss,
        z_rl_cosine_mean=jnp.mean(cosine_values),
        z_rl_cosine_std=jnp.std(cosine_values),
    )


def _batch_shuffle(z_rl: at.Float[at.Array, "b d"]) -> at.Float[at.Array, "b d"]:
    if z_rl.shape[0] <= 1:
        return jnp.zeros_like(z_rl)
    return jnp.roll(z_rl, shift=1, axis=0)


def _off_diagonal_cosines(z_rl: at.Float[at.Array, "b d"]) -> at.Float[at.Array, " k"]:
    if z_rl.shape[0] <= 1:
        return jnp.zeros((1,), dtype=z_rl.dtype)
    z_norm = z_rl / jnp.maximum(jnp.linalg.norm(z_rl, axis=-1, keepdims=True), 1e-8)
    cosine = z_norm @ z_norm.T
    off_diag_mask = ~jnp.eye(z_rl.shape[0], dtype=bool)
    return cosine[off_diag_mask]
