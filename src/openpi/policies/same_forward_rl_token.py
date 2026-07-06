from __future__ import annotations

import logging
import pathlib
from typing import Any

import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.shared import nnx_utils


def _drop_language(prefix_hidden: tuple[Any, Any], observation: _model.Observation) -> tuple[Any, Any]:
    prefix_out, prefix_mask = prefix_hidden
    if observation.tokenized_prompt is None:
        return prefix_out, prefix_mask
    image_token_count = prefix_out.shape[1] - observation.tokenized_prompt.shape[1]
    return prefix_out[:, :image_token_count], prefix_mask[:, :image_token_count]


class SameForwardRLTokenEncoder:
    """Encode z_rl from the same VLA forward pass used to produce actions.

    The lower+right RLToken model was trained with the full four-camera slot
    layout, but only the low and right-wrist slots marked valid. This encoder
    preserves that layout while sourcing the tokens from the main cam4 VLA
    forward pass instead of running a separate visual encoder.
    """

    def __init__(
        self,
        autoencoder: Any,
        *,
        target_slots: tuple[str, ...] = ("base_1_rgb", "right_wrist_0_rgb"),
    ) -> None:
        self._autoencoder = autoencoder
        self._target_slots = target_slots
        try:
            self._encode = nnx_utils.module_jit(autoencoder.encode)
        except ValueError:
            # Unit tests use lightweight fake encoders that are not NNX modules.
            self._encode = autoencoder.encode

    def encode(
        self,
        prefix_hidden: tuple[Any, Any],
        observation: _model.Observation,
    ) -> Any:
        prefix_out, prefix_mask = _drop_language(prefix_hidden, observation)
        slot_names = tuple(observation.images.keys())
        missing = [slot for slot in self._target_slots if slot not in slot_names]
        if missing:
            raise ValueError(f"same-forward RLToken missing image slots: {missing}; available={slot_names}")
        if len(slot_names) == 0:
            raise ValueError("same-forward RLToken requires image slots")
        if int(prefix_out.shape[1]) % len(slot_names) != 0:
            raise ValueError(
                f"image token length {prefix_out.shape[1]} is not divisible by slot count {len(slot_names)}"
            )

        tokens_per_slot = int(prefix_out.shape[1]) // len(slot_names)
        rebuilt = jnp.zeros_like(prefix_out)
        rebuilt_mask = jnp.zeros_like(prefix_mask)
        for slot in self._target_slots:
            slot_index = slot_names.index(slot)
            start = slot_index * tokens_per_slot
            end = start + tokens_per_slot
            rebuilt = rebuilt.at[:, start:end, :].set(prefix_out[:, start:end, :])
            rebuilt_mask = rebuilt_mask.at[:, start:end].set(prefix_mask[:, start:end])
        return self._encode(jax.lax.stop_gradient(rebuilt), rebuilt_mask)


def load_same_forward_rl_token_encoder(
    *,
    config_name: str,
    checkpoint_dir: str | pathlib.Path,
) -> SameForwardRLTokenEncoder:
    from openpi.training import config as train_config

    cfg = train_config.get_config(config_name)
    checkpoint = pathlib.Path(checkpoint_dir)
    logging.info("Loading same-forward RLToken autoencoder config=%s checkpoint=%s", config_name, checkpoint)
    model = cfg.model.load(_model.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
    autoencoder = getattr(model, "rl_token_autoencoder", None)
    if autoencoder is None:
        raise ValueError(f"Config {config_name!r} checkpoint {checkpoint} does not contain rl_token_autoencoder")
    return SameForwardRLTokenEncoder(autoencoder)
