from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy

def _split_actions_and_prefix_hidden(result):
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, None


def _drop_language_from_prefix_hidden(prefix_hidden, observation: _model.Observation):
    prefix_out, prefix_mask = prefix_hidden
    if observation.tokenized_prompt is None:
        return prefix_out, prefix_mask
    image_token_count = prefix_out.shape[1] - observation.tokenized_prompt.shape[1]
    return prefix_out[:, :image_token_count], prefix_mask[:, :image_token_count]


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        same_forward_rl_token_encoder: Any | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._same_forward_rl_token_encoder = same_forward_rl_token_encoder

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._guided_inference = nnx_utils.module_jit(model.guided_inference)
            self._sample_actions_with_prefix_hidden = (
                nnx_utils.module_jit(model.sample_actions_with_prefix_hidden)
                if hasattr(model, "sample_actions_with_prefix_hidden")
                else None
            )
            self._guided_inference_with_prefix_hidden = (
                nnx_utils.module_jit(model.guided_inference_with_prefix_hidden)
                if hasattr(model, "guided_inference_with_prefix_hidden")
                else None
            )
            self._sample_actions_with_rl_token = (
                nnx_utils.module_jit(model.sample_actions_with_rl_token)
                if hasattr(model, "sample_actions_with_rl_token")
                else None
            )
            self._guided_inference_with_rl_token = (
                nnx_utils.module_jit(model.guided_inference_with_rl_token)
                if hasattr(model, "guided_inference_with_rl_token")
                else None
            )
            self._embed_prefix_hidden = (
                nnx_utils.module_jit(model.embed_prefix_hidden, static_argnames=("drop_language",))
                if hasattr(model, "embed_prefix_hidden")
                else None
            )
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, prev_action: np.ndarray | None = None, use_rtc: bool = False, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise
        guided_kwargs = dict(sample_kwargs)

        observation = _model.Observation.from_dict(inputs)
        model_has_rl_token_autoencoder = getattr(self._model, "rl_token_autoencoder", None) is not None
        needs_rl_token = (
            not self._is_pytorch_model
            and (
                model_has_rl_token_autoencoder
                or self._same_forward_rl_token_encoder is not None
            )
        )
        sample_actions_fn = self._sample_actions
        guided_inference_fn = self._guided_inference
        sample_actions_returns_rl_token = False
        guided_inference_returns_rl_token = False
        if needs_rl_token:
            sample_with_rl_token = getattr(self, "_sample_actions_with_rl_token", None)
            guided_with_rl_token = getattr(self, "_guided_inference_with_rl_token", None)
            sample_with_prefix = getattr(self, "_sample_actions_with_prefix_hidden", None)
            guided_with_prefix = getattr(self, "_guided_inference_with_prefix_hidden", None)
            prefer_same_forward = self._same_forward_rl_token_encoder is not None
            if model_has_rl_token_autoencoder and sample_with_rl_token is not None and not prefer_same_forward:
                sample_actions_fn = sample_with_rl_token
                sample_actions_returns_rl_token = True
            elif sample_with_prefix is not None:
                sample_actions_fn = sample_with_prefix
            else:
                sample_kwargs["return_prefix_hidden"] = True
            if model_has_rl_token_autoencoder and guided_with_rl_token is not None and not prefer_same_forward:
                guided_inference_fn = guided_with_rl_token
                guided_inference_returns_rl_token = True
            elif guided_with_prefix is not None:
                guided_inference_fn = guided_with_prefix
            else:
                guided_kwargs["return_prefix_hidden"] = True

        start_time = time.monotonic()
        prefix_hidden = None
        z_rl = None
        if use_rtc:
            if prev_action is None:
                origin_actions, token_result = _split_actions_and_prefix_hidden(
                    sample_actions_fn(sample_rng_or_pytorch_device, observation, **sample_kwargs)
                )
                if sample_actions_returns_rl_token:
                    z_rl = token_result
                else:
                    prefix_hidden = token_result
                outputs = {
                    "state": inputs["state"],
                    "actions": origin_actions,
                    "origin_actions": origin_actions,
                }
            else:
                prev_action = jnp.asarray(prev_action)[np.newaxis, ...]  # Add batch dimension
                origin_actions, token_result = _split_actions_and_prefix_hidden(
                    guided_inference_fn(sample_rng_or_pytorch_device, prev_action, observation, **guided_kwargs)
                )
                if guided_inference_returns_rl_token:
                    z_rl = token_result
                else:
                    prefix_hidden = token_result
                outputs = {
                    "state": inputs["state"],
                    "actions": origin_actions,
                    "origin_actions": origin_actions,
                }
        else:
            origin_actions, token_result = _split_actions_and_prefix_hidden(
                sample_actions_fn(sample_rng_or_pytorch_device, observation, **sample_kwargs)
            )
            if sample_actions_returns_rl_token:
                z_rl = token_result
            else:
                prefix_hidden = token_result
            outputs = {
                "state": inputs["state"],
                "actions": origin_actions,
                "origin_actions": origin_actions,
            }
        if needs_rl_token:
            if z_rl is None:
                if self._same_forward_rl_token_encoder is not None:
                    if prefix_hidden is None:
                        prefix_hidden = self._model.embed_prefix_hidden(observation, drop_language=False)
                    z_rl = self._same_forward_rl_token_encoder.encode(prefix_hidden, observation)
                else:
                    if prefix_hidden is None:
                        prefix_hidden = self._model.embed_prefix_hidden(observation, drop_language=True)
                    else:
                        prefix_hidden = _drop_language_from_prefix_hidden(prefix_hidden, observation)
                    prefix_out, prefix_mask = prefix_hidden
                    z_rl = self._model.rl_token_autoencoder.encode(jax.lax.stop_gradient(prefix_out), prefix_mask)
            outputs["z_rl"] = z_rl
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def infer_rl_token(self, obs: dict) -> dict:
        """Encode only the RL token and transformed state, without action sampling.

        Replay re-encoding needs deterministic visual tokens, not sampled action
        chunks. `infer()` prefers `sample_actions_with_rl_token()` when available,
        which advances the policy RNG and can make repeated z-only conversion
        harder to audit. This path uses the model prefix directly and never calls
        the diffusion/action sampler.
        """
        if self._is_pytorch_model:
            raise ValueError("RL token-only inference is only supported for JAX policies")
        if self._embed_prefix_hidden is None:
            raise ValueError("policy model does not expose embed_prefix_hidden")

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)
        prefix_hidden = self._embed_prefix_hidden(observation, drop_language=False)
        if self._same_forward_rl_token_encoder is not None:
            z_rl = self._same_forward_rl_token_encoder.encode(prefix_hidden, observation)
            z_rl_source = "vla_same_forward"
        else:
            if getattr(self._model, "rl_token_autoencoder", None) is None:
                raise ValueError("policy model does not have rl_token_autoencoder")
            prefix_out, prefix_mask = _drop_language_from_prefix_hidden(prefix_hidden, observation)
            z_rl = self._model.rl_token_autoencoder.encode(jax.lax.stop_gradient(prefix_out), prefix_mask)
            z_rl_source = "model_rl_token_autoencoder"
        return {
            "state": np.asarray(inputs["state"][0, ...]),
            "z_rl": np.asarray(z_rl[0, ...]),
            "z_rl_source": z_rl_source,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
