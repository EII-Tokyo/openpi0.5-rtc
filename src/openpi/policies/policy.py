from collections.abc import Sequence
import dataclasses
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
from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


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
        self._input_transforms_raw = list(transforms)
        self._input_transform = _transforms.compose(transforms)
        subtask_transforms = [
            dataclasses.replace(t, for_subtask_generation=True)
            if isinstance(t, _transforms.TokenizePrompt)
            else t
            for t in transforms
        ]
        self._input_transform_subtask = _transforms.compose(subtask_transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._tokenizer = None
        for transform in self._input_transforms_raw:
            if isinstance(transform, _transforms.TokenizePrompt):
                self._tokenizer = transform.tokenizer
                break
        if self._tokenizer is None:
            raise ValueError("Policy requires a TokenizePrompt transform with a configured tokenizer.")

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._sample_subtask_tokens = getattr(model, "sample_subtask_tokens", None)
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._guided_inference = nnx_utils.module_jit(model.guided_inference)
            if hasattr(model, "sample_subtask_tokens"):
                self._sample_subtask_tokens = nnx_utils.module_jit(
                    model.sample_subtask_tokens,
                    static_argnames=(
                        "temperature",
                        "eos_token_id",
                        "max_text_token_id",
                        "debug_top_logits",
                    ),
                )
            else:
                self._sample_subtask_tokens = None
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

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        if use_rtc:
            if prev_action is None:
                origin_actions = self._sample_actions(sample_rng_or_pytorch_device, _model.Observation.from_dict(inputs), **self._sample_kwargs)
                outputs = {
                    "state": inputs["state"],
                    "actions": origin_actions,
                    "origin_actions": origin_actions,
                }
            else:
                prev_action = jnp.asarray(prev_action)[np.newaxis, ...]  # Add batch dimension
                origin_actions = self._guided_inference(sample_rng_or_pytorch_device, prev_action, _model.Observation.from_dict(inputs), **self._sample_kwargs)
                outputs = {
                    "state": inputs["state"],
                    "actions": origin_actions,
                    "origin_actions": origin_actions,
                }
        else:
            origin_actions = self._sample_actions(sample_rng_or_pytorch_device, _model.Observation.from_dict(inputs), **self._sample_kwargs)
            outputs = {
                "state": inputs["state"],
                "actions": origin_actions,
                "origin_actions": origin_actions,
            }
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

    def infer_subtask(
        self,
        obs: dict,
        *,
        temperature: float = 0.0,
        max_text_token_id: int = 240000,
        debug_top_logits: bool = False,
    ) -> dict:
        """Autoregressively decode high-level subtask text from image+prompt inputs."""
        if self._sample_subtask_tokens is None:
            raise NotImplementedError("Current model does not support autoregressive subtask decoding.")

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform_subtask(inputs)

        if self._is_pytorch_model:
            raise NotImplementedError("infer_subtask is currently implemented for JAX PI0/PI05 models.")

        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        tokens = self._sample_subtask_tokens(
            sample_rng,
            _model.Observation.from_dict(inputs),
            temperature=temperature,
            eos_token_id=self._tokenizer.eos_token_id,
            max_text_token_id=max_text_token_id,
            debug_top_logits=debug_top_logits,
        )
        token_ids = np.asarray(tokens[0], dtype=np.int32)
        # Trim after EOS/zero pad.
        eos = self._tokenizer.eos_token_id
        stop = len(token_ids)
        for idx, tid in enumerate(token_ids.tolist()):
            if tid == eos or tid == 0:
                stop = idx + (1 if tid == eos else 0)
                break
        token_ids = token_ids[:stop]
        text = self._tokenizer.decode(token_ids.tolist()) if len(token_ids) else ""
        return {
            "subtask_tokens": token_ids,
            "subtask_text": text,
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
