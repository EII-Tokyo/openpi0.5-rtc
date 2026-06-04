from collections.abc import Callable, Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi.serving import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi.data import transforms as _transforms
from openpi.models import model as _model
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
        observation_transform: Callable[[_model.Observation], _model.Observation] | None = None,
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
            sample_kwargs: Additional keyword arguments to pass to model.sample_action_chunk.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._observation_transform = observation_transform or (lambda observation: observation)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_action_chunk = model.sample_action_chunk
            self._sample_action_chunk_with_inference_time_rtc = model.sample_action_chunk_with_inference_time_rtc
            self._sample_action_chunk_with_training_time_rtc = model.sample_action_chunk_with_training_time_rtc
        else:
            # JAX model setup
            self._sample_action_chunk = nnx_utils.module_jit(model.sample_action_chunk)
            self._sample_action_chunk_with_inference_time_rtc = nnx_utils.module_jit(model.sample_action_chunk_with_inference_time_rtc)
            self._sample_action_chunk_with_training_time_rtc = nnx_utils.module_jit(model.sample_action_chunk_with_training_time_rtc)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(
        self,
        obs: dict,
        prev_action: np.ndarray | None = None,
        noise: np.ndarray | None = None,
        *,
        chunking_mode: str | None = None,
        action_prefix: np.ndarray | None = None,
        handoff_delay_steps: int | None = None,
    ) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device.
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        def _batch_action(value: np.ndarray | None):
            if value is None:
                return None
            if self._is_pytorch_model:
                batched = torch.from_numpy(np.asarray(value)).to(self._pytorch_device)
            else:
                batched = jnp.asarray(value)
            if batched.ndim == 2:
                batched = batched[None, ...]
            return batched

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        if chunking_mode is None:
            chunking_mode = "sync"
        if chunking_mode not in {"sync", "inference_time", "training_time"}:
            raise ValueError(f"Unknown chunking_mode={chunking_mode!r}")

        def _inference_time_sampling_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
            return dict(kwargs)

        def _plain_sampling_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
            kwargs = dict(kwargs)
            for key in ("replan_start_step", "handoff_delay_steps", "guidance_scale"):
                kwargs.pop(key, None)
            return kwargs

        observation = self._observation_transform(_model.Observation.from_dict(inputs))
        start_time = time.monotonic()
        if chunking_mode == "sync":
            origin_actions = self._sample_action_chunk(
                sample_rng_or_pytorch_device,
                observation,
                **_plain_sampling_kwargs(sample_kwargs),
            )
        elif chunking_mode == "inference_time":
            batched_prev_action = _batch_action(prev_action)
            if batched_prev_action is None:
                origin_actions = self._sample_action_chunk(
                    sample_rng_or_pytorch_device,
                    observation,
                    **_plain_sampling_kwargs(sample_kwargs),
                )
            else:
                origin_actions = self._sample_action_chunk_with_inference_time_rtc(
                    sample_rng_or_pytorch_device,
                    batched_prev_action,
                    observation,
                    **_inference_time_sampling_kwargs(sample_kwargs),
                )
        else:
            batched_action_prefix = _batch_action(action_prefix)
            if batched_action_prefix is None or handoff_delay_steps is None:
                origin_actions = self._sample_action_chunk(
                    sample_rng_or_pytorch_device,
                    observation,
                    **_plain_sampling_kwargs(sample_kwargs),
                )
            else:
                rtc_kwargs = _plain_sampling_kwargs(sample_kwargs)
                origin_actions = self._sample_action_chunk_with_training_time_rtc(
                    sample_rng_or_pytorch_device,
                    observation,
                    action_prefix=batched_action_prefix,
                    handoff_delay_steps=handoff_delay_steps,
                    **rtc_kwargs,
                )

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
