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
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to the RLT context samplers.
            metadata: Additional metadata to store with the policy.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._observation_transform = observation_transform or (lambda observation: observation)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        if not hasattr(model, "sample_action_chunk_with_rlt_context"):
            raise NotImplementedError("Policy requires sample_action_chunk_with_rlt_context().")
        if not hasattr(model, "sample_action_chunk_with_inference_time_rtc_context"):
            raise NotImplementedError("Policy requires sample_action_chunk_with_inference_time_rtc_context().")
        self._sample_action_chunk_with_rlt_context = nnx_utils.module_jit(model.sample_action_chunk_with_rlt_context)
        self._sample_action_chunk_with_inference_time_rtc_context = nnx_utils.module_jit(
            model.sample_action_chunk_with_inference_time_rtc_context
        )
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
        return_rlt_state: bool = False,
    ) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)

        def _batch_action(value: np.ndarray | None):
            if value is None:
                return None
            batched = jnp.asarray(value)
            if batched.ndim == 2:
                batched = batched[None, ...]
            return batched

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        if chunking_mode is None:
            chunking_mode = "inference_time"
        if chunking_mode != "inference_time":
            raise ValueError("Policy only supports chunking_mode='inference_time'.")

        def _inference_time_sampling_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
            return dict(kwargs)

        def _plain_sampling_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
            kwargs = dict(kwargs)
            for key in ("replan_start_step", "handoff_delay_steps", "guidance_scale"):
                kwargs.pop(key, None)
            return kwargs

        observation = self._observation_transform(_model.Observation.from_dict(inputs))
        start_time = time.monotonic()
        batched_prev_action = _batch_action(prev_action)
        if batched_prev_action is None:
            fused_outputs = self._sample_action_chunk_with_rlt_context(
                sample_rng,
                observation,
                **_plain_sampling_kwargs(sample_kwargs),
            )
        else:
            fused_outputs = self._sample_action_chunk_with_inference_time_rtc_context(
                sample_rng,
                batched_prev_action,
                observation,
                **_inference_time_sampling_kwargs(sample_kwargs),
            )
        model_actions = fused_outputs["actions"]
        rlt_outputs = {
            "rlt_embeddings": fused_outputs["rlt_embeddings"],
            "rlt_mask": fused_outputs["rlt_mask"],
        }

        batched_model_outputs = {
            "state": inputs["state"],
            "actions": model_actions,
        }
        model_time = time.monotonic() - start_time
        model_outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), batched_model_outputs)
        transformed_outputs = self._output_transform(model_outputs)
        transformed_outputs["model_actions"] = model_outputs["actions"]
        transformed_outputs["model_state"] = model_outputs["state"]
        if return_rlt_state:
            transformed_outputs.update(jax.tree.map(lambda x: np.asarray(x[0, ...]), rlt_outputs))
        transformed_outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return transformed_outputs

    def transform_model_outputs(
        self,
        *,
        state: np.ndarray,
        actions: np.ndarray,
    ) -> dict:
        model_outputs = {
            "state": state,
            "actions": actions,
        }
        transformed_outputs = self._output_transform(model_outputs)
        transformed_outputs["model_actions"] = actions
        transformed_outputs["model_state"] = state
        return transformed_outputs

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
