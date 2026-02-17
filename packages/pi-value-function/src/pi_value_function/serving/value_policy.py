import logging
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi_client import base_policy
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from pi_value_function.pi_value import PiValue

logger = logging.getLogger(__name__)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    return bool(value)


class ValuePolicy(base_policy.BasePolicy):
    def __init__(
        self,
        model: PiValue,
        tokenizer: _tokenizer.Gemma3Tokenizer,
        *,
        return_distribution: bool = False,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._rng = jax.random.PRNGKey(0)
        self._return_distribution_default = return_distribution

        # Create JIT-compiled inference function for speed
        # Extract graphdef and state for stateless JIT function
        graphdef, state = nnx.split(model)

        @jax.jit
        def _jit_predict_value(rng_key, observation_dict, state):
            # Reconstruct model from state
            model_instance = nnx.merge(graphdef, state)
            return model_instance.predict_value(rng_key, observation_dict)

        @jax.jit
        def _jit_forward(rng_key, observation_dict, state):
            model_instance = nnx.merge(graphdef, state)
            logits = model_instance.forward(rng_key, observation_dict, train=False)
            probs = jax.nn.softmax(logits, axis=-1)
            value_support = jnp.linspace(model_instance.value_min, model_instance.value_max, model_instance.value_dims)
            expected_value = jnp.sum(probs * value_support, axis=-1)
            return expected_value, probs

        self._jit_predict_value = _jit_predict_value
        self._jit_forward = _jit_forward
        self._model_state = state

        logger.info("Value policy initialized with JIT-compiled inference")

    def infer(
        self,
        obs: Dict,
        prev_action: Optional[Dict] = None,
        use_rtc: bool = False,
        return_distribution: Optional[bool] = None,
    ) -> Dict:
        # 1. Map client image keys to model image keys
        # Model keys based on train.py: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
        # Client keys based on main.py: exterior_image_1_left, wrist_image_left, (maybe exterior_image_2_left)
        
        # Mapping strategy:
        # exterior_image_1_left -> base_0_rgb
        # wrist_image_left -> left_wrist_0_rgb
        # exterior_image_2_left -> right_wrist_0_rgb (if present)
        
        images_map = {
            "exterior_image_1_left": "base_0_rgb",
            "wrist_image_left": "left_wrist_0_rgb",
            "exterior_image_2_left": "right_wrist_0_rgb",
            "left_image": "base_0_rgb", # Fallback if main.py changes
            "wrist_image": "left_wrist_0_rgb",
            "right_image": "right_wrist_0_rgb",
        }
        
        images = {}
        image_masks = {}
        
        # Iterate over keys looking for images
        for key, value in obs.items():
            if not isinstance(key, str): continue
            
            clean_key = key.replace("observation/", "")
            
            target_key = None
            if clean_key in images_map:
                target_key = images_map[clean_key]
            elif "image" in clean_key:
                # Fallback: keep original name if likely an image but not mapped
                target_key = clean_key
            
            if target_key:
                # Preprocess image
                img = value
                
                # Ensure float32 [-1, 1]
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0 * 2.0 - 1.0
                
                # Check resolution needed? PiValue uses SigLIP which is 224x224.
                # Assuming client might send arbitrary size, we should resize to 224x224
                # However, JAX resize might be slow on CPU if client didn't do it.
                # Ideally client does it. But we can do it here to be safe.
                # Using openpi.shared.image_tools if available or just assume 224.
                # main.py from client does resize to 224.
                
                # Add batch dim if missing
                if img.ndim == 3:
                    img = img[None, ...]
                
                images[target_key] = jnp.array(img)
                image_masks[target_key] = jnp.ones(img.shape[:1], dtype=bool)

        # 2. State
        joint_position = obs.get("observation/joint_position")
        gripper_position = obs.get("observation/gripper_position")
        
        state = None
        if joint_position is not None:
             jp = jnp.array(joint_position)
             gp = jnp.array(gripper_position) if gripper_position is not None else jnp.array([0.0])
             
             if jp.ndim == 1: jp = jp[None, ...]
             if gp.ndim == 1: gp = gp[None, ...]
             
             state = jnp.concatenate([jp, gp], axis=-1)
        else:
            # Create dummy state if missing, though unlikely for real bot
            state = jnp.zeros((1, 8)) # Assuming 7 DoF + gripper ? Or just use dummy size

        # 3. Prompt
        prompt = obs.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0] # Take first if list
            
        tokenized_prompt_ids, tokenized_prompt_mask = self._tokenizer.tokenize(prompt)
        
        # Add batch dim
        tokenized_prompt_ids = jnp.array(tokenized_prompt_ids[None, ...])
        tokenized_prompt_mask = jnp.array(tokenized_prompt_mask[None, ...])

        # 4. Create Observation
        observation = _model.Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt_ids,
            tokenized_prompt_mask=tokenized_prompt_mask
        )

        # 5. Determine output mode.
        # `return_distribution` can be set:
        # 1) per-request via infer arg (if called directly), or
        # 2) per-request via obs["return_distribution"] / obs["observation/return_distribution"], or
        # 3) by server default from CLI.
        obs_return_distribution = obs.get("return_distribution")
        if obs_return_distribution is None:
            obs_return_distribution = obs.get("observation/return_distribution")
        if return_distribution is None:
            if obs_return_distribution is None:
                return_distribution = self._return_distribution_default
            else:
                return_distribution = _coerce_bool(obs_return_distribution)

        # 6. Run Inference
        self._rng, key = jax.random.split(self._rng)

        if return_distribution:
            expected_value, probs = self._jit_forward(key, observation, self._model_state)
            value_float = float(expected_value[0])
            distribution = np.array(probs[0])  # (201,)
            return {
                "value": value_float,
                "distribution": distribution.tolist(),
            }
        else:
            predicted_value = self._jit_predict_value(key, observation, self._model_state)
            value_float = float(predicted_value[0])
            return {
                "value": value_float,
            }
