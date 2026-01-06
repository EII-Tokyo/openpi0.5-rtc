import logging
from typing import Dict, Optional
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import einops
from flax import nnx

from openpi_client import base_policy
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.shared import image_tools
from pi_value_function.pi_value import PiValue

logger = logging.getLogger(__name__)

class ValuePolicy(base_policy.BasePolicy):
    def __init__(self, model: PiValue, tokenizer: _tokenizer.Gemma3Tokenizer):
        self._model = model
        self._tokenizer = tokenizer
        self._rng = jax.random.PRNGKey(0)

        # Create JIT-compiled inference function for speed
        # Extract graphdef and state for stateless JIT function
        graphdef, state = nnx.split(model)

        @jax.jit
        def _jit_predict_value(rng_key, observation_dict, state):
            # Reconstruct model from state
            model_instance = nnx.merge(graphdef, state)
            return model_instance.predict_value(rng_key, observation_dict)

        self._jit_predict_value = _jit_predict_value
        self._model_state = state

        logger.info("Value policy initialized with JIT-compiled inference")

    def infer(self, obs: Dict, prev_action: Optional[Dict] = None, use_rtc: bool = False) -> Dict:
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

        # 5. Run Inference
        self._rng, key = jax.random.split(self._rng)

        # Use JIT-compiled inference for speed
        predicted_value = self._jit_predict_value(key, observation, self._model_state)

        # Convert to python float
        value_float = float(predicted_value[0])
        
        return {
            "value": value_float,
            # Echo back timestamp or other meta if needed
        }
