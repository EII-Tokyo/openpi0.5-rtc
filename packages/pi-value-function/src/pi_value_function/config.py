from typing import TYPE_CHECKING
import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models.model import Observation
from pi_value_function.value_model_base import BaseValueModelConfig, ValueModelType
import openpi.shared.array_typing as at
from openpi.models import model as _model

if TYPE_CHECKING:
    from pi_value_function.pi_value import PiValue


@dataclasses.dataclass(frozen=True)
class PiValueConfig(BaseValueModelConfig):
    """ Configuration for the Pi value model. 
        By default, the value model shares a backbone with the action model,
        just like in the original Pi*0.6 paper.
    """

    value_dims: int = 1
    value_min: float = -1.0
    value_max: float = 0.0
    dtype: str = "bfloat16"
    gemma_variant: str = "gemma-3-270m"
    siglip_variant: str = "siglip2-so400m-patch16-384"
    
    def model_type(self) -> str:
        # return ValueModelType.SHARED_BACKBONE
        return "pi_value"
    
    def create(self, rng: at.KeyArrayLike) -> "PiValue":
        from pi_value_function.pi_value import PiValue

        return PiValue(self, rngs=nnx.Rngs(rng))
    
    def inputs_spec(self, *, batch_size: int = 1) -> Observation:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_) #TODO: Do i need image masking?
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, 32], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, 48], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, 48], bool),
            )
        return observation_spec