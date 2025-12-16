import abc
import dataclasses
import enum

from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch
import typing as tp
import openpi.shared.array_typing as at

from openpi.models.model import ArrayT, Observation, preprocess_observation

# Define Values type for categorical distribution
# vd (value_dims): number of bins in the categorical distribution
# Paper uses 201 bins over range (-1, 0) for normalized returns
Values = at.Float[ArrayT, "*b vd"]

class ValueModelType(enum.Enum):
    """Enum for different types of value models."""

    SHARED_BACKBONE = "shared_backbone"
    SMALL_BACKBONE = "small_backbone"

@dataclasses.dataclass(frozen=True)
class BaseValueModelConfig(abc.ABC):
    """Configuration shared by all value models."""

    # Value distribution parameters
    value_dims: int  # Number of bins in categorical distribution (e.g., 201)
    value_min: float  # Minimum value in distribution (e.g., -1.0)
    value_max: float  # Maximum value in distribution (e.g., 0.0)
    
    @property
    @abc.abstractmethod
    def model_type(self) -> str:
        """The value model type."""
    
    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseValueModel":
        """Create a new value model."""
    
    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> Observation:
        """Returns input spec (observation) for the model."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def value_support(self) -> at.Float[at.Array, "vd"]:
        """Returns the value support (bin centers) for the categorical distribution."""
        return jnp.linspace(self.value_min, self.value_max, self.value_dims)
    
class BaseValueModel(nnx.Module, abc.ABC):
    """Base class for all value model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (value_dims, value_min, value_max).
    """

    value_dims: int = 201  # Number of bins in categorical distribution
    value_min: float = -1.0  # Minimum value in distribution
    value_max: float = 0.0  # Maximum value in distribution

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        returns: at.Float[at.Array, "*b"],
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b"]:
        """Compute cross-entropy loss between predicted value distribution and target.

        Args:
            rng: Random key for any stochastic operations
            observation: Input observation (images, state, language)
            returns: Monte Carlo returns normalized to [value_min, value_max]
            train: Whether in training mode

        Returns:
            Loss per sample in batch
        """
        ...

    @abc.abstractmethod
    def forward(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b vd"]:
        """Forward pass to get value distribution logits.

        Args:
            rng: Random key for dropout and other stochastic operations
            observation: Input observation (images, state, language)
            train: Whether in training mode

        Returns:
            Logits over value bins, shape (*batch, value_dims)
        """
        ...

    def predict_value(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
    ) -> at.Float[at.Array, "*b"]:
        """Predict expected value from the categorical distribution.

        Args:
            rng: Random key
            observation: Input observation

        Returns:
            Expected value (mean of distribution), shape (*batch,)
        """
        logits = self.forward(rng, observation, train=False)
        probs = jax.nn.softmax(logits, axis=-1)
        value_support = jnp.linspace(self.value_min, self.value_max, self.value_dims)
        expected_value = jnp.sum(probs * value_support, axis=-1)
        return expected_value

    def value_support(self) -> at.Float[at.Array, "vd"]:
        """Returns the value support (bin centers) for the categorical distribution."""
        return jnp.linspace(self.value_min, self.value_max, self.value_dims)
