import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
from typing_extensions import override

from openpi.models import pi0_config
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.pi0_rl import Pi0RL


@dataclasses.dataclass(frozen=True)
class Pi0RLConfig(pi0_config.Pi0Config):
    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0RL":
        from openpi.models.pi0_rl import Pi0RL

        if self.rl_token is None:
            raise ValueError("Pi0RLConfig requires rl_token to be configured.")
        return Pi0RL(self, rngs=nnx.Rngs(rng))
