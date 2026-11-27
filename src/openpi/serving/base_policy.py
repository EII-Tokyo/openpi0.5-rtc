import abc
from typing import Dict


class PolicyWaitingForObservation(RuntimeError):
    """No action is available until the high-level policy has a fresh result."""


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
