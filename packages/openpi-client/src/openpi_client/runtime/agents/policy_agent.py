from typing_extensions import override

from openpi_client import base_policy as _base_policy
from openpi_client.runtime import agent as _agent


class PolicyAgent(_agent.Agent):
    """An agent that uses a policy to determine actions."""

    def __init__(self, policy: _base_policy.BasePolicy) -> None:
        self._policy = policy

    @override
    def get_action(self, observation: dict) -> dict:
        return self._policy.infer(observation)

    def get_rlt_frame_token(self, observation: dict) -> dict | None:
        infer_rl_token = getattr(self._policy, "infer_rl_token", None)
        if infer_rl_token is None:
            return None
        return infer_rl_token(observation)

    def rlt_actor_status(self) -> dict | None:
        status = getattr(self._policy, "rlt_actor_status", None)
        if status is None:
            return None
        return status()

    def flush_action_cache(self, reason: str | None = None) -> None:
        flush = getattr(self._policy, "flush_action_cache", None)
        if flush is not None:
            flush(reason)

    def set_rlt_gate(self, *, enabled: bool, epoch: int, reason: str | None = None) -> None:
        setter = getattr(self._policy, "set_rlt_gate", None)
        if setter is not None:
            setter(enabled=enabled, epoch=epoch, reason=reason)
        else:
            self.flush_action_cache(reason)

    def reset(self) -> None:
        self._policy.reset()
