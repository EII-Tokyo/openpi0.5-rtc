"""Pure state helpers for stable Physics Inspector startup."""

from dataclasses import dataclass
from enum import Enum, auto
import math
from typing import Mapping


def target_change_is_isolated(
    before: Mapping[str, float],
    after: Mapping[str, float],
    operated_joint: str,
    requested_target: float,
    tolerance: float = 1e-9,
) -> bool:
    """Return whether only the requested joint acquired the requested target."""

    if before.keys() != after.keys() or operated_joint not in before:
        return False
    if not math.isclose(
        after[operated_joint],
        requested_target,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        return False
    return all(
        math.isclose(after[name], value, rel_tol=0.0, abs_tol=tolerance)
        for name, value in before.items()
        if name != operated_joint
    )


@dataclass
class LoadingStability:
    """Require a run of zero-pending samples before declaring USD stable."""

    required_samples: int
    consecutive_zero: int = 0

    def observe(self, pending_files: int) -> bool:
        self.consecutive_zero = self.consecutive_zero + 1 if pending_files == 0 else 0
        return self.consecutive_zero >= self.required_samples


class RecoveryDecision(Enum):
    """Decision returned for each observed Inspector state."""

    KEEP_MONITORING = auto()
    RECOVER = auto()
    FAIL = auto()


@dataclass
class RecoveryGuard:
    """Permit exactly one native Inspector recovery."""

    recoveries: int = 0

    def observe(self, disabled: bool) -> RecoveryDecision:
        if not disabled:
            return RecoveryDecision.KEEP_MONITORING
        if self.recoveries == 0:
            self.recoveries = 1
            return RecoveryDecision.RECOVER
        return RecoveryDecision.FAIL
