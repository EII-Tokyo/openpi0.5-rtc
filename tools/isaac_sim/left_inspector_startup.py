"""Pure state helpers for stable Physics Inspector startup."""

from dataclasses import dataclass
from enum import Enum, auto


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
