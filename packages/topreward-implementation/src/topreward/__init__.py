"""TOPReward: Token Probabilities as Hidden Zero-Shot Rewards."""

from topreward.advantage import AdvantageConfig
from topreward.advantage import compute_advantages
from topreward.advantage import compute_episode_advantages
from topreward.model import TOPRewardModel
from topreward.progress import estimate_progress
from topreward.reward import build_prompt
from topreward.reward import compute_reward
from topreward.serving.policy import TOPRewardPolicy
from topreward.success_detector import detect_success

__all__ = [
    "AdvantageConfig",
    "TOPRewardModel",
    "TOPRewardPolicy",
    "build_prompt",
    "compute_advantages",
    "compute_episode_advantages",
    "compute_reward",
    "detect_success",
    "estimate_progress",
]
