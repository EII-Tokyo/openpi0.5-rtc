from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

from topreward.model import TOPRewardModel

PROMPT_TEMPLATE_TRUE_OR_NOT = (
    "<|vision_start|><|video_pad|><|vision_end|> The above video shows a robot manipulation trajectory that completes the "
    "following task: {instruction}. Decide whether the above statement is True or not. "
    "The answer is:"
)

PROMPT_TEMPLATE_TRUE_OR_FALSE = (
    "<|vision_start|><|video_pad|><|vision_end|> The above video shows a robot manipulation trajectory that completes the "
    "following task: {instruction}. Decide whether the above statement is True or False. "
    "The answer is:"
)

def build_prompt(instruction: str, score_mode: str = "true_log_prob") -> str:
    """Build the raw-text TOPReward prompt without chat template wrapping."""
    clean_instruction = instruction.strip()
    if not clean_instruction:
        raise ValueError("instruction must not be empty")

    if score_mode == "true_log_prob":
        template = PROMPT_TEMPLATE_TRUE_OR_NOT
    elif score_mode == "true_minus_false":
        template = PROMPT_TEMPLATE_TRUE_OR_FALSE
    else:
        raise ValueError(f"Unsupported score_mode='{score_mode}'. Use 'true_log_prob' or 'true_minus_false'.")

    return template.format(instruction=clean_instruction)


def compute_reward(
    model: TOPRewardModel,
    frames: Sequence[Image.Image],
    instruction: str,
    prefix_end: int,
    score_mode: str = "true_log_prob",
    true_token: str = "True",
    false_token: str = "False",
    max_prefix_frames: int | None = None,
) -> float:
    """Compute reward for a trajectory prefix frames[:prefix_end]."""
    if not frames:
        raise ValueError("frames must not be empty")
    if prefix_end < 1 or prefix_end > len(frames):
        raise ValueError(f"prefix_end must be in [1, {len(frames)}], got {prefix_end}")

    prompt = build_prompt(instruction, score_mode=score_mode)
    prefix_frames = list(frames[:prefix_end])
    if max_prefix_frames is not None and len(prefix_frames) > max_prefix_frames:
        indices = np.linspace(0, len(prefix_frames) - 1, max_prefix_frames, dtype=int)
        prefix_frames = [prefix_frames[i] for i in indices]

    if score_mode == "true_log_prob":
        if hasattr(model, "get_log_prob_for_token"):
            return model.get_log_prob_for_token(prefix_frames, prompt, true_token)
        return model.get_log_prob(prefix_frames, prompt)

    if score_mode == "true_minus_false":
        if hasattr(model, "get_log_probs_for_tokens"):
            scores = model.get_log_probs_for_tokens(prefix_frames, prompt, [true_token, false_token])
            return float(scores[true_token] - scores[false_token])

        true_score = model.get_log_prob_for_token(prefix_frames, prompt, true_token)
        false_score = model.get_log_prob_for_token(prefix_frames, prompt, false_token)
        return float(true_score - false_score)

    raise ValueError(f"Unsupported score_mode='{score_mode}'. Use 'true_log_prob' or 'true_minus_false'.")
