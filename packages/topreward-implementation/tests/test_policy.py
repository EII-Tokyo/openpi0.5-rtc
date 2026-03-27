from __future__ import annotations

import numpy as np
from PIL import Image

from topreward.serving.policy import TOPRewardPolicy
from topreward.serving.real_time import RealTimeConfig


class FakeModel:
    def get_log_prob(self, video_frames: list[Image.Image], prompt: str) -> float:
        del prompt
        return float(len(video_frames))


def _frame() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_policy_scores_and_returns_value_alias() -> None:
    policy = TOPRewardPolicy(
        model=FakeModel(),
        instruction="test task",
        config=RealTimeConfig(score_interval=1, subsample_factor=1, max_buffer_frames=16),
    )

    out1 = policy.infer({"observation/exterior_image_1_left": _frame(), "prompt": "test task"})
    out2 = policy.infer({"observation/exterior_image_1_left": _frame(), "prompt": "test task"})

    assert out1["status"] == "scored"
    assert out1["value"] == out1["progress"]
    assert out2["status"] == "scored"
    assert out2["frame_idx"] == 2
    assert float(out2["progress"]) >= 0.0


def test_policy_missing_frame_status() -> None:
    policy = TOPRewardPolicy(model=FakeModel(), instruction="task")
    out = policy.infer({"prompt": "task"})
    assert out["status"] == "missing_frame"


def test_policy_resets_on_prompt_change() -> None:
    policy = TOPRewardPolicy(
        model=FakeModel(),
        instruction="task a",
        config=RealTimeConfig(score_interval=1, subsample_factor=1, max_buffer_frames=16),
        reset_on_prompt_change=True,
    )

    out1 = policy.infer({"observation/exterior_image_1_left": _frame(), "prompt": "task a"})
    out2 = policy.infer({"observation/exterior_image_1_left": _frame(), "prompt": "task b"})

    assert out1["frame_idx"] == 1
    assert out2["frame_idx"] == 1
    assert out2["instruction"] == "task b"
