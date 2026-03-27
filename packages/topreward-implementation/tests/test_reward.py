from __future__ import annotations

from PIL import Image

from topreward.advantage import AdvantageConfig
from topreward.advantage import compute_advantages
from topreward.advantage import compute_episode_advantages
from topreward.progress import estimate_progress
from topreward.reward import build_prompt
from topreward.reward import compute_reward
from topreward.success_detector import detect_success


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []
        self.token_calls: list[tuple[int, str, str]] = []

    def get_log_prob(self, video_frames: list[Image.Image], prompt: str) -> float:
        self.calls.append((len(video_frames), prompt))
        return float(len(video_frames))

    def get_log_prob_for_token(self, video_frames: list[Image.Image], prompt: str, token: str) -> float:
        self.token_calls.append((len(video_frames), prompt, token))
        if token == "True":
            return float(len(video_frames))
        if token == "False":
            return float(len(video_frames)) - 2.0
        return float(len(video_frames)) - 1.0

    def get_log_probs_for_tokens(self, video_frames: list[Image.Image], prompt: str, tokens: list[str]) -> dict[str, float]:
        return {token: self.get_log_prob_for_token(video_frames, prompt, token) for token in tokens}


def _make_frames(num_frames: int) -> list[Image.Image]:
    return [Image.new("RGB", (4, 4), color=(i, i, i)) for i in range(num_frames)]


def test_build_prompt_contains_instruction_without_chat_wrapper() -> None:
    prompt = build_prompt("pick up the battery")
    assert "pick up the battery" in prompt
    assert "system" not in prompt.lower()
    assert "assistant" not in prompt.lower()


def test_build_prompt_true_minus_false_mode_uses_true_or_false_wording() -> None:
    prompt = build_prompt("pick up the battery", score_mode="true_minus_false")
    assert "True or False" in prompt


def test_compute_reward_uses_prefix_and_prompt() -> None:
    model = FakeModel()
    frames = _make_frames(5)

    score = compute_reward(model, frames, "test task", prefix_end=3)

    assert score == 3.0
    assert model.token_calls
    prefix_len, prompt, token = model.token_calls[-1]
    assert prefix_len == 3
    assert "test task" in prompt
    assert token == "True"


def test_compute_reward_true_minus_false_mode() -> None:
    model = FakeModel()
    frames = _make_frames(5)

    score = compute_reward(model, frames, "test task", prefix_end=3, score_mode="true_minus_false")

    assert score == 2.0
    assert model.token_calls


def test_estimate_progress_samples_prefixes_and_normalizes() -> None:
    model = FakeModel()
    frames = _make_frames(8)

    progress = estimate_progress(model, frames, "demo", k=4)

    assert progress["timestamps"] == [1, 3, 5, 8]
    assert progress["raw_rewards"] == [1.0, 3.0, 5.0, 8.0]
    assert progress["normalized"][0] == 0.0
    assert abs(progress["normalized"][-1] - 1.0) < 1e-6


def test_estimate_progress_supports_fixed_prefix_stride() -> None:
    model = FakeModel()
    frames = _make_frames(8)

    progress = estimate_progress(model, frames, "demo", prefix_stride=3)

    assert progress["timestamps"] == [1, 4, 7, 8]
    assert progress["raw_rewards"] == [1.0, 4.0, 7.0, 8.0]
    assert progress["normalized"][0] == 0.0
    assert abs(progress["normalized"][-1] - 1.0) < 1e-6


def test_detect_success_averages_last_prefixes() -> None:
    model = FakeModel()
    frames = _make_frames(6)

    result = detect_success(model, frames, "demo", n_final_frames=3, threshold=4.5)

    assert result["final_prefixes"] == [4, 5, 6]
    assert result["score"] == 5.0
    assert result["is_success"] is True


def test_compute_advantages_tiers_and_weights() -> None:
    cfg = AdvantageConfig(good_threshold=0.5, bad_threshold=-0.2, good_weight=2.0, neutral_weight=1.0, bad_weight=0.1)
    progress = [0.0, 0.7, 0.1, 0.15]

    out = compute_advantages(progress, cfg)

    assert out["tiers"] == ["good", "bad", "neutral"]
    assert out["weights"] == [2.0, 0.1, 1.0]
    assert len(out["deltas"]) == 3


def test_compute_episode_advantages_pipeline() -> None:
    model = FakeModel()
    frames = _make_frames(5)
    cfg = AdvantageConfig()

    out = compute_episode_advantages(
        model=model,
        frames=frames,
        instruction="demo",
        action_timestamps=[1, 2, 3, 4, 5],
        config=cfg,
    )

    assert out["timestamps"] == [1, 2, 3, 4, 5]
    assert len(out["normalized_progress"]) == 5
    assert len(out["weights"]) == 4
