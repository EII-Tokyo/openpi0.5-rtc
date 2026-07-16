from __future__ import annotations

from pathlib import Path


def test_inference_pipeline_returns_standard_absolute_action_chunk() -> None:
    config = Path("src/openpi/training/config.py").read_text()
    policy = Path("src/openpi/policies/aloha_policy.py").read_text()
    report = Path("reports/aloha_isaac_replay/action_provenance/inference_transform.md").read_text()
    assert "transforms.AbsoluteActions" in config
    assert "class AlohaOutputs" in policy
    assert "def _encode_actions(" in policy
    assert "model output in training/action space" in report
    assert "standard ALOHA-like" in report
    assert "absolute" in report
