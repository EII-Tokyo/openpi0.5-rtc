from __future__ import annotations

from pathlib import Path


def test_rtc_guidance_space_is_normalized_delta_like_from_standard_actions() -> None:
    broker = Path("packages/openpi-client/src/openpi_client/action_chunk_broker.py").read_text()
    report = Path("reports/aloha_isaac_replay/action_provenance/rtc_transform.md").read_text()
    assert "scaled = self._joint_signs * (prev_actions - obs[\"state\"][:14])" in broker
    assert "q01" in broker and "q99" in broker
    assert "norm_action[:, 6] = last_origin_actions[:, 6]" in broker
    assert "norm_action[:, 13] = last_origin_actions[:, 13]" in broker
    assert "joint_signs * (prev_actions - obs[\"state\"])" in report
    assert "q01/q99 normalization" in report
