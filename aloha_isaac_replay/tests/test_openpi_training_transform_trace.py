from __future__ import annotations

from pathlib import Path


def test_training_pipeline_traces_hdf5_action_to_delta_target() -> None:
    config = Path("src/openpi/training/config.py").read_text()
    policy = Path("src/openpi/policies/aloha_policy.py").read_text()
    report = Path("reports/aloha_isaac_replay/action_provenance/training_transform.md").read_text()
    assert "\"actions\": \"action\"" in config
    assert "transforms.DeltaActions" in config
    assert "delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)" in config
    assert "outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)]" in config
    assert "def _encode_actions_inv" in policy
    assert "training_arm_target = flip(HDF5_action_arm) - flip(HDF5_state_arm)" in report
    assert "DeltaActions" in report
