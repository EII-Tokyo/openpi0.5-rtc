from __future__ import annotations

from pathlib import Path


def test_runtime_passes_same_action_to_robot_and_recorder_subscriber() -> None:
    runtime = Path("packages/openpi-client/src/openpi_client/runtime/runtime.py").read_text()
    report = Path("reports/aloha_isaac_replay/action_provenance/action_source.md").read_text()
    apply_idx = runtime.index("self._environment.apply_action(action)")
    subscriber_idx = runtime.index("subscriber.on_step(observation[\"origin_observation\"], action)")
    assert apply_idx < subscriber_idx
    assert "self._last_action = action.get(\"actions\")" in runtime
    assert "then passes the same action to subscribers" in report
