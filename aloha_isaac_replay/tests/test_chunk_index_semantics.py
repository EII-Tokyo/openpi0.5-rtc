from __future__ import annotations

import json
from pathlib import Path


def test_action_chunk_broker_slices_one_executed_step_per_runtime_tick() -> None:
    broker = Path("packages/openpi-client/src/openpi_client/action_chunk_broker.py").read_text()
    summary = json.loads(Path("reports/aloha_isaac_replay/action_provenance/summary.json").read_text())
    assert "self._s = 25" in broker
    assert "self._d = 10" in broker
    assert "def _slice_result_cache" in broker
    assert "sliced[key] = value[self._cur_step, ...]" in broker
    assert summary["chunk_length"] == 25
    assert summary["rtc_s"] == 25
    assert summary["rtc_d"] == 10
