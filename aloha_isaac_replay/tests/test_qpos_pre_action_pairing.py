from __future__ import annotations

import json
from pathlib import Path


def test_qpos_pre_action_pairing_is_recorded() -> None:
    summary = json.loads(Path("reports/aloha_isaac_replay/action_provenance/summary.json").read_text())
    assert summary["hdf5_write_timing"] == "pre-action qpos observation paired with post-observation emitted command"
    assert summary["qpos_timing"] == "observed before action is applied in Runtime._step"

