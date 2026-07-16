from __future__ import annotations

import pytest

from aloha_isaac_replay.runtime.eula_status import IsaacEulaProbe


@pytest.mark.manual
def test_isaac_runtime_smoke_requires_manual_eula_probe() -> None:
    pytest.skip("Requires launching Isaac SimulationApp. Run scripts/replay_aloha_qpos_arm_only.py --probe-runtime manually.")


def test_eula_probe_result_shape_is_explicit() -> None:
    probe = IsaacEulaProbe(["python"], 1, "", "Do you accept the EULA? (Yes/No):", True)
    assert probe.manual_action_required is True

