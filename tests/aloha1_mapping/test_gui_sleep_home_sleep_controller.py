from __future__ import annotations

import pytest

from tools.aloha1_mapping.gui_sleep_home_sleep_controller import GuiSleepHomeSleepController
from tools.aloha1_mapping.gui_sleep_home_sleep_controller import compose_arm_target
from tools.aloha1_mapping.gui_sleep_home_sleep_controller import build_gui_button_samples


def test_disarmed_button_request_is_digital_only() -> None:
    controller = GuiSleepHomeSleepController(real_armed=False)
    decision = controller.request_run(digital_at_sleep=True, real_ready=True)
    assert decision.status == "DIGITAL_ONLY_READY"
    assert decision.real_commands_allowed is False


def test_armed_request_requires_real_ready_gate() -> None:
    controller = GuiSleepHomeSleepController(real_armed=True)
    decision = controller.request_run(digital_at_sleep=True, real_ready=False)
    assert decision.status == "BLOCKED_REAL_GATE"
    assert decision.real_commands_allowed is False


def test_armed_request_allows_real_only_after_all_gates() -> None:
    controller = GuiSleepHomeSleepController(real_armed=True)
    decision = controller.request_run(digital_at_sleep=True, real_ready=True)
    assert decision.status == "DIGITAL_AND_REAL_READY"
    assert decision.real_commands_allowed is True


def test_button_rejects_non_sleep_digital_state() -> None:
    controller = GuiSleepHomeSleepController(real_armed=False)
    with pytest.raises(ValueError, match="digital articulation must be at Sleep"):
        controller.request_run(digital_at_sleep=False, real_ready=False)


def test_compose_arm_target_preserves_non_arm_dofs() -> None:
    target = compose_arm_target([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [1, 2, 3, 4, 5, 6])
    assert target == [1, 2, 3, 4, 5, 6, 0.7, 0.8]


def test_gui_button_is_one_smooth_cycle_about_three_seconds_each_way_at_50_hz() -> None:
    samples = build_gui_button_samples(
        sleep=[0, -1.8, 1.6, 0, -1.8, 0],
        home=[0, -0.96, 1.16, 0, -0.3, 0],
        command_hz=50,
        move_seconds=3.5,
    )
    assert len(samples) == 351
    assert samples[0]["segment"] == "sleep_to_home"
    assert samples[175]["segment"] == "sleep_to_home"
    assert samples[175]["q_rad"] == [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
    assert samples[176]["segment"] == "home_to_sleep"
    assert samples[-1]["segment"] == "home_to_sleep"
    assert samples[-1]["time_ns"] == 7_000_000_000
    assert samples[0]["q_rad"] == samples[175]["q_rad"] or samples[0]["q_rad"] == [0.0, -1.8, 1.6, 0.0, -1.8, 0.0]
    # The first and last increments are tiny: smooth-step starts and ends at
    # zero velocity instead of issuing a hard target jump.
    first_delta = abs(samples[1]["q_rad"][1] - samples[0]["q_rad"][1])
    middle_delta = abs(samples[88]["q_rad"][1] - samples[87]["q_rad"][1])
    last_delta = abs(samples[-1]["q_rad"][1] - samples[-2]["q_rad"][1])
    assert first_delta < middle_delta
    assert last_delta < middle_delta
