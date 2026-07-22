from __future__ import annotations

import pytest

from aloha_isaac_replay.validation.fixed_bottle_grasp import (
    BOTTLE_DROPPED,
    BOTTLE_SLIPPED,
    COLLIDER_PENETRATION,
    GRIP_FORCE_INSUFFICIENT,
    INITIAL_PENETRATION,
    NO_CONTACT,
    ONE_FINGER_CONTACT_ONLY,
    PASS_FIXED_BOTTLE_GRASP,
    classify_trial,
    summarize_trials,
)


def _metrics(**overrides):
    metrics = {
        "reset_stable": True,
        "initial_penetration": False,
        "nan_or_inf": False,
        "joint_limit_or_effort_violation": False,
        "left_contact": True,
        "right_contact": True,
        "max_contact_force_n": 3.0,
        "lift_height_m": 0.095,
        "left_table_during_hold": True,
        "touched_table_during_hold": False,
        "max_slip_m": 0.003,
        "collider_penetration": False,
        "control_timeout": False,
    }
    metrics.update(overrides)
    return metrics


def test_classify_trial_success_requires_all_fixed_bottle_gates():
    result = classify_trial(_metrics())

    assert result.success is True
    assert result.reason == PASS_FIXED_BOTTLE_GRASP


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"initial_penetration": True}, INITIAL_PENETRATION),
        ({"left_contact": False, "right_contact": False}, NO_CONTACT),
        ({"left_contact": True, "right_contact": False}, ONE_FINGER_CONTACT_ONLY),
        ({"right_contact": True, "left_contact": False}, ONE_FINGER_CONTACT_ONLY),
        ({"max_contact_force_n": 0.0}, GRIP_FORCE_INSUFFICIENT),
        ({"lift_height_m": 0.02}, BOTTLE_DROPPED),
        ({"touched_table_during_hold": True}, BOTTLE_DROPPED),
        ({"max_slip_m": 0.02}, BOTTLE_SLIPPED),
        ({"collider_penetration": True}, COLLIDER_PENETRATION),
    ],
)
def test_classify_trial_reports_specific_failure_reason(overrides, expected_reason):
    result = classify_trial(_metrics(**overrides))

    assert result.success is False
    assert result.reason == expected_reason


def test_summarize_trials_passes_only_at_19_of_20_successes():
    rows = [classify_trial(_metrics()).to_dict() for _ in range(19)]
    rows.append(classify_trial(_metrics(max_slip_m=0.03)).to_dict())

    summary = summarize_trials(rows)

    assert summary["success_count"] == 19
    assert summary["failure_count"] == 1
    assert summary["final_conclusion"] == PASS_FIXED_BOTTLE_GRASP
    assert summary["failure_reason_counts"] == {BOTTLE_SLIPPED: 1}


def test_summarize_trials_fails_below_19_successes():
    rows = [classify_trial(_metrics()).to_dict() for _ in range(18)]
    rows.extend(classify_trial(_metrics(lift_height_m=0.01)).to_dict() for _ in range(2))

    summary = summarize_trials(rows)

    assert summary["success_count"] == 18
    assert summary["failure_count"] == 2
    assert summary["final_conclusion"] == "FAIL_CONTACT"
