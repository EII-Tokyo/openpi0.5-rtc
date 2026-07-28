from __future__ import annotations

import pytest

from tools.aloha1_mapping.gripper_force_diagnosis import classify_normal_force
from tools.aloha1_mapping.gripper_force_diagnosis import required_normal_force_each
from tools.aloha1_mapping.gripper_force_diagnosis import select_lowest_sufficient_preload
from tools.aloha1_mapping.gripper_force_diagnosis import summarize_preload_trials


def _trial(delta_m: float, left: list[float], right: list[float]) -> dict:
    return {
        "status": "PASS",
        "delta_m": delta_m,
        "left_stable_normal_force_n": left,
        "right_stable_normal_force_n": right,
        "left_target_error_m": [delta_m] * len(left),
        "right_target_error_m": [delta_m] * len(right),
        "finite": True,
        "fresh_reset": True,
    }


def test_required_normal_force_each_matches_two_sided_coulomb_reference() -> None:
    result = required_normal_force_each(mass_kg=0.020, friction=0.7, gravity_m_s2=9.81)

    assert result == pytest.approx(0.14014285714285715)


def test_force_curve_summary_uses_minimum_stable_force_on_each_side() -> None:
    trials = [
        _trial(0.001, [0.18, 0.17, 0.19], [0.16, 0.15, 0.17]),
        _trial(0.001, [0.20, 0.18, 0.19], [0.17, 0.16, 0.18]),
    ]

    summary = summarize_preload_trials(trials, minimum_repeats=2)

    assert summary["delta_m"] == pytest.approx(0.001)
    assert summary["trial_count"] == 2
    assert summary["left"]["minimum_stable_normal_force_n"] == pytest.approx(0.17)
    assert summary["right"]["minimum_stable_normal_force_n"] == pytest.approx(0.15)
    assert summary["left_right_asymmetry_ratio"] == pytest.approx(0.15 / 0.17)
    assert summary["all_finite"] is True
    assert summary["all_fresh_resets"] is True


def test_normal_force_classification_and_preload_selection() -> None:
    curves = [
        {
            "delta_m": 0.0,
            "complete": True,
            "left": {"minimum_stable_normal_force_n": 0.05},
            "right": {"minimum_stable_normal_force_n": 0.04},
        },
        {
            "delta_m": 0.0005,
            "complete": True,
            "left": {"minimum_stable_normal_force_n": 0.16},
            "right": {"minimum_stable_normal_force_n": 0.15},
        },
        {
            "delta_m": 0.001,
            "complete": True,
            "left": {"minimum_stable_normal_force_n": 0.20},
            "right": {"minimum_stable_normal_force_n": 0.18},
        },
    ]
    result = classify_normal_force(curves, required_each_n=0.140142857)

    assert result["NORMAL_FORCE_STATUS"] == "SUFFICIENT"
    assert result["lowest_sufficient_preload_m"] == pytest.approx(0.0005)
    assert select_lowest_sufficient_preload(curves, 0.140142857) == pytest.approx(0.0005)


def test_normal_force_not_observable_is_not_called_insufficient() -> None:
    result = classify_normal_force([], required_each_n=0.140142857)

    assert result["NORMAL_FORCE_STATUS"] == "NOT_OBSERVABLE"


def test_failed_reset_cannot_make_preload_curve_complete_or_sufficient() -> None:
    passing = _trial(0.001, [0.20, 0.19], [0.20, 0.18])
    failed = {
        **_trial(0.001, [0.30, 0.30], [0.30, 0.30]),
        "status": "FAIL",
        "failure": "bilateral_contact_not_found",
    }

    summary = summarize_preload_trials(
        [passing, failed],
        minimum_repeats=2,
    )
    result = classify_normal_force([summary], required_each_n=0.14)

    assert summary["complete"] is False
    assert summary["successful_trial_count"] == 1
    assert summary["failed_trial_count"] == 1
    assert result["NORMAL_FORCE_STATUS"] == "INCONCLUSIVE"
