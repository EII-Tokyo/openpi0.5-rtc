from __future__ import annotations

from tools.audit_aloha1_bottle_velocity_consistency import classify_velocity_semantics


def test_velocity_semantics_verified_when_baseline_is_aligned() -> None:
    assert classify_velocity_semantics(
        baseline_aligned=True,
        initialize_aligned=False,
        initialize_runtime_pass=False,
        recreate_aligned=False,
        recreate_runtime_pass=False,
    ) == "VERIFIED"


def test_velocity_semantics_classifies_effective_lifecycle_fix() -> None:
    assert classify_velocity_semantics(
        baseline_aligned=False,
        initialize_aligned=True,
        initialize_runtime_pass=True,
        recreate_aligned=False,
        recreate_runtime_pass=False,
    ) == "KINEMATIC_TRANSITION_ISSUE"
    assert classify_velocity_semantics(
        baseline_aligned=False,
        initialize_aligned=False,
        initialize_runtime_pass=False,
        recreate_aligned=True,
        recreate_runtime_pass=True,
    ) == "STALE_TENSOR_VIEW"


def test_velocity_semantics_stays_inconclusive_when_neither_fix_works() -> None:
    assert classify_velocity_semantics(
        baseline_aligned=False,
        initialize_aligned=False,
        initialize_runtime_pass=False,
        recreate_aligned=False,
        recreate_runtime_pass=True,
    ) == "INCONCLUSIVE"
