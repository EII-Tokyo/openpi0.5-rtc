from __future__ import annotations

from tools import audit_aloha1_bottle_velocity_consistency as velocity_audit
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


def test_readback_boundary_is_localized_without_claiming_internal_root_cause() -> None:
    status = velocity_audit.classify_readback_responsibility(
        exact_tensor_path=True,
        tensor_direct_transform_max_delta_m=0.0,
        tensor_usd_linear_velocity_max_delta_m_s=0.0,
        transform_velocity_alignment=False,
    )

    assert status == "VERIFIED_LOCAL_PHYSX_VELOCITY_TRANSFORM_DISAGREEMENT"
