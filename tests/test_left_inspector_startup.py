from tools.isaac_sim.left_inspector_startup import (
    LoadingStability,
    RecoveryDecision,
    RecoveryGuard,
)


def test_loading_requires_consecutive_zero_pending_samples():
    stability = LoadingStability(required_samples=3)

    assert not stability.observe(2)
    assert not stability.observe(0)
    assert not stability.observe(1)
    assert not stability.observe(0)
    assert not stability.observe(0)
    assert stability.observe(0)


def test_recovery_guard_allows_only_one_disabled_recovery():
    guard = RecoveryGuard()

    assert guard.observe(disabled=False) is RecoveryDecision.KEEP_MONITORING
    assert guard.observe(disabled=True) is RecoveryDecision.RECOVER
    assert guard.observe(disabled=False) is RecoveryDecision.KEEP_MONITORING
    assert guard.observe(disabled=True) is RecoveryDecision.FAIL
