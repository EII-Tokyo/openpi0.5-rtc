from pathlib import Path

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


def test_runtime_script_has_required_order_and_safety_contract():
    source = Path("tools/isaac_sim/open_left_physics_inspector.py").read_text()

    assert source.index('"perspective_camera"') < source.index(
        '"show_physics_inspector"'
    )
    assert "get_stage_loading_status" in source
    assert "enable_inspector_authoring_mode" in source
    assert "MAX_RECOVERIES = 1" in source
    for forbidden in (
        "set_joint_value",
        "set_joint_position",
        "set_drive_target",
        ".play(",
        "save_stage",
    ):
        assert forbidden not in source


def test_runtime_script_configures_verified_dual_panel_handoff():
    source = Path("tools/isaac_sim/open_left_physics_inspector.py").read_text()

    for required in (
        'TABLE_COLLIDER = "/World/environment/worldBody/user_confirmed_table"',
        'TABLE_INSPECTOR_WINDOW_TITLE = "Physics Inspector: ###PhysicsInspector2"',
        "add_inspector_window()",
        "PhysXInspectorModelControlType.JOINT_DRIVE",
        "get_enable_quasi_static_mode_model().set_value(True)",
        "get_fix_articulation_base_model().set_value(True)",
        "get_enable_gravity_model().set_value(False)",
        "CODEX_TABLE_INSPECTOR_READY",
        "CODEX_DUAL_INSPECTOR_ACCEPTED",
    ):
        assert required in source
