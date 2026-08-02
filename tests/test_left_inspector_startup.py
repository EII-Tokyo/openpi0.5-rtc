from pathlib import Path

from tools.isaac_sim.left_inspector_startup import (
    LoadingStability,
    RecoveryDecision,
    RecoveryGuard,
    target_change_is_isolated,
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


def test_target_change_is_isolated_to_requested_joint():
    before = {"waist": 0.0, "shoulder": -55.004, "elbow": 66.463}
    after = {"waist": 0.0, "shoulder": 20.0, "elbow": 66.463}

    assert target_change_is_isolated(before, after, "shoulder", 20.0)


def test_target_change_rejects_launcher_multi_selection_propagation():
    before = {"waist": 0.0, "shoulder": -55.004, "elbow": 66.463}
    after = {"waist": 10.8, "shoulder": 10.8, "elbow": 10.8}

    assert not target_change_is_isolated(before, after, "shoulder", 10.8)


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


def test_runtime_script_configures_verified_single_panel_handoff():
    source = Path("tools/isaac_sim/open_left_physics_inspector.py").read_text()

    for required in (
        'TABLE_COLLIDER = "/World/environment/worldBody/user_confirmed_table"',
        "INSPECTED_PATHS = (LEFT_ARTICULATION_ROOT, TABLE_COLLIDER)",
        "_expanded_inspected_paths",
        "prim.IsA(UsdPhysics.Joint)",
        "prim.HasAPI(UsdPhysics.RigidBodyAPI)",
        "prim.HasAPI(UsdPhysics.CollisionAPI)",
        "PhysXInspectorModelControlType.JOINT_DRIVE",
        "get_enable_quasi_static_mode_model().set_value(True)",
        "get_fix_articulation_base_model().set_value(True)",
        "get_enable_gravity_model().set_value(False)",
        "CODEX_INSPECTOR_SELECTION_READY",
        "CODEX_SINGLE_INSPECTOR_ACCEPTED",
    ):
        assert required in source
    assert "add_inspector_window" not in source


def test_runtime_clears_transient_selection_only_after_association():
    source = Path("tools/isaac_sim/open_left_physics_inspector.py").read_text()

    assert "async def _clear_transient_selection" in source
    assert "set_selected_prim_paths([], False)" in source
    assert "_handler_selection.get_selection()" in source
    assert "CODEX_INSPECTOR_INTERACTION_SELECTION_CLEARED" in source
    assert source.index("_inspector_toolbar._select_current()") < source.index(
        "await _clear_transient_selection"
    )
