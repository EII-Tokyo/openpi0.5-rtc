"""Open the approved ALOHA Stage and prepare follower_left Physics Inspector."""

import asyncio
import traceback

import omni.kit.actions.core
import omni.kit.app
import omni.timeline
import omni.ui
import omni.usd
import omni.physxsupportui.bindings._physxSupportUi as pxsupportui
from pxr import UsdPhysics

from tools.isaac_sim.left_inspector_startup import (
    LoadingStability,
    RecoveryDecision,
    RecoveryGuard,
)


TARGET_STAGE = (
    "/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/"
    "diagnostics/cad_derived_full_body_colliders/1.0/"
    "aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_diagnostic.usda"
)
LEFT_ARTICULATION_ROOT = "/World/follower_left/vx300s_left/root_joint"
INSPECTOR_WINDOW_TITLE = "Physics Inspector: ###PhysicsInspector1"
EXPECTED_JOINT_ROWS = 13
STABLE_LOADING_SAMPLES = 5
LOADING_TIMEOUT_UPDATES = 2400
ACCEPTANCE_UPDATES = 180
MAX_RECOVERIES = 1


async def _wait_for_stable_loading(app, context, phase: str) -> tuple[str, int, int]:
    stability = LoadingStability(required_samples=STABLE_LOADING_SAMPLES)
    last_status = ("", 0, -1)
    for update in range(LOADING_TIMEOUT_UPDATES):
        await app.next_update_async()
        last_status = context.get_stage_loading_status()
        _, _, pending_files = last_status
        if stability.observe(pending_files):
            print(
                "CODEX_STAGE_LOADING_STABLE "
                f"phase={phase} update={update + 1} status={last_status}",
                flush=True,
            )
            return last_status
    raise TimeoutError(
        f"Stage loading did not stabilize during {phase}: status={last_status}"
    )


def _collect_inspector_rows(model) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    pending = list(model.get_item_children(None) or [])
    seen: set[int] = set()
    while pending and len(seen) < 256:
        item = pending.pop(0)
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        name_model = model.get_item_value_model(item, 0)
        path_model = model.get_item_value_model(item, 1)
        name = name_model.get_value_as_string() if name_model else ""
        path = path_model.get_value_as_string() if path_model else ""
        if name or path:
            rows.append((name, path))
        pending.extend(model.get_item_children(item) or [])
    return rows


async def _bind_left_articulation(app, context, inspector_window) -> None:
    context.get_selection().set_selected_prim_paths([LEFT_ARTICULATION_ROOT], False)
    inspector_window.visible = True
    inspector_window._inspector_toolbar._select_current()
    for _ in range(20):
        await app.next_update_async()


async def _prepare_left_inspector() -> None:
    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    context = omni.usd.get_context()
    action_registry = omni.kit.actions.core.get_action_registry()
    inspector_window = None

    try:
        await app.next_update_async()
        timeline.stop()
        opened = context.open_stage(TARGET_STAGE)
        print(
            f"CODEX_STAGE_OPEN_REQUEST opened={opened} target={TARGET_STAGE}",
            flush=True,
        )
        if not opened:
            raise RuntimeError("Isaac Sim rejected the approved Stage open request")

        perspective_result = action_registry.execute_action(
            "omni.kit.viewport.actions", "perspective_camera"
        )
        print(f"CODEX_VIEW_PERSPECTIVE result={perspective_result}", flush=True)
        await _wait_for_stable_loading(app, context, "initial")

        stage_url = context.get_stage_url()
        stage = context.get_stage()
        root_prim = stage.GetPrimAtPath(LEFT_ARTICULATION_ROOT) if stage else None
        root_valid = bool(root_prim and root_prim.IsValid())
        articulation_api = bool(
            root_valid and root_prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        )
        print(f"CODEX_STAGE_URL {stage_url}", flush=True)
        print(
            "CODEX_LEFT_ARTICULATION "
            f"path={LEFT_ARTICULATION_ROOT} valid={root_valid} "
            f"articulation_api={articulation_api}",
            flush=True,
        )
        if stage_url != TARGET_STAGE:
            raise RuntimeError(f"Unexpected current Stage URL: {stage_url}")
        if not articulation_api:
            raise RuntimeError("Approved follower_left articulation root is missing")

        action_registry.execute_action(
            "omni.physx.supportui", "show_physics_inspector"
        )
        for _ in range(30):
            await app.next_update_async()
            inspector_window = omni.ui.Workspace.get_window(INSPECTOR_WINDOW_TITLE)
            if inspector_window is not None:
                break
        if inspector_window is None:
            raise RuntimeError("Physics Inspector window was not created")

        await _bind_left_articulation(app, context, inspector_window)
        guard = RecoveryGuard()
        for update in range(ACCEPTANCE_UPDATES):
            await app.next_update_async()
            state = inspector_window._supportui_private.get_inspector_state()
            decision = guard.observe(state == pxsupportui.PhysXInspectorModelState.DISABLED)
            if decision is RecoveryDecision.RECOVER:
                if guard.recoveries > MAX_RECOVERIES:
                    raise AssertionError("RecoveryGuard exceeded MAX_RECOVERIES")
                print(
                    f"CODEX_INSPECTOR_RECOVERY attempt={guard.recoveries} update={update + 1}",
                    flush=True,
                )
                inspector_window._supportui_private.enable_inspector_authoring_mode()
                await _wait_for_stable_loading(app, context, "recovery")
                await _bind_left_articulation(app, context, inspector_window)
            elif decision is RecoveryDecision.FAIL:
                print(
                    "CODEX_INSPECTOR_RECOVERY_FAILED reason=second_disabled_state",
                    flush=True,
                )
                return

        final_state = inspector_window._supportui_private.get_inspector_state()
        selected_label = (
            inspector_window._inspector_toolbar.label_selection.model.get_value_as_string()
        )
        rows = _collect_inspector_rows(inspector_window._model_inspector)
        joint_rows = [(name, path) for name, path in rows if "/joints/" in path]
        print(
            "CODEX_INSPECTOR_READY "
            f"visible={inspector_window.visible} selected={selected_label}",
            flush=True,
        )
        print(
            f"CODEX_INSPECTOR_ROWS total={len(rows)} joint_rows={len(joint_rows)}",
            flush=True,
        )
        for name, path in joint_rows[:20]:
            print(f"CODEX_INSPECTOR_JOINT name={name} path={path}", flush=True)
        if final_state == pxsupportui.PhysXInspectorModelState.DISABLED:
            raise RuntimeError("Inspector ended the acceptance window DISABLED")
        if selected_label != LEFT_ARTICULATION_ROOT:
            raise RuntimeError(f"Inspector selected unexpected path: {selected_label}")
        if len(joint_rows) < EXPECTED_JOINT_ROWS:
            raise RuntimeError(
                f"Inspector exposed {len(joint_rows)} joint rows; expected at least "
                f"{EXPECTED_JOINT_ROWS}"
            )
        print(
            "CODEX_INSPECTOR_ACCEPTED "
            f"state={final_state.name} recoveries={guard.recoveries}",
            flush=True,
        )
    except Exception as exc:
        print(f"CODEX_STARTUP_FAILED type={type(exc).__name__} message={exc}", flush=True)
        traceback.print_exc()
    finally:
        timeline.stop()
        print(f"CODEX_TIMELINE_STOPPED {not timeline.is_playing()}", flush=True)


asyncio.ensure_future(_prepare_left_inspector())
