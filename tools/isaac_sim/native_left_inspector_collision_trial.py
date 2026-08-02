"""Run one disposable collision trial through native Physics Inspector."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import traceback
from typing import Any

import omni.kit.actions.core
import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.ui
import omni.usd
import omni.physxsupportui.bindings._physxSupportUi as pxsupportui
from omni.physx import get_physx_simulation_interface
from pxr import PhysxSchema, PhysicsSchemaTools, Sdf, Usd, UsdGeom, UsdPhysics

from tools.isaac_sim.left_table_collision_gate import (
    ALLOWED_TIP_ROOTS,
    MAX_CONTACT_SEPARATION_M,
    TrialMetrics,
    evaluate_trial,
)
from tools.isaac_sim.left_inspector_startup import target_change_is_isolated
from tools.isaac_sim.verify_left_table_collision import (
    LEFT_BODY_PATHS,
    _capture_verified_contact,
    _live_finger_geometry,
    _preflight,
)


TARGET_STAGE = (
    "/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/"
    "diagnostics/cad_derived_full_body_colliders/1.0/"
    "aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_diagnostic.usda"
)
EXPECTED_STAGE_SHA256 = (
    "165093c3e7bf359b2ef5dbb595feb4ed976b194844830e70f387d6b882c1d6f2"
)
LEFT_ARTICULATION_ROOT = "/World/follower_left/vx300s_left/root_joint"
TABLE_COLLIDER = "/World/environment/worldBody/user_confirmed_table"
INSPECTED_PATHS = (LEFT_ARTICULATION_ROOT, TABLE_COLLIDER)
LEFT_ROBOT_ROOT = "/World/follower_left/vx300s_left"
SHOULDER_JOINT = "/World/follower_left/vx300s_left/joints/shoulder"
INSPECTOR_WINDOW_TITLE = "Physics Inspector: ###PhysicsInspector1"
EXPECTED_JOINT_ROWS = 13
APPROACH_TARGET_DEG = 20.0
HOLD_TARGET_DEG = 30.0
LOADING_TIMEOUT_UPDATES = 2400
INSPECTOR_TIMEOUT_UPDATES = 1200
SIMULATION_TIMEOUT_UPDATES = 2400


def _sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_matches(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def _target_pair(pair: tuple[str, str]) -> bool:
    first, second = pair
    return (
        _path_matches(first, TABLE_COLLIDER)
        and any(_path_matches(second, root) for root in ALLOWED_TIP_ROOTS)
    ) or (
        _path_matches(second, TABLE_COLLIDER)
        and any(_path_matches(first, root) for root in ALLOWED_TIP_ROOTS)
    )


def _disallowed_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    result = []
    for first, second in pairs:
        first_tip = any(_path_matches(first, root) for root in ALLOWED_TIP_ROOTS)
        second_tip = any(_path_matches(second, root) for root in ALLOWED_TIP_ROOTS)
        other = second if first_tip else first if second_tip else ""
        if (
            other
            and _path_matches(other, "/World/environment")
            and not _path_matches(other, TABLE_COLLIDER)
        ):
            result.append((first, second))
    return sorted(set(result))


def _contact_report(interface: Any) -> list[dict[str, Any]]:
    headers, data = interface.get_contact_report()
    result = []
    for header in headers:
        first = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
        second = str(PhysicsSchemaTools.intToSdfPath(header.collider1))
        start = int(header.contact_data_offset)
        stop = start + int(header.num_contact_data)
        separations = [float(datum.separation) for datum in list(data)[start:stop]]
        result.append(
            {
                "pair": (first, second),
                "minimum_separation_m": min(separations, default=math.inf),
                "contact_data_count": len(separations),
            }
        )
    return result


async def _wait_for_loading(app: Any, context: Any) -> None:
    stable = 0
    for _ in range(LOADING_TIMEOUT_UPDATES):
        await app.next_update_async()
        pending = int(context.get_stage_loading_status()[2])
        stable = stable + 1 if pending == 0 else 0
        if stable >= 5:
            return
    raise TimeoutError(f"Stage loading did not stabilize: {context.get_stage_loading_status()}")


def _collect_rows(model: Any) -> list[tuple[str, str]]:
    rows = []
    pending = list(model.get_item_children(None) or [])
    seen: set[int] = set()
    while pending and len(seen) < 256:
        item = pending.pop(0)
        if id(item) in seen:
            continue
        seen.add(id(item))
        name_model = model.get_item_value_model(item, 0)
        path_model = model.get_item_value_model(item, 1)
        name = name_model.get_value_as_string() if name_model else ""
        path = path_model.get_value_as_string() if path_model else ""
        if name or path:
            rows.append((name, path))
        pending.extend(model.get_item_children(item) or [])
    return rows


def _expanded_inspected_paths(stage: Any) -> list[str]:
    paths = list(INSPECTED_PATHS)
    robot = stage.GetPrimAtPath(LEFT_ROBOT_ROOT)
    for prim in Usd.PrimRange(robot):
        if (
            prim.IsA(UsdPhysics.Joint)
            or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            or prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or prim.HasAPI(UsdPhysics.CollisionAPI)
        ):
            path = str(prim.GetPath())
            if path not in paths:
                paths.append(path)
    return paths


async def _bind_paths(app: Any, context: Any, window: Any, stage: Any) -> list[str]:
    paths = _expanded_inspected_paths(stage)
    context.get_selection().set_selected_prim_paths(paths, False)
    window.visible = True
    window._inspector_toolbar._select_current()
    for _ in range(20):
        await app.next_update_async()
    return paths


async def _clear_transient_selection(
    app: Any, context: Any, window: Any
) -> dict[str, list[str]]:
    context.get_selection().set_selected_prim_paths([], False)
    for _ in range(20):
        await app.next_update_async()
    stage_selection = list(context.get_selection().get_selected_prim_paths())
    inspector_selection = list(window._handler_selection.get_selection() or [])
    if stage_selection or inspector_selection:
        raise RuntimeError(
            "native Inspector interaction selection did not clear: "
            f"stage={stage_selection} inspector={inspector_selection}"
        )
    return {
        "stage_selection_after_clear": stage_selection,
        "inspector_selection_after_clear": inspector_selection,
    }


def _drive_targets(
    stage: Any, joint_rows: list[tuple[str, str]]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, path in joint_rows:
        prim = stage.GetPrimAtPath(path)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            drive = UsdPhysics.DriveAPI.Get(prim, "linear")
        else:
            continue
        attr = drive.GetTargetPositionAttr()
        value = attr.Get() if attr else None
        if value is not None:
            result[name] = float(value)
    return result


class _InspectorValueModel:
    def __init__(self, value: float):
        self.as_float = value


def _configure_left(window: Any) -> None:
    model = window._model_inspector
    model.get_control_type_model().set_value(
        str(int(pxsupportui.PhysXInspectorModelControlType.JOINT_DRIVE))
    )
    model.get_enable_quasi_static_mode_model().set_value(True)
    model.get_fix_articulation_base_model().set_value(True)
    model.get_enable_gravity_model().set_value(False)


def _begin_contact_reporting(stage: Any) -> dict[str, Any]:
    layer = Sdf.Layer.CreateAnonymous("native_left_inspector_contact_report")
    stage.GetSessionLayer().subLayerPaths.append(layer.identifier)
    previous_target = stage.GetEditTarget()
    stage.SetEditTarget(Usd.EditTarget(layer))
    scene_prim = stage.GetPrimAtPath("/World/PhysicsScene")
    quasistatic = PhysxSchema.PhysxSceneQuasistaticAPI.Apply(scene_prim)
    quasistatic.CreateEnableQuasistaticAttr().Set(True)
    for body_path in LEFT_BODY_PATHS:
        body = stage.GetPrimAtPath(body_path)
        report_api = PhysxSchema.PhysxContactReportAPI.Apply(body)
        report_api.CreateThresholdAttr().Set(0)
    return {
        "layer": layer,
        "previous_target": previous_target,
        "interface": get_physx_simulation_interface(),
    }


def _new_accumulator() -> dict[str, Any]:
    return {
        "contact_pairs": set(),
        "disallowed_pairs": set(),
        "minimum_target_separation_m": math.inf,
        "minimum_table_local_finger_z_m": math.inf,
        "maximum_visual_collision_error_m": 0.0,
        "physical_contact_steps": 0,
        "native_steps": 0,
        "samples": [],
    }


def _sample_step(stage: Any, interface: Any, accumulator: dict[str, Any]) -> None:
    contacts = _contact_report(interface)
    pairs = [row["pair"] for row in contacts]
    accumulator["contact_pairs"].update(pairs)
    accumulator["disallowed_pairs"].update(_disallowed_pairs(pairs))
    target_separations = [
        row["minimum_separation_m"]
        for row in contacts
        if _target_pair(row["pair"])
    ]
    separation = min(target_separations, default=math.inf)
    geometry = _live_finger_geometry(stage)
    accumulator["minimum_target_separation_m"] = min(
        accumulator["minimum_target_separation_m"], separation
    )
    accumulator["minimum_table_local_finger_z_m"] = min(
        accumulator["minimum_table_local_finger_z_m"],
        geometry["minimum_table_local_finger_z_m"],
    )
    accumulator["maximum_visual_collision_error_m"] = max(
        accumulator["maximum_visual_collision_error_m"],
        geometry["maximum_visual_collision_error_m"],
    )
    accumulator["native_steps"] += 1
    physical = math.isfinite(separation) and separation <= MAX_CONTACT_SEPARATION_M
    accumulator["physical_contact_steps"] += int(physical)
    if accumulator["native_steps"] in (1, 30, 60, 90, 120, 150, 180):
        accumulator["samples"].append(
            {
                "native_step": accumulator["native_steps"],
                "minimum_target_separation_m": separation,
                "minimum_table_local_finger_z_m": geometry[
                    "minimum_table_local_finger_z_m"
                ],
                "maximum_visual_collision_error_m": geometry[
                    "maximum_visual_collision_error_m"
                ],
                "physical_contact": physical,
            }
        )


async def _wait_native_run(
    app: Any,
    simulation: Any,
    stage: Any,
    interface: Any,
    accumulator: dict[str, Any],
) -> None:
    started = False
    previous_remaining = None
    for _ in range(SIMULATION_TIMEOUT_UPDATES):
        await app.next_update_async()
        task = simulation._sub_async_sim_run
        if task is not None:
            started = True
            remaining = float(simulation._simulation_time)
            if previous_remaining is not None and remaining < previous_remaining - 1e-12:
                _sample_step(stage, interface, accumulator)
            previous_remaining = remaining
        elif started:
            return
    raise TimeoutError(
        "native Inspector authoring simulation did not start or finish within bound"
    )


async def _run_trial() -> None:
    app = omni.kit.app.get_app()
    context = omni.usd.get_context()
    timeline = omni.timeline.get_timeline_interface()
    output_dir = Path(os.environ["CODEX_NATIVE_TRIAL_OUTPUT_DIR"]).resolve()
    trial_index = int(os.environ.get("CODEX_NATIVE_TRIAL_INDEX", "1"))
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "trial.json"
    screenshot_path = output_dir / "native_verified_contact.png"
    stage_file = Path(TARGET_STAGE)
    report: dict[str, Any] = {
        "trial_index": trial_index,
        "status": "FAIL",
        "stage_path": TARGET_STAGE,
        "stage_sha256": EXPECTED_STAGE_SHA256,
        "stage_saved": False,
        "real_robot_touched": False,
        "failure_reasons": [],
    }
    edit_state = None
    try:
        timeline.stop()
        if _sha256(stage_file) != EXPECTED_STAGE_SHA256:
            raise RuntimeError("frozen Stage hash changed before native trial")
        if not context.open_stage(TARGET_STAGE):
            raise RuntimeError("Full Kit rejected the approved Stage")
        await _wait_for_loading(app, context)
        stage = context.get_stage()
        if context.get_stage_url() != TARGET_STAGE:
            raise RuntimeError(f"unexpected Stage URL: {context.get_stage_url()}")
        if str(stage.GetDefaultPrim().GetPath()) != "/World":
            raise RuntimeError("unexpected default prim")
        if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
            raise RuntimeError("Stage is not Z-up")
        if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
            raise RuntimeError("Stage is not meter-scaled")
        preflight = _preflight(stage)
        edit_state = _begin_contact_reporting(stage)

        registry = omni.kit.actions.core.get_action_registry()
        registry.execute_action("omni.physx.supportui", "show_physics_inspector")
        left_window = None
        for _ in range(INSPECTOR_TIMEOUT_UPDATES):
            await app.next_update_async()
            left_window = omni.ui.Workspace.get_window(INSPECTOR_WINDOW_TITLE)
            if left_window is not None:
                break
        if left_window is None:
            raise RuntimeError("native Physics Inspector window was not created")
        selected_paths = await _bind_paths(app, context, left_window, stage)
        _configure_left(left_window)

        selected_label = (
            left_window._inspector_toolbar.label_selection.model.get_value_as_string()
        )
        if selected_label != f"{LEFT_ARTICULATION_ROOT} (+{len(selected_paths) - 1})":
            raise RuntimeError(
                f"native Inspector did not keep both selected paths: {selected_label}"
            )

        rows = _collect_rows(left_window._model_inspector)
        joint_rows = [(name, path) for name, path in rows if "/joints/" in path]
        if len(joint_rows) < EXPECTED_JOINT_ROWS:
            raise RuntimeError(
                f"native Inspector exposed {len(joint_rows)} joint rows"
            )
        state = left_window._supportui_private.get_inspector_state()
        if state != pxsupportui.PhysXInspectorModelState.AUTHORING:
            raise RuntimeError(f"native Inspector not in AUTHORING: {state}")

        selection_evidence = await _clear_transient_selection(
            app, context, left_window
        )
        rows_after_clear = _collect_rows(left_window._model_inspector)
        joint_rows_after_clear = [
            (name, path) for name, path in rows_after_clear if "/joints/" in path
        ]
        if len(joint_rows_after_clear) < EXPECTED_JOINT_ROWS:
            raise RuntimeError(
                "native Inspector lost joint rows after selection clear: "
                f"{len(joint_rows_after_clear)}"
            )

        shoulder = stage.GetPrimAtPath(SHOULDER_JOINT)
        drive = UsdPhysics.DriveAPI.Get(shoulder, "angular")
        target_attr = drive.GetTargetPositionAttr()
        previous_target = float(target_attr.Get())
        simulation = left_window._inspector._inspector_simulation
        approach = _new_accumulator()
        drive_targets_before = _drive_targets(stage, joint_rows_after_clear)
        probe_target = previous_target + 1.0
        left_window._inspector_panel._delegate_tree._on_value_changed(
            _InspectorValueModel(probe_target), SHOULDER_JOINT
        )
        drive_targets_after_single_joint_edit = _drive_targets(
            stage, joint_rows_after_clear
        )
        single_joint_target_isolated = target_change_is_isolated(
            drive_targets_before,
            drive_targets_after_single_joint_edit,
            "shoulder",
            probe_target,
        )
        if not single_joint_target_isolated:
            raise RuntimeError(
                "native Inspector propagated the shoulder target to other joints"
            )
        left_window._inspector_panel._delegate_tree._on_value_changed(
            _InspectorValueModel(previous_target), SHOULDER_JOINT
        )
        drive_targets_after_probe_restore = _drive_targets(
            stage, joint_rows_after_clear
        )
        if drive_targets_after_probe_restore != drive_targets_before:
            raise RuntimeError("native Inspector shoulder probe did not restore")

        omni.kit.commands.execute(
            "ChangeProperty",
            prop_path=target_attr.GetPath(),
            value=APPROACH_TARGET_DEG,
            prev=previous_target,
        )
        await _wait_native_run(
            app, simulation, stage, edit_state["interface"], approach
        )

        hold = _new_accumulator()
        omni.kit.commands.execute(
            "ChangeProperty",
            prop_path=target_attr.GetPath(),
            value=HOLD_TARGET_DEG,
            prev=APPROACH_TARGET_DEG,
        )
        await _wait_native_run(app, simulation, stage, edit_state["interface"], hold)

        joint_state = PhysxSchema.JointStateAPI.Get(shoulder, "angular")
        realized_deg = float(joint_state.GetPositionAttr().Get())
        final_target_error_rad = math.radians(HOLD_TARGET_DEG - realized_deg)
        all_pairs = sorted(approach["contact_pairs"] | hold["contact_pairs"])
        disallowed = sorted(
            approach["disallowed_pairs"] | hold["disallowed_pairs"]
        )
        metrics = TrialMetrics(
            contact_pairs=all_pairs,
            minimum_target_separation_m=min(
                approach["minimum_target_separation_m"],
                hold["minimum_target_separation_m"],
            ),
            minimum_table_local_finger_z_m=min(
                approach["minimum_table_local_finger_z_m"],
                hold["minimum_table_local_finger_z_m"],
            ),
            maximum_visual_collision_error_m=max(
                approach["maximum_visual_collision_error_m"],
                hold["maximum_visual_collision_error_m"],
            ),
            final_target_error_rad=final_target_error_rad,
            persistent_contact_steps=int(hold["physical_contact_steps"]),
            finite=all(
                math.isfinite(value)
                for value in (
                    realized_deg,
                    approach["minimum_table_local_finger_z_m"],
                    hold["minimum_table_local_finger_z_m"],
                )
            ),
            within_joint_limits=-106.0 <= realized_deg <= 72.0,
            ccd_effective=(
                preflight["scene"]["enable_ccd"]
                and not preflight["scene"]["enable_gpu_dynamics"]
                and preflight["left_ccd_body_count"] == len(LEFT_BODY_PATHS)
            ),
            disallowed_tip_contacts=disallowed,
            physx_errors=[],
        )
        decision = evaluate_trial(metrics)
        report.update(
            {
                **decision,
                "initial_shoulder_target_deg": previous_target,
                "approach_shoulder_target_deg": APPROACH_TARGET_DEG,
                "commanded_shoulder_target_deg": HOLD_TARGET_DEG,
                "realized_shoulder_deg": realized_deg,
                "approach_native_steps": int(approach["native_steps"]),
                "hold_native_steps": int(hold["native_steps"]),
                "hold_physical_contact_steps": int(hold["physical_contact_steps"]),
                "approach_samples": approach["samples"],
                "hold_samples": hold["samples"],
                "joint_row_count": len(joint_rows),
                "inspector_selected_paths": selected_paths,
                **selection_evidence,
                "drive_targets_before": drive_targets_before,
                "drive_targets_after_single_joint_edit": (
                    drive_targets_after_single_joint_edit
                ),
                "drive_targets_after_probe_restore": (
                    drive_targets_after_probe_restore
                ),
                "single_joint_target_isolated": single_joint_target_isolated,
                "preflight": preflight,
            }
        )
        if report["status"] == "PASS":
            _capture_verified_contact(app, screenshot_path)
            report["screenshot"] = str(screenshot_path)
            report["screenshot_nonempty"] = (
                screenshot_path.exists() and screenshot_path.stat().st_size > 0
            )
            if not report["screenshot_nonempty"]:
                report["status"] = "FAIL"
                report["failure_reasons"].append("missing_native_contact_screenshot")
    except Exception as exc:
        report["failure_reasons"].append(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        timeline.stop()
        if edit_state is not None:
            stage = context.get_stage()
            stage.SetEditTarget(edit_state["previous_target"])
        report["stage_sha256_after"] = _sha256(stage_file)
        if report["stage_sha256_after"] != EXPECTED_STAGE_SHA256:
            report["status"] = "FAIL"
            report["failure_reasons"].append("stage_hash_changed")
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=list) + "\n",
            encoding="utf-8",
        )
        print(
            f"CODEX_NATIVE_INSPECTOR_TRIAL_{report['status']} "
            f"trial={trial_index} report={report_path}",
            flush=True,
        )
        await app.next_update_async()
        app.post_quit()


asyncio.ensure_future(_run_trial())
