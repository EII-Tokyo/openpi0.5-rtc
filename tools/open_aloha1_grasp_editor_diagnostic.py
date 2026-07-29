#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Open an isolated ALOHA Grasp Editor session without authoring the source Stage."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
STAGE_PATH = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0"
    / "aloha1_signal_correspondence_workcell.usda"
)
BOTTLE_USD = ROOT / "assets/bottle_500ml/isaac/bottle_500ml_sim.usd"
EXPECTED_STAGE_SHA256 = (
    "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
)
EXPECTED_BOTTLE_SHA256 = (
    "16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e"
)
EXTENSION_ID = "isaacsim.robot_setup.grasp_editor"
EXTENSION_VERSION = "2.0.20"
WINDOW_TITLE = "Grasp Editor"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _move_isaac_to_workspace_two() -> None:
    query = subprocess.run(
        ["xdotool", "search", "--name", "Isaac Sim"],
        check=False,
        capture_output=True,
        text=True,
    )
    window_ids = [
        value.strip() for value in query.stdout.splitlines() if value.strip()
    ]
    if window_ids:
        subprocess.run(
            ["xdotool", "set_desktop_for_window", window_ids[-1], "1"],
            check=False,
        )


def _assert_frozen_diagnostic_stage(
    get_current_stage,
    frozen_stage,
    diagnostic_layer,
    *,
    allow_missing_stage: bool = False,
) -> None:
    current_stage = get_current_stage()
    if allow_missing_stage and current_stage is None:
        return
    if current_stage is not frozen_stage:
        raise RuntimeError(
            "omni.usd current USD Stage escaped the frozen diagnostic Stage"
        )
    actual_layer = frozen_stage.GetEditTarget().GetLayer()
    if actual_layer is not diagnostic_layer:
        actual_identifier = getattr(
            actual_layer,
            "identifier",
            repr(actual_layer),
        )
        raise RuntimeError(
            "Grasp Editor diagnostic edit target escaped anonymous layer: "
            f"{actual_identifier}"
        )


def _guarded_app_update(
    app,
    get_current_stage,
    frozen_stage,
    diagnostic_layer,
) -> None:
    _assert_frozen_diagnostic_stage(
        get_current_stage,
        frozen_stage,
        diagnostic_layer,
    )
    app.update()
    _assert_frozen_diagnostic_stage(
        get_current_stage,
        frozen_stage,
        diagnostic_layer,
        allow_missing_stage=not app.is_running(),
    )


def _restore_previous_edit_target(stage, previous_edit_target) -> None:
    previous_layer = previous_edit_target.GetLayer()
    stage.SetEditTarget(previous_edit_target)
    restored_layer = stage.GetEditTarget().GetLayer()
    if restored_layer is not previous_layer:
        raise RuntimeError(
            "failed to restore exact previous edit target layer"
        )


def _remove_exact_session_sublayer(
    session_layer,
    diagnostic_layer_identifier: str,
) -> None:
    occurrences = list(session_layer.subLayerPaths).count(
        diagnostic_layer_identifier
    )
    if occurrences != 1:
        raise RuntimeError(
            "expected exactly one anonymous diagnostic session sublayer, "
            f"found {occurrences}"
        )
    session_layer.subLayerPaths.remove(diagnostic_layer_identifier)
    if diagnostic_layer_identifier in session_layer.subLayerPaths:
        raise RuntimeError("anonymous diagnostic session sublayer remained")


def _assert_file_hash(
    path: Path,
    expected_sha256: str,
    label: str,
    sha256,
) -> None:
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{label} changed during GUI diagnostic: {actual_sha256}"
        )


def _assert_root_dirty_state(root_layer, root_dirty_before) -> None:
    if root_layer.dirty != root_dirty_before:
        raise RuntimeError("source Stage root dirty state changed")


def _assert_root_serialized_specs(root_layer, root_specs_before) -> None:
    if root_layer.ExportToString() != root_specs_before:
        raise RuntimeError("source Stage root specs changed")


def _run_cleanup_steps(
    steps,
    *,
    primary_error: BaseException | None = None,
    primary_traceback=None,
) -> None:
    cleanup_errors: list[Exception] = []
    for label, step in steps:
        try:
            step()
        except Exception as error:
            error.add_note(f"Grasp Editor cleanup step: {label}")
            cleanup_errors.append(error)

    cleanup_group = None
    if cleanup_errors:
        cleanup_group = ExceptionGroup(
            "Grasp Editor diagnostic cleanup failures",
            cleanup_errors,
        )
    if primary_error is not None:
        if cleanup_group is not None:
            if primary_error.__cause__ is None:
                raise primary_error.with_traceback(
                    primary_traceback
                ) from cleanup_group
            primary_error.grasp_editor_cleanup_errors = cleanup_group
            primary_error.add_note(
                "Additional Grasp Editor cleanup failures are available in "
                "exception.grasp_editor_cleanup_errors"
            )
        raise primary_error.with_traceback(primary_traceback)
    if cleanup_group is not None:
        raise cleanup_group


def _cleanup_diagnostic_session(
    *,
    stage,
    previous_edit_target,
    session_layer,
    diagnostic_layer_identifier,
    root_layer,
    root_dirty_before,
    root_specs_before,
    app,
    stage_path: Path,
    source_stage_sha256_before: str,
    bottle_path: Path,
    bottle_sha256_before: str,
    sha256=_sha256,
    primary_error: BaseException | None = None,
    primary_traceback=None,
) -> None:
    steps = []
    if stage is not None and previous_edit_target is not None:
        steps.append(
            (
                "restore previous edit target",
                lambda: _restore_previous_edit_target(
                    stage,
                    previous_edit_target,
                ),
            )
        )
    if (
        session_layer is not None
        and diagnostic_layer_identifier is not None
    ):
        steps.append(
            (
                "remove anonymous session sublayer",
                lambda: _remove_exact_session_sublayer(
                    session_layer,
                    diagnostic_layer_identifier,
                ),
            )
        )
    if root_layer is not None and root_dirty_before is not None:
        steps.append(
            (
                "verify source root dirty state",
                lambda: _assert_root_dirty_state(
                    root_layer,
                    root_dirty_before,
                ),
            )
        )
    if root_layer is not None and root_specs_before is not None:
        steps.append(
            (
                "verify source root serialized specs",
                lambda: _assert_root_serialized_specs(
                    root_layer,
                    root_specs_before,
                ),
            )
        )
    steps.extend(
        [
            ("close SimulationApp", app.close),
            (
                "verify source Stage hash",
                lambda: _assert_file_hash(
                    stage_path,
                    source_stage_sha256_before,
                    "source Stage",
                    sha256,
                ),
            ),
            (
                "verify Bottle500 hash",
                lambda: _assert_file_hash(
                    bottle_path,
                    bottle_sha256_before,
                    "Bottle500 diagnostic USD",
                    sha256,
                ),
            ),
        ]
    )
    _run_cleanup_steps(
        steps,
        primary_error=primary_error,
        primary_traceback=primary_traceback,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=STAGE_PATH)
    parser.add_argument("--bottle-usd", type=Path, default=BOTTLE_USD)
    parser.add_argument(
        "--no-move-to-startup-workspace",
        action="store_true",
        help="Keep the Isaac window on its current workspace.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage_path = args.stage.resolve()
    bottle_path = args.bottle_usd.resolve()
    source_stage_sha256_before = _sha256(stage_path)
    if source_stage_sha256_before != EXPECTED_STAGE_SHA256:
        raise RuntimeError("approved Stage hash no longer matches")
    bottle_sha256_before = _sha256(bottle_path)
    if bottle_sha256_before != EXPECTED_BOTTLE_SHA256:
        raise RuntimeError("Bottle500 diagnostic USD hash no longer matches")

    import isaacsim

    app = isaacsim.SimulationApp({"headless": False})
    stage = None
    root_layer = None
    session_layer = None
    diagnostic_layer = None
    previous_edit_target = None
    root_dirty_before = None
    root_specs_before = None
    diagnostic_layer_identifier = None
    primary_error = None
    primary_traceback = None
    result = 0
    try:
        from isaacsim.core.utils.stage import open_stage
        import omni.kit.actions.core
        import omni.kit.app
        import omni.ui
        import omni.usd
        from pxr import Gf
        from pxr import Sdf
        from pxr import UsdGeom

        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate(
            EXTENSION_ID,
            True,  # noqa: FBT003 - local Kit binding is positional-only.
        )
        app.update()
        enabled_id = manager.get_enabled_extension_id(EXTENSION_ID)
        extension = manager.get_extension_dict(enabled_id)
        version = extension.get("package", {}).get("version")
        if str(version) != EXTENSION_VERSION:
            raise RuntimeError(
                f"local Grasp Editor version mismatch: {version}"
            )
        if not open_stage(str(stage_path)):
            raise RuntimeError(f"failed to open Stage: {stage_path}")
        app.update()

        def get_current_stage():
            return omni.usd.get_context().get_stage()

        stage = get_current_stage()
        root_layer = stage.GetRootLayer()
        session_layer = stage.GetSessionLayer()
        previous_edit_target = stage.GetEditTarget()
        previous_edit_target_identifier = (
            previous_edit_target.GetLayer().identifier
        )
        root_identifier = root_layer.identifier
        session_identifier = session_layer.identifier
        root_dirty_before = root_layer.dirty
        root_specs_before = root_layer.ExportToString()

        diagnostic_layer = Sdf.Layer.CreateAnonymous(
            "ALOHA1GraspEditorDiagnostic"
        )
        diagnostic_layer_identifier = diagnostic_layer.identifier
        session_layer.subLayerPaths.append(diagnostic_layer_identifier)
        stage.SetEditTarget(diagnostic_layer)
        _assert_frozen_diagnostic_stage(
            get_current_stage,
            stage,
            diagnostic_layer,
        )

        task_frame = UsdGeom.Xform.Define(
            stage,
            "/World/ALOHA1GraspEditorSession/W_T",
        )
        task_frame.AddTranslateOp().Set(
            Gf.Vec3d(0.0, 0.0, -0.0909000015258789)
        )
        bottle = stage.DefinePrim(
            "/World/ALOHA1GraspEditorSession/Bottle500",
            "Xform",
        )
        bottle.GetReferences().AddReference(
            Sdf.AssetPath(str(bottle_path)),
            Sdf.Path("/Bottle500"),
        )
        bottle.SetCustomDataByKey(
            "aloha1:classification",
            "DIAGNOSTIC_SESSION_ONLY_NOT_FINAL",
        )
        _guarded_app_update(
            app,
            get_current_stage,
            stage,
            diagnostic_layer,
        )

        action_id = f"CreateUIExtension:{WINDOW_TITLE}"
        action = omni.kit.actions.core.get_action_registry().get_action(
            enabled_id,
            action_id,
        )
        if action is None:
            action = omni.kit.actions.core.get_action_registry().get_action(
                EXTENSION_ID,
                action_id,
            )
        if action is None:
            raise RuntimeError("Grasp Editor action was not registered")
        _assert_frozen_diagnostic_stage(
            get_current_stage,
            stage,
            diagnostic_layer,
        )
        action.execute()
        _assert_frozen_diagnostic_stage(
            get_current_stage,
            stage,
            diagnostic_layer,
        )
        for _ in range(10):
            _guarded_app_update(
                app,
                get_current_stage,
                stage,
                diagnostic_layer,
            )
        window = omni.ui.Workspace.get_window(WINDOW_TITLE)
        if window is None or not window.visible:
            raise RuntimeError("Grasp Editor window did not become visible")
        if not args.no_move_to_startup_workspace:
            time.sleep(1.0)
            _move_isaac_to_workspace_two()

        print(f"Stage: {stage_path}")
        print(f"Stage SHA-256: {EXPECTED_STAGE_SHA256}")
        print(f"Extension: {enabled_id}")
        print(f"Extension version: {version}")
        print(f"Window: {WINDOW_TITLE}")
        print(f"Previous edit target: {previous_edit_target_identifier}")
        print(f"Root layer: {root_identifier}")
        print(f"Session layer: {session_identifier}")
        print(f"Anonymous diagnostic layer: {diagnostic_layer_identifier}")
        print(f"Root dirty before diagnostic: {root_dirty_before}")
        print("Session classification: DIAGNOSTIC_SESSION_ONLY_NOT_FINAL")
        while app.is_running():
            _guarded_app_update(
                app,
                get_current_stage,
                stage,
                diagnostic_layer,
            )
    except BaseException as error:
        primary_error = error
        primary_traceback = error.__traceback__
    finally:
        _cleanup_diagnostic_session(
            stage=stage,
            previous_edit_target=previous_edit_target,
            session_layer=session_layer,
            diagnostic_layer_identifier=diagnostic_layer_identifier,
            root_layer=root_layer,
            root_dirty_before=root_dirty_before,
            root_specs_before=root_specs_before,
            app=app,
            stage_path=stage_path,
            source_stage_sha256_before=source_stage_sha256_before,
            bottle_path=bottle_path,
            bottle_sha256_before=bottle_sha256_before,
            primary_error=primary_error,
            primary_traceback=primary_traceback,
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
