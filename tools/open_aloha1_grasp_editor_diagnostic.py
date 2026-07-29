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
    if _sha256(stage_path) != EXPECTED_STAGE_SHA256:
        raise RuntimeError("approved Stage hash no longer matches")
    if _sha256(bottle_path) != EXPECTED_BOTTLE_SHA256:
        raise RuntimeError("Bottle500 diagnostic USD hash no longer matches")

    import isaacsim

    app = isaacsim.SimulationApp({"headless": False})
    try:
        from isaacsim.core.utils.stage import open_stage
        import omni.kit.actions.core
        import omni.kit.app
        import omni.ui
        import omni.usd
        from pxr import Gf
        from pxr import Sdf
        from pxr import Usd
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

        stage = omni.usd.get_context().get_stage()
        session_layer = stage.GetSessionLayer()
        session_identifier = session_layer.identifier
        with Usd.EditContext(stage, session_layer):
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
        app.update()

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
        action.execute()
        for _ in range(10):
            app.update()
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
        print(f"Session layer: {session_identifier}")
        print("Session classification: DIAGNOSTIC_SESSION_ONLY_NOT_FINAL")
        while app.is_running():
            app.update()
        return 0
    finally:
        if _sha256(stage_path) != EXPECTED_STAGE_SHA256:
            raise RuntimeError("source Stage changed during GUI diagnostic")
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
