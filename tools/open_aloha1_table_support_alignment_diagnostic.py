#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Open the isolated ALOHA table/support alignment Stage with runtime MCP."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
STAGE_PATH = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "table_support_alignment/1.0/"
    "aloha1_table_support_aligned_workcell.usda"
)
EXPECTED_STAGE_SHA256 = (
    "2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c"
)
CONTROL_EXTENSION_ID = "isaac.sim.mcp_extension"
CONTROL_EXTENSION_VERSION = "0.4.1"
CONTROL_EXTENSION_PARENT = Path(
    "/home/eii/isaac_mcp_setup/repos/isaacsim-mcp-server"
)
CONTROL_EXTENSION_PATH = (
    CONTROL_EXTENSION_PARENT / CONTROL_EXTENSION_ID
)
PYTHON_EXTENSION_ID = "isaacsim.code_editor.vscode"
PYTHON_EXTENSION_VERSION = "1.1.0"
WINDOW_TITLE = "Isaac Sim - ALOHA Table Support Alignment"

CAMERAS = {
    "overview": {
        "path": "/World/ALOHA1TableSupportAlignmentSession/Cameras/Overview",
        "eye": (-1.15, 2.75, 1.35),
        "target": (0.0, 0.0, 0.20),
        "focal_length": 32.0,
    },
    "support_side": {
        "path": (
            "/World/ALOHA1TableSupportAlignmentSession/"
            "Cameras/SupportSide"
        ),
        "eye": (0.0, 1.65, 0.16),
        "target": (0.0, 0.0, 0.055),
        "focal_length": 48.0,
    },
    "left_base_side": {
        "path": (
            "/World/ALOHA1TableSupportAlignmentSession/"
            "Cameras/LeftBaseSide"
        ),
        "eye": (-0.47, 1.10, 0.16),
        "target": (-0.47, -0.02, 0.055),
        "focal_length": 55.0,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _author_session_cameras(stage) -> None:
    from pxr import Gf
    from pxr import UsdGeom

    stage.SetEditTarget(stage.GetSessionLayer())
    UsdGeom.Xform.Define(
        stage,
        "/World/ALOHA1TableSupportAlignmentSession",
    )
    UsdGeom.Xform.Define(
        stage,
        "/World/ALOHA1TableSupportAlignmentSession/Cameras",
    )
    for record in CAMERAS.values():
        camera = UsdGeom.Camera.Define(stage, record["path"])
        camera.CreateFocalLengthAttr().Set(record["focal_length"])
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
        matrix = (
            Gf.Matrix4d()
            .SetLookAt(
                Gf.Vec3d(*record["eye"]),
                Gf.Vec3d(*record["target"]),
                Gf.Vec3d(0.0, 0.0, 1.0),
            )
            .GetInverse()
        )
        xformable = UsdGeom.Xformable(camera.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(matrix)


def _set_active_camera(path: str) -> None:
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Sdf

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("active viewport is unavailable")
    viewport.camera_path = Sdf.Path(path)


def main() -> None:
    stage_path = STAGE_PATH.resolve(strict=True)
    stage_hash_before = _sha256(stage_path)
    if stage_hash_before != EXPECTED_STAGE_SHA256:
        raise RuntimeError(
            f"diagnostic Stage hash mismatch: {stage_hash_before}"
        )

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": False,
            "window_title": WINDOW_TITLE,
            "window_width": 1980,
            "window_height": 1120,
        }
    )
    try:
        import carb
        import omni.kit.app
        import omni.timeline
        import omni.usd

        from examples.aloha_isaac.scripts.open_workcell_gui import _move_current_process_window_to_workspace
        from tools.open_aloha1_grasp_editor_diagnostic import _enable_extension_exact

        manager = omni.kit.app.get_app().get_extension_manager()
        control_extension = _enable_extension_exact(
            manager,
            extension_id=CONTROL_EXTENSION_ID,
            expected_version=CONTROL_EXTENSION_VERSION,
            extension_parent=CONTROL_EXTENSION_PARENT,
            expected_extension_path=CONTROL_EXTENSION_PATH,
        )
        python_extension = _enable_extension_exact(
            manager,
            extension_id=PYTHON_EXTENSION_ID,
            expected_version=PYTHON_EXTENSION_VERSION,
        )
        context = omni.usd.get_context()
        if not context.open_stage(str(stage_path)):
            raise RuntimeError(f"failed to open Stage: {stage_path}")
        for _ in range(8):
            app.update()

        stage = context.get_stage()
        if str(stage.GetDefaultPrim().GetPath()) != "/World":
            raise RuntimeError("diagnostic Stage default prim is not /World")
        required = (
            "/World/follower_left/vx300s_left/root_joint",
            "/World/follower_right/vx300s_right/root_joint",
            "/World/environment/worldBody/user_confirmed_table",
        )
        missing = [
            path for path in required if not stage.GetPrimAtPath(path).IsValid()
        ]
        if missing:
            raise RuntimeError(f"diagnostic Stage missing prims: {missing}")

        _author_session_cameras(stage)
        _set_active_camera(CAMERAS["overview"]["path"])
        omni.timeline.get_timeline_interface().pause()
        for _ in range(8):
            app.update()
        moved = _move_current_process_window_to_workspace(2)
        print(
            "ALOHA_TABLE_SUPPORT_ALIGNMENT_RUNTIME="
            + json.dumps(
                {
                    "status": "READY",
                    "stage": str(stage_path),
                    "stage_sha256": stage_hash_before,
                    "root_prim": "/World",
                    "sublayers": list(
                        stage.GetRootLayer().subLayerPaths
                    ),
                    "required_prims": list(required),
                    "timeline_playing": (
                        omni.timeline.get_timeline_interface().is_playing()
                    ),
                    "workspace_two": moved,
                    "control_extension": control_extension,
                    "python_extension": python_extension,
                    "cameras": CAMERAS,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        while app.is_running():
            app.update()
            time.sleep(0.001)
    finally:
        stage_hash_after = _sha256(stage_path)
        if stage_hash_after != stage_hash_before:
            carb.log_error(
                "table/support diagnostic source Stage hash changed: "
                f"{stage_hash_before} -> {stage_hash_after}"
            )
        app.close()


if __name__ == "__main__":
    main()
