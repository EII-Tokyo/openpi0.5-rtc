"""Open the exact local episode-0 Stage and load replay in Script Editor.

Intended for Isaac Sim's ``--exec`` option.  This loader does not save the
Stage, enable ROS, or communicate with a real robot.
"""

from __future__ import annotations

import asyncio
from pathlib import Path


REPOSITORY_ROOT = Path("/home/eii/project/openpi0.5-rtc-reward-learning")
STAGE_PATH = (
    REPOSITORY_ROOT
    / "remote_isaac_assets/aloha1_bottle_server/attempt1/remote_stream_cap_stage.usda"
)
REPLAY_PATH = REPOSITORY_ROOT / "isaac_script/replay_episode0_labeled.py"
REQUIRED_PRIMS = (
    "/World/ALOHA1RemoteBottleSession/Bottle500",
    "/World/ALOHA1RemoteBottleSession/BottleCap",
    "/World/follower_left/vx300s_left/root_joint",
    "/World/follower_right/vx300s_right/root_joint",
)


async def launch_episode0() -> None:
    import carb.settings
    import omni.kit.app
    import omni.timeline
    import omni.usd
    import omni.ui as ui

    app = omni.kit.app.get_app()
    context = omni.usd.get_context()
    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()

    for path in (STAGE_PATH, REPLAY_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    print(f"[Episode0LocalLoader] opening stage={STAGE_PATH}", flush=True)
    result = await context.open_stage_async(str(STAGE_PATH))
    opened = result[0] if isinstance(result, tuple) else result
    if opened is False:
        raise RuntimeError(f"open_stage_async failed: {result!r}")

    stage = None
    for _ in range(600):
        stage = context.get_stage()
        if stage is not None and all(stage.GetPrimAtPath(path).IsValid() for path in REQUIRED_PRIMS):
            break
        await app.next_update_async()
    else:
        missing = [] if stage is None else [
            path for path in REQUIRED_PRIMS if not stage.GetPrimAtPath(path).IsValid()
        ]
        raise RuntimeError(f"episode-0 Stage did not become ready; missing={missing}")

    root_identifier = Path(stage.GetRootLayer().realPath).resolve()
    if root_identifier != STAGE_PATH.resolve():
        raise RuntimeError(
            f"wrong Stage loaded: actual={root_identifier}, expected={STAGE_PATH.resolve()}"
        )

    timeline.pause()
    print(
        f"[Episode0LocalLoader] READY stage={root_identifier} "
        f"required_prims={len(REQUIRED_PRIMS)}",
        flush=True,
    )

    # --exec runs as soon as Kit reports app-ready, while the Full App bundle
    # is still building its workspace. Let that UI initialization settle before
    # constructing Script Editor to avoid a large concurrent startup peak.
    for _ in range(240):
        await app.next_update_async()

    ui.Workspace.show_window("Script Editor", True)
    window = None
    for _ in range(120):
        window = ui.Workspace.get_window("Script Editor")
        if window is not None and window._script_editor_widget is not None:
            break
        await app.next_update_async()
    else:
        raise RuntimeError("Script Editor window did not become ready")

    # The user's persisted workspace may have "Execute File On Reload" on.
    # Disable it before loading so this helper never starts robot replay on the
    # user's behalf; the Script Editor Run button remains the explicit trigger.
    carb.settings.get_settings().set(
        "/persistent/exts/omni.kit.window.script_editor/executeOnReload",
        False,
    )
    window._script_editor_widget.load_script(str(REPLAY_PATH))
    await app.next_update_async()
    loaded_path = Path(window._script_editor_widget.get_script_path()).resolve()
    if loaded_path != REPLAY_PATH.resolve():
        raise RuntimeError(
            f"wrong Script Editor file: actual={loaded_path}, expected={REPLAY_PATH.resolve()}"
        )
    window.visible = True
    print(
        f"[Episode0LocalLoader] EDITOR_READY script={loaded_path} executed=false",
        flush=True,
    )


async def guarded_launch() -> None:
    try:
        await launch_episode0()
    except Exception as error:
        print(
            f"[Episode0LocalLoader] FAIL {type(error).__name__}: {error}",
            flush=True,
        )
        raise


asyncio.ensure_future(guarded_launch())
