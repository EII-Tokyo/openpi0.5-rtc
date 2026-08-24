"""Run labeled episode 0 in the currently open Isaac Sim Stage.

Execute this file from Isaac Sim's Script Editor.  It creates an anonymous
session layer, never saves the Stage, never connects to ROS, and pauses the
timeline after frame 917.
"""

from __future__ import annotations

import asyncio
import builtins
import importlib.util
import os
import time
from pathlib import Path


HXZ_REPOSITORY_ROOT = Path("/home/eii/project/openpi0.5-rtc-reward-learning")
ALOHA_REPOSITORY_ROOT = Path("/home/eii/openpi0.5-rtc-reward-learning")
EXPECTED_FRAME_COUNT = 918
EXPECTED_FREQUENCY_HZ = 50.0
TASK_KEY = "_aloha_episode0_replay_task"
SESSION_LAYER_KEY = "_aloha_episode0_replay_session_layer"


def resolve_repository_root() -> Path:
    """Locate the source tree even when Script Editor executes a /tmp copy."""
    override = os.environ.get("OPENPI_REPOSITORY_ROOT")
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend(
        (
            Path(__file__).resolve().parents[1],
            HXZ_REPOSITORY_ROOT,
            ALOHA_REPOSITORY_ROOT,
        )
    )

    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        core_path = (
            candidate
            / "remote_isaac_assets/aloha1_bottle_server/attempt1/replays/episode_0"
            / "episode0_replay_core.py"
        )
        checked.append(str(core_path))
        if core_path.is_file():
            return candidate
    raise FileNotFoundError(
        "cannot locate episode-0 replay source tree; checked: " + ", ".join(checked)
    )


REPOSITORY_ROOT = resolve_repository_root()
BUNDLE_DIR = REPOSITORY_ROOT / "remote_isaac_assets/aloha1_bottle_server/attempt1/replays/episode_0"
CORE_PATH = BUNDLE_DIR / "episode0_replay_core.py"


def load_core():
    spec = importlib.util.spec_from_file_location("aloha_episode0_replay_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load replay core: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def acquire_replay_layer(stage, sdf):
    """Reuse this Stage's replay layer so reruns do not invalidate PhysX."""
    previous_identifier = getattr(builtins, SESSION_LAYER_KEY, None)
    paths = stage.GetSessionLayer().subLayerPaths
    if previous_identifier in paths:
        previous_layer = sdf.Layer.Find(previous_identifier)
        if previous_layer is not None:
            return previous_layer
        paths.remove(previous_identifier)

    layer = sdf.Layer.CreateAnonymous("episode_0_labeled_replay.usda")
    paths.append(layer.identifier)
    setattr(builtins, SESSION_LAYER_KEY, layer.identifier)
    return layer


async def replay() -> None:
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from omni.physx import get_physx_interface
    from pxr import Sdf, Usd

    core = load_core()
    _, payload = core.load_bundle(BUNDLE_DIR)
    usd_context = omni.usd.get_context()
    app = omni.kit.app.get_app()
    stage = usd_context.get_stage()
    if stage is None:
        raise RuntimeError("no active USD Stage")
    expected_stage = stage.GetPrimAtPath(core.BOTTLE).IsValid() and stage.GetPrimAtPath(core.CAP).IsValid()
    if not expected_stage:
        raise RuntimeError("open remote_stream_cap_stage.usda before running episode 0")

    frame_count = int(payload["action"].shape[0])
    frequency_hz = float(payload["frequency_hz"])
    if frame_count != EXPECTED_FRAME_COUNT or frequency_hz != EXPECTED_FREQUENCY_HZ:
        raise RuntimeError(
            "episode-0 replay contract mismatch: "
            f"frames={frame_count}, frequency_hz={frequency_hz}"
        )

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_stopped():
        timeline.stop()
        await app.next_update_async()
        await app.next_update_async()

    layer = acquire_replay_layer(stage, Sdf)
    previous_edit_target = stage.GetEditTarget()
    stage.SetEditTarget(Usd.EditTarget(layer))
    root_layer_identifier = stage.GetRootLayer().identifier
    frame = -1
    try:
        # This must happen before the first Play and before rigid-body tensor
        # views are initialized. Switching Dynamic -> Kinematic afterwards
        # leaves the existing view with incompatible velocity semantics.
        core.prepare_kinematic_objects(stage)
        timeline.play()
        await app.next_update_async()

        left = SingleArticulation(
            core.LEFT_EE.rsplit("/", 1)[0] + "/root_joint",
            name="episode0_left",
            reset_xform_properties=False,
        )
        right = SingleArticulation(
            core.RIGHT_EE.rsplit("/", 1)[0] + "/root_joint",
            name="episode0_right",
            reset_xform_properties=False,
        )
        bottle = SingleRigidPrim(core.BOTTLE, name="episode0_bottle", reset_xform_properties=False)
        cap = SingleRigidPrim(core.CAP, name="episode0_cap", reset_xform_properties=False)
        left.initialize()
        right.initialize()
        bottle.initialize()
        cap.initialize()

        # Keep the timeline paused during teleport replay so the stream loader's
        # normal physics callback cannot overwrite the recorded state.
        timeline.pause()
        runner = core.Episode0Replay(stage, left, right, bottle, cap, payload)
        runner.reset()

        print(
            f"[Episode0Replay] START {frame_count} frames @ "
            f"{frequency_hz:g} Hz; {core.CLASSIFICATION}"
        )
        start = time.monotonic()
        for frame in range(frame_count):
            current_stage = usd_context.get_stage()
            if (
                current_stage is None
                or current_stage.GetRootLayer().identifier != root_layer_identifier
            ):
                raise RuntimeError("USD Stage changed while replay was running")
            try:
                runner.apply_robot_frame(frame)
                get_physx_interface().update_transformations(True, True, False, False)
                runner.apply_objects_and_metadata(frame)
            except Exception as error:
                raise RuntimeError(
                    f"frame {frame}/{frame_count - 1} failed; "
                    "the PhysX/articulation view may have been invalidated"
                ) from error
            if frame in (0, 174, 650, 800, frame_count - 1):
                print(f"[Episode0Replay] frame={frame} state={payload['state'][frame]}")
            deadline = start + (frame + 1) / frequency_hz
            await asyncio.sleep(max(0.0, deadline - time.monotonic()))
        print(
            f"[Episode0Replay] PASS frames={runner.frames_applied}, "
            f"max_attach_jump_m={runner.max_attach_jump_m:.9f}"
        )
    except asyncio.CancelledError:
        print(f"[Episode0Replay] CANCELLED after frame={frame}")
        raise
    except Exception as error:
        print(f"[Episode0Replay] FAIL frame={frame}: {type(error).__name__}: {error}")
        raise
    finally:
        timeline.pause()
        current_stage = usd_context.get_stage()
        if (
            current_stage is not None
            and current_stage.GetRootLayer().identifier == root_layer_identifier
        ):
            current_stage.SetEditTarget(previous_edit_target)


previous_task = getattr(builtins, TASK_KEY, None)


async def launch() -> None:
    if previous_task is not None and not previous_task.done():
        previous_task.cancel()
        try:
            await previous_task
        except asyncio.CancelledError:
            pass
    await replay()


setattr(builtins, TASK_KEY, asyncio.ensure_future(launch()))
