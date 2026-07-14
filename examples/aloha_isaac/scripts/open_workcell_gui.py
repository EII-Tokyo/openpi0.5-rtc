from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from examples.aloha_isaac.scripts.apply_aloha_initial_pose import split_real_start_pose_for_isaac_articulations


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_USD = (
    REPO_ROOT
    / "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose"
    / "aloha2_menagerie_scene_deep_black_real_start_pose.usd"
)
DEFAULT_LEFT_ARTICULATION_ROOT = "/scene/left_base_link/left_base_link"
DEFAULT_RIGHT_ARTICULATION_ROOT = "/scene/right_base_link/right_base_link"


def _set_real_start_pose_on_initialized_articulations(left: Any, right: Any) -> bool:
    if not getattr(left, "handles_initialized", False) or not getattr(right, "handles_initialized", False):
        return False

    left_pose, right_pose = split_real_start_pose_for_isaac_articulations()
    zeros = [0.0] * len(left_pose)
    left.set_joint_positions(left_pose)
    right.set_joint_positions(right_pose)
    left.set_joint_velocities(zeros)
    right.set_joint_velocities(zeros)
    return True


def _apply_real_start_pose_to_articulations() -> tuple[object, object, object]:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation

    world = World(stage_units_in_meters=1.0)
    left = SingleArticulation(DEFAULT_LEFT_ARTICULATION_ROOT, name="aloha_left")
    right = SingleArticulation(DEFAULT_RIGHT_ARTICULATION_ROOT, name="aloha_right")
    world.scene.add(left)
    world.scene.add(right)

    left_pose, right_pose = split_real_start_pose_for_isaac_articulations()
    zeros = [0.0] * len(left_pose)
    left.set_joints_default_state(positions=left_pose, velocities=zeros)
    right.set_joints_default_state(positions=right_pose, velocities=zeros)

    world.reset()
    world.pause()
    if not _set_real_start_pose_on_initialized_articulations(left, right):
        raise RuntimeError("ALOHA articulations were not initialized after World.reset().")
    return world, left, right


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the generated ALOHA Isaac workcell USD in Isaac Sim GUI.")
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument(
        "--no-real-start-pose",
        action="store_true",
        help="Open the USD without forcing the imported ALOHA articulations to the real START_ARM_POSE.",
    )
    args = parser.parse_args()

    usd_path = args.usd.resolve()
    if not usd_path.exists():
        raise FileNotFoundError(f"USD stage does not exist: {usd_path}")

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": False, "window_title": "Isaac Sim - ALOHA Workcell"})
    try:
        import omni.kit.app
        import omni.usd

        context = omni.usd.get_context()
        if not context.open_stage(str(usd_path)):
            raise RuntimeError(f"Isaac failed to open stage: {usd_path}")
        for _ in range(5):
            app.update()

        articulations = None
        if not args.no_real_start_pose:
            articulations = _apply_real_start_pose_to_articulations()
            print("Applied real START_ARM_POSE to ALOHA articulations.")

        kit_app = omni.kit.app.get_app()
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        while kit_app.is_running():
            app.update()
            if articulations is not None and not timeline.is_playing():
                _, left, right = articulations
                _set_real_start_pose_on_initialized_articulations(left, right)
    finally:
        app.close()


if __name__ == "__main__":
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    main()
