from __future__ import annotations

import argparse
import json
from pathlib import Path

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect runtime DOF/body names for the generated ALOHA Isaac USDs.")
    parser.add_argument("--left-usd", default="assets/isaac/original_stationary_aloha/generated/vx300s_left.usd")
    parser.add_argument("--right-usd", default="assets/isaac/original_stationary_aloha/generated/vx300s_right.usd")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp(dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG))
    try:
        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation

        World.clear_instance()
        stage_utils.create_new_stage()
        world = World(stage_units_in_meters=1.0, backend="numpy", device="cpu")
        stage_utils.add_reference_to_stage(usd_path=str(Path(args.left_usd).resolve()), prim_path="/World/left")
        stage_utils.add_reference_to_stage(usd_path=str(Path(args.right_usd).resolve()), prim_path="/World/right")
        left = world.scene.add(SingleArticulation(prim_path="/World/left/root_joint/root_joint", name="left_vx300s"))
        right = world.scene.add(SingleArticulation(prim_path="/World/right/root_joint/root_joint", name="right_vx300s"))
        world.reset()
        payload = {}
        for side, art in {"left": left, "right": right}.items():
            view = art._articulation_view
            physics_view = view._physics_view
            body_names = list(view.body_names)
            dof_names = list(art.dof_names)
            payload[side] = {
                "prim_path": art.prim_path,
                "num_dof": int(art.num_dof),
                "num_bodies": int(art.num_bodies),
                "dof_names": dof_names,
                "body_names": body_names,
                "ee_body_candidates": [name for name in body_names if "ee" in name or "gripper" in name],
                "link_indices": {name: int(view.get_body_index(name)) for name in body_names},
                "physics_view_pose_methods": [
                    name
                    for name in dir(physics_view)
                    if "transform" in name.lower() or "pose" in name.lower() or "link" in name.lower()
                ],
            }
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    finally:
        app.close(skip_cleanup=True)


if __name__ == "__main__":
    raise SystemExit(main())
