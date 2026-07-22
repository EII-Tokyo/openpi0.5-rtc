from __future__ import annotations

from pathlib import Path

from isaacsim import SimulationApp

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG


def main() -> int:
    cfg = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
    cfg.update({"headless": True, "disable_viewport_updates": True, "fast_shutdown": True})
    app = SimulationApp(cfg)
    try:
        import isaacsim.core.utils.stage as stage_utils
        import omni.kit.app
        from pxr import Usd, UsdGeom

        usd = Path(
            "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/"
            "aloha2_menagerie_scene_deep_black_real_start_pose.usd"
        ).resolve()
        stage_utils.open_stage(str(usd))
        for _ in range(10):
            omni.kit.app.get_app().update()
        stage = stage_utils.get_current_stage()
        target_path = "/scene/worldBody/__9/collisions/__9/__9"
        prim = stage.GetPrimAtPath(target_path)
        print("INSPECT stage", usd, flush=True)
        print("INSPECT path", target_path, flush=True)
        print("INSPECT exists", bool(prim and prim.IsValid()), flush=True)
        if not prim or not prim.IsValid():
            return 2
        print("INSPECT type", prim.GetTypeName(), flush=True)
        print("INSPECT name", prim.GetName(), flush=True)
        print("INSPECT parent", prim.GetParent().GetPath(), flush=True)
        print("INSPECT applied_schemas", list(prim.GetAppliedSchemas()), flush=True)
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=False,
        )
        box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        print("INSPECT bbox_empty", box.IsEmpty(), flush=True)
        if not box.IsEmpty():
            mn = box.GetMin()
            mx = box.GetMax()
            print("INSPECT bbox_min", [float(mn[i]) for i in range(3)], flush=True)
            print("INSPECT bbox_max", [float(mx[i]) for i in range(3)], flush=True)
            print("INSPECT bbox_size", [float(mx[i] - mn[i]) for i in range(3)], flush=True)
            print("INSPECT bbox_center", [float((mx[i] + mn[i]) * 0.5) for i in range(3)], flush=True)
        print("INSPECT attrs", flush=True)
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if any(key in name.lower() for key in ("display", "purpose", "visibility", "physics", "collision")):
                try:
                    value = attr.Get()
                except Exception as exc:
                    value = f"<read failed: {exc}>"
                print("INSPECT attr", name, repr(value), flush=True)
        print("INSPECT ancestor_chain", flush=True)
        cursor = prim
        while cursor and cursor.IsValid():
            print("INSPECT ancestor", cursor.GetPath(), cursor.GetTypeName(), list(cursor.GetAppliedSchemas()), flush=True)
            if str(cursor.GetPath()) == "/":
                break
            cursor = cursor.GetParent()
        print("INSPECT siblings", flush=True)
        for child in prim.GetParent().GetChildren():
            print("INSPECT sibling", child.GetPath(), child.GetTypeName(), list(child.GetAppliedSchemas()), flush=True)
        print("INSPECT camera_like_prims", flush=True)
        for candidate in stage.Traverse():
            text = str(candidate.GetPath()).lower()
            if candidate.GetTypeName() == "Camera" or "camera" in text or "d405" in text:
                print("INSPECT camera_like", candidate.GetPath(), candidate.GetTypeName(), list(candidate.GetAppliedSchemas()), flush=True)
        return 0
    finally:
        app.close(skip_cleanup=True)


if __name__ == "__main__":
    raise SystemExit(main())
