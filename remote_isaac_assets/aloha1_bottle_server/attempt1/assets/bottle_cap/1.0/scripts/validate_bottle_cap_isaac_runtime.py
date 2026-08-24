from __future__ import annotations

import json
import math
from pathlib import Path

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
STAGE_PATH = ROOT / "tests/bottle_cap_integration_test.usda"
REPORT_PATH = ROOT / "reports/bottle_cap_isaac_runtime_validation.json"


def bbox(stage, path: str) -> dict[str, list[float]]:
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    box = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return {"min": [float(v) for v in box.GetMin()], "max": [float(v) for v in box.GetMax()]}


def main() -> None:
    app = SimulationApp({"headless": True, "width": 640, "height": 480, "multi_gpu": False, "limit_cpu_threads": 8})
    try:
        import omni.timeline
        import omni.usd
        from pxr import PhysxSchema, UsdPhysics

        context = omni.usd.get_context()
        if not context.open_stage(str(STAGE_PATH)):
            raise RuntimeError(f"Isaac Sim could not open {STAGE_PATH}")
        for _ in range(60):
            app.update()
        stage = context.get_stage()
        cap = stage.GetPrimAtPath("/World/BottleCap")
        bottle = stage.GetPrimAtPath("/World/Bottle500")
        before = bbox(stage, "/World/BottleCap")

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(60):
            app.update()
        timeline.stop()
        for _ in range(10):
            app.update()
        after = bbox(stage, "/World/BottleCap")

        cap_colliders = [prim for prim in stage.Traverse() if str(prim.GetPath()).startswith("/World/BottleCap/") and prim.HasAPI(UsdPhysics.CollisionAPI)]
        materials = [stage.GetPrimAtPath(f"/World/PhysicsMaterials/{name}") for name in ("BottleSurface_TEMP", "CapSurface_TEMP", "TableSurface_TEMP", "GripperPad_TEMP")]
        finite_bounds = all(math.isfinite(value) for value in before["min"] + before["max"] + after["min"] + after["max"])
        max_motion = max(abs(after[k][i] - before[k][i]) for k in ("min", "max") for i in range(3))
        checks = {
            "stage_opened": stage is not None,
            "bottle_valid": bool(bottle),
            "cap_valid": bool(cap),
            "cap_physx_rigid_body_api": cap.HasAPI(PhysxSchema.PhysxRigidBodyAPI),
            "cap_collision_count_17": len(cap_colliders) == 17,
            "four_physx_materials": all(bool(material) and material.HasAPI(PhysxSchema.PhysxMaterialAPI) for material in materials),
            "finite_bounds": finite_bounds,
            "kinematic_hold_stable": max_motion < 1e-7,
        }
        report = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "classification": "ISAAC_SIM_RUNTIME_LOAD_PHASE_1_2_DIAGNOSTIC",
            "stage": str(STAGE_PATH),
            "checks": checks,
            "cap_bounds_before": before,
            "cap_bounds_after": after,
            "max_bbox_motion_m": max_motion,
            "physics_updates": 60,
            "note": "Kinematic hold is intentional until the cap-thread constraint is implemented in a later phase.",
        }
        REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        if report["status"] != "PASS":
            raise SystemExit(1)
    finally:
        app.close()


if __name__ == "__main__":
    main()
