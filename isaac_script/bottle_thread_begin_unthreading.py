"""Explicitly transition BottleCap from THREADED to UNTHREADING.

Run in Isaac Sim 5.1 Script Editor while Timeline is Paused.  This opens the
0--12 mm axial travel, removes the Revolute limits using the USD Physics schema
defaults (-inf/+inf), enables the right-hand Rack-and-Pinion coupling, and
leaves Timeline Paused.  It does not create a Drive or save the Stage.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import traceback
from datetime import datetime

import omni.timeline
import omni.usd
from pxr import Sdf, UsdPhysics

SESSION = "/World/ALOHA1RemoteBottleSession"
BOTTLE = f"{SESSION}/Bottle500"
CAP = f"{SESSION}/BottleCap"
SLIDER = f"{SESSION}/BottleThreadSlider"
JOINT_SCOPE = f"{SESSION}/BottleThreadJoints"
PRISMATIC = f"{JOINT_SCOPE}/ThreadPrismatic"
REVOLUTE = f"{JOINT_SCOPE}/ThreadRevolute"
COUPLING = f"{JOINT_SCOPE}/RightHandThreadCoupling"
TRAVEL_M = 0.012
REPORT_DIR = (
    "/home/eii/openpi0.5-rtc-reward-learning/remote_isaac_assets/"
    "aloha1_bottle_server/attempt1/reports/lula_joint_diagnostics"
)
LATEST_REPORT_PATH = os.path.join(REPORT_DIR, "bottle_thread_begin_unthreading_result.json")


def _set_joint_enabled(stage, path: str, enabled: bool) -> None:
    stage.GetPrimAtPath(path).CreateAttribute(
        "physics:jointEnabled", Sdf.ValueTypeNames.Bool
    ).Set(enabled)


def _set_limit(stage, path: str, lower: float, upper: float) -> None:
    prim = stage.GetPrimAtPath(path)
    prim.CreateAttribute("physics:lowerLimit", Sdf.ValueTypeNames.Float).Set(lower)
    prim.CreateAttribute("physics:upperLimit", Sdf.ValueTypeNames.Float).Set(upper)


async def begin_bottle_thread_unthreading() -> dict:
    timeline = omni.timeline.get_timeline_interface()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped = os.path.join(REPORT_DIR, f"bottle_thread_begin_unthreading_{timestamp}.json")
    report = {
        "status": "STARTED",
        "classification": "ISAAC_SIM_5_1_EXPLICIT_UNTHREADING_TRANSITION",
        "stage_saved": False,
        "ros_used": False,
        "real_robot_touched": False,
    }
    try:
        if timeline.is_playing():
            raise RuntimeError("Timeline must be Paused before enabling UNTHREADING")
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("no active USD Stage")
        for path in (BOTTLE, CAP, SLIDER, JOINT_SCOPE, PRISMATIC, REVOLUTE, COUPLING):
            if not stage.GetPrimAtPath(path):
                raise RuntimeError(f"required prim is missing: {path}")
        state_before = stage.GetPrimAtPath(JOINT_SCOPE).GetCustomDataByKey("threadState")
        if state_before != "THREADED":
            raise RuntimeError(f"expected THREADED, got {state_before!r}")
        for path in (BOTTLE, CAP, SLIDER):
            body = UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path))
            if not body or bool(body.GetKinematicEnabledAttr().Get()):
                raise RuntimeError(f"UNTHREADING requires Dynamic rigid body: {path}")
        revolute_prim = stage.GetPrimAtPath(REVOLUTE)
        if UsdPhysics.DriveAPI.Get(revolute_prim, "angular"):
            raise RuntimeError("remove the ThreadRevolute angular Drive before UNTHREADING")

        # Keep the coupling disabled until both constituent joints are open.
        _set_joint_enabled(stage, COUPLING, False)
        _set_limit(stage, PRISMATIC, 0.0, TRAVEL_M)
        _set_limit(stage, REVOLUTE, float("-inf"), float("inf"))
        _set_joint_enabled(stage, PRISMATIC, True)
        _set_joint_enabled(stage, REVOLUTE, True)
        _set_joint_enabled(stage, COUPLING, True)
        for path in (JOINT_SCOPE, COUPLING):
            prim = stage.GetPrimAtPath(path)
            prim.SetCustomDataByKey("threadState", "UNTHREADING")
            prim.SetCustomDataByKey("transitionInProgress", True)
            prim.SetCustomDataByKey("tightHoldMode", "NONE")

        p_limits = [float(stage.GetPrimAtPath(PRISMATIC).GetAttribute(name).Get()) for name in ("physics:lowerLimit", "physics:upperLimit")]
        r_limits = [float(stage.GetPrimAtPath(REVOLUTE).GetAttribute(name).Get()) for name in ("physics:lowerLimit", "physics:upperLimit")]
        checks = {
            "state_is_unthreading": stage.GetPrimAtPath(JOINT_SCOPE).GetCustomDataByKey("threadState") == "UNTHREADING",
            "prismatic_open_0_to_12mm": p_limits == [0.0, TRAVEL_M],
            "revolute_unlimited": math.isinf(r_limits[0]) and r_limits[0] < 0 and math.isinf(r_limits[1]) and r_limits[1] > 0,
            "all_three_joints_enabled": all(bool(stage.GetPrimAtPath(path).GetAttribute("physics:jointEnabled").Get()) for path in (PRISMATIC, REVOLUTE, COUPLING)),
            "no_angular_drive": not bool(UsdPhysics.DriveAPI.Get(revolute_prim, "angular")),
            "timeline_paused": not timeline.is_playing(),
        }
        report.update({
            "status": "PASS" if all(checks.values()) else "FAIL",
            "state_before": state_before,
            "state_after": "UNTHREADING",
            "prismatic_limits_m": p_limits,
            "revolute_limits_deg": r_limits,
            "checks": checks,
            "handoff": "UNTHREADING_READY; PRESS_PLAY_ONLY_WHEN_INTENTIONAL",
        })
        if report["status"] != "PASS":
            raise RuntimeError(f"UNTHREADING readback failed: {checks}")
    except Exception as exc:
        timeline.pause()
        report["status"] = "EXCEPTION"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc().splitlines()[-30:]
    finally:
        timeline.pause()
        os.makedirs(REPORT_DIR, exist_ok=True)
        for path in (timestamped, LATEST_REPORT_PATH):
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        print(f"UNTHREADING report: {LATEST_REPORT_PATH}", flush=True)
    return report


asyncio.ensure_future(begin_bottle_thread_unthreading())
