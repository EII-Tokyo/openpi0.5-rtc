"""Commit the current BottleCap mechanism to RELEASED while Paused.

This script only changes the three thread joints and metadata.  It does not
move the cap, inject velocity, save the Stage, or use ROS/real hardware.
Normally run it after the cap has reached the release threshold.
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from datetime import datetime

import omni.timeline
import omni.usd
from pxr import Sdf, UsdPhysics

SESSION = "/World/ALOHA1RemoteBottleSession"
CAP = f"{SESSION}/BottleCap"
JOINT_SCOPE = f"{SESSION}/BottleThreadJoints"
PRISMATIC = f"{JOINT_SCOPE}/ThreadPrismatic"
REVOLUTE = f"{JOINT_SCOPE}/ThreadRevolute"
COUPLING = f"{JOINT_SCOPE}/RightHandThreadCoupling"
REPORT_DIR = (
    "/home/eii/openpi0.5-rtc-reward-learning/remote_isaac_assets/"
    "aloha1_bottle_server/attempt1/reports/lula_joint_diagnostics"
)
LATEST_REPORT_PATH = os.path.join(REPORT_DIR, "bottle_thread_set_released_result.json")


async def set_bottle_thread_released() -> dict:
    timeline = omni.timeline.get_timeline_interface()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped = os.path.join(REPORT_DIR, f"bottle_thread_set_released_{timestamp}.json")
    report = {
        "status": "STARTED",
        "classification": "ISAAC_SIM_5_1_EXPLICIT_RELEASED_COMMIT",
        "stage_saved": False,
        "ros_used": False,
        "real_robot_touched": False,
    }
    try:
        if timeline.is_playing():
            raise RuntimeError("Timeline must be Paused before committing RELEASED")
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("no active USD Stage")
        for path in (CAP, JOINT_SCOPE, PRISMATIC, REVOLUTE, COUPLING):
            if not stage.GetPrimAtPath(path):
                raise RuntimeError(f"required prim is missing: {path}")
        state_before = stage.GetPrimAtPath(JOINT_SCOPE).GetCustomDataByKey("threadState")
        if state_before not in ("UNTHREADING", "THREADED"):
            raise RuntimeError(f"cannot commit RELEASED from {state_before!r}")
        for path in (COUPLING, REVOLUTE, PRISMATIC):
            stage.GetPrimAtPath(path).CreateAttribute(
                "physics:jointEnabled", Sdf.ValueTypeNames.Bool
            ).Set(False)
        for path in (JOINT_SCOPE, COUPLING):
            prim = stage.GetPrimAtPath(path)
            prim.SetCustomDataByKey("threadState", "RELEASED")
            prim.SetCustomDataByKey("transitionInProgress", False)
            prim.SetCustomDataByKey("tightHoldMode", "NONE")
        cap_kinematic = bool(
            UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(CAP))
            .GetKinematicEnabledAttr()
            .Get()
        )
        enabled = {
            path: bool(stage.GetPrimAtPath(path).GetAttribute("physics:jointEnabled").Get())
            for path in (PRISMATIC, REVOLUTE, COUPLING)
        }
        checks = {
            "state_is_released": stage.GetPrimAtPath(JOINT_SCOPE).GetCustomDataByKey("threadState") == "RELEASED",
            "all_thread_joints_disabled": not any(enabled.values()),
            "cap_is_dynamic": not cap_kinematic,
            "timeline_paused": not timeline.is_playing(),
        }
        report.update({
            "status": "PASS" if all(checks.values()) else "FAIL",
            "state_before": state_before,
            "state_after": "RELEASED",
            "joint_enabled": enabled,
            "checks": checks,
            "handoff": "RELEASED; BOTTLECAP_IS_PHYSICALLY_INDEPENDENT",
        })
        if report["status"] != "PASS":
            raise RuntimeError(f"RELEASED readback failed: {checks}")
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
        print(f"RELEASED report: {LATEST_REPORT_PATH}", flush=True)
    return report


asyncio.ensure_future(set_bottle_thread_released())
