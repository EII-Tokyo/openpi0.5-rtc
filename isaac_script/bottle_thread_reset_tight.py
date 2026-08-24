"""Restore BottleCap to a locked THREADED state in Isaac Sim 5.1.

Run in Script Editor while Timeline is Paused and the grippers are clear.
The Stage is not saved; ROS and real hardware are not used.  Bottle500,
BottleCap, and BottleThreadSlider remain Dynamic.  Joint limits—not a Drive—
hold the cap closed.
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from datetime import datetime

import omni.kit.app
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
REPORT_DIR = (
    "/home/eii/openpi0.5-rtc-reward-learning/remote_isaac_assets/"
    "aloha1_bottle_server/attempt1/reports/lula_joint_diagnostics"
)
LATEST_REPORT_PATH = os.path.join(REPORT_DIR, "bottle_thread_reset_tight_result.json")


def _set_joint_enabled(stage, path: str, enabled: bool) -> None:
    stage.GetPrimAtPath(path).CreateAttribute(
        "physics:jointEnabled", Sdf.ValueTypeNames.Bool
    ).Set(enabled)


def _set_limit(stage, path: str, lower: float, upper: float) -> None:
    prim = stage.GetPrimAtPath(path)
    prim.CreateAttribute("physics:lowerLimit", Sdf.ValueTypeNames.Float).Set(lower)
    prim.CreateAttribute("physics:upperLimit", Sdf.ValueTypeNames.Float).Set(upper)


def _set_dynamic(stage, path: str) -> None:
    UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path)).CreateKinematicEnabledAttr().Set(False)


def _set_state(stage, state: str) -> None:
    for path in (JOINT_SCOPE, COUPLING):
        prim = stage.GetPrimAtPath(path)
        prim.SetCustomDataByKey("threadState", state)
        prim.SetCustomDataByKey("transitionInProgress", False)
        prim.SetCustomDataByKey("tightHoldMode", "LOCKED_JOINT_LIMITS")
        prim.SetCustomDataByKey("tightHoldCalibrationStatus", "NOT_APPLICABLE")
        for key in ("releaseExtensionM", "releaseThresholdM"):
            prim.ClearCustomDataByKey(key)


def _kinematic(stage, path: str) -> bool:
    return bool(
        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path))
        .GetKinematicEnabledAttr()
        .Get()
    )


async def reset_bottle_thread_tight() -> dict:
    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_report = os.path.join(
        REPORT_DIR, f"bottle_thread_reset_tight_{timestamp}.json"
    )
    report = {
        "status": "STARTED",
        "classification": "ISAAC_SIM_5_1_LOCKED_THREADED_RESET",
        "stage_saved": False,
        "ros_used": False,
        "real_robot_touched": False,
    }
    try:
        if timeline.is_playing():
            raise RuntimeError("Timeline must be Paused before resetting the thread")
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("no active USD Stage")
        expected = {
            BOTTLE: None,
            CAP: None,
            SLIDER: None,
            JOINT_SCOPE: "Scope",
            PRISMATIC: "PhysicsPrismaticJoint",
            REVOLUTE: "PhysicsRevoluteJoint",
            COUPLING: "PhysxPhysicsRackAndPinionJoint",
        }
        for path, expected_type in expected.items():
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                raise RuntimeError(f"required prim is missing: {path}")
            if expected_type and prim.GetTypeName() != expected_type:
                raise RuntimeError(
                    f"{path} has type {prim.GetTypeName()}, expected {expected_type}"
                )
        for path in (BOTTLE, CAP, SLIDER):
            if not stage.GetPrimAtPath(path).HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"RigidBodyAPI missing: {path}")

        report["state_before"] = stage.GetPrimAtPath(JOINT_SCOPE).GetCustomDataByKey(
            "threadState"
        )
        # Stop restores the authored tight transforms. Disable first so stale
        # solver impulses cannot act during the reset.
        for path in (COUPLING, REVOLUTE, PRISMATIC):
            _set_joint_enabled(stage, path, False)
        timeline.stop()
        for _ in range(8):
            await app.next_update_async()

        for path in (BOTTLE, CAP, SLIDER):
            _set_dynamic(stage, path)
        revolute_prim = stage.GetPrimAtPath(REVOLUTE)
        stale_drive_removed = bool(UsdPhysics.DriveAPI.Get(revolute_prim, "angular"))
        if stale_drive_removed:
            revolute_prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")

        _set_limit(stage, PRISMATIC, 0.0, 0.0)
        _set_limit(stage, REVOLUTE, 0.0, 0.0)
        _set_joint_enabled(stage, PRISMATIC, True)
        _set_joint_enabled(stage, REVOLUTE, True)
        _set_joint_enabled(stage, COUPLING, False)
        _set_state(stage, "THREADED")

        readback = {
            "state": stage.GetPrimAtPath(JOINT_SCOPE).GetCustomDataByKey("threadState"),
            "bottle_kinematic": _kinematic(stage, BOTTLE),
            "cap_kinematic": _kinematic(stage, CAP),
            "slider_kinematic": _kinematic(stage, SLIDER),
            "prismatic_enabled": bool(stage.GetPrimAtPath(PRISMATIC).GetAttribute("physics:jointEnabled").Get()),
            "prismatic_limits_m": [float(stage.GetPrimAtPath(PRISMATIC).GetAttribute(name).Get()) for name in ("physics:lowerLimit", "physics:upperLimit")],
            "revolute_enabled": bool(stage.GetPrimAtPath(REVOLUTE).GetAttribute("physics:jointEnabled").Get()),
            "revolute_limits_deg": [float(stage.GetPrimAtPath(REVOLUTE).GetAttribute(name).Get()) for name in ("physics:lowerLimit", "physics:upperLimit")],
            "coupling_enabled": bool(stage.GetPrimAtPath(COUPLING).GetAttribute("physics:jointEnabled").Get()),
            "angular_drive_present": bool(UsdPhysics.DriveAPI.Get(revolute_prim, "angular")),
        }
        checks = {
            "state_is_threaded": readback["state"] == "THREADED",
            "all_bodies_dynamic": not any(readback[key] for key in ("bottle_kinematic", "cap_kinematic", "slider_kinematic")),
            "prismatic_locked_zero": readback["prismatic_enabled"] and readback["prismatic_limits_m"] == [0.0, 0.0],
            "revolute_locked_zero": readback["revolute_enabled"] and readback["revolute_limits_deg"] == [0.0, 0.0],
            "coupling_disabled": not readback["coupling_enabled"],
            "no_angular_drive": not readback["angular_drive_present"],
            "timeline_paused": not timeline.is_playing(),
        }
        report.update({
            "status": "PASS" if all(checks.values()) else "FAIL",
            "stale_angular_drive_removed": stale_drive_removed,
            "readback": readback,
            "checks": checks,
            "handoff": "THREADED_LOCKED_READY_FOR_ORDINARY_PLAY",
        })
        if report["status"] != "PASS":
            raise RuntimeError(f"locked THREADED readback failed: {checks}")
    except Exception as exc:
        timeline.pause()
        report["status"] = "EXCEPTION"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc().splitlines()[-30:]
    finally:
        timeline.pause()
        os.makedirs(REPORT_DIR, exist_ok=True)
        for output in (timestamped_report, LATEST_REPORT_PATH):
            with open(output, "w", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        print(f"Bottle thread reset report: {LATEST_REPORT_PATH}", flush=True)
    return report


asyncio.ensure_future(reset_bottle_thread_tight())
