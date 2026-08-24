"""Measure left/right gripper contact points on Bottle500 for two seconds.

Run inside Isaac Sim 5.1 Script Editor after the grasp is established and the
Timeline is Paused (not Stopped).  The startup loader must already have applied
ContactReportAPI before the first PhysX Play.  This diagnostic changes no pose,
joint target, material, rigid-body mode, or Stage layer; it only plays the
existing simulation for a measured two-second observation and returns to
Paused.

Contact-point count is a per-physics-step solver/manifold quantity.  It can
change as collision features and solver manifolds change, so the report uses
median, p95, maximum, and contact-presence fraction instead of treating one
frame's count as a permanent number of physical contact patches.
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from datetime import datetime

import numpy as np
import omni.kit.app
import omni.physx
import omni.timeline
import omni.usd
from isaacsim.core.prims import SingleRigidPrim
from omni.physx import get_physx_simulation_interface
from pxr import PhysxSchema, PhysicsSchemaTools, UsdPhysics


BOTTLE = "/World/ALOHA1RemoteBottleSession/Bottle500"
CAP = "/World/ALOHA1RemoteBottleSession/BottleCap"
LEFT_ROBOT = "/World/follower_left/vx300s_left"
LEFT_FINGER = f"{LEFT_ROBOT}/follower_left_left_finger_link"
RIGHT_FINGER = f"{LEFT_ROBOT}/follower_left_right_finger_link"
LEFT_ARTICULATION = f"{LEFT_ROBOT}/root_joint"
GRIPPER_MATERIAL = "/World/BottleTaskPhysicsMaterials/GripperPad_TEMP"

CONTACT_REPORT_PATHS = (LEFT_ARTICULATION, LEFT_FINGER, RIGHT_FINGER, BOTTLE)
OBSERVATION_DURATION_S = 2.0
MAX_RENDER_UPDATES = 2000
POSITION_DEDUP_DECIMALS = 6

REPORT_DIR = (
    "/home/eii/openpi0.5-rtc-reward-learning/remote_isaac_assets/"
    "aloha1_bottle_server/attempt1/reports/lula_joint_diagnostics"
)
LATEST_REPORT = os.path.join(
    REPORT_DIR, "bottle_gripper_contact_points_diagnostic_latest.json"
)

_CONTACT_POINT_TASK = None
_CONTACT_POINT_PHYSICS_SUBSCRIPTION = None
_CONTACT_POINT_CONTACT_SUBSCRIPTION = None


def _decode_path(encoded_path: int) -> str:
    if not encoded_path:
        return ""
    try:
        return str(PhysicsSchemaTools.intToSdfPath(encoded_path))
    except Exception:
        return ""


def _array(value):
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None


def _json_value(value):
    array = _array(value)
    if array is not None:
        return array.tolist()
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return str(value)


def _magnitude(value) -> float:
    array = _array(value)
    if array is None or array.size == 0:
        return 0.0
    return float(np.linalg.norm(array))


def _scalar(value):
    array = _array(value)
    if array is None or array.size == 0:
        return None
    return float(array[0])


def _quat_angle_deg(first, second) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    return float(
        np.degrees(2.0 * np.arccos(np.clip(abs(np.dot(first, second)), -1.0, 1.0)))
    )


def _quat_rotate_wxyz(quaternion, vector) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q /= np.linalg.norm(q)
    w = q[0]
    xyz = q[1:]
    vector = np.asarray(vector, dtype=np.float64)
    return (
        (2.0 * w * w - 1.0) * vector
        + 2.0 * np.dot(xyz, vector) * xyz
        + 2.0 * w * np.cross(xyz, vector)
    )


def _world_point_to_body_local(position_world, body_position, body_orientation):
    q = np.asarray(body_orientation, dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rotation.T @ (
        np.asarray(position_world, dtype=np.float64)
        - np.asarray(body_position, dtype=np.float64)
    )


def _impulse_weighted_centroid(details):
    valid = []
    for detail in details:
        position = detail.get("position_world_m")
        if not isinstance(position, list) or len(position) < 3:
            continue
        valid.append(
            (
                np.asarray(position[:3], dtype=np.float64),
                max(float(detail.get("impulse_magnitude", 0.0)), 0.0),
            )
        )
    if not valid:
        return None
    positive = [(position, weight) for position, weight in valid if weight > 0.0]
    if positive:
        total = sum(weight for _, weight in positive)
        return sum(position * weight for position, weight in positive) / total
    return np.mean(np.asarray([position for position, _ in valid]), axis=0)


def _classify(paths) -> str | None:
    if not any(path.startswith(BOTTLE) for path in paths):
        return None
    if any(path.startswith(LEFT_FINGER) for path in paths):
        return "left"
    if any(path.startswith(RIGHT_FINGER) for path in paths):
        return "right"
    if any(path.startswith(LEFT_ROBOT) for path in paths):
        return "nonfinger"
    return None


def _summary(values, total_steps: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    present = array[array > 0]
    return {
        "physics_steps": int(total_steps),
        "steps_with_contact": int(np.count_nonzero(array > 0)),
        "contact_presence_fraction": float(np.count_nonzero(array > 0) / total_steps)
        if total_steps
        else 0.0,
        "median_points_per_step": float(np.median(array)) if array.size else 0.0,
        "median_points_when_present": float(np.median(present)) if present.size else 0.0,
        "p95_points_per_step": float(np.percentile(array, 95)) if array.size else 0.0,
        "maximum_points_per_step": int(np.max(array)) if array.size else 0,
    }


async def run_contact_point_diagnostic() -> dict:
    global _CONTACT_POINT_PHYSICS_SUBSCRIPTION
    global _CONTACT_POINT_CONTACT_SUBSCRIPTION

    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    stage = omni.usd.get_context().get_stage()
    physics = {"elapsed_s": 0.0, "step": 0}
    points_by_step = {}
    event_count = {"left": 0, "right": 0, "nonfinger": 0}
    motion_samples = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_report = os.path.join(
        REPORT_DIR, f"bottle_gripper_contact_points_diagnostic_{timestamp}.json"
    )
    report = {
        "status": "STARTED",
        "classification": "ISAAC_SIM_5_1_BOTTLE_GRIPPER_CONTACT_POINTS",
        "observation_duration_s": OBSERVATION_DURATION_S,
        "position_dedup_resolution_m": 10.0 ** (-POSITION_DEDUP_DECIMALS),
        "stage_saved": False,
        "poses_commanded": False,
        "joint_targets_commanded": False,
        "materials_changed": False,
        "rigid_body_modes_changed": False,
        "ros_used": False,
        "real_robot_touched": False,
    }

    try:
        if stage is None:
            raise RuntimeError("no active USD Stage")
        if timeline.is_playing():
            raise RuntimeError("Timeline must be Paused before contact observation")
        if timeline.is_stopped():
            raise RuntimeError(
                "Timeline is Stopped; establish the grasp, Pause, then run this diagnostic"
            )

        for path in CONTACT_REPORT_PATHS:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                raise RuntimeError(f"required Contact Report prim is missing: {path}")
            if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                raise RuntimeError(
                    f"ContactReportAPI is absent on {path}; restart through the project loader "
                    "before the first PhysX Play"
                )
        bottle_prim = stage.GetPrimAtPath(BOTTLE)
        bottle_rigid = UsdPhysics.RigidBodyAPI(bottle_prim)
        report["preflight"] = {
            "timeline_time_s": float(timeline.get_current_time()),
            "bottle_kinematic": bool(bottle_rigid.GetKinematicEnabledAttr().Get()),
            "contact_report_paths": list(CONTACT_REPORT_PATHS),
        }
        material_prim = stage.GetPrimAtPath(GRIPPER_MATERIAL)
        if not material_prim or not material_prim.IsValid():
            raise RuntimeError(f"gripper material is missing: {GRIPPER_MATERIAL}")
        report["preflight"]["gripper_material_profile"] = {
            "path": GRIPPER_MATERIAL,
            "static_friction": float(
                material_prim.GetAttribute("physics:staticFriction").Get()
            ),
            "dynamic_friction": float(
                material_prim.GetAttribute("physics:dynamicFriction").Get()
            ),
            "friction_combine_mode": str(
                material_prim.GetAttribute(
                    "physxMaterial:frictionCombineMode"
                ).Get()
            ),
            "compliant_contact_acceleration_spring": bool(
                material_prim.GetAttribute(
                    "physxMaterial:compliantContactAccelerationSpring"
                ).Get()
            ),
            "compliant_contact_stiffness": float(
                material_prim.GetAttribute(
                    "physxMaterial:compliantContactStiffness"
                ).Get()
            ),
            "compliant_contact_damping": float(
                material_prim.GetAttribute(
                    "physxMaterial:compliantContactDamping"
                ).Get()
            ),
        }

        def on_physics_step(dt: float) -> None:
            physics["elapsed_s"] += float(dt)
            physics["step"] += 1

        def on_contact_report(contact_headers, contact_data) -> None:
            step = int(physics["step"])
            step_entry = points_by_step.setdefault(
                step, {"left": {}, "right": {}, "nonfinger": {}}
            )
            for header in contact_headers:
                paths = tuple(
                    sorted(
                        {
                            path
                            for path in (
                                _decode_path(getattr(header, "actor0", 0)),
                                _decode_path(getattr(header, "actor1", 0)),
                                _decode_path(getattr(header, "collider0", 0)),
                                _decode_path(getattr(header, "collider1", 0)),
                            )
                            if path
                        }
                    )
                )
                side = _classify(paths)
                if side is None:
                    continue
                offset = int(getattr(header, "contact_data_offset", 0))
                count = int(getattr(header, "num_contact_data", 0))
                if count > 0:
                    event_count[side] += 1
                for index in range(offset, min(offset + count, len(contact_data))):
                    datum = contact_data[index]
                    position = _array(getattr(datum, "position", None))
                    if position is None or position.size < 3:
                        key = ("missing_position", index, len(step_entry[side]))
                    else:
                        key = tuple(
                            np.round(position[:3], POSITION_DEDUP_DECIMALS).tolist()
                        )
                    detail = {
                        "position_world_m": _json_value(
                            getattr(datum, "position", None)
                        ),
                        "normal_world": _json_value(getattr(datum, "normal", None)),
                        "impulse": _json_value(getattr(datum, "impulse", None)),
                        "impulse_magnitude": _magnitude(
                            getattr(datum, "impulse", None)
                        ),
                        "separation": _json_value(
                            getattr(datum, "separation", None)
                        ),
                        "paths": list(paths),
                    }
                    existing = step_entry[side].get(key)
                    if existing is None or detail["impulse_magnitude"] > existing[
                        "impulse_magnitude"
                    ]:
                        step_entry[side][key] = detail

        _CONTACT_POINT_PHYSICS_SUBSCRIPTION = (
            omni.physx.get_physx_interface().subscribe_physics_step_events(
                on_physics_step
            )
        )
        _CONTACT_POINT_CONTACT_SUBSCRIPTION = (
            get_physx_simulation_interface().subscribe_contact_report_events(
                on_contact_report
            )
        )

        timeline.play()
        await app.next_update_async()
        bottle_body = SingleRigidPrim(BOTTLE, "grasp_rotation_bottle")
        cap_body = SingleRigidPrim(CAP, "grasp_rotation_cap")
        bottle_body.initialize()
        cap_body.initialize()
        bottle_start_position, bottle_start_orientation = bottle_body.get_world_pose()
        cap_start_position, cap_start_orientation = cap_body.get_world_pose()
        render_updates = 0
        while physics["elapsed_s"] < OBSERVATION_DURATION_S:
            await app.next_update_async()
            render_updates += 1
            bottle_position, bottle_orientation = bottle_body.get_world_pose()
            cap_position, cap_orientation = cap_body.get_world_pose()
            bottle_angular_velocity = np.asarray(
                bottle_body.get_angular_velocity(), dtype=np.float64
            )
            cap_angular_velocity = np.asarray(
                cap_body.get_angular_velocity(), dtype=np.float64
            )
            bottle_axis_world = _quat_rotate_wxyz(
                bottle_orientation, [0.0, 0.0, 1.0]
            )
            motion_samples.append(
                {
                    "physics_elapsed_s": float(physics["elapsed_s"]),
                    "bottle_world_rotation_deg": _quat_angle_deg(
                        bottle_start_orientation, bottle_orientation
                    ),
                    "cap_world_rotation_deg": _quat_angle_deg(
                        cap_start_orientation, cap_orientation
                    ),
                    "bottle_cap_relative_rotation_deg": _quat_angle_deg(
                        bottle_orientation, cap_orientation
                    ),
                    "bottle_angular_speed_rad_s": float(
                        np.linalg.norm(bottle_angular_velocity)
                    ),
                    "cap_angular_speed_rad_s": float(
                        np.linalg.norm(cap_angular_velocity)
                    ),
                    "bottle_axial_angular_speed_rad_s": float(
                        np.dot(bottle_angular_velocity, bottle_axis_world)
                    ),
                    "bottle_world_position_m": np.asarray(
                        bottle_position, dtype=np.float64
                    ).tolist(),
                    "bottle_world_orientation_wxyz": np.asarray(
                        bottle_orientation, dtype=np.float64
                    ).tolist(),
                    "cap_world_position_m": np.asarray(
                        cap_position, dtype=np.float64
                    ).tolist(),
                }
            )
            if render_updates >= MAX_RENDER_UPDATES:
                raise RuntimeError("insufficient physics progress during contact observation")
        timeline.pause()
        for _ in range(3):
            await app.next_update_async()

        total_steps = int(physics["step"])
        per_step = []
        counts = {"left": [], "right": [], "nonfinger": []}
        impulses = {"left": [], "right": [], "nonfinger": []}
        separations = {"left": [], "right": [], "nonfinger": []}
        representative_points = {"left": [], "right": [], "nonfinger": []}
        for step in range(1, total_steps + 1):
            entry = points_by_step.get(
                step, {"left": {}, "right": {}, "nonfinger": {}}
            )
            row = {"physics_step": step}
            for side in ("left", "right", "nonfinger"):
                details = list(entry[side].values())
                count = len(details)
                counts[side].append(count)
                impulses[side].append(sum(item["impulse_magnitude"] for item in details))
                separations[side].extend(
                    separation
                    for separation in (
                        _scalar(item.get("separation")) for item in details
                    )
                    if separation is not None
                )
                row[f"{side}_unique_contact_points"] = count
                row[f"{side}_impulse_magnitude_sum"] = impulses[side][-1]
                for detail in details:
                    if len(representative_points[side]) < 30:
                        representative_points[side].append(detail)
            per_step.append(row)

        summary = {}
        physics_dt_s = float(physics["elapsed_s"] / total_steps) if total_steps else 0.0
        for side in ("left", "right", "nonfinger"):
            impulse_array = np.asarray(impulses[side], dtype=np.float64)
            separation_array = np.asarray(separations[side], dtype=np.float64)
            penetration_array = np.maximum(-separation_array, 0.0)
            summary[side] = {
                **_summary(counts[side], total_steps),
                "contact_event_headers_with_points": int(event_count[side]),
                "median_impulse_magnitude_sum_per_step": float(np.median(impulse_array))
                if impulse_array.size
                else 0.0,
                "p95_impulse_magnitude_sum_per_step": float(np.percentile(impulse_array, 95))
                if impulse_array.size
                else 0.0,
                "maximum_impulse_magnitude_sum_per_step": float(np.max(impulse_array))
                if impulse_array.size
                else 0.0,
                "median_equivalent_force_n": float(np.median(impulse_array) / physics_dt_s)
                if impulse_array.size and physics_dt_s > 0.0
                else 0.0,
                "p95_equivalent_force_n": float(np.percentile(impulse_array, 95) / physics_dt_s)
                if impulse_array.size and physics_dt_s > 0.0
                else 0.0,
                "minimum_contact_separation_m": float(np.min(separation_array))
                if separation_array.size
                else None,
                "maximum_equivalent_pad_compression_m": float(np.max(penetration_array))
                if penetration_array.size
                else None,
                "median_equivalent_pad_compression_m": float(np.median(penetration_array))
                if penetration_array.size
                else None,
            }
        bilateral_steps = sum(
            1
            for left, right in zip(counts["left"], counts["right"])
            if left > 0 and right > 0
        )
        motion_summary = {
            "bottle_world_rotation_deg": float(
                max((sample["bottle_world_rotation_deg"] for sample in motion_samples), default=0.0)
            ),
            "cap_world_rotation_deg": float(
                max((sample["cap_world_rotation_deg"] for sample in motion_samples), default=0.0)
            ),
            "maximum_bottle_cap_relative_rotation_deg": float(
                max((sample["bottle_cap_relative_rotation_deg"] for sample in motion_samples), default=0.0)
            ),
            "maximum_bottle_angular_speed_rad_s": float(
                max((sample["bottle_angular_speed_rad_s"] for sample in motion_samples), default=0.0)
            ),
            "maximum_cap_angular_speed_rad_s": float(
                max((sample["cap_angular_speed_rad_s"] for sample in motion_samples), default=0.0)
            ),
            "maximum_abs_bottle_axial_angular_speed_rad_s": float(
                max((abs(sample["bottle_axial_angular_speed_rad_s"]) for sample in motion_samples), default=0.0)
            ),
            "final_bottle_angular_speed_rad_s": float(
                motion_samples[-1]["bottle_angular_speed_rad_s"] if motion_samples else 0.0
            ),
            "final_bottle_axial_angular_speed_rad_s": float(
                motion_samples[-1]["bottle_axial_angular_speed_rad_s"] if motion_samples else 0.0
            ),
        }
        # Representative points pool data across time and must never be used
        # to infer bilateral alignment. Pair the two sides at the same physics
        # step, then express both centroids in the nearest measured Bottle
        # frame. Bottle-local +Z is the long/axial direction in this asset.
        synchronized_pairs = []
        for step in range(1, total_steps + 1):
            entry = points_by_step.get(step)
            if not entry:
                continue
            left_centroid = _impulse_weighted_centroid(list(entry["left"].values()))
            right_centroid = _impulse_weighted_centroid(list(entry["right"].values()))
            if left_centroid is None or right_centroid is None or not motion_samples:
                continue
            step_time = float(step * physics["elapsed_s"] / max(total_steps, 1))
            pose_sample = min(
                motion_samples,
                key=lambda sample: abs(sample["physics_elapsed_s"] - step_time),
            )
            left_local = _world_point_to_body_local(
                left_centroid,
                pose_sample["bottle_world_position_m"],
                pose_sample["bottle_world_orientation_wxyz"],
            )
            right_local = _world_point_to_body_local(
                right_centroid,
                pose_sample["bottle_world_position_m"],
                pose_sample["bottle_world_orientation_wxyz"],
            )
            synchronized_pairs.append(
                {
                    "physics_step": step,
                    "physics_elapsed_s": step_time,
                    "left_centroid_bottle_local_m": left_local.tolist(),
                    "right_centroid_bottle_local_m": right_local.tolist(),
                    "axial_difference_m": float(left_local[2] - right_local[2]),
                    "absolute_axial_difference_m": float(abs(left_local[2] - right_local[2])),
                }
            )
        axial_differences = np.asarray(
            [pair["absolute_axial_difference_m"] for pair in synchronized_pairs],
            dtype=np.float64,
        )
        synchronized_geometry_summary = {
            "coordinate_contract": "Bottle-local +Z is the bottle long axis",
            "paired_physics_steps": len(synchronized_pairs),
            "median_absolute_axial_difference_m": float(np.median(axial_differences))
            if axial_differences.size
            else None,
            "p95_absolute_axial_difference_m": float(np.percentile(axial_differences, 95))
            if axial_differences.size
            else None,
            "maximum_absolute_axial_difference_m": float(np.max(axial_differences))
            if axial_differences.size
            else None,
        }
        report.update(
            {
                "status": "RECORDED",
                "physics_elapsed_s": float(physics["elapsed_s"]),
                "mean_physics_dt_s": physics_dt_s,
                "physics_step_count": total_steps,
                "render_updates": render_updates,
                "summary": summary,
                "bilateral_contact": {
                    "steps": bilateral_steps,
                    "fraction": float(bilateral_steps / total_steps)
                    if total_steps
                    else 0.0,
                },
                "motion_summary": motion_summary,
                "motion_samples": motion_samples,
                "synchronized_bilateral_geometry": synchronized_geometry_summary,
                "synchronized_bilateral_pairs": synchronized_pairs,
                "per_step": per_step,
                "representative_contact_points": representative_points,
                "interpretation": (
                    "Use median/p95/presence over physics steps. A single-frame point count "
                    "is not a persistent physical patch count."
                ),
            }
        )
    except Exception as exc:
        report["status"] = "EXCEPTION"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc().splitlines()[-30:]
    finally:
        timeline.pause()
        _CONTACT_POINT_PHYSICS_SUBSCRIPTION = None
        _CONTACT_POINT_CONTACT_SUBSCRIPTION = None
        for _ in range(3):
            await app.next_update_async()
        os.makedirs(REPORT_DIR, exist_ok=True)
        for output_path in (timestamped_report, LATEST_REPORT):
            with open(output_path, "w", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        print(f"Bottle/gripper contact report: {LATEST_REPORT}", flush=True)
    return report


_CONTACT_POINT_TASK = asyncio.ensure_future(run_contact_point_diagnostic())
