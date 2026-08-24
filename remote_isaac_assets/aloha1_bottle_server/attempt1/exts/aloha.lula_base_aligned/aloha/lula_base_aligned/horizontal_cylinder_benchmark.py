"""Isolated horizontal-cylinder grasp benchmark for the ALOHA Lula panel.

The benchmark deliberately reuses the production Bottle grasp trajectory and
left ALOHA gripper.  Only Session-Layer collision geometry/material opinions
and runtime simulation dt are changed.  The authored Stage is never saved.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import traceback
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import omni.kit.app
import omni.physx
import omni.timeline
import omni.usd
from isaacsim.core.api.physics_context import PhysicsContext
from isaacsim.core.prims import SingleRigidPrim
from omni.physx import get_physx_simulation_interface
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


BOTTLE_PATH = "/World/ALOHA1RemoteBottleSession/Bottle500"
CAP_PATH = "/World/ALOHA1RemoteBottleSession/BottleCap"
SLIDER_PATH = "/World/ALOHA1RemoteBottleSession/BottleThreadSlider"
THREAD_JOINT_PATHS = (
    "/World/ALOHA1RemoteBottleSession/BottleThreadJoints/ThreadPrismatic",
    "/World/ALOHA1RemoteBottleSession/BottleThreadJoints/ThreadRevolute",
    "/World/ALOHA1RemoteBottleSession/BottleThreadJoints/RightHandThreadCoupling",
)
LEFT_ROBOT_PATH = "/World/follower_left/vx300s_left"
LEFT_FINGER_PATH = f"{LEFT_ROBOT_PATH}/follower_left_left_finger_link"
RIGHT_FINGER_PATH = f"{LEFT_ROBOT_PATH}/follower_left_right_finger_link"
GRIPPER_MATERIAL_PATH = "/World/BottleTaskPhysicsMaterials/GripperPad_TEMP"
CYLINDER_PATH = f"{BOTTLE_PATH}/HorizontalCylinderBenchmark"

CYLINDER_LENGTH_M = 0.180
CYLINDER_RADIUS_M = 0.032
# Keep the production Grasp Editor plane, but put that plane at the Cylinder's
# own lower-end one-third station.  For a centered 180 mm Cylinder this means
# that its COM is 30 mm beyond the pinch plane.
GRASP_STATION_LOCAL_Z_M = 0.206 / 3.0
CYLINDER_CENTER_LOCAL_Z_M = GRASP_STATION_LOCAL_Z_M + CYLINDER_LENGTH_M / 6.0
CYLINDER_LOW_END_LOCAL_Z_M = CYLINDER_CENTER_LOCAL_Z_M - CYLINDER_LENGTH_M / 2.0
TARGET_GRASP_FRACTION_FROM_LOW_END = 1.0 / 3.0
TARGET_GRASP_AXIAL_TOLERANCE_M = 0.002
CYLINDER_MASS_KG = 0.200
CYLINDER_INERTIA_AXIAL_KG_M2 = 0.5 * CYLINDER_MASS_KG * CYLINDER_RADIUS_M**2
CYLINDER_INERTIA_RADIAL_KG_M2 = (
    CYLINDER_MASS_KG
    * (3.0 * CYLINDER_RADIUS_M**2 + CYLINDER_LENGTH_M**2)
    / 12.0
)
DEFAULT_CENTER_WORLD_M = np.asarray([-0.202153, -0.046512, 0.034], dtype=np.float64)
DEFAULT_YAW_DEG = -11.754
CONTROL_HZ = 50.0
HOLD_DURATION_S = 2.0
BENCHMARK_MATERIAL_PROFILE = {
    "static_friction": 2.0,
    "dynamic_friction": 1.5,
    "friction_combine_mode": "max",
    "compliant_contact_acceleration_spring": True,
    "compliant_contact_stiffness": 1000.0,
    "compliant_contact_damping": 64.0,
}


def _quat_normalize(value) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if norm <= 1.0e-12:
        raise RuntimeError("zero-length quaternion")
    return result / norm


def _quat_rotation(value) -> np.ndarray:
    w, x, y, z = _quat_normalize(value)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_multiply(left, right) -> np.ndarray:
    lw, lx, ly, lz = _quat_normalize(left)
    rw, rx, ry, rz = _quat_normalize(right)
    return _quat_normalize(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _quat_conjugate(value) -> np.ndarray:
    w, x, y, z = _quat_normalize(value)
    return np.asarray([w, -x, -y, -z], dtype=np.float64)


def _rotation_vector_world(initial_orientation, final_orientation) -> np.ndarray:
    relative = _quat_multiply(final_orientation, _quat_conjugate(initial_orientation))
    if relative[0] < 0.0:
        relative = -relative
    vector_norm = float(np.linalg.norm(relative[1:]))
    if vector_norm <= 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(relative[0]))
    return relative[1:] * (angle / vector_norm)


def _full_orientation_metrics(
    initial_orientation, final_orientation, finger_line_world
) -> Dict[str, object]:
    rotation_vector = _rotation_vector_world(initial_orientation, final_orientation)
    cylinder_axis = _quat_rotation(initial_orientation)[:, 2]
    finger_axis = np.asarray(finger_line_world, dtype=np.float64)
    finger_axis /= max(float(np.linalg.norm(finger_axis)), 1.0e-12)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return {
        "rotation_vector_world_rad": rotation_vector.tolist(),
        "total_rotation_deg": math.degrees(float(np.linalg.norm(rotation_vector))),
        "signed_rotation_about_finger_line_deg": math.degrees(
            float(np.dot(rotation_vector, finger_axis))
        ),
        "signed_rotation_about_initial_cylinder_axis_deg": math.degrees(
            float(np.dot(rotation_vector, cylinder_axis))
        ),
        "signed_rotation_about_world_up_deg": math.degrees(
            float(np.dot(rotation_vector, world_up))
        ),
        "finger_line_world": finger_axis.tolist(),
        "initial_cylinder_axis_world": cylinder_axis.tolist(),
    }


def _axis_metrics(initial_orientation, final_orientation) -> Dict[str, object]:
    initial_axis = _quat_rotation(initial_orientation)[:, 2]
    final_axis = _quat_rotation(final_orientation)[:, 2]
    direction_dot = float(np.clip(abs(np.dot(initial_axis, final_axis)), 0.0, 1.0))
    direction_change_deg = math.degrees(math.acos(direction_dot))
    elevation_from_table_deg = math.degrees(
        math.asin(float(np.clip(abs(final_axis[2]), 0.0, 1.0)))
    )
    return {
        "initial_axis_world": initial_axis.tolist(),
        "final_axis_world": final_axis.tolist(),
        "axis_direction_change_deg": direction_change_deg,
        "axis_elevation_from_table_deg": elevation_from_table_deg,
    }


def _array(value):
    try:
        return np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None


def _decode_path(value: int) -> str:
    if not value:
        return ""
    try:
        from pxr import PhysicsSchemaTools

        return str(PhysicsSchemaTools.intToSdfPath(value))
    except Exception:
        return ""


class _ContactCollector:
    def __init__(self, owner, body: SingleRigidPrim) -> None:
        self.owner = owner
        self.body = body
        self.phase = "idle"
        self.physics_step = 0
        self.hold_elapsed_s = 0.0
        self.hold_steps = 0
        self.points: Dict[int, Dict[str, Dict[Tuple[float, float, float], dict]]] = {}
        self.paths = set()
        self.errors: List[str] = []
        self._physics_subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(
            self._on_physics_step
        )
        self._contact_subscription = get_physx_simulation_interface().subscribe_contact_report_events(
            self._on_contact
        )

    def reset_trial(self) -> None:
        self.phase = "approach_close_lift"
        self.physics_step = 0
        self.hold_elapsed_s = 0.0
        self.hold_steps = 0
        self.points = {}
        self.paths = set()
        self.errors = []

    def begin_hold(self) -> None:
        self.phase = "hold"
        self.hold_elapsed_s = 0.0
        self.hold_steps = 0
        self.points = {}
        self.paths = set()

    def close(self) -> None:
        self._physics_subscription = None
        self._contact_subscription = None

    def _on_physics_step(self, dt: float) -> None:
        self.physics_step += 1
        if self.phase == "hold":
            self.hold_elapsed_s += float(dt)
            self.hold_steps += 1

    def _classify(self, paths: Tuple[str, ...]):
        if not any(path.startswith(BOTTLE_PATH) for path in paths):
            return None
        if any(path.startswith(LEFT_FINGER_PATH) for path in paths):
            return "left"
        if any(path.startswith(RIGHT_FINGER_PATH) for path in paths):
            return "right"
        if any(path.startswith(LEFT_ROBOT_PATH) for path in paths):
            return "nonfinger"
        return None

    def _on_contact(self, headers, data) -> None:
        if self.phase != "hold":
            return
        for header in headers:
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
            side = self._classify(paths)
            if side not in ("left", "right", "nonfinger"):
                continue
            self.paths.add((side, paths))
            offset = int(getattr(header, "contact_data_offset", 0))
            count = int(getattr(header, "num_contact_data", 0))
            for index in range(offset, min(offset + count, len(data))):
                try:
                    datum = data[index]
                    position = _array(getattr(datum, "position", None))
                    if position is None or position.size < 3:
                        continue
                    body_position, body_orientation = self.body.get_world_pose()
                    local = _quat_rotation(body_orientation).T @ (
                        position[:3] - np.asarray(body_position, dtype=np.float64)
                    )
                    impulse = _array(getattr(datum, "impulse", None))
                    impulse_magnitude = (
                        float(np.linalg.norm(impulse)) if impulse is not None else 0.0
                    )
                    key = tuple(np.round(local[:3], 6).tolist())
                    per_step = self.points.setdefault(
                        self.physics_step, {"left": {}, "right": {}, "nonfinger": {}}
                    )
                    current = per_step[side].get(key)
                    detail = {
                        "position_cylinder_local_m": local[:3].tolist(),
                        "impulse_magnitude": impulse_magnitude,
                        "paths": list(paths),
                    }
                    if current is None or impulse_magnitude > current["impulse_magnitude"]:
                        per_step[side][key] = detail
                except Exception as exc:
                    if len(self.errors) < 20:
                        self.errors.append(f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _weighted_centroid(details: List[dict]):
        if not details:
            return None
        positions = np.asarray(
            [item["position_cylinder_local_m"] for item in details], dtype=np.float64
        )
        weights = np.asarray(
            [max(float(item["impulse_magnitude"]), 0.0) for item in details],
            dtype=np.float64,
        )
        if float(np.sum(weights)) > 0.0:
            return np.average(positions, axis=0, weights=weights)
        return np.mean(positions, axis=0)

    def summarize_hold(self) -> Dict[str, object]:
        count_summary = {}
        all_details = {}
        for side in ("left", "right", "nonfinger"):
            counts = []
            details = []
            for step in range(max(self.physics_step - self.hold_steps + 1, 0), self.physics_step + 1):
                step_details = list(
                    self.points.get(step, {}).get(side, {}).values()
                )
                counts.append(len(step_details))
                details.extend(step_details)
            count_array = np.asarray(counts, dtype=np.float64)
            centroid = self._weighted_centroid(details)
            count_summary[side] = {
                "median_points_per_physics_step": float(np.median(count_array)) if count_array.size else 0.0,
                "p95_points_per_physics_step": float(np.percentile(count_array, 95)) if count_array.size else 0.0,
                "max_points_per_physics_step": int(np.max(count_array)) if count_array.size else 0,
                "contact_step_fraction": float(np.mean(count_array > 0.0)) if count_array.size else 0.0,
                "impulse_weighted_centroid_cylinder_local_m": None if centroid is None else centroid.tolist(),
            }
            all_details[side] = details
        left_centroid = count_summary["left"]["impulse_weighted_centroid_cylinder_local_m"]
        right_centroid = count_summary["right"]["impulse_weighted_centroid_cylinder_local_m"]
        axial_mismatch = None
        if left_centroid is not None and right_centroid is not None:
            axial_mismatch = abs(float(left_centroid[2]) - float(right_centroid[2]))
        return {
            "physics_steps": int(self.hold_steps),
            "elapsed_s": float(self.hold_elapsed_s),
            "counts": count_summary,
            "left_right_axial_centroid_mismatch_m": axial_mismatch,
            "active_paths": [
                {"side": side, "paths": list(paths)}
                for side, paths in sorted(self.paths, key=lambda item: (item[0], item[1]))
            ],
            "collector_errors": list(self.errors),
        }


def _get_attr_value(prim, name: str):
    attr = prim.GetAttribute(name)
    return attr.Get() if attr else None


def _set_attr(prim, name: str, type_name, value) -> None:
    attr = prim.GetAttribute(name)
    if not attr:
        attr = prim.CreateAttribute(name, type_name)
    attr.Set(value)


def _material_profile(prim) -> Dict[str, object]:
    return {
        "static_friction": float(_get_attr_value(prim, "physics:staticFriction") or 0.0),
        "dynamic_friction": float(_get_attr_value(prim, "physics:dynamicFriction") or 0.0),
        "friction_combine_mode": str(_get_attr_value(prim, "physxMaterial:frictionCombineMode")),
        "compliant_contact_acceleration_spring": bool(
            _get_attr_value(prim, "physxMaterial:compliantContactAccelerationSpring")
        ),
        "compliant_contact_stiffness": float(
            _get_attr_value(prim, "physxMaterial:compliantContactStiffness") or 0.0
        ),
        "compliant_contact_damping": float(
            _get_attr_value(prim, "physxMaterial:compliantContactDamping") or 0.0
        ),
    }


def _apply_material_profile(prim, profile: Dict[str, object]) -> None:
    """Author a runtime-only gripper material profile in the active edit target."""
    _set_attr(
        prim,
        "physics:staticFriction",
        Sdf.ValueTypeNames.Float,
        float(profile["static_friction"]),
    )
    _set_attr(
        prim,
        "physics:dynamicFriction",
        Sdf.ValueTypeNames.Float,
        float(profile["dynamic_friction"]),
    )
    _set_attr(
        prim,
        "physxMaterial:frictionCombineMode",
        Sdf.ValueTypeNames.Token,
        str(profile["friction_combine_mode"]),
    )
    _set_attr(
        prim,
        "physxMaterial:compliantContactAccelerationSpring",
        Sdf.ValueTypeNames.Bool,
        bool(profile["compliant_contact_acceleration_spring"]),
    )
    _set_attr(
        prim,
        "physxMaterial:compliantContactStiffness",
        Sdf.ValueTypeNames.Float,
        float(profile["compliant_contact_stiffness"]),
    )
    _set_attr(
        prim,
        "physxMaterial:compliantContactDamping",
        Sdf.ValueTypeNames.Float,
        float(profile["compliant_contact_damping"]),
    )


def _write_report(path: str, report: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def _finger_collision_prims(stage):
    result = []
    for root_path in (LEFT_FINGER_PATH, RIGHT_FINGER_PATH):
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            raise RuntimeError(f"missing finger rigid body {root_path}")
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                result.append(prim)
    if not result:
        raise RuntimeError("no left/right finger collision shapes were found")
    return result


def _finger_mesh_collision_prims(stage):
    result = [prim for prim in _finger_collision_prims(stage) if prim.IsA(UsdGeom.Mesh)]
    if not result:
        raise RuntimeError("no left/right finger Mesh collision shapes were found")
    return result


def _set_finger_mesh_approximation(stage, approximation: str) -> List[dict]:
    if approximation not in ("convexHull", "convexDecomposition"):
        raise RuntimeError(f"unsupported finger approximation {approximation}")
    readback = []
    for prim in _finger_mesh_collision_prims(stage):
        mesh_collision = (
            UsdPhysics.MeshCollisionAPI(prim)
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            else UsdPhysics.MeshCollisionAPI.Apply(prim)
        )
        mesh_collision.CreateApproximationAttr().Set(approximation)
        readback.append(
            {
                "path": prim.GetPath().pathString,
                "authored": approximation,
                "readback": str(mesh_collision.GetApproximationAttr().Get()),
                "type": prim.GetTypeName(),
            }
        )
    if not all(item["readback"] == approximation for item in readback):
        raise RuntimeError(f"finger approximation readback failed: {readback}")
    return readback


def _set_torsional_patch_radius(stage, radius_m: float) -> List[str]:
    if not 0.0 <= radius_m <= 0.010:
        raise RuntimeError("torsional patch radius must be within [0, 0.010] m")
    prims = _finger_collision_prims(stage)
    cylinder_prim = stage.GetPrimAtPath(CYLINDER_PATH)
    if not cylinder_prim or not cylinder_prim.IsValid():
        raise RuntimeError(f"missing benchmark cylinder {CYLINDER_PATH}")
    prims.append(cylinder_prim)
    paths = []
    for prim in prims:
        api = (
            PhysxSchema.PhysxCollisionAPI(prim)
            if prim.HasAPI(PhysxSchema.PhysxCollisionAPI)
            else PhysxSchema.PhysxCollisionAPI.Apply(prim)
        )
        api.CreateTorsionalPatchRadiusAttr().Set(float(radius_m))
        api.CreateMinTorsionalPatchRadiusAttr().Set(float(radius_m))
        paths.append(prim.GetPath().pathString)
    return paths


def _configure_cylinder(stage) -> Dict[str, object]:
    state: Dict[str, object] = {
        "colliders": [],
        "nonfinger_robot_colliders": [],
        "visibility": [],
        "joints": [],
        "rigid_gravity": [],
        "finger_mesh_approximations": [],
    }
    material_prim = stage.GetPrimAtPath(GRIPPER_MATERIAL_PATH)
    if not material_prim or not material_prim.IsValid():
        raise RuntimeError(f"missing material {GRIPPER_MATERIAL_PATH}")
    state["material_profile"] = _material_profile(material_prim)

    for prim in _finger_mesh_collision_prims(stage):
        api = (
            UsdPhysics.MeshCollisionAPI(prim)
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            else UsdPhysics.MeshCollisionAPI.Apply(prim)
        )
        state["finger_mesh_approximations"].append(
            (prim.GetPath().pathString, api.GetApproximationAttr().Get())
        )

    bottle_prim = stage.GetPrimAtPath(BOTTLE_PATH)
    if not bottle_prim or not bottle_prim.IsValid():
        raise RuntimeError(f"missing rigid body {BOTTLE_PATH}")

    for root_path in (BOTTLE_PATH, CAP_PATH, SLIDER_PATH):
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            raise RuntimeError(f"missing benchmark component {root_path}")
        for prim in Usd.PrimRange(root):
            if prim.GetPath().pathString == CYLINDER_PATH:
                continue
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                api = UsdPhysics.CollisionAPI(prim)
                original = bool(api.GetCollisionEnabledAttr().Get())
                state["colliders"].append((prim.GetPath().pathString, original))
                api.CreateCollisionEnabledAttr().Set(False)

    # Layer 1 of this benchmark isolates the real ALOHA finger collision
    # meshes.  Disable every other left-robot collider so wrist/palm approach
    # geometry cannot prevent us from measuring the requested fingertip
    # Convex Decomposition contact.  This is deliberately not an end-to-end
    # collision-safe grasp claim; the report records the isolation explicitly.
    robot_root = stage.GetPrimAtPath(LEFT_ROBOT_PATH)
    if not robot_root or not robot_root.IsValid():
        raise RuntimeError(f"missing left robot {LEFT_ROBOT_PATH}")
    finger_roots = (LEFT_FINGER_PATH, RIGHT_FINGER_PATH)
    for prim in Usd.PrimRange(robot_root):
        path = prim.GetPath().pathString
        if any(path == root or path.startswith(root + "/") for root in finger_roots):
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            api = UsdPhysics.CollisionAPI(prim)
            original = bool(api.GetCollisionEnabledAttr().Get())
            state["nonfinger_robot_colliders"].append((path, original))
            api.CreateCollisionEnabledAttr().Set(False)

    for root_path in (f"{BOTTLE_PATH}/Visuals", CAP_PATH, SLIDER_PATH):
        prim = stage.GetPrimAtPath(root_path)
        if prim and prim.IsValid() and prim.IsA(UsdGeom.Imageable):
            imageable = UsdGeom.Imageable(prim)
            original = imageable.GetVisibilityAttr().Get()
            state["visibility"].append((root_path, str(original)))
            imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    for path in THREAD_JOINT_PATHS:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            joint = UsdPhysics.Joint(prim)
            original = bool(joint.GetJointEnabledAttr().Get())
            state["joints"].append((path, original))
            joint.CreateJointEnabledAttr().Set(False)

    for path in (CAP_PATH, SLIDER_PATH):
        prim = stage.GetPrimAtPath(path)
        attr_name = "physxRigidBody:disableGravity"
        original = _get_attr_value(prim, attr_name)
        state["rigid_gravity"].append((path, original))
        _set_attr(prim, attr_name, Sdf.ValueTypeNames.Bool, True)

    mass_api = (
        UsdPhysics.MassAPI(bottle_prim)
        if bottle_prim.HasAPI(UsdPhysics.MassAPI)
        else UsdPhysics.MassAPI.Apply(bottle_prim)
    )
    mass_attributes = {
        "mass": mass_api.GetMassAttr(),
        "density": mass_api.GetDensityAttr(),
        "center_of_mass": mass_api.GetCenterOfMassAttr(),
        "diagonal_inertia": mass_api.GetDiagonalInertiaAttr(),
        "principal_axes": mass_api.GetPrincipalAxesAttr(),
    }
    state["bottle_mass_properties"] = {
        name: attribute.Get() for name, attribute in mass_attributes.items()
    }
    mass_api.CreateMassAttr().Set(CYLINDER_MASS_KG)
    mass_api.CreateDensityAttr().Set(0.0)
    mass_api.CreateCenterOfMassAttr().Set(
        Gf.Vec3f(0.0, 0.0, CYLINDER_CENTER_LOCAL_Z_M)
    )
    mass_api.CreateDiagonalInertiaAttr().Set(
        Gf.Vec3f(
            CYLINDER_INERTIA_RADIAL_KG_M2,
            CYLINDER_INERTIA_RADIAL_KG_M2,
            CYLINDER_INERTIA_AXIAL_KG_M2,
        )
    )
    mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    _apply_material_profile(material_prim, BENCHMARK_MATERIAL_PROFILE)

    if stage.GetPrimAtPath(CYLINDER_PATH):
        stage.RemovePrim(CYLINDER_PATH)
    cylinder = UsdGeom.Cylinder.Define(stage, CYLINDER_PATH)
    cylinder.CreateAxisAttr().Set(UsdGeom.Tokens.z)
    cylinder.CreateHeightAttr().Set(CYLINDER_LENGTH_M)
    cylinder.CreateRadiusAttr().Set(CYLINDER_RADIUS_M)
    cylinder.CreateDisplayColorAttr().Set([Gf.Vec3f(0.12, 0.55, 0.95)])
    cylinder.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, CYLINDER_CENTER_LOCAL_Z_M))
    cylinder_prim = cylinder.GetPrim()
    UsdPhysics.CollisionAPI.Apply(cylinder_prim).CreateCollisionEnabledAttr().Set(True)
    UsdShade.MaterialBindingAPI.Apply(cylinder_prim).Bind(UsdShade.Material(material_prim))

    torsional_state = []
    for prim in _finger_collision_prims(stage):
        api = (
            PhysxSchema.PhysxCollisionAPI(prim)
            if prim.HasAPI(PhysxSchema.PhysxCollisionAPI)
            else PhysxSchema.PhysxCollisionAPI.Apply(prim)
        )
        torsional_state.append(
            (
                prim.GetPath().pathString,
                api.GetTorsionalPatchRadiusAttr().Get(),
                api.GetMinTorsionalPatchRadiusAttr().Get(),
            )
        )
    state["finger_torsional_patch"] = torsional_state

    state["benchmark_material_profile"] = _material_profile(material_prim)
    return state


def _restore_configuration(stage, state: Dict[str, object]) -> None:
    if stage.GetPrimAtPath(CYLINDER_PATH):
        stage.RemovePrim(CYLINDER_PATH)
    for path, value in state.get("finger_mesh_approximations", []):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        api = (
            UsdPhysics.MeshCollisionAPI(prim)
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            else UsdPhysics.MeshCollisionAPI.Apply(prim)
        )
        if value is None:
            api.GetApproximationAttr().Clear()
        else:
            api.CreateApproximationAttr().Set(value)
    for path, value in state.get("colliders", []):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(bool(value))
    for path, value in state.get("nonfinger_robot_colliders", []):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(bool(value))
    for path, value in state.get("visibility", []):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(value)
    for path, value in state.get("joints", []):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            UsdPhysics.Joint(prim).CreateJointEnabledAttr().Set(bool(value))
    for path, value in state.get("rigid_gravity", []):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            _set_attr(
                prim,
                "physxRigidBody:disableGravity",
                Sdf.ValueTypeNames.Bool,
                bool(value) if value is not None else False,
            )
    bottle_prim = stage.GetPrimAtPath(BOTTLE_PATH)
    mass_values = state.get("bottle_mass_properties", {})
    if bottle_prim and bottle_prim.IsValid():
        mass_api = UsdPhysics.MassAPI.Apply(bottle_prim)
        mass_attributes = {
            "mass": mass_api.CreateMassAttr(),
            "density": mass_api.CreateDensityAttr(),
            "center_of_mass": mass_api.CreateCenterOfMassAttr(),
            "diagonal_inertia": mass_api.CreateDiagonalInertiaAttr(),
            "principal_axes": mass_api.CreatePrincipalAxesAttr(),
        }
        for name, attribute in mass_attributes.items():
            value = mass_values.get(name)
            if value is None:
                attribute.Clear()
            else:
                attribute.Set(value)
    for path, radius, minimum_radius in state.get("finger_torsional_patch", []):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            if radius is None:
                api.GetTorsionalPatchRadiusAttr().Clear()
            else:
                api.CreateTorsionalPatchRadiusAttr().Set(float(radius))
            if minimum_radius is None:
                api.GetMinTorsionalPatchRadiusAttr().Clear()
            else:
                api.CreateMinTorsionalPatchRadiusAttr().Set(float(minimum_radius))
    material_prim = stage.GetPrimAtPath(GRIPPER_MATERIAL_PATH)
    profile = state.get("material_profile", {})
    if material_prim and material_prim.IsValid() and profile:
        _set_attr(
            material_prim,
            "physics:staticFriction",
            Sdf.ValueTypeNames.Float,
            float(profile["static_friction"]),
        )
        _set_attr(
            material_prim,
            "physics:dynamicFriction",
            Sdf.ValueTypeNames.Float,
            float(profile["dynamic_friction"]),
        )
        _set_attr(
            material_prim,
            "physxMaterial:frictionCombineMode",
            Sdf.ValueTypeNames.Token,
            str(profile["friction_combine_mode"]),
        )
        _set_attr(
            material_prim,
            "physxMaterial:compliantContactAccelerationSpring",
            Sdf.ValueTypeNames.Bool,
            bool(profile["compliant_contact_acceleration_spring"]),
        )
        _set_attr(
            material_prim,
            "physxMaterial:compliantContactStiffness",
            Sdf.ValueTypeNames.Float,
            float(profile["compliant_contact_stiffness"]),
        )
        _set_attr(
            material_prim,
            "physxMaterial:compliantContactDamping",
            Sdf.ValueTypeNames.Float,
            float(profile["compliant_contact_damping"]),
        )


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _aggregate(trials: List[dict]) -> dict:
    passed = [trial for trial in trials if trial.get("workflow_status") == "PASS"]
    scored = [trial for trial in passed if "axis_metrics" in trial]
    tilt = np.asarray(
        [trial["axis_metrics"]["axis_direction_change_deg"] for trial in scored],
        dtype=np.float64,
    )
    elevation = np.asarray(
        [trial["axis_metrics"]["axis_elevation_from_table_deg"] for trial in scored],
        dtype=np.float64,
    )
    mimic = np.asarray(
        [trial.get("hold_final_mimic_residual_m", np.nan) for trial in scored],
        dtype=np.float64,
    )
    axial = np.asarray(
        [
            trial.get("contact_hold", {}).get("left_right_axial_centroid_mismatch_m")
            for trial in scored
            if trial.get("contact_hold", {}).get("left_right_axial_centroid_mismatch_m")
            is not None
        ],
        dtype=np.float64,
    )
    total_hold_rotation = np.asarray(
        [
            trial.get("hold_orientation_metrics", {}).get("total_rotation_deg", np.nan)
            for trial in scored
        ],
        dtype=np.float64,
    )
    finger_line_hold_rotation = np.asarray(
        [
            abs(
                trial.get("hold_orientation_metrics", {}).get(
                    "signed_rotation_about_finger_line_deg", np.nan
                )
            )
            for trial in scored
        ],
        dtype=np.float64,
    )
    bilateral = np.asarray(
        [trial.get("hold_bilateral_contact_fraction", np.nan) for trial in scored],
        dtype=np.float64,
    )

    def stats(values):
        values = values[np.isfinite(values)] if values.size else values
        if not values.size:
            return None
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    strict_passes = [trial for trial in scored if trial.get("strict_quality_pass")]
    return {
        "trial_count": len(trials),
        "workflow_pass_count": len(passed),
        "strict_quality_pass_count": len(strict_passes),
        "axis_direction_change_deg": stats(tilt),
        "axis_elevation_from_table_deg": stats(elevation),
        "mimic_residual_m": stats(mimic),
        "axial_contact_centroid_mismatch_m": stats(axial),
        "total_hold_rotation_deg": stats(total_hold_rotation),
        "abs_finger_line_hold_rotation_deg": stats(finger_line_hold_rotation),
        "bilateral_contact_fraction": stats(bilateral),
    }


async def run_benchmark(owner, request: dict, result_path: str) -> None:
    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("no active USD Stage")
    if timeline.is_playing():
        timeline.pause()
        for _ in range(3):
            await app.next_update_async()

    approximation_request = request.get("finger_approximations")
    approximation_sweep = approximation_request is not None
    patch_radii_request = request.get("torsional_patch_radii_m")
    patch_sweep = patch_radii_request is not None
    if approximation_sweep and patch_sweep:
        raise RuntimeError("finger approximation and torsional patch sweeps are mutually exclusive")
    frequencies = [int(value) for value in request.get("physics_hz", [50, 100, 200])]
    approximations = []
    repeats = int(
        request.get(
            "repeats_per_approximation"
            if approximation_sweep
            else ("repeats_per_patch" if patch_sweep else "repeats_per_frequency"),
            10,
        )
    )
    if approximation_sweep:
        approximations = [str(value) for value in approximation_request]
        expected_approximations = ["convexHull", "convexDecomposition"]
        if approximations != expected_approximations:
            raise RuntimeError(
                f"finger approximation sweep must be exactly {expected_approximations}"
            )
        if frequencies != [200]:
            raise RuntimeError("finger approximation sweep requires physics_hz=[200]")
        patch_radii = []
    elif patch_sweep:
        patch_radii = [float(value) for value in patch_radii_request]
        expected_patch_radii = [0.0, 0.0025, 0.005, 0.010]
        if patch_radii != expected_patch_radii:
            raise RuntimeError(
                f"torsional patch sweep must be exactly {expected_patch_radii} m"
            )
        if frequencies != [200]:
            raise RuntimeError("torsional patch sweep requires physics_hz=[200]")
    else:
        patch_radii = []
        if frequencies != [50, 100, 200]:
            raise RuntimeError("benchmark physics_hz must be exactly [50, 100, 200]")
    if repeats < 1 or repeats > 10:
        raise RuntimeError("repeats_per_frequency must be in [1, 10]")
    center = np.asarray(request.get("center_world_m", DEFAULT_CENTER_WORLD_M), dtype=np.float64)
    if center.shape != (3,):
        raise RuntimeError("center_world_m must contain three values")
    yaw_deg = float(request.get("yaw_deg", DEFAULT_YAW_DEG))

    # The startup loader's local World object may already have been garbage
    # collected in a long-running streaming process.  Reuse the authored
    # PhysicsScene directly without applying PhysicsContext defaults; doing so
    # preserves gravity, solver type, GPU dynamics, and stabilization.
    physics_context = PhysicsContext(set_defaults=False)
    original_physics_dt = float(physics_context.get_physics_dt())
    original_rendering_dt = 1.0 / CONTROL_HZ
    report = {
        "status": "RUNNING",
        "classification": (
            "ALOHA_HORIZONTAL_CYLINDER_ONE_THIRD_CONVEX_DECOMPOSITION_AB"
            if approximation_sweep
            else (
                "ALOHA_HORIZONTAL_CYLINDER_TORSIONAL_PATCH_SWEEP"
                if patch_sweep
                else "ALOHA_HORIZONTAL_DYNAMIC_CYLINDER_GRASP_BENCHMARK"
            )
        ),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "stage_saved": False,
        "ros_used": False,
        "real_robot_touched": False,
        "horizontal_on_table": True,
        "cylinder": {
            "geometry": "UsdGeom.Cylinder analytic collider",
            "axis_local": "Z",
            "length_m": CYLINDER_LENGTH_M,
            "radius_m": CYLINDER_RADIUS_M,
            "mass_kg": CYLINDER_MASS_KG,
            "center_of_mass_local_m": [0.0, 0.0, CYLINDER_CENTER_LOCAL_Z_M],
            "diagonal_inertia_kg_m2": [
                CYLINDER_INERTIA_RADIAL_KG_M2,
                CYLINDER_INERTIA_RADIAL_KG_M2,
                CYLINDER_INERTIA_AXIAL_KG_M2,
            ],
            "principal_axes_wxyz": [1.0, 0.0, 0.0, 0.0],
            "mass_properties_explicit": True,
            "center_local_z_m": CYLINDER_CENTER_LOCAL_Z_M,
            "low_end_local_z_m": CYLINDER_LOW_END_LOCAL_Z_M,
            "grasp_station_local_z_m": GRASP_STATION_LOCAL_Z_M,
            "grasp_fraction_from_low_end": TARGET_GRASP_FRACTION_FROM_LOW_END,
            "grasp_offset_from_com_m": GRASP_STATION_LOCAL_Z_M
            - CYLINDER_CENTER_LOCAL_Z_M,
            "expected_gravity_torque_nm": CYLINDER_MASS_KG
            * 9.81
            * abs(GRASP_STATION_LOCAL_Z_M - CYLINDER_CENTER_LOCAL_Z_M),
            "centered_at_grasp_station": False,
            "center_world_m": center.tolist(),
            "yaw_deg": yaw_deg,
        },
        "control_hz": CONTROL_HZ,
        "physics_hz": frequencies,
        "torsional_patch_radii_m": patch_radii,
        "finger_approximations": approximations if approximation_sweep else None,
        "repeats_per_approximation"
        if approximation_sweep
        else ("repeats_per_patch" if patch_sweep else "repeats_per_frequency"): repeats,
        "hold_duration_s": HOLD_DURATION_S,
        "quality_gates": {
            "lift_height_min_m": 0.10,
            "axis_direction_change_max_deg": 1.0 if patch_sweep else 3.0,
            "axis_elevation_from_table_max_deg": 1.0 if patch_sweep else 3.0,
            "final_angular_speed_max_deg_s": 1.0,
            "mimic_residual_max_m": 0.0025,
            "bilateral_contact_fraction_min": 0.8,
            "axial_contact_centroid_mismatch_max_m": 0.005,
            "actual_contact_station_tolerance_m": TARGET_GRASP_AXIAL_TOLERANCE_M,
            "total_hold_rotation_max_deg": 1.0,
            "finger_line_hold_rotation_max_deg": 1.0,
        },
        "approximation_results"
        if approximation_sweep
        else ("patch_results" if patch_sweep else "frequency_results"): [],
    }
    _write_report(result_path, report)

    configuration_state = None
    collector = None
    try:
        owner._clear_bottle_visible_pose_overrides()
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            requested_restore_profile = request.get("restore_material_profile")
            if requested_restore_profile is not None:
                required_profile_keys = {
                    "static_friction",
                    "dynamic_friction",
                    "friction_combine_mode",
                    "compliant_contact_acceleration_spring",
                    "compliant_contact_stiffness",
                    "compliant_contact_damping",
                }
                missing = sorted(required_profile_keys - set(requested_restore_profile))
                if missing:
                    raise RuntimeError(
                        f"restore_material_profile missing keys: {missing}"
                    )
                _apply_material_profile(
                    stage.GetPrimAtPath(GRIPPER_MATERIAL_PATH),
                    requested_restore_profile,
                )
            configuration_state = _configure_cylinder(stage)
        report["material_before"] = configuration_state["material_profile"]
        report["benchmark_material"] = configuration_state[
            "benchmark_material_profile"
        ]
        report["collision_isolation"] = {
            "classification": "FINGERTIP_CONTACT_LAYER_NOT_END_TO_END_PATH_VALIDATION",
            "actual_finger_colliders_enabled": True,
            "disabled_nonfinger_left_robot_collider_count": len(
                configuration_state["nonfinger_robot_colliders"]
            ),
        }
        for _ in range(4):
            await app.next_update_async()

        group_specs = (
            [
                {"physics_hz": 200, "finger_approximation": approximation}
                for approximation in approximations
            ]
            if approximation_sweep
            else ([
                {"physics_hz": 200, "torsional_patch_radius_m": radius_m}
                for radius_m in patch_radii
            ]
            if patch_sweep
            else [{"physics_hz": value} for value in frequencies])
        )

        async def restart_physx_and_reload_arm() -> None:
            """Rebuild Isaac 5.1 tensor views after changing PhysicsScene dt."""
            timeline.stop()
            for _ in range(3):
                await app.next_update_async()
            timeline.play()
            for _ in range(5):
                await app.next_update_async()
            timeline.pause()
            for _ in range(3):
                await app.next_update_async()
            owner._articulation = None
            owner._load_left_arm()

        results_key = (
            "approximation_results"
            if approximation_sweep
            else ("patch_results" if patch_sweep else "frequency_results")
        )
        for group_spec in group_specs:
            physics_hz = int(group_spec["physics_hz"])
            patch_paths = None
            approximation_readback = None
            if approximation_sweep:
                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    approximation_readback = _set_finger_mesh_approximation(
                        stage, str(group_spec["finger_approximation"])
                    )
            elif patch_sweep:
                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    patch_paths = _set_torsional_patch_radius(
                        stage, float(group_spec["torsional_patch_radius_m"])
                    )
            physics_context.set_physics_dt(
                dt=1.0 / float(physics_hz),
                substeps=max(int(round(float(physics_hz) / CONTROL_HZ)), 1),
            )
            # Both the dt and collider approximation are consumed by PhysX at
            # scene creation.  A full Stop -> Play cycle is required in 5.1;
            # merely pausing and replacing the Python wrapper leaves its
            # backend view uninitialized (the previous experiment's fault).
            await restart_physx_and_reload_arm()
            if collector is not None:
                collector.close()
            owner._ensure_grasp_contact_monitor()
            body = SingleRigidPrim(
                BOTTLE_PATH,
                name=f"horizontal_cylinder_benchmark_body_{physics_hz}",
                reset_xform_properties=False,
            )
            body.initialize()
            collector = _ContactCollector(owner, body)
            group_result = {
                "requested_physics_hz": physics_hz,
                "actual_physics_dt_s": float(physics_context.get_physics_dt()),
                "rendering_dt_s": original_rendering_dt,
                "trials": [],
            }
            if patch_sweep:
                group_result["torsional_patch_radius_m"] = float(
                    group_spec["torsional_patch_radius_m"]
                )
                group_result["torsional_patch_prim_paths"] = patch_paths
            if approximation_sweep:
                group_result["finger_approximation"] = str(
                    group_spec["finger_approximation"]
                )
                group_result["finger_approximation_readback"] = approximation_readback
            report[results_key].append(group_result)
            for trial_index in range(1, repeats + 1):
                trial = {
                    "trial": trial_index,
                    "requested_physics_hz": physics_hz,
                    "status": "RUNNING",
                }
                if patch_sweep:
                    trial["torsional_patch_radius_m"] = float(
                        group_spec["torsional_patch_radius_m"]
                    )
                if approximation_sweep:
                    trial["finger_approximation"] = str(
                        group_spec["finger_approximation"]
                    )
                group_result["trials"].append(trial)
                _write_report(result_path, report)
                try:
                    collector.reset_trial()
                    owner._requested_random_pose_override = (
                        center.copy(),
                        math.radians(yaw_deg),
                    )
                    await owner._randomize_bottle_transaction()
                    random_result = _load_json(
                        os.path.join(
                            os.path.dirname(result_path),
                            "random_bottle_pose_latest.json",
                        )
                    )
                    if random_result.get("status") != "PASS":
                        raise RuntimeError(
                            f"fixed horizontal placement failed: {random_result.get('error')}"
                        )
                    initial_readback = random_result["accepted_pose"]["component_readback"][
                        BOTTLE_PATH
                    ]
                    initial_orientation = initial_readback["orientation_wxyz"]
                    trial["initial_component_readback"] = initial_readback

                    await owner._auto_grasp_lift_transaction()
                    grasp_result = _load_json(
                        os.path.join(
                            os.path.dirname(result_path),
                            "auto_random_grasp_lift_latest.json",
                        )
                    )
                    trial["workflow_status"] = grasp_result.get("status")
                    trial["workflow_error"] = grasp_result.get("error")
                    trial["lift_delta_m"] = grasp_result.get("bottle_lift_delta_m")
                    trial["close_readback"] = grasp_result.get("close_readback")
                    trial["post_lift_mimic_residual_m"] = grasp_result.get(
                        "final_mimic_residual_m"
                    )
                    if grasp_result.get("status") != "PASS":
                        raise RuntimeError(
                            f"grasp/lift workflow failed: {grasp_result.get('error')}"
                        )

                    collector.begin_hold()
                    hold_start_position, hold_start_orientation = body.get_world_pose()
                    # In the validated top-down grasp, the finger-closing line
                    # is perpendicular to both the horizontal Cylinder axis
                    # and world up.  Use that torque axis directly; creating
                    # independent rigid tensor views for articulation links can
                    # invalidate the owner's articulation backend.
                    hold_start_cylinder_axis = _quat_rotation(
                        hold_start_orientation
                    )[:, 2]
                    hold_start_finger_line = np.cross(
                        hold_start_cylinder_axis,
                        np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
                    )
                    if float(np.linalg.norm(hold_start_finger_line)) <= 1.0e-6:
                        raise RuntimeError("horizontal Cylinder does not define a finger torque axis")
                    hold_start_finger_line /= float(np.linalg.norm(hold_start_finger_line))
                    timeline.play()
                    hold_updates = 0
                    hold_contact_samples = []
                    hold_mimic_samples = []
                    try:
                        while collector.hold_elapsed_s < HOLD_DURATION_S:
                            await app.next_update_async()
                            hold_updates += 1
                            if hold_updates > 1000:
                                raise RuntimeError("2 s hold observation timed out")
                            _, left_index, right_index, positions = owner._get_gripper_state()
                            hold_mimic_samples.append(
                                abs(float(positions[left_index]) + float(positions[right_index]))
                            )
                            hold_contact_samples.append(
                                bool(owner._grasp_left_contact and owner._grasp_right_contact)
                            )
                    finally:
                        timeline.pause()
                    for _ in range(3):
                        await app.next_update_async()
                    hold_final_position, hold_final_orientation = body.get_world_pose()
                    _, left_index, right_index, final_positions = owner._get_gripper_state()
                    final_mimic = abs(
                        float(final_positions[left_index])
                        + float(final_positions[right_index])
                    )
                    contact_hold = collector.summarize_hold()
                    axis_metrics = _axis_metrics(
                        initial_orientation, hold_final_orientation
                    )
                    table_to_hold_orientation = _full_orientation_metrics(
                        initial_orientation,
                        hold_start_orientation,
                        hold_start_finger_line,
                    )
                    hold_orientation = _full_orientation_metrics(
                        hold_start_orientation,
                        hold_final_orientation,
                        hold_start_finger_line,
                    )
                    lift_delta = np.asarray(
                        grasp_result["bottle_lift_delta_m"], dtype=np.float64
                    )
                    bilateral_fraction = (
                        float(np.mean(hold_contact_samples))
                        if hold_contact_samples
                        else 0.0
                    )
                    axial_mismatch = contact_hold[
                        "left_right_axial_centroid_mismatch_m"
                    ]
                    contact_station_errors = {}
                    for side in ("left", "right"):
                        centroid = contact_hold["counts"][side][
                            "impulse_weighted_centroid_cylinder_local_m"
                        ]
                        contact_station_errors[side] = (
                            None
                            if centroid is None
                            else abs(float(centroid[2]) - GRASP_STATION_LOCAL_Z_M)
                        )
                    try:
                        final_angular_velocity = np.asarray(
                            body.get_angular_velocity(), dtype=np.float64
                        )
                        final_angular_speed_deg_s = math.degrees(
                            float(np.linalg.norm(final_angular_velocity))
                        )
                    except Exception:
                        final_angular_velocity = np.full(3, np.nan, dtype=np.float64)
                        final_angular_speed_deg_s = float("nan")
                    angle_gate_deg = 1.0 if patch_sweep else 3.0
                    strict_checks = {
                        "lift_height": float(lift_delta[2]) >= 0.10,
                        "axis_direction": axis_metrics[
                            "axis_direction_change_deg"
                        ]
                        <= angle_gate_deg,
                        "axis_elevation": axis_metrics[
                            "axis_elevation_from_table_deg"
                        ]
                        <= angle_gate_deg,
                        "final_angular_speed": (
                            math.isfinite(final_angular_speed_deg_s)
                            and final_angular_speed_deg_s <= 1.0
                        ),
                        "mimic": final_mimic <= 0.0025,
                        "bilateral_contact_fraction": bilateral_fraction >= 0.8,
                        "axial_contact_centroid": axial_mismatch is not None
                        and axial_mismatch <= 0.005,
                        "actual_contact_station": all(
                            error is not None
                            and error <= TARGET_GRASP_AXIAL_TOLERANCE_M
                            for error in contact_station_errors.values()
                        ),
                        "total_hold_rotation": hold_orientation["total_rotation_deg"]
                        <= 1.0,
                        "finger_line_hold_rotation": abs(
                            hold_orientation["signed_rotation_about_finger_line_deg"]
                        )
                        <= 1.0,
                    }
                    trial.update(
                        {
                            "status": "COMPLETE",
                            "hold_updates": hold_updates,
                            "hold_start_position_m": np.asarray(
                                hold_start_position, dtype=np.float64
                            ).tolist(),
                            "hold_start_orientation_wxyz": _quat_normalize(
                                hold_start_orientation
                            ).tolist(),
                            "hold_start_finger_line_world": hold_start_finger_line.tolist(),
                            "hold_final_position_m": np.asarray(
                                hold_final_position, dtype=np.float64
                            ).tolist(),
                            "hold_position_delta_m": (
                                np.asarray(hold_final_position, dtype=np.float64)
                                - np.asarray(hold_start_position, dtype=np.float64)
                            ).tolist(),
                            "hold_final_orientation_wxyz": _quat_normalize(
                                hold_final_orientation
                            ).tolist(),
                            "axis_metrics": axis_metrics,
                            "table_to_hold_orientation_metrics": table_to_hold_orientation,
                            "hold_orientation_metrics": hold_orientation,
                            "hold_final_angular_velocity_rad_s": (
                                final_angular_velocity.tolist()
                            ),
                            "hold_final_angular_speed_deg_s": (
                                final_angular_speed_deg_s
                            ),
                            "hold_final_mimic_residual_m": final_mimic,
                            "hold_mimic_residual_median_m": float(
                                np.median(np.asarray(hold_mimic_samples))
                            )
                            if hold_mimic_samples
                            else None,
                            "hold_bilateral_contact_fraction": bilateral_fraction,
                            "contact_hold": contact_hold,
                            "contact_station_error_m": contact_station_errors,
                            "strict_quality_checks": strict_checks,
                            "strict_quality_pass": all(strict_checks.values()),
                        }
                    )
                except Exception as exc:
                    trial.update(
                        {
                            "status": "EXCEPTION",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc().splitlines()[-30:],
                        }
                    )
                finally:
                    timeline.pause()
                    owner._requested_random_pose_override = None
                    # A failed lift can leave the cylinder between the fingers.
                    # Move the complete rigid assembly back to the table while
                    # it is kinematic before the next trial tries to open the
                    # gripper.  Without this recovery, trial 1 can physically
                    # obstruct every later trial and make them non-independent.
                    try:
                        owner._clear_bottle_visible_pose_overrides()
                        # Move the kinematic test assembly away before opening
                        # or resetting.  A failed close can otherwise leave the
                        # Cylinder wedged between decomposed finger shapes and
                        # contaminate every later repeat.
                        await owner._place_bottle_assembly(
                            np.asarray([0.0, 0.0, center[2]], dtype=np.float64),
                            0.0,
                        )
                        await owner._open_left_gripper_transaction()
                        await owner._reset_left_sleep_from_button()
                        await owner._place_bottle_assembly(
                            center.copy(), math.radians(yaw_deg)
                        )
                        trial["inter_trial_recovery"] = "PASS"
                    except Exception as recovery_exc:
                        trial["inter_trial_recovery"] = "EXCEPTION"
                        trial["inter_trial_recovery_error"] = (
                            f"{type(recovery_exc).__name__}: {recovery_exc}"
                        )
                    group_result["aggregate"] = _aggregate(
                        group_result["trials"]
                    )
                    _write_report(result_path, report)

        report["status"] = "COMPLETE"
        report["completed_at"] = datetime.now().isoformat(timespec="seconds")
    except BaseException as exc:
        report["status"] = "EXCEPTION"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc().splitlines()[-50:]
    finally:
        timeline.pause()
        owner._requested_random_pose_override = None
        if collector is not None:
            collector.close()
        try:
            physics_context.set_physics_dt(
                dt=original_physics_dt,
                substeps=1,
            )
            report["restored_physics_dt_s"] = float(
                physics_context.get_physics_dt()
            )
            report["restored_rendering_dt_s"] = original_rendering_dt
        except Exception as exc:
            report["dt_restore_error"] = f"{type(exc).__name__}: {exc}"
        if configuration_state is not None:
            try:
                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    _restore_configuration(stage, configuration_state)
                report["material_after_restore"] = _material_profile(
                    stage.GetPrimAtPath(GRIPPER_MATERIAL_PATH)
                )
                report["configuration_restored"] = True
            except Exception as exc:
                report["configuration_restored"] = False
                report["configuration_restore_error"] = f"{type(exc).__name__}: {exc}"
        try:
            await restart_physx_and_reload_arm()
            owner._set_bottle_assembly_kinematic(True)
            await owner._open_left_gripper_transaction()
            await owner._reset_left_sleep_from_button()
            await owner._reset_bottle_initial_pose_transaction()
            report["scene_reset_attempted"] = True
        except Exception as exc:
            report["scene_reset_attempted"] = False
            report["scene_reset_error"] = f"{type(exc).__name__}: {exc}"
        report["timeline_paused"] = not timeline.is_playing()
        report["stage_saved"] = False
        report["ros_used"] = False
        report["real_robot_touched"] = False
        _write_report(result_path, report)
