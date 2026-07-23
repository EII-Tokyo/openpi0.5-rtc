"""Pure OpenUSD helpers for A19 joint-state/body-XForm coherence."""

from __future__ import annotations

import math
from typing import Any

from pxr import Gf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdPhysics

POSITION_TOLERANCE_M = 1.0e-6
ORIENTATION_TOLERANCE_DEG = 1.0e-4
_JOINT_TYPES = {
    "PhysicsFixedJoint",
    "PhysicsRevoluteJoint",
    "PhysicsPrismaticJoint",
}
_AXES = {
    "X": Gf.Vec3d(1.0, 0.0, 0.0),
    "Y": Gf.Vec3d(0.0, 1.0, 0.0),
    "Z": Gf.Vec3d(0.0, 0.0, 1.0),
}


def _finite(value: object, *, label: str, joint_path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{joint_path}: invalid {label}: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{joint_path}: non-finite {label}: {value!r}")
    return result


def _single_body_target(
    joint: UsdPhysics.Joint, side: int, *, required: bool
) -> str | None:
    path = str(joint.GetPrim().GetPath())
    targets = (
        joint.GetBody0Rel().GetTargets()
        if side == 0
        else joint.GetBody1Rel().GetTargets()
    )
    if len(targets) > 1 or (required and len(targets) != 1):
        raise ValueError(
            f"{path}: expected {'one' if required else 'zero or one'} body{side} "
            f"target, got {list(targets)}"
        )
    return str(targets[0]) if targets else None


def _local_transform(joint: UsdPhysics.Joint, side: int) -> Gf.Matrix4d:
    prim = joint.GetPrim()
    path = str(prim.GetPath())
    position_attribute = (
        joint.GetLocalPos0Attr() if side == 0 else joint.GetLocalPos1Attr()
    )
    rotation_attribute = (
        joint.GetLocalRot0Attr() if side == 0 else joint.GetLocalRot1Attr()
    )
    position = position_attribute.Get()
    rotation = rotation_attribute.Get()
    if position is None or rotation is None:
        raise ValueError(f"{path}: missing body{side} local joint frame")
    values = [
        *position,
        rotation.GetReal(),
        *rotation.GetImaginary(),
    ]
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"{path}: non-finite body{side} local joint frame")
    matrix = Gf.Matrix4d(1.0)
    matrix.SetTranslate(Gf.Vec3d(position))
    quaternion = Gf.Quatd(
        float(rotation.GetReal()), Gf.Vec3d(rotation.GetImaginary())
    ).GetNormalized()
    matrix.SetRotateOnly(quaternion)
    return matrix


def _body_world_transform(
    stage: Usd.Stage,
    cache: UsdGeom.XformCache,
    target: str | None,
    *,
    joint_path: str,
) -> Gf.Matrix4d:
    if target is None:
        return Gf.Matrix4d(1.0)
    prim = stage.GetPrimAtPath(target)
    if not prim.IsValid():
        raise ValueError(f"{joint_path}: missing body prim: {target}")
    matrix = Gf.Matrix4d(cache.GetLocalToWorldTransform(prim))
    determinant = float(matrix.GetDeterminant())
    if not math.isfinite(determinant) or abs(determinant) < 1.0e-12:
        raise ValueError(f"{joint_path}: singular body transform: {target}")
    return matrix


def _motion_transform(joint: UsdPhysics.Joint) -> Gf.Matrix4d:
    prim = joint.GetPrim()
    path = str(prim.GetPath())
    joint_type = prim.GetTypeName()
    motion = Gf.Matrix4d(1.0)
    if joint_type == "PhysicsFixedJoint":
        return motion
    if joint_type not in _JOINT_TYPES:
        raise ValueError(f"{path}: unsupported joint type: {joint_type!r}")
    axis_attribute = prim.GetAttribute("physics:axis")
    axis = axis_attribute.Get() if axis_attribute.IsValid() else None
    if axis not in _AXES:
        raise ValueError(f"{path}: unsupported joint axis: {axis!r}")
    if joint_type == "PhysicsRevoluteJoint":
        attribute = prim.GetAttribute("state:angular:physics:position")
        value = attribute.Get() if attribute.IsValid() else None
        angle_degrees = _finite(value, label="angular state", joint_path=path)
        motion.SetRotate(Gf.Rotation(_AXES[axis], angle_degrees))
        return motion
    attribute = prim.GetAttribute("state:linear:physics:position")
    value = attribute.Get() if attribute.IsValid() else None
    distance_m = _finite(value, label="linear state", joint_path=path)
    motion.SetTranslate(_AXES[axis] * distance_m)
    return motion


def _orientation_error_degrees(
    expected: Gf.Matrix4d, observed: Gf.Matrix4d
) -> float:
    left = expected.ExtractRotationQuat().GetNormalized()
    right = observed.ExtractRotationQuat().GetNormalized()
    left_values = [left.GetReal(), *left.GetImaginary()]
    right_values = [right.GetReal(), *right.GetImaginary()]
    dot = abs(
        sum(
            float(left_value) * float(right_value)
            for left_value, right_value in zip(
                left_values, right_values, strict=True
            )
        )
    )
    return math.degrees(2.0 * math.acos(min(max(dot, -1.0), 1.0)))


def _coherence_frames(
    stage: Usd.Stage, joint_prim: Usd.Prim
) -> tuple[Gf.Matrix4d, Gf.Matrix4d, Gf.Matrix4d]:
    path = str(joint_prim.GetPath())
    if joint_prim.GetTypeName() not in _JOINT_TYPES:
        raise ValueError(
            f"{path}: unsupported joint type: {joint_prim.GetTypeName()!r}"
        )
    joint = UsdPhysics.Joint(joint_prim)
    body0 = _single_body_target(joint, 0, required=False)
    body1 = _single_body_target(joint, 1, required=True)
    cache = UsdGeom.XformCache()
    body0_world = _body_world_transform(
        stage, cache, body0, joint_path=path
    )
    body1_world = _body_world_transform(
        stage, cache, body1, joint_path=path
    )
    desired_body1_world_frame = (
        _motion_transform(joint) * _local_transform(joint, 0) * body0_world
    )
    observed_body1_world_frame = _local_transform(joint, 1) * body1_world
    return desired_body1_world_frame, observed_body1_world_frame, body1_world


def measure_joint_state_coherence(
    stage: Usd.Stage, joint_prim: Usd.Prim
) -> dict[str, object]:
    """Measure current body1 joint frame against body0 plus authored state."""
    desired, observed, _body1_world = _coherence_frames(stage, joint_prim)
    position_residual = float(
        (
            desired.ExtractTranslation() - observed.ExtractTranslation()
        ).GetLength()
    )
    orientation_residual = _orientation_error_degrees(desired, observed)
    if not math.isfinite(position_residual) or not math.isfinite(
        orientation_residual
    ):
        raise ValueError(f"{joint_prim.GetPath()}: non-finite coherence residual")
    return {
        "joint_path": str(joint_prim.GetPath()),
        "joint_type": joint_prim.GetTypeName(),
        "position_residual_m": position_residual,
        "orientation_residual_deg": orientation_residual,
        "position_tolerance_m": POSITION_TOLERANCE_M,
        "orientation_tolerance_deg": ORIENTATION_TOLERANCE_DEG,
        "ok": position_residual <= POSITION_TOLERANCE_M
        and orientation_residual <= ORIENTATION_TOLERANCE_DEG,
    }


def repair_body1_local_frame(
    stage: Usd.Stage, joint_prim: Usd.Prim
) -> dict[str, object]:
    """Solve only body1 localPos/localRot while preserving bodies and state."""
    before = measure_joint_state_coherence(stage, joint_prim)
    desired, _observed, body1_world = _coherence_frames(stage, joint_prim)
    repaired_local1 = desired * body1_world.GetInverse()
    determinant = float(repaired_local1.GetDeterminant())
    if not math.isfinite(determinant) or abs(determinant) < 1.0e-12:
        raise ValueError(f"{joint_prim.GetPath()}: invalid repaired local frame")
    translation = repaired_local1.ExtractTranslation()
    rotation = repaired_local1.ExtractRotationQuat().GetNormalized()
    values = [*translation, rotation.GetReal(), *rotation.GetImaginary()]
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"{joint_prim.GetPath()}: non-finite repaired local frame")
    joint = UsdPhysics.Joint(joint_prim)
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(translation))
    joint.CreateLocalRot1Attr().Set(
        Gf.Quatf(
            float(rotation.GetReal()),
            Gf.Vec3f(rotation.GetImaginary()),
        )
    )
    after = measure_joint_state_coherence(stage, joint_prim)
    if not after["ok"]:
        raise ValueError(
            f"{joint_prim.GetPath()}: repaired frame outside tolerance: {after}"
        )
    return {"before": before, "after": after}


def audit_stage_joint_state_coherence(
    stage: Usd.Stage, joint_paths: list[str] | None = None
) -> dict[str, Any]:
    """Fail-closed coherence audit for an explicit inventory or all joints."""
    if joint_paths is None:
        prims = [
            prim for prim in stage.Traverse() if prim.GetTypeName() in _JOINT_TYPES
        ]
    else:
        prims = [stage.GetPrimAtPath(path) for path in joint_paths]
    records: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for prim in prims:
        path = str(prim.GetPath()) if prim.IsValid() else "<invalid>"
        try:
            records.append(measure_joint_state_coherence(stage, prim))
        except (RuntimeError, TypeError, ValueError) as exc:
            errors.append({"joint_path": path, "message": str(exc)})
    max_position = max(
        (float(record["position_residual_m"]) for record in records),
        default=math.inf,
    )
    max_orientation = max(
        (float(record["orientation_residual_deg"]) for record in records),
        default=math.inf,
    )
    ok = (
        len(records) == len(prims)
        and not errors
        and all(record["ok"] is True for record in records)
    )
    return {
        "ok": ok,
        "joint_count": len(prims),
        "records": records,
        "errors": errors,
        "max_position_residual_m": max_position,
        "max_orientation_residual_deg": max_orientation,
        "position_tolerance_m": POSITION_TOLERANCE_M,
        "orientation_tolerance_deg": ORIENTATION_TOLERANCE_DEG,
    }
