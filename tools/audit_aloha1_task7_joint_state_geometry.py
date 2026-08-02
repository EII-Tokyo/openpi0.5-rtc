#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Read-only body-transform reconciliation for Task 7 joint-state findings."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FROZEN_STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0"
    / "aloha1_cad_derived_full_body_collider_gripper_decomposition_"
    "tabletop_zero_z_up_meters_diagnostic.usda"
)
FROZEN_STAGE_SHA256 = (
    "327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9"
)
CANDIDATES = {
    "follower_left": (
        ROOT
        / "assets/Trossen/ALOHA1/1.0/diagnostics/"
        "cad_derived_task7_rule_candidates/1.0/Trossen/vx300s_left/1.0/"
        "vx300s_left.usda",
        "/vx300s_left",
    ),
    "follower_right": (
        ROOT
        / "assets/Trossen/ALOHA1/1.0/diagnostics/"
        "cad_derived_task7_rule_candidates/1.0/Trossen/vx300s_right/1.0/"
        "vx300s_right.usda",
        "/vx300s_right",
    ),
}
JOINT_NAMES = (
    "elbow",
    "left_finger",
    "right_finger",
    "shoulder",
    "wrist_angle",
)
OUTPUT_JSON = (
    ROOT
    / "reports/aloha1_mapping/aloha1_task7_joint_state_geometry_audit.json"
)
OUTPUT_MD = OUTPUT_JSON.with_suffix(".md")
RULE_SOURCE = (
    ROOT
    / ".venv_issac/lib/python3.11/site-packages/isaacsim/exts/"
    "isaacsim.asset.validation/isaacsim/asset/validation/joint_rules.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix(value: Any) -> list[list[float]]:
    matrix = value.GetMatrix() if hasattr(value, "GetMatrix") else value
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def _vec3(value: Any) -> list[float]:
    return [float(value[index]) for index in range(3)]


def _quat(value: Any) -> list[float]:
    imaginary = value.GetImaginary()
    return [
        float(value.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    ]


def _normalized_angle_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _state_transform(joint_prim: Any, position: float) -> Any:
    from pxr import Gf
    from pxr import UsdPhysics

    axis_vectors = {
        "X": Gf.Vec3d(1.0, 0.0, 0.0),
        "Y": Gf.Vec3d(0.0, 1.0, 0.0),
        "Z": Gf.Vec3d(0.0, 0.0, 1.0),
    }
    result = Gf.Transform()
    axis = str(joint_prim.GetAttribute("physics:axis").Get())
    if joint_prim.IsA(UsdPhysics.RevoluteJoint):
        result.SetRotation(Gf.Rotation(axis_vectors[axis], position))
    elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
        result.SetTranslation(axis_vectors[axis] * position)
    else:
        raise ValueError(f"unsupported joint type: {joint_prim.GetTypeName()}")
    return result


def _geometry_state(joint_prim: Any, expected_0: Any, expected_1: Any) -> float:
    from pxr import Gf
    from pxr import UsdPhysics

    axis_vectors = {
        "X": Gf.Vec3d(1.0, 0.0, 0.0),
        "Y": Gf.Vec3d(0.0, 1.0, 0.0),
        "Z": Gf.Vec3d(0.0, 0.0, 1.0),
    }
    relative_matrix = expected_1.GetMatrix() * expected_0.GetMatrix().GetInverse()
    relative = Gf.Transform(relative_matrix)
    axis = axis_vectors[str(joint_prim.GetAttribute("physics:axis").Get())]
    if joint_prim.IsA(UsdPhysics.RevoluteJoint):
        rotation = relative.GetRotation()
        signed = rotation.GetAngle()
        if Gf.Dot(rotation.GetAxis(), axis) < 0.0:
            signed = -signed
        return _normalized_angle_degrees(float(signed))
    if joint_prim.IsA(UsdPhysics.PrismaticJoint):
        return float(Gf.Dot(relative.GetTranslation(), axis))
    raise ValueError(f"unsupported joint type: {joint_prim.GetTypeName()}")


def _residual(state: Any, expected_0: Any, expected_1: Any) -> dict[str, float]:
    actual = state * expected_0
    translation = float(
        (actual.GetTranslation() - expected_1.GetTranslation()).GetLength()
    )
    actual_quat = actual.GetRotation().GetQuat().GetNormalized()
    expected_quat = expected_1.GetRotation().GetQuat().GetNormalized()
    direct = (_quat(actual_quat)[0] - _quat(expected_quat)[0]) ** 2
    direct += sum(
        (left - right) ** 2
        for left, right in zip(
            _quat(actual_quat)[1:], _quat(expected_quat)[1:], strict=True
        )
    )
    negated = (_quat(actual_quat)[0] + _quat(expected_quat)[0]) ** 2
    negated += sum(
        (left + right) ** 2
        for left, right in zip(
            _quat(actual_quat)[1:], _quat(expected_quat)[1:], strict=True
        )
    )
    return {
        "translation_m": translation,
        "quaternion_l2": float(min(direct, negated) ** 0.5),
        "validator_translation_tolerance_m": 1.0e-4,
        "validator_quaternion_tolerance": 1.0e-3,
        "finite": math.isfinite(translation),
    }


def _source_layer(prim: Any) -> dict[str, Any]:
    stack = list(prim.GetPrimStack())
    if not stack:
        raise RuntimeError(f"joint has no prim stack: {prim.GetPath()}")
    defining = stack[-1]
    return {
        "layer": str(defining.layer.identifier),
        "path": str(defining.path),
        "stack": [
            {"layer": str(spec.layer.identifier), "path": str(spec.path)}
            for spec in stack
        ],
    }


def _joint_record(stage: Any, root_path: str, name: str) -> dict[str, Any]:
    from isaacsim.asset.validation.joint_rules import get_world_body_transform
    from pxr import PhysxSchema
    from pxr import UsdGeom
    from pxr import UsdPhysics

    prim_path = f"{root_path}/joints/{name}"
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"missing joint: {prim_path}")
    joint = UsdPhysics.Joint(prim)
    cache = UsdGeom.XformCache()
    expected_0 = get_world_body_transform(stage, cache, joint, body0base=False)
    expected_1 = get_world_body_transform(stage, cache, joint, body0base=True)
    joint_type = prim.GetTypeName()
    state_axis = "angular" if prim.IsA(UsdPhysics.RevoluteJoint) else "linear"
    state_api = PhysxSchema.JointStateAPI(prim, state_axis)
    authored_position = float(state_api.GetPositionAttr().Get() or 0.0)
    geometry_position = _geometry_state(prim, expected_0, expected_1)
    drive = UsdPhysics.DriveAPI(prim, state_axis)
    target = drive.GetTargetPositionAttr().Get() if drive else None
    source = _source_layer(prim)
    residual_before = _residual(
        _state_transform(prim, authored_position),
        expected_0,
        expected_1,
    )
    residual_geometry = _residual(
        _state_transform(prim, geometry_position),
        expected_0,
        expected_1,
    )
    return {
        "rule": "JointHasCorrectTransformAndState",
        "prim_path": prim_path,
        "joint_type": joint_type,
        "axis": str(prim.GetAttribute("physics:axis").Get()),
        "body0_targets": [str(path) for path in joint.GetBody0Rel().GetTargets()],
        "body1_targets": [str(path) for path in joint.GetBody1Rel().GetTargets()],
        "local_pos0": _vec3(joint.GetLocalPos0Attr().Get()),
        "local_pos1": _vec3(joint.GetLocalPos1Attr().Get()),
        "local_rot0_wxyz": _quat(joint.GetLocalRot0Attr().Get()),
        "local_rot1_wxyz": _quat(joint.GetLocalRot1Attr().Get()),
        "expected_transform_from_body0": _matrix(expected_1),
        "expected_transform_from_body1": _matrix(expected_0),
        "authored_state_position": authored_position,
        "drive_target_position": float(target) if target is not None else None,
        "geometry_derived_state_position": geometry_position,
        "state_units": "degrees" if state_axis == "angular" else "meters",
        "residual_before": residual_before,
        "residual_geometry_candidate": residual_geometry,
        "geometry_candidate_matches_validator": (
            residual_geometry["translation_m"] <= 1.0e-4
            and residual_geometry["quaternion_l2"] <= 1.0e-3
        ),
        "source_layer": source["layer"],
        "source_prim_path": source["path"],
        "prim_stack": source["stack"],
        "usd_modified": False,
    }


def build_report() -> dict[str, Any]:
    from pxr import Usd

    frozen = FROZEN_STAGE.resolve(strict=True)
    frozen_before = _sha256(frozen)
    if frozen_before != FROZEN_STAGE_SHA256:
        raise RuntimeError("frozen Stage hash mismatch")
    joints: list[dict[str, Any]] = []
    candidates: dict[str, Any] = {}
    for follower, (candidate_path, root_path) in CANDIDATES.items():
        candidate = candidate_path.resolve(strict=True)
        before = _sha256(candidate)
        stage = Usd.Stage.Open(str(candidate), Usd.Stage.LoadAll)
        if stage is None:
            raise RuntimeError(f"unable to open {candidate}")
        follower_records = [
            {"follower": follower, **_joint_record(stage, root_path, name)}
            for name in JOINT_NAMES
        ]
        joints.extend(follower_records)
        after = _sha256(candidate)
        candidates[follower] = {
            "absolute_path": str(candidate),
            "sha256_before": before,
            "sha256_after": after,
            "modified": before != after,
        }
    frozen_after = _sha256(frozen)
    all_geometry_candidates_match = all(
        item["geometry_candidate_matches_validator"] for item in joints
    )
    return {
        "schema_version": 1,
        "status": "PASS" if all_geometry_candidates_match else "PARTIAL",
        "finding_count": len(joints),
        "stage": {
            "absolute_path": str(frozen),
            "sha256_before": frozen_before,
            "sha256_after": frozen_after,
            "modified": frozen_before != frozen_after,
        },
        "candidate_stages": candidates,
        "joints": joints,
        "all_geometry_candidates_match_validator": all_geometry_candidates_match,
        "candidate_authoring_allowed": False,
        "candidate_authoring_blocker": (
            "READ_ONLY_AUDIT_REQUIRES_SOURCE_AND_RUNTIME_COMPARISON"
        ),
        "local_rule_source": {
            "absolute_path": str(RULE_SOURCE.resolve(strict=True)),
            "sha256": _sha256(RULE_SOURCE.resolve(strict=True)),
            "class": "JointHasCorrectTransformAndState",
            "tolerances": {"translation_m": 1.0e-4, "quaternion_l2": 1.0e-3},
        },
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "asset_validation": "1.1.0",
        },
        "final_or_default_asset_modified": False,
        "task8": "NOT_RUN",
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 Task 7 joint-state geometry audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Findings: `{report['finding_count']}`",
        "- Candidate authoring: `NOT_RUN`",
        "- Task 8: `NOT_RUN`",
        "",
        "| Follower | Joint | Authored | Geometry-derived | Translation residual | Quaternion residual |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        [
            "| {follower} | `{joint}` | {authored:.9g} | {geometry:.9g} | "
            "{translation:.9g} | {rotation:.9g} |".format(
                follower=item["follower"],
                joint=Path(item["prim_path"]).name,
                authored=item["authored_state_position"],
                geometry=item["geometry_derived_state_position"],
                translation=item["residual_geometry_candidate"]["translation_m"],
                rotation=item["residual_geometry_candidate"]["quaternion_l2"],
            )
            for item in report["joints"]
        ]
    )
    lines.extend(
        [
            "",
            "The geometry-derived values reproduce the installed 1.1.0 rule's "
            "body-transform equation only. No state, drive, body transform, or "
            "asset was authored. Runtime/home/source comparison remains required "
            "before any candidate may be built.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    report = build_report()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUTPUT_MD.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(OUTPUT_JSON.resolve())}))
    return 0


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    exit_code = 1
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        extension_id = "isaacsim.asset.validation"
        if not manager.is_extension_enabled(extension_id):
            manager.set_extension_enabled_immediate(extension_id, True)  # noqa: FBT003
        if not manager.is_extension_enabled(extension_id):
            raise RuntimeError(f"required extension disabled: {extension_id}")
        exit_code = main()
    except Exception:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
