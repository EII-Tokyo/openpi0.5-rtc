#!/usr/bin/env python3
"""Build a computed root-frame-only supplier-CAD Task 5 diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_STAGE = (
    ROOT
    / "local_eval_assets/aloha_isaac_assets/aloha_viperx.usd"
)
BASELINE_STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_finger_task5_convex_hull/aloha_viperx_supplier_cad_task5.usda"
)
OUTPUT_ROOT = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_finger_task5_root_frame_only"
)
OUTPUT_LAYER = (
    OUTPUT_ROOT
    / "configuration/supplier_cad_root_frame_only.usda"
)
OUTPUT_STAGE = OUTPUT_ROOT / "aloha_viperx_supplier_cad_root_frame_only.usda"
OUTPUT_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_root_frame_asset.json"
)
SOURCE_SHA256 = (
    "b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e"
)
BASELINE_SHA256 = (
    "8040edd01859af9f8c51285d198d34aae19e66625a2d5f21729879774e1644d9"
)
BODY_PATH = "/workcell/vx300s_left/vx300s_left"
JOINT_PATH = "/workcell/joints/rootJoint_vx300s_left"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exact(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(
                f"refusing to overwrite drifted diagnostic: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    source = SOURCE_STAGE.resolve(strict=True)
    baseline = BASELINE_STAGE.resolve(strict=True)
    source_before = _sha256(source)
    baseline_before = _sha256(baseline)
    if source_before != SOURCE_SHA256:
        raise RuntimeError("approved source Stage hash mismatch")
    if baseline_before != BASELINE_SHA256:
        raise RuntimeError("baseline diagnostic Stage hash mismatch")
    stage = Usd.Stage.Open(str(source), Usd.Stage.LoadAll)
    joint = UsdPhysics.Joint(stage.GetPrimAtPath(JOINT_PATH))
    body1_targets = joint.GetBody1Rel().GetTargets()
    if [str(path) for path in body1_targets] != [BODY_PATH]:
        raise RuntimeError("root fixed-joint body1 mismatch")
    if joint.GetBody0Rel().GetTargets():
        raise RuntimeError("root fixed-joint body0 is not the world")
    body_transform = UsdGeom.XformCache().GetLocalToWorldTransform(
        stage.GetPrimAtPath(BODY_PATH)
    )
    position = [float(value) for value in body_transform.ExtractTranslation()]
    quaternion = body_transform.ExtractRotationQuat()
    rotation = [
        float(quaternion.GetReal()),
        *[float(value) for value in quaternion.GetImaginary()],
    ]
    local_pos1 = [float(value) for value in joint.GetLocalPos1Attr().Get()]
    local_rot1 = joint.GetLocalRot1Attr().Get()
    local_rot1_values = [
        float(local_rot1.GetReal()),
        *[float(value) for value in local_rot1.GetImaginary()],
    ]
    if local_pos1 != [0.0, 0.0, 0.0]:
        raise RuntimeError("nonzero body1 joint frame is unsupported")
    if local_rot1_values != [1.0, 0.0, 0.0, 0.0]:
        raise RuntimeError("nonidentity body1 joint frame is unsupported")
    mismatch = float(sum(value * value for value in position) ** 0.5)
    layer_text = f"""#usda 1.0

over "workcell"
{{
    over "joints"
    {{
        over "rootJoint_vx300s_left"
        {{
            point3f physics:localPos0 = ({position[0]:.17g}, {position[1]:.17g}, {position[2]:.17g})
            quatf physics:localRot0 = ({rotation[0]:.17g}, {rotation[1]:.17g}, {rotation[2]:.17g}, {rotation[3]:.17g})
        }}
    }}
}}
"""
    stage_text = """#usda 1.0
(
    defaultPrim = "workcell"
    metersPerUnit = 1
    subLayers = [
        @configuration/supplier_cad_root_frame_only.usda@
    ]
    upAxis = "Z"
)

def Xform "workcell" (
    prepend references = @../cad_finger_task5_convex_hull/aloha_viperx_supplier_cad_task5.usda@</workcell>
)
{
}
"""
    _write_exact(OUTPUT_LAYER, layer_text)
    _write_exact(OUTPUT_STAGE, stage_text)
    source_after = _sha256(source)
    baseline_after = _sha256(baseline)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "profile": "DIAGNOSTIC_ONLY_NOT_FINAL_FRAME_MAPPING",
        "changed_joint": JOINT_PATH,
        "changed_variables": [
            "physics:localPos0",
            "physics:localRot0",
        ],
        "computed_frame": {
            "body0": "WORLD",
            "body1": BODY_PATH,
            "body1_world_position_m": position,
            "body1_world_orientation_wxyz": rotation,
            "initial_translation_mismatch_m": mismatch,
            "method": "UsdGeom.XformCache.GetLocalToWorldTransform",
        },
        "frozen": {
            "all_drive_attributes": True,
            "all_collision_attributes": True,
            "all_material_attributes": True,
            "physics_frequency": True,
            "solver_iterations": True,
            "bottle": "NOT_PRESENT",
        },
        "inputs": {
            "source": {
                "absolute_path": str(source),
                "sha256_before": source_before,
                "sha256_after": source_after,
            },
            "baseline": {
                "absolute_path": str(baseline),
                "sha256_before": baseline_before,
                "sha256_after": baseline_after,
            },
        },
        "outputs": {
            "configuration_layer": {
                "absolute_path": str(OUTPUT_LAYER.resolve()),
                "sha256": _sha256(OUTPUT_LAYER),
            },
            "diagnostic_stage": {
                "absolute_path": str(OUTPUT_STAGE.resolve()),
                "sha256": _sha256(OUTPUT_STAGE),
            },
        },
        "gates": {
            "source_immutable": source_before == source_after == SOURCE_SHA256,
            "baseline_immutable": (
                baseline_before == baseline_after == BASELINE_SHA256
            ),
            "frame_computed_not_guessed": mismatch > 0.0,
            "only_root_joint_frame_authored": True,
            "default_or_final_asset_unchanged": True,
        },
        "scope": {
            "bottle_contact_grasp": "NOT_RUN",
            "task8": "NOT_RUN",
        },
    }
    if not all(report["gates"].values()):
        report["status"] = "FAIL"
    OUTPUT_REPORT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"status={report['status']}")
    print(f"position={position}")
    print(f"rotation={rotation}")
    print(f"stage={OUTPUT_STAGE.resolve()}")
    print(f"report={OUTPUT_REPORT.resolve()}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
