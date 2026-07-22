#!/usr/bin/env python3
"""Create a visual-only clean ALOHA1 stage baked into the verified home pose.

This script is intentionally a preview bridge between A15 visual assembly and a
future clean articulation rebuild.  It does not author clean joints, physics,
controllers, cameras, replay, or training readiness.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.aloha_isaac.scripts.open_workcell_gui import (  # noqa: E402
    _apply_real_start_pose_to_articulations,
)


DEFAULT_CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml")
DEFAULT_VALIDATION_JSON = Path(
    "aloha_isaac_rebuild/artifacts/validation/a15_home_pose_preview_audit.json"
)


def _start_isaac_headless():
    from isaacsim import SimulationApp

    return SimulationApp({"headless": True})


def _extract_home_pose_world_transforms(source_usd: Path) -> tuple[dict[str, object], dict]:
    import omni.usd
    from pxr import UsdGeom

    from aloha_isaac_rebuild.scripts.create_aloha_stationary_style_rebuild_stages import _source_components

    context = omni.usd.get_context()
    if not context.open_stage(str(source_usd)):
        raise RuntimeError(f"Isaac failed to open source stage: {source_usd}")

    _apply_real_start_pose_to_articulations()
    stage = context.get_stage()
    components = _source_components(stage)
    cache = UsdGeom.XformCache()
    transforms = {}
    records = []
    for component in components:
        prim = stage.GetPrimAtPath(component.source_component_path)
        if not prim.IsValid():
            raise RuntimeError(f"Missing source component after home pose: {component.source_component_path}")
        matrix = cache.GetLocalToWorldTransform(prim)
        transforms[component.source_component_path] = matrix
        translation = matrix.ExtractTranslation()
        rotation = matrix.ExtractRotationQuat().GetNormalized()
        records.append(
            {
                "source_component_path": component.source_component_path,
                "resource_name": component.resource_name,
                "assembly_link_path": component.assembly_link_path,
                "translation_m": [float(value) for value in translation],
                "rotation_quat_wxyz": [
                    float(rotation.GetReal()),
                    *(float(value) for value in rotation.GetImaginary()),
                ],
            }
        )
    return transforms, {
        "source_usd": str(source_usd),
        "pose": "REAL_RUNTIME_RESET_POSE",
        "component_count": len(records),
        "records": records,
    }


def _quat_angle_error_deg(left: object, right: object) -> float:
    import math

    left = left.GetNormalized()
    right = right.GetNormalized()
    left_vec = [float(left.GetReal()), *(float(value) for value in left.GetImaginary())]
    right_vec = [float(right.GetReal()), *(float(value) for value in right.GetImaginary())]
    dot = abs(sum(a * b for a, b in zip(left_vec, right_vec, strict=True)))
    dot = min(max(dot, -1.0), 1.0)
    return math.degrees(2.0 * math.acos(dot))


def create_home_pose_preview(config_path: Path, validation_json: Path) -> dict:
    from pxr import Sdf, UsdGeom

    from aloha_isaac_rebuild.scripts.create_aloha_stationary_style_rebuild_stages import (
        _build_visual_assembly,
        _define_stage,
        _reference_path,
        _source_components,
        _write_resource_libraries,
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = (REPO_ROOT / config["source_aloha1_usd"]).resolve()
    output_path = REPO_ROOT / config["outputs"].get(
        "a15_home_pose_preview",
        "aloha_isaac_rebuild/scenes/a15_aloha_home_pose_preview.usda",
    )
    output_path = output_path.resolve()

    transforms, transform_report = _extract_home_pose_world_transforms(source_path)

    import omni.usd

    source_stage = omni.usd.get_context().get_stage()
    components = _source_components(source_stage)
    source_ref = _reference_path(output_path, source_path)
    stage, root = _define_stage(output_path, "A15_home_pose_visual_preview", config)
    _write_resource_libraries(
        stage,
        source_stage,
        source_ref,
        components,
        include_visuals=True,
        include_colliders=False,
        include_mesh_catalog=True,
    )
    _build_visual_assembly(
        stage,
        source_stage,
        components,
        include_collisions=False,
        component_world_transforms=transforms,
    )
    root.SetCustomDataByKey("acceptance", "visual-only home pose preview; clean articulation not authored")
    root.SetCustomDataByKey("home_pose_transform_report_json", json.dumps(transform_report, indent=2, sort_keys=True))
    root.CreateAttribute("aloha:visualOnly", Sdf.ValueTypeNames.Bool).Set(True)
    root.CreateAttribute("aloha:posePreview", Sdf.ValueTypeNames.Bool).Set(True)
    root.CreateAttribute("aloha:homePoseBaked", Sdf.ValueTypeNames.Bool).Set(True)
    root.CreateAttribute("aloha:cleanArticulationAuthored", Sdf.ValueTypeNames.Bool).Set(False)

    cache = UsdGeom.XformCache()
    residual_records = []
    max_translation_residual_m = 0.0
    max_rotation_residual_deg = 0.0
    for component in components:
        if component.assembly_link_path is None:
            continue
        link = stage.GetPrimAtPath(component.assembly_link_path)
        if not link.IsValid():
            raise RuntimeError(f"Missing baked clean link: {component.assembly_link_path}")
        link.CreateAttribute("aloha:visualOnly", Sdf.ValueTypeNames.Bool).Set(True)
        link.CreateAttribute("aloha:posePreview", Sdf.ValueTypeNames.Bool).Set(True)
        link.CreateAttribute("aloha:homePoseBaked", Sdf.ValueTypeNames.Bool).Set(True)
        link.CreateAttribute("aloha:cleanArticulationAuthored", Sdf.ValueTypeNames.Bool).Set(False)

        source_matrix = transforms[component.source_component_path]
        baked_matrix = cache.GetLocalToWorldTransform(link)
        source_translation = source_matrix.ExtractTranslation()
        baked_translation = baked_matrix.ExtractTranslation()
        translation_residual_m = (source_translation - baked_translation).GetLength()
        rotation_residual_deg = _quat_angle_error_deg(
            source_matrix.ExtractRotationQuat(),
            baked_matrix.ExtractRotationQuat(),
        )
        max_translation_residual_m = max(max_translation_residual_m, float(translation_residual_m))
        max_rotation_residual_deg = max(max_rotation_residual_deg, float(rotation_residual_deg))
        residual_records.append(
            {
                "source_component_path": component.source_component_path,
                "assembly_link_path": component.assembly_link_path,
                "translation_residual_m": float(translation_residual_m),
                "rotation_residual_deg": float(rotation_residual_deg),
            }
        )
    stage.GetRootLayer().Save()

    validation_json.parent.mkdir(parents=True, exist_ok=True)
    residual_tolerance_m = 1.0e-9
    residual_tolerance_deg = 1.0e-5
    resource_libraries_hidden = {
        "/visuals": UsdGeom.Imageable(stage.GetPrimAtPath("/visuals")).ComputeVisibility(),
        "/colliders": UsdGeom.Imageable(stage.GetPrimAtPath("/colliders")).ComputeVisibility(),
    }
    ok = (
        resource_libraries_hidden["/visuals"] == "invisible"
        and resource_libraries_hidden["/colliders"] == "invisible"
        and max_translation_residual_m <= residual_tolerance_m
        and max_rotation_residual_deg <= residual_tolerance_deg
    )
    validation = {
        "ok": ok,
        "output_usd": str(output_path),
        "source_usd": str(source_path),
        "pose": "REAL_RUNTIME_RESET_POSE",
        "component_count": len(components),
        "resource_libraries_hidden": resource_libraries_hidden,
        "physics_ready": False,
        "training_eligible": False,
        "clean_articulation_authored": False,
        "status": "PASS_VISUAL_HOME_POSE_PREVIEW_ONLY_NO_RUNTIME_SEMANTICS",
        "max_translation_residual_m": max_translation_residual_m,
        "max_rotation_residual_deg": max_rotation_residual_deg,
        "residual_tolerance_m": residual_tolerance_m,
        "residual_tolerance_deg": residual_tolerance_deg,
        "transform_residuals": residual_records,
        "transform_report": transform_report,
    }
    validation_json.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_VALIDATION_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    app = _start_isaac_headless()
    try:
        result = create_home_pose_preview(args.config, args.json_output)
        print(json.dumps({k: v for k, v in result.items() if k != "transform_report"}, indent=2, sort_keys=True))
    finally:
        app.close()


if __name__ == "__main__":
    main()
