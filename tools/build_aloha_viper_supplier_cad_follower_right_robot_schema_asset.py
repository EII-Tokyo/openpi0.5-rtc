#!/usr/bin/env python3
# ruff: noqa: FBT003, PLC0415
"""Build an isolated RobotRules-only wrapper for supplier-CAD follower_right."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any

from PIL import Image
from PIL import ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RIGHT_ASSET = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/follower_vx300s/follower_right/"
    "follower_right.usd"
)
PHYSICAL_RIGHT_STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "supplier_cad_follower_right/1.0/supplier_cad_follower_right.usda"
)
SOURCE_GEOMETRY_LAYER = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "supplier_cad_follower_right/1.0/geometry/"
    "supplier_cad_follower_right_geometry.usda"
)
THUMBNAIL_SOURCE = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "follower_right_pose_evidence/attempt4_final/screenshots_raw/"
    "full_arm_oblique_home_reference_raw.png"
)
OUTPUT_ROOT = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "supplier_cad_follower_right_robot_schema/1.0"
)
OUTPUT_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_supplier_cad_follower_right_robot_schema_asset.json"
)

EXPECTED_HASHES = {
    "source_right_asset": (
        "86d850cea5b35fb2969d3a78834317b51e2ac0d301f09aaaa9dad191f9bb3d5d"
    ),
    "physical_right_stage": (
        "95c7878f794f5f557b70997a2240b6476836b8ffbeed5a4992cb114a169487ea"
    ),
    "source_geometry_layer": (
        "168ad2705541ea17afe46f4fe2389a53f4ab660f5f4df09b25fff7fa5007fd65"
    ),
    "thumbnail_source": (
        "26e66a9de101024051957de123c9fe699508e732dbad78d70a0d97fa88b27d97"
    ),
}

ROOT_PRIM = "/follower_right"
PRODUCT_PRIM = "/follower_right/vx300s_right"
ROBOT_NAME = "supplier_cad_follower_right_robot_schema"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str) -> str:
    actual = _sha256(path.resolve(strict=True))
    if actual != expected:
        raise RuntimeError(f"protected hash mismatch: {path}")
    return actual


def _relative(target: Path, owner: Path) -> str:
    return Path(
        os.path.relpath(target.resolve(), owner.resolve().parent)
    ).as_posix()


def _normalized_usda_text(text: str) -> str:
    return text.rstrip() + "\n"


def _normalize_usda(path: Path) -> None:
    path.write_text(
        _normalized_usda_text(path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )


def _create_thumbnail(*, wrapper: Path) -> dict[str, Any]:
    source_hash = _require_hash(
        THUMBNAIL_SOURCE,
        EXPECTED_HASHES["thumbnail_source"],
    )
    output = wrapper.parent / ".thumbs/256x256" / f"{wrapper.name}.png"
    output.parent.mkdir(parents=True)
    with Image.open(THUMBNAIL_SOURCE) as opened:
        image = ImageOps.fit(
            opened.convert("RGB"),
            (256, 256),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        image.save(output, format="PNG", optimize=False, compress_level=9)
    with Image.open(output) as opened:
        resolution = [opened.width, opened.height]
        mode = opened.mode
    if resolution != [256, 256] or mode != "RGB":
        raise RuntimeError("thumbnail readback failed")
    return {
        "absolute_path": str(output.resolve()),
        "sha256": _sha256(output),
        "resolution": resolution,
        "mode": mode,
        "source_absolute_path": str(THUMBNAIL_SOURCE.resolve()),
        "source_sha256": source_hash,
        "source_visual_review": "PASS",
        "scope": "ASSET_BROWSER_METADATA_NOT_PHYSICS_ACCEPTANCE",
    }


def build(*, output_root: Path, output_report: Path) -> dict[str, Any]:
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics
    from usd.schema.isaac import robot_schema

    protected = {
        "source_right_asset": _require_hash(
            SOURCE_RIGHT_ASSET,
            EXPECTED_HASHES["source_right_asset"],
        ),
        "physical_right_stage": _require_hash(
            PHYSICAL_RIGHT_STAGE,
            EXPECTED_HASHES["physical_right_stage"],
        ),
        "source_geometry_layer": _require_hash(
            SOURCE_GEOMETRY_LAYER,
            EXPECTED_HASHES["source_geometry_layer"],
        ),
    }
    if output_root.exists():
        raise FileExistsError(f"schema diagnostic already exists: {output_root}")
    output_root.mkdir(parents=True)
    wrapper = output_root / f"{ROBOT_NAME}.usda"
    configuration = output_root / "configuration"
    configuration.mkdir()
    schema_layer_path = configuration / f"{ROBOT_NAME}_robot.usda"
    schema_layer = Sdf.Layer.CreateNew(str(schema_layer_path))
    if schema_layer is None:
        raise RuntimeError("unable to create schema layer")
    schema_layer.Save()

    stage = Usd.Stage.CreateNew(str(wrapper))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, ROOT_PRIM).GetPrim()
    product = UsdGeom.Xform.Define(stage, PRODUCT_PRIM).GetPrim()
    if not product.GetReferences().AddReference(
        _relative(SOURCE_RIGHT_ASSET, wrapper),
        Sdf.Path("/follower_right"),
    ):
        raise RuntimeError("unable to reference follower_right source")
    stage.GetRootLayer().subLayerPaths = [
        _relative(schema_layer_path, wrapper),
        _relative(SOURCE_GEOMETRY_LAYER, wrapper),
    ]
    stage.SetDefaultPrim(root)
    stage.GetRootLayer().Save()

    schema_stage = Usd.Stage.Open(str(wrapper), Usd.Stage.LoadAll)
    if schema_stage is None:
        raise RuntimeError("unable to compose schema wrapper")
    edit_layer = next(
        (
            layer
            for layer in schema_stage.GetLayerStack(includeSessionLayers=False)
            if layer.realPath
            and Path(layer.realPath).resolve() == schema_layer_path.resolve()
        ),
        None,
    )
    if edit_layer is None:
        raise RuntimeError("schema layer missing from layer stack")
    schema_stage.SetEditTarget(edit_layer)
    schema_root = schema_stage.GetDefaultPrim()
    robot_schema.ApplyRobotAPI(schema_root)

    generic_deactivated = []
    for link in (
        "follower_right_left_finger_link",
        "follower_right_right_finger_link",
    ):
        for role in ("visuals", "collisions"):
            path = f"{PRODUCT_PRIM}/{link}/{role}/gripper_finger"
            if not schema_stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(f"generic finger path missing: {path}")
            schema_stage.OverridePrim(path).SetActive(False)
            generic_deactivated.append(path)

    links = []
    joints = []
    product = schema_stage.GetPrimAtPath(PRODUCT_PRIM)
    for prim in Usd.PrimRange(product, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            path = str(prim.GetPath())
            robot_schema.ApplyLinkAPI(schema_stage.GetPrimAtPath(path))
            links.append(path)
        if prim.IsA(UsdPhysics.Joint):
            path = str(prim.GetPath())
            robot_schema.ApplyJointAPI(schema_stage.GetPrimAtPath(path))
            joints.append(path)
    if len(links) < 10 or len(joints) < 10:
        raise RuntimeError("right schema link/joint inventory is incomplete")

    for relation, targets in (
        (robot_schema.Relations.ROBOT_LINKS, links),
        (robot_schema.Relations.ROBOT_JOINTS, joints),
    ):
        relationship = schema_root.GetRelationship(relation.name)
        relationship.ClearTargets(True)
        for path in targets:
            relationship.AddTarget(
                Sdf.Path(path),
                Usd.ListPositionBackOfPrependList,
            )
    edit_layer.Save()
    _normalize_usda(wrapper)
    _normalize_usda(schema_layer_path)
    thumbnail = _create_thumbnail(wrapper=wrapper)

    readback_stage = Usd.Stage.Open(str(wrapper), Usd.Stage.LoadAll)
    readback_root = readback_stage.GetDefaultPrim()
    robot_api = readback_root.HasAPI(robot_schema.Classes.ROBOT_API.value)
    readback_links = [
        str(path)
        for path in readback_root.GetRelationship(
            robot_schema.Relations.ROBOT_LINKS.name
        ).GetTargets()
    ]
    readback_joints = [
        str(path)
        for path in readback_root.GetRelationship(
            robot_schema.Relations.ROBOT_JOINTS.name
        ).GetTargets()
    ]
    if not robot_api or readback_links != links or readback_joints != joints:
        raise RuntimeError("right schema readback failed")
    if any(
        readback_stage.GetPrimAtPath(path).IsActive()
        for path in generic_deactivated
    ):
        raise RuntimeError("generic finger remained active")

    unchanged = {
        "source_right_asset": _sha256(SOURCE_RIGHT_ASSET)
        == protected["source_right_asset"],
        "physical_right_stage": _sha256(PHYSICAL_RIGHT_STAGE)
        == protected["physical_right_stage"],
        "source_geometry_layer": _sha256(SOURCE_GEOMETRY_LAYER)
        == protected["source_geometry_layer"],
    }
    if not all(unchanged.values()):
        raise RuntimeError("protected input changed during schema build")

    report = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "ROBOTRULES_SCHEMA_ONLY_DIAGNOSTIC",
        "wrapper": {
            "absolute_path": str(wrapper.resolve()),
            "sha256": _sha256(wrapper),
            "default_prim": str(readback_root.GetPath()),
        },
        "schema_layer": {
            "absolute_path": str(schema_layer_path.resolve()),
            "sha256": _sha256(schema_layer_path),
        },
        "source_right_asset": {
            "absolute_path": str(SOURCE_RIGHT_ASSET.resolve()),
            "sha256": protected["source_right_asset"],
            "modified": False,
        },
        "source_geometry_layer": {
            "absolute_path": str(SOURCE_GEOMETRY_LAYER.resolve()),
            "sha256": protected["source_geometry_layer"],
            "modified": False,
        },
        "physical_right_stage": {
            "absolute_path": str(PHYSICAL_RIGHT_STAGE.resolve()),
            "sha256": protected["physical_right_stage"],
            "modified": False,
        },
        "physical_right_stage_included": False,
        "generic_fingers_deactivated": generic_deactivated,
        "thumbnail": thumbnail,
        "readback": {
            "robot_api": robot_api,
            "robot_links": readback_links,
            "robot_joints": readback_joints,
        },
        "protected_inputs_unchanged": unchanged,
        "isaac_version_contract": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "asset_validation": "1.1.0",
        },
        "final_default_collider_modified": False,
        "real_robot_connected": False,
        "remote_192_168_1_103_accessed": False,
        "task8": "NOT_RUN",
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_report.with_suffix(".md").write_text(
        "\n".join(
            [
                "# Supplier-CAD follower_right Robot Schema asset",
                "",
                "- Status: `PASS`",
                "- Scope: `ROBOTRULES_SCHEMA_ONLY_DIAGNOSTIC`",
                f"- Root prim: `{readback_root.GetPath()}`",
                f"- Robot links: `{len(readback_links)}`",
                f"- Robot joints: `{len(readback_joints)}`",
                "- Physical follower_right Stage included: `false`",
                "- Final/default collider modified: `false`",
                "- Task 8: `NOT_RUN`",
                "",
                "This isolated wrapper exists only to validate RobotAPI, "
                "ordered link/joint relationships, naming and thumbnail "
                "packaging without the physical diagnostic overrides.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report


def main() -> int:
    report = build(output_root=OUTPUT_ROOT, output_report=OUTPUT_REPORT)
    print(f"status={report['status']}")
    print(f"wrapper={report['wrapper']['absolute_path']}")
    print(f"report={OUTPUT_REPORT.resolve()}")
    return 0


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    exit_code = 1
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        for extension_id in (
            "isaacsim.robot.schema",
            "isaacsim.asset.validation",
        ):
            if not manager.is_extension_enabled(extension_id):
                manager.set_extension_enabled_immediate(extension_id, True)
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
