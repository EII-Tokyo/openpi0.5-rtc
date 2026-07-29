#!/usr/bin/env python3
# ruff: noqa: FBT003, PLC0415
"""Build the schema-only RobotRules target for supplier-CAD Task 7."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TASK5_STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_finger_task5_bottle/"
    "aloha_viperx_supplier_cad_bottle_task5.usda"
)
PHYSICAL_ASSET_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task7_robot_asset_v1_3.json"
)
CONFIGURATION_LAYER = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "supplier_cad_follower_left/1.3/configuration/"
    "supplier_cad_follower_left_configuration.usda"
)
OUTPUT_ROOT = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "supplier_cad_follower_left_robot_schema/1.0"
)
OUTPUT_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task7_robot_schema_asset.json"
)
ROBOT_PATH = "/supplier_cad_follower_left"
ROBOT_NAME = "supplier_cad_follower_left_robot_schema"
EXPECTED_TASK5_HASH = (
    "62697e4b25a7ec82234cc9ebd79d4a6d530a6ead0165519cbd275c0fa3f32178"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(target: Path, owner: Path) -> str:
    return Path(
        os.path.relpath(target.resolve(), owner.resolve().parent)
    ).as_posix()


def build(
    *,
    output_root: Path,
    output_report: Path,
) -> dict[str, Any]:
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom
    from usd.schema.isaac import robot_schema

    task5_hash = _sha256(TASK5_STAGE.resolve(strict=True))
    if task5_hash != EXPECTED_TASK5_HASH:
        raise RuntimeError("protected Task 5 Stage hash mismatch")
    physical_report = json.loads(
        PHYSICAL_ASSET_REPORT.resolve(strict=True).read_text(
            encoding="utf-8"
        )
    )
    expected_configuration_hash = physical_report["files"][
        "configuration"
    ]["sha256"]
    configuration_hash = _sha256(CONFIGURATION_LAYER.resolve(strict=True))
    if configuration_hash != expected_configuration_hash:
        raise RuntimeError("Task 7 configuration layer hash mismatch")
    if output_root.exists():
        raise FileExistsError(
            f"schema diagnostic output already exists: {output_root}"
        )
    output_root.mkdir(parents=True)
    wrapper = output_root / f"{ROBOT_NAME}.usda"

    stage = Usd.Stage.CreateNew(str(wrapper))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, ROBOT_PATH).GetPrim()
    if not root.GetReferences().AddReference(
        _relative(TASK5_STAGE, wrapper),
        Sdf.Path("/workcell"),
    ):
        raise RuntimeError("unable to reference Task 5 Stage")
    stage.GetRootLayer().subLayerPaths = [
        _relative(CONFIGURATION_LAYER, wrapper)
    ]
    stage.SetDefaultPrim(root)
    stage.GetRootLayer().Save()

    readback_stage = Usd.Stage.Open(str(wrapper), Usd.Stage.LoadAll)
    readback_root = readback_stage.GetDefaultPrim()
    if not readback_root.HasAPI(robot_schema.Classes.ROBOT_API.value):
        raise RuntimeError("schema wrapper RobotAPI readback failed")
    links = [
        str(path)
        for path in readback_root.GetRelationship(
            robot_schema.Relations.ROBOT_LINKS.name
        ).GetTargets()
    ]
    joints = [
        str(path)
        for path in readback_root.GetRelationship(
            robot_schema.Relations.ROBOT_JOINTS.name
        ).GetTargets()
    ]
    if not links or not joints:
        raise RuntimeError("schema wrapper robot relationships are empty")
    if _sha256(TASK5_STAGE) != task5_hash:
        raise RuntimeError("Task 5 Stage changed during schema build")
    if _sha256(CONFIGURATION_LAYER) != configuration_hash:
        raise RuntimeError("configuration changed during schema build")

    report = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "ROBOTRULES_SCHEMA_ONLY_DIAGNOSTIC",
        "wrapper": {
            "absolute_path": str(wrapper.resolve()),
            "sha256": _sha256(wrapper),
            "default_prim": str(readback_root.GetPath()),
        },
        "task5_stage": {
            "absolute_path": str(TASK5_STAGE.resolve()),
            "sha256": task5_hash,
            "modified": False,
        },
        "configuration": {
            "absolute_path": str(CONFIGURATION_LAYER.resolve()),
            "sha256": configuration_hash,
            "modified": False,
        },
        "readback": {
            "robot_api": True,
            "robot_links": links,
            "robot_joints": joints,
        },
        "excluded_layers": [
            physical_report["files"]["physics"]["absolute_path"]
        ],
        "reason": (
            "RobotRules validates RobotAPI and ordered relationships without "
            "the diagnostic physics overrides required by PhysicsRules."
        ),
        "final_default_collider_modified": False,
        "task8": "NOT_RUN",
    }
    output_report = output_report.resolve()
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-report", type=Path, default=OUTPUT_REPORT)
    args = parser.parse_args()
    report = build(
        output_root=args.output_root.resolve(),
        output_report=args.output_report,
    )
    print(f"status={report['status']}")
    print(f"wrapper={report['wrapper']['absolute_path']}")
    print(f"report={args.output_report.resolve()}")
    return 0


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    exit_code = 1
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        extension_id = "isaacsim.robot.schema"
        if not manager.is_extension_enabled(extension_id):
            manager.set_extension_enabled_immediate(extension_id, True)
        if not manager.is_extension_enabled(extension_id):
            raise RuntimeError(
                f"required extension disabled: {extension_id}"
            )
        exit_code = main()
    except Exception:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
