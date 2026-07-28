#!/usr/bin/env python3
"""Build the calibration-pending ALOHA 1 workcell and camera interfaces."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
from typing import Any

import yaml

from tools.aloha1_mapping.workcell_config import build_workcell_plan


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _custom_status(prim: Any, status: str) -> None:
    from pxr import Sdf

    prim.CreateAttribute(
        "aloha:calibrationStatus",
        Sdf.ValueTypeNames.String,
        custom=True,
    ).Set(status)


def write_workcell(plan: dict[str, Any], *, report_path: Path) -> dict[str, Any]:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from pxr import Gf
        from pxr import Sdf
        from pxr import Usd
        from pxr import UsdGeom
        from pxr import UsdPhysics

        output = Path(plan["stage"])
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            layer = Sdf.Layer.FindOrOpen(str(output))
            if layer is None:
                raise RuntimeError(f"unable to reopen workcell: {output}")
            layer.Clear()
            stage = Usd.Stage.Open(layer)
        else:
            stage = Usd.Stage.CreateNew(str(output))
        if stage is None:
            raise RuntimeError(f"unable to create workcell: {output}")
        root = UsdGeom.Xform.Define(stage, "/aloha1_workcell").GetPrim()
        stage.SetDefaultPrim(root)
        _custom_status(root, "PARTIAL")
        root.CreateAttribute(
            "aloha:coordinateConvention",
            Sdf.ValueTypeNames.String,
            custom=True,
        ).Set(plan["coordinate_convention"])

        references = []
        for robot in plan["robots"]:
            prim_path = f"/aloha1_workcell/Robots/{robot['name']}"
            xform = UsdGeom.Xform.Define(stage, prim_path)
            relative = os.path.relpath(robot["usd"], output.parent)
            if not xform.GetPrim().GetReferences().AddReference(
                relative, f"/{robot['name']}"
            ):
                raise RuntimeError(f"unable to reference {robot['usd']}")
            xform.AddTranslateOp(opSuffix="workcellPlacement").Set(
                Gf.Vec3d(*robot["translation_m"])
            )
            _custom_status(xform.GetPrim(), robot["transform_status"])
            references.append(
                {"prim": prim_path, "usd": robot["usd"], "kind": "follower"}
            )

        leader_variants = root.GetVariantSets().AddVariantSet("Leaders")
        for selection in ("disabled", "enabled"):
            leader_variants.AddVariant(selection)
            leader_variants.SetVariantSelection(selection)
            with leader_variants.GetVariantEditContext():
                container = UsdGeom.Xform.Define(
                    stage, "/aloha1_workcell/Leaders"
                ).GetPrim()
                container.SetActive(selection == "enabled")
                if selection == "enabled":
                    for leader in plan["leaders"]:
                        leader_path = (
                            f"/aloha1_workcell/Leaders/{leader['name']}"
                        )
                        leader_xform = UsdGeom.Xform.Define(stage, leader_path)
                        relative = os.path.relpath(
                            leader["usd"], output.parent
                        )
                        if not leader_xform.GetPrim().GetReferences().AddReference(
                            relative, f"/{leader['name']}"
                        ):
                            raise RuntimeError(
                                f"unable to reference {leader['usd']}"
                            )
                        leader_xform.AddTranslateOp(
                            opSuffix="workcellPlacement"
                        ).Set(
                            Gf.Vec3d(*leader["translation_m"])
                        )
                        _custom_status(
                            leader_xform.GetPrim(),
                            leader["transform_status"],
                        )
        leader_variants.SetVariantSelection(plan["leader_variant_default"])

        objects_root = "/aloha1_workcell/WorkcellObjects"
        UsdGeom.Xform.Define(stage, objects_root)
        object_prims = []
        for item in plan["workcell_objects"]:
            object_path = f"{objects_root}/{item['name']}"
            object_prim = UsdGeom.Xform.Define(stage, object_path).GetPrim()
            _custom_status(object_prim, item["status"])
            object_prim.CreateAttribute(
                "aloha:collisionEnabled",
                Sdf.ValueTypeNames.Bool,
                custom=True,
            ).Set(item["collision_enabled"])
            object_prims.append(object_path)
            if item["name"] == "table":
                guide = UsdGeom.Cube.Define(
                    stage, f"{object_path}/visual_reference"
                )
                guide.CreateSizeAttr(1.0)
                guide.AddScaleOp().Set(Gf.Vec3d(*item["dimensions_m"]))
                guide.AddTranslateOp().Set(
                    Gf.Vec3d(0.0, 0.0, -item["dimensions_m"][2] / 2)
                )
                guide.CreatePurposeAttr(UsdGeom.Tokens.guide)
                _custom_status(
                    guide.GetPrim(), "VISUAL_REFERENCE_ONLY_NO_COLLISION"
                )

        sensor_root = "/aloha1_workcell/Sensors"
        UsdGeom.Xform.Define(stage, sensor_root)
        camera_prims = []
        for camera in plan["cameras"]:
            camera_path = f"{sensor_root}/{camera['name']}"
            camera_prim = UsdGeom.Camera.Define(stage, camera_path).GetPrim()
            _custom_status(camera_prim, camera["calibration_status"])
            camera_prim.CreateAttribute(
                "aloha:renderEligible",
                Sdf.ValueTypeNames.Bool,
                custom=True,
            ).Set(camera["render_eligible"])
            camera_prim.CreateAttribute(
                "aloha:resolutionWidth",
                Sdf.ValueTypeNames.Int,
                custom=True,
            ).Set(camera["resolution_wh"][0])
            camera_prim.CreateAttribute(
                "aloha:resolutionHeight",
                Sdf.ValueTypeNames.Int,
                custom=True,
            ).Set(camera["resolution_wh"][1])
            camera_prims.append(camera_path)

        scene = UsdPhysics.Scene.Define(
            stage, "/aloha1_workcell/PhysicsScene"
        )
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr(9.81)
        stage.GetRootLayer().Save()
        report = {
            "schema_version": 1,
            "status": "PARTIAL",
            "stage": str(output.resolve()),
            "follower_references": references,
            "leader_variant": plan["leader_variant_default"],
            "object_prims": object_prims,
            "camera_prims": camera_prims,
            "hard_blockers": plan["hard_blockers"],
        }
        _write_json(report_path, report)
    except Exception as error:
        _write_json(
            report_path,
            {
                "schema_version": 1,
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        app.close()
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--enable-leaders", action="store_true")
    arguments = parser.parse_args(argv)
    root = arguments.project_root.resolve(strict=True)
    enable_leaders = arguments.enable_leaders or os.environ.get(
        "ENABLE_LEADERS", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    plan = build_workcell_plan(root, enable_leaders=enable_leaders)
    (root / "configs/aloha1_cameras.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": plan["schema_version"],
                "status": plan["status"],
                "cameras": plan["cameras"],
                "hard_blocker": plan["hard_blockers"][
                    "camera_calibration"
                ],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (root / "configs/aloha1_observation_schema.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": plan["schema_version"],
                "status": plan["status"],
                **plan["observation"],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    _write_json(
        root / "reports/aloha1_mapping/camera_validation.json",
        {
            "schema_version": 1,
            "status": "PARTIAL",
            "logical_camera_count": len(plan["cameras"]),
            "logical_names_unique": len(
                {item["name"] for item in plan["cameras"]}
            )
            == len(plan["cameras"]),
            "observation_order_matches_camera_order": (
                [item["name"] for item in plan["cameras"]]
                == plan["observation"]["camera_order"]
            ),
            "resolution_interface_contract_wh": [640, 480],
            "calibration_pending": [
                item["name"]
                for item in plan["cameras"]
                if item["calibration_status"] == "CALIBRATION_PENDING"
            ],
        },
    )
    write_workcell(
        plan,
        report_path=root
        / "reports/aloha1_mapping/workcell_manifest.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
