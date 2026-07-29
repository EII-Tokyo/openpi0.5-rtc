#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Run one fresh-process Isaac 5.1 Hydra protoPath diagnostic variant."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
from typing import Any

from tools.aloha1_mapping.hydra_protopath_diagnosis import PROTOPATH_SETTINGS
from tools.aloha1_mapping.signal_correspondence import RUNTIME_SPECS

ROOT = Path(__file__).resolve().parents[1]
FROZEN_STAGE = (
    ROOT / "assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda"
)
EXPECTED_STAGE_SHA256 = "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
REQUIRED_PRIMS = (
    "/World",
    "/World/environment",
    "/World/follower_left",
    "/World/follower_left/vx300s_left/root_joint",
    "/World/follower_right",
    "/World/follower_right/vx300s_right/root_joint",
)
ROBOT_ROOTS = (
    "/World/follower_left/vx300s_left",
    "/World/follower_right/vx300s_right",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _setting_record(settings: Any, path: str) -> dict[str, Any]:
    value = settings.get(path)
    return {
        "path": path,
        "exists": value is not None,
        "python_type": type(value).__name__ if value is not None else None,
        "value": value,
    }


def _cpu_stage_inventory(stage: Any) -> dict[str, Any]:
    from pxr import Usd
    from pxr import UsdGeom

    instance_proxies = []
    visual_meshes = []
    all_meshes = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if prim.IsInstanceProxy():
            instance_proxies.append(path)
        if prim.IsA(UsdGeom.Mesh):
            all_meshes.append(path)
            if (
                "/visuals/" in path
                and "/collisions/" not in path
                and UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
            ):
                visual_meshes.append(path)
    references = [
        {
            "prim_path": str(prim.GetPath()),
            "references_metadata": str(prim.GetMetadata("references")),
        }
        for prim in stage.Traverse()
        if prim.HasAuthoredReferences()
    ]
    return {
        "root_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim().IsValid() else None,
        "root_layer": str(Path(stage.GetRootLayer().realPath).resolve()),
        "root_sublayers": list(stage.GetRootLayer().subLayerPaths),
        "references": references,
        "required_prims": {path: stage.GetPrimAtPath(path).IsValid() for path in REQUIRED_PRIMS},
        "prototype_count": len(stage.GetPrototypes()),
        "prototype_paths": sorted(str(prim.GetPath()) for prim in stage.GetPrototypes()),
        "instance_proxy_count": len(instance_proxies),
        "instance_proxy_paths": sorted(instance_proxies),
        "all_mesh_count": len(all_meshes),
        "visible_visual_mesh_count": len(visual_meshes),
        "visible_visual_mesh_paths": sorted(visual_meshes),
    }


def _fabric_inventory() -> dict[str, Any]:
    import omni.usd
    import usdrt

    try:
        stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
        prims = list(stage.Traverse())
        meshes = []
        prototype_records = []
        for prim in prims:
            path = str(prim.GetPath())
            if str(prim.GetTypeName()) == "Mesh":
                meshes.append(path)
            attributes = {}
            for name in ("_protoPath", "_protoIndex", "_protoPrimType"):
                attribute = prim.GetAttribute(name)
                if attribute and attribute.IsValid():
                    try:
                        value = attribute.Get()
                    except Exception as error:  # pragma: no cover - runtime binding
                        value = f"READBACK_ERROR:{type(error).__name__}:{error}"
                    if value is not None:
                        attributes[name] = str(value)
            if attributes:
                prototype_records.append(
                    {
                        "prim_path": path,
                        "attributes": attributes,
                    }
                )
        return {
            "status": "PASS",
            "prim_count": len(prims),
            "mesh_count": len(meshes),
            "mesh_paths": sorted(meshes),
            "prototype_attribute_count": len(prototype_records),
            "prototype_attribute_records": sorted(
                prototype_records,
                key=lambda record: record["prim_path"],
            ),
        }
    except Exception as error:  # pragma: no cover - runtime binding
        return {
            "status": "PARTIAL",
            "error": f"{type(error).__name__}: {error}",
            "prim_count": None,
            "mesh_count": None,
            "mesh_paths": [],
            "prototype_attribute_count": None,
            "prototype_attribute_records": [],
        }


def _apply_native_screenshot_visibility_trigger(stage: Any) -> list[str]:
    """Reproduce the current screenshot pipeline's session visibility edits."""
    from pxr import Usd
    from pxr import UsdGeom

    changed = []
    stage.SetEditTarget(stage.GetSessionLayer())
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Imageable):
                continue
            path = str(prim.GetPath())
            if not path.startswith(ROBOT_ROOTS):
                continue
            imageable = UsdGeom.Imageable(prim)
            if "/collisions" in path or "/sites" in path:
                imageable.MakeInvisible()
            else:
                imageable.MakeVisible()
            changed.append(path)
    return sorted(set(changed))


def _initialize_two_follower_world() -> tuple[Any, dict[str, Any]]:
    """Match the current screenshot pipeline's one-reset articulation setup."""
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation

    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
    )
    world.get_physics_context().set_solve_articulation_contact_last(True)
    articulations = {}
    for robot, spec in RUNTIME_SPECS.items():
        articulation = SingleArticulation(
            prim_path=spec["articulation_path"],
            name=f"hydra_protopath_{robot}",
            reset_xform_properties=False,
        )
        world.scene.add(articulation)
        articulations[robot] = articulation
    world.reset()
    return world, articulations


def _materialize_visual_stage(
    source_path: Path,
    layer_path: Path,
    wrapper_path: Path,
) -> dict[str, Any]:
    from pxr import Gf
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom

    source = Usd.Stage.Open(str(source_path))
    if source is None:
        raise RuntimeError(f"failed to open source for materialization: {source_path}")
    layer_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = Usd.Stage.CreateNew(str(layer_path))
    UsdGeom.Xform.Define(materialized, "/World")
    scope = UsdGeom.Scope.Define(materialized, "/World/HydraProtoPathDiagnosticMaterialized")
    scope.GetPrim().SetCustomDataByKey("diagnosticOnly", True)  # noqa: FBT003
    scope.GetPrim().SetCustomDataByKey("physicsSchemasCopied", False)  # noqa: FBT003
    scope.GetPrim().SetCustomDataByKey("collisionSchemasCopied", False)  # noqa: FBT003
    for root_path in ROBOT_ROOTS:
        over = materialized.OverridePrim(root_path)
        UsdGeom.Imageable(over).CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    cache = UsdGeom.XformCache()
    records = []
    for prim in Usd.PrimRange(source.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        source_path_text = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Mesh) or "/visuals/" not in source_path_text:
            continue
        if "/collisions/" in source_path_text:
            continue
        if not source_path_text.startswith(ROBOT_ROOTS):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        source_mesh = UsdGeom.Mesh(prim)
        points = source_mesh.GetPointsAttr().Get() or []
        transform = cache.GetLocalToWorldTransform(prim)
        world_points = [transform.Transform(point) for point in points]
        destination_path = f"/World/HydraProtoPathDiagnosticMaterialized/mesh_{len(records):03d}"
        destination = UsdGeom.Mesh.Define(materialized, destination_path)
        destination.CreatePointsAttr(world_points)
        destination.CreateFaceVertexCountsAttr(source_mesh.GetFaceVertexCountsAttr().Get() or [])
        destination.CreateFaceVertexIndicesAttr(source_mesh.GetFaceVertexIndicesAttr().Get() or [])
        destination.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        destination.CreateDoubleSidedAttr(True)  # noqa: FBT003 - USD API positional.
        color = (
            Gf.Vec3f(0.18, 0.48, 0.72) if source_path_text.startswith(ROBOT_ROOTS[0]) else Gf.Vec3f(0.86, 0.47, 0.16)
        )
        destination.CreateDisplayColorPrimvar(
            UsdGeom.Tokens.constant,
        ).Set([color])
        records.append(
            {
                "source_prim": source_path_text,
                "destination_prim": destination_path,
                "source_instance_proxy": prim.IsInstanceProxy(),
                "point_count": len(world_points),
                "face_count": len(source_mesh.GetFaceVertexCountsAttr().Get() or []),
            }
        )
    materialized.GetRootLayer().Save()
    wrapper = Sdf.Layer.CreateNew(str(wrapper_path))
    wrapper.defaultPrim = "World"
    wrapper.subLayerPaths = [
        str(layer_path),
        str(source_path),
    ]
    wrapper.Save()
    return {
        "source_stage": str(source_path),
        "source_stage_sha256": _sha256(source_path),
        "materialized_layer": str(layer_path),
        "materialized_layer_sha256": _sha256(layer_path),
        "wrapper_stage": str(wrapper_path),
        "wrapper_stage_sha256": _sha256(wrapper_path),
        "mesh_count": len(records),
        "records": records,
        "source_robot_visibility_override": "invisible_visual_only",
        "physics_schemas_copied": False,
        "collision_schemas_copied": False,
    }


def _capture_native_view(app: Any, destination: Path) -> None:
    from isaacsim.sensors.camera import Camera
    import numpy as np
    from omni.kit.viewport.utility import capture_viewport_to_file
    from omni.kit.viewport.utility import get_active_viewport
    import omni.usd
    from pxr import Gf
    from pxr import Sdf
    from pxr import UsdGeom
    from pxr import UsdLux

    from tools.aloha1_mapping.isaac_screenshot import look_at_orientation_wxyz

    stage = omni.usd.get_context().get_stage()
    # D is a visual-geometry diagnostic.  Hide the referenced environment only
    # for this session capture so the rack cannot occlude the materialized
    # follower meshes; no authored layer or physics schema is changed.
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/environment")).MakeInvisible()
    camera_path = "/World/HydraProtoPathDiagnosticCamera"
    camera = Camera(
        prim_path=camera_path,
        name="hydra_protopath_diagnostic_camera",
        resolution=(960, 720),
        frequency=60,
    )
    camera.initialize()
    camera.set_clipping_range(0.01, 10.0)
    dome = UsdLux.DomeLight.Define(stage, "/World/HydraProtoPathDiagnosticDome")
    dome.CreateIntensityAttr(900.0)
    dome.CreateColorAttr(Gf.Vec3f(0.92, 0.94, 1.0))
    # The materialized diagnostic meshes are authored in their computed world
    # transforms.  This view targets the measured follower-left materialized
    # AABB midpoint and keeps a 20% framing margin over the Task 7A oblique
    # direction so the whole robot remains inspectable.
    target = np.asarray([-0.2740307041, -0.0190000013, 0.2511500064])
    position = np.asarray([-2.431, -1.852, 1.804])
    camera.set_world_pose(
        position=position,
        orientation=look_at_orientation_wxyz(position, target),
        camera_axes="usd",
    )
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("no active viewport")
    # viewport.utility 1.1.2 honors the camera_path property used by the
    # project's proven Isaac 5.1 screenshot pipeline.
    viewport.camera_path = Sdf.Path(camera.prim_path)
    for _ in range(30):
        app.update()
    destination.parent.mkdir(parents=True, exist_ok=True)
    helper = capture_viewport_to_file(viewport, file_path=str(destination))
    previous = -1
    stable = 0
    for _ in range(360):
        app.update()
        if not destination.exists():
            continue
        size = destination.stat().st_size
        if size > 0 and size == previous:
            stable += 1
        else:
            stable = 0
        previous = size
        if stable >= 3:
            break
    # Installed viewport.utility 1.1.2 documents a future-like return object;
    # its own tests additionally flush the renderer capture queue before
    # asserting that the file exists.
    import omni.kit.renderer_capture

    omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
    for _ in range(10):
        app.update()
    del helper
    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(f"viewport capture failed: {destination}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--expected-stage-sha256", required=True)
    parser.add_argument("--overrides-json", default="{}")
    parser.add_argument("--materialize-visual-instances", action="store_true")
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path, required=True)
    args = parser.parse_args()
    stage_path = args.stage.resolve(strict=True)
    if stage_path != FROZEN_STAGE.resolve():
        raise ValueError("only the user-approved signal-correspondence Stage is accepted")
    if _sha256(stage_path) != args.expected_stage_sha256:
        raise RuntimeError("frozen Stage SHA-256 mismatch before SimulationApp")
    overrides = json.loads(args.overrides_json)
    if len(overrides) > 1:
        raise ValueError("each variant may change at most one setting")

    from isaacsim import SimulationApp

    started = time.monotonic()
    app = SimulationApp(
        {
            "headless": True,
            "width": 960,
            "height": 720,
            "create_new_stage": False,
        }
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "variant_id": args.variant_id,
        "status": "FAIL",
        "isaac_sim_version": "5.1.0.0",
        "kit_version": "107.3.3",
        "hydra_usdrt_delegate_version": "7.5.1",
        "usdrt_scenegraph_version": "7.6.1",
        "frozen_stage": str(stage_path),
        "frozen_stage_sha256_before": _sha256(stage_path),
        "setting_overrides": overrides,
        "materialize_visual_instances": args.materialize_visual_instances,
        "fresh_load_count": 0,
        "real_robot_connected": False,
        "remote_192_168_1_103_accessed": False,
        "task_8": "NOT_RUN",
    }
    exit_code = 1
    try:
        import carb.settings
        import omni.usd

        settings = carb.settings.get_settings()
        before = {path: _setting_record(settings, path) for path in PROTOPATH_SETTINGS.values()}
        report["settings_before"] = before
        unsupported = [path for path in overrides if not before[path]["exists"]]
        if unsupported:
            raise RuntimeError(f"unsupported local settings: {unsupported}")
        for path, value in overrides.items():
            if not isinstance(value, bool):
                raise TypeError(f"only bool setting overrides are accepted: {path}")
            settings.set_bool(path, value)
        report["settings_effective"] = {path: _setting_record(settings, path) for path in PROTOPATH_SETTINGS.values()}

        effective_stage = stage_path
        if args.materialize_visual_instances:
            materialized_root = args.diagnostic_root.resolve()
            materialized_root.mkdir(parents=True, exist_ok=True)
            layer_path = materialized_root / "materialized_visuals.usda"
            wrapper_path = materialized_root / "materialized_workcell.usda"
            report["materialization"] = _materialize_visual_stage(
                stage_path,
                layer_path,
                wrapper_path,
            )
            effective_stage = wrapper_path

        context = omni.usd.get_context()
        if not context.open_stage(str(effective_stage)):
            raise RuntimeError(f"failed to open diagnostic Stage: {effective_stage}")
        report["fresh_load_count"] = 1
        for _ in range(120):
            app.update()
        stage = context.get_stage()
        clone_handles: dict[str, list[tuple[str, str]]] = {}
        if not args.materialize_visual_instances:
            report["native_visibility_trigger"] = {
                "status": "APPLIED_SESSION_ONLY",
                "changed_paths": _apply_native_screenshot_visibility_trigger(stage),
            }
            from tools.capture_aloha1_signal_correspondence_screenshots import _create_arm_visual_clones
            from tools.capture_aloha1_signal_correspondence_screenshots import _create_diagnostic_arm_materials

            materials = _create_diagnostic_arm_materials(stage)
            clone_manifest, clone_handles = _create_arm_visual_clones(
                stage,
                materials,
                tuple(RUNTIME_SPECS),
            )
            report["exact_visual_clones_control_trigger"] = {
                "status": "SESSION_ONLY_CONTROL_TRIGGER",
                "mesh_count": len(clone_manifest),
                "physics_schema_applied": False,
                "collision_schema_applied": False,
                "final_asset_fix": False,
                "records": clone_manifest,
            }
            for _ in range(60):
                app.update()
        else:
            report["native_visibility_trigger"] = {
                "status": "NOT_APPLIED_MATERIALIZED_VISUAL_COMPARISON",
                "changed_paths": [],
            }
            report["exact_visual_clones_control_trigger"] = {
                "status": "NOT_APPLIED_IN_VARIANT_D",
                "mesh_count": 0,
                "physics_schema_applied": False,
                "collision_schema_applied": False,
                "final_asset_fix": False,
                "records": [],
            }
        world, articulations = _initialize_two_follower_world()
        report["world_reset"] = {
            "status": "PASS",
            "articulation_count": len(articulations),
            "articulation_paths": {robot: spec["articulation_path"] for robot, spec in RUNTIME_SPECS.items()},
            "solve_articulation_contact_last": True,
            "reset_count": 1,
            "physics_composition_changed": False,
            "source_stage_saved": False,
        }
        for _ in range(30):
            world.step(render=True)
        from omni.physx import get_physx_interface

        # Exact installed-5.1 transform synchronization used by the current
        # signal screenshot pipeline.
        get_physx_interface().update_transformations(True, True, False, False)  # noqa: FBT003
        report["world_reset"]["physx_fabric_transform_sync_count"] = 1
        if clone_handles:
            from tools.capture_aloha1_signal_correspondence_screenshots import _update_arm_visual_clones

            _update_arm_visual_clones(stage, clone_handles)
            report["exact_visual_clones_control_trigger"]["post_physics_world_point_refresh_count"] = sum(
                len(pairs) for pairs in clone_handles.values()
            )
            for _ in range(30):
                app.update()
        report["effective_stage"] = str(effective_stage)
        report["effective_stage_sha256"] = _sha256(effective_stage)
        report["cpu_usd"] = _cpu_stage_inventory(stage)
        report["fabric_usdrt"] = _fabric_inventory()
        _capture_native_view(app, args.screenshot.resolve())
        report["screenshot"] = {
            "absolute_path": str(args.screenshot.resolve()),
            "sha256": _sha256(args.screenshot.resolve()),
            "size_bytes": args.screenshot.resolve().stat().st_size,
        }
        report["frozen_stage_sha256_after"] = _sha256(stage_path)
        report["source_stage_unchanged"] = (
            report["frozen_stage_sha256_before"] == report["frozen_stage_sha256_after"] == args.expected_stage_sha256
        )
        report["status"] = (
            "PASS" if report["source_stage_unchanged"] and all(report["cpu_usd"]["required_prims"].values()) else "FAIL"
        )
        exit_code = 0 if report["status"] == "PASS" else 1
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        report["runtime_seconds"] = time.monotonic() - started
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        app.close()
    print(
        json.dumps(
            {
                "variant": args.variant_id,
                "status": report["status"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return exit_code


def run() -> int:
    return main()


if __name__ == "__main__":
    raise SystemExit(run())
