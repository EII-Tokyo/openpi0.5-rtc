"""Render matched-view bottle CAD comparison screenshots in Blender."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import bpy
from mathutils import Matrix
from mathutils import Vector

RESOLUTION = (1280, 960)
PROJECT_COLOR = (0.08, 0.62, 0.38, 1.0)
REFERENCE_COLOR = (0.95, 0.38, 0.08, 1.0)


def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_obj(path: Path) -> tuple[list[Vector], list[list[int]]]:
    vertices = []
    triangles = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("v "):
            vertices.append(Vector(tuple(float(v) for v in line.split()[1:])))
        elif line.startswith("f "):
            indices = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
            if len(indices) != 3:
                raise RuntimeError(f"non-triangle OBJ face in {path}")
            triangles.append(indices)
    if not vertices or not triangles:
        raise RuntimeError(f"empty OBJ geometry: {path}")
    return vertices, triangles


def _reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in list(collection):
            if datablock.users == 0:
                collection.remove(datablock)


def _material(
    name: str,
    color: tuple[float, float, float, float],
) -> Any:
    material = bpy.data.materials.new(name=name)
    material.diffuse_color = color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = 0.31
    principled.inputs["Metallic"].default_value = 0.03
    return material


def _canonicalize_vertices(
    vertices: list[Vector],
    *,
    source_axis: str,
) -> tuple[list[Vector], Matrix]:
    rotation = Matrix.Rotation(math.radians(90.0), 4, "X") if source_axis == "+Y" else Matrix.Identity(4)
    rotated = [rotation @ vertex for vertex in vertices]
    x_min = min(point.x for point in rotated)
    x_max = max(point.x for point in rotated)
    y_min = min(point.y for point in rotated)
    y_max = max(point.y for point in rotated)
    z_min = min(point.z for point in rotated)
    translation = Matrix.Translation(
        Vector(
            (
                -(x_min + x_max) / 2.0,
                -(y_min + y_max) / 2.0,
                -z_min,
            )
        )
    )
    transform = translation @ rotation
    return [transform @ vertex for vertex in vertices], transform


def _make_object(
    *,
    asset_id: str,
    obj_path: Path,
    source_axis: str,
    material: Any,
) -> tuple[Any, Matrix, dict[str, list[float]]]:
    vertices, triangles = _read_obj(obj_path)
    canonical, transform = _canonicalize_vertices(
        vertices,
        source_axis=source_axis,
    )
    mesh = bpy.data.meshes.new(f"{asset_id}_mesh")
    mesh.from_pydata([tuple(point) for point in canonical], [], triangles)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(asset_id, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    bounds = {
        "min_m": [min(point[index] for point in canonical) for index in range(3)],
        "max_m": [max(point[index] for point in canonical) for index in range(3)],
    }
    return obj, transform, bounds


def _matrix_rows(matrix: Matrix) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _look_at(camera: Any, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def _scene_setup() -> Any:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = RESOLUTION[0]
    scene.render.resolution_y = RESOLUTION[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.display.shading.background_type = "VIEWPORT"
    scene.display.shading.background_color = (0.025, 0.032, 0.045)
    scene.world.color = (0.025, 0.032, 0.045)
    camera_data = bpy.data.cameras.new("BottleCADReviewCamera")
    camera_data.type = "ORTHO"
    camera_data.clip_start = 0.01
    camera_data.clip_end = 10.0
    camera = bpy.data.objects.new("BottleCADReviewCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    return camera


def _bbox_corners(obj: Any) -> list[Vector]:
    return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]


def _project_bbox(
    *,
    obj: Any,
    camera: Any,
) -> dict[str, float]:
    from bpy_extras.object_utils import world_to_camera_view

    scene = bpy.context.scene
    projected = [world_to_camera_view(scene, camera, point) for point in _bbox_corners(obj)]
    return {
        "xmin": min(point.x for point in projected) * RESOLUTION[0],
        "xmax": max(point.x for point in projected) * RESOLUTION[0],
        "ymin": (1.0 - max(point.y for point in projected)) * RESOLUTION[1],
        "ymax": (1.0 - min(point.y for point in projected)) * RESOLUTION[1],
    }


def main() -> None:
    args = _parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    output_root = args.output_root.resolve()
    raw_root = output_root / "screenshots_raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "PASS":
        raise RuntimeError("tessellation manifest is not PASS")

    _reset_scene()
    camera = _scene_setup()
    materials = {
        "project_main_bottle": _material(
            "ProjectMainBottleMaterial",
            PROJECT_COLOR,
        ),
        "downloaded_reference_bottle": _material(
            "DownloadedReferenceBottleMaterial",
            REFERENCE_COLOR,
        ),
    }
    objects = {}
    transforms = {}
    canonical_bounds = {}
    for asset_id, record in manifest["assets"].items():
        obj, transform, bounds = _make_object(
            asset_id=asset_id,
            obj_path=Path(record["obj_path"]).resolve(strict=True),
            source_axis=record["source_long_axis"],
            material=materials[asset_id],
        )
        objects[asset_id] = obj
        transforms[asset_id] = _matrix_rows(transform)
        canonical_bounds[asset_id] = bounds

    views = {
        "front": {
            "camera_location_m": [0.0, -0.70, 0.105],
            "target_m": [0.0, 0.0, 0.105],
            "orthographic_scale_m": 0.310,
        },
        "isometric": {
            "camera_location_m": [0.38, -0.50, 0.32],
            "target_m": [0.0, 0.0, 0.105],
            "orthographic_scale_m": 0.320,
        },
    }
    captures = [
        {
            "capture_id": "project_main_front",
            "view_id": "front",
            "visible_assets": ["project_main_bottle"],
        },
        {
            "capture_id": "downloaded_reference_front",
            "view_id": "front",
            "visible_assets": ["downloaded_reference_bottle"],
        },
        {
            "capture_id": "project_main_isometric",
            "view_id": "isometric",
            "visible_assets": ["project_main_bottle"],
        },
        {
            "capture_id": "downloaded_reference_isometric",
            "view_id": "isometric",
            "visible_assets": ["downloaded_reference_bottle"],
        },
        {
            "capture_id": "comparison_front",
            "view_id": "front",
            "visible_assets": [
                "project_main_bottle",
                "downloaded_reference_bottle",
            ],
            "pair_offset_x_m": 0.060,
            "orthographic_scale_m": 0.310,
        },
        {
            "capture_id": "comparison_isometric",
            "view_id": "isometric",
            "visible_assets": [
                "project_main_bottle",
                "downloaded_reference_bottle",
            ],
            "pair_offset_x_m": 0.060,
            "orthographic_scale_m": 0.350,
        },
    ]

    capture_records = []
    for capture in captures:
        for asset_id, obj in objects.items():
            obj.hide_render = asset_id not in capture["visible_assets"]
            obj.hide_viewport = obj.hide_render
            obj.location.x = 0.0
        if len(capture["visible_assets"]) == 2:
            offset = float(capture["pair_offset_x_m"])
            objects["project_main_bottle"].location.x = -offset
            objects["downloaded_reference_bottle"].location.x = offset

        view = views[capture["view_id"]]
        camera.location = Vector(view["camera_location_m"])
        target = Vector(view["target_m"])
        _look_at(camera, target)
        camera.data.ortho_scale = float(
            capture.get(
                "orthographic_scale_m",
                view["orthographic_scale_m"],
            )
        )
        raw_path = raw_root / f"{capture['capture_id']}_raw.png"
        bpy.context.scene.render.filepath = str(raw_path)
        bpy.ops.render.render(write_still=True)
        capture_records.append(
            {
                **capture,
                "status": "NOT_RUN",
                "raw_path": str(raw_path),
                "raw_sha256": _sha256(raw_path),
                "resolution": list(RESOLUTION),
                "camera": {
                    "projection": "ORTHOGRAPHIC",
                    "location_m": [float(value) for value in camera.location],
                    "target_m": [float(value) for value in target],
                    "rotation_euler_rad": [float(value) for value in camera.rotation_euler],
                    "orthographic_scale_m": float(camera.data.ortho_scale),
                },
                "object_locations_m": {
                    asset_id: [float(value) for value in objects[asset_id].location]
                    for asset_id in capture["visible_assets"]
                },
                "projected_bbox_px": {
                    asset_id: _project_bbox(
                        obj=objects[asset_id],
                        camera=camera,
                    )
                    for asset_id in capture["visible_assets"]
                },
            }
        )

    metadata = {
        "schema_version": 1,
        "status": "NOT_RUN",
        "scope": "CAD_VISUAL_COMPARISON_ONLY_NOT_PHYSICS_VALIDATION",
        "renderer": {
            "blender_version": bpy.app.version_string,
            "engine": bpy.context.scene.render.engine,
            "resolution": list(RESOLUTION),
        },
        "tessellation_manifest_path": str(manifest_path),
        "tessellation_manifest_sha256": _sha256(manifest_path),
        "canonicalization": {
            "project_main_bottle": {
                "source_axis": "+Z",
                "display_axis": "+Z",
                "transform": transforms["project_main_bottle"],
            },
            "downloaded_reference_bottle": {
                "source_axis": "+Y",
                "display_axis": "+Z",
                "transform": transforms["downloaded_reference_bottle"],
                "rotation_description": ("+90 degrees about X maps CAD +Y to display +Z"),
            },
        },
        "canonical_bounds_m": canonical_bounds,
        "captures": capture_records,
    }
    metadata_path = output_root / "render_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("status=PASS")
    print(f"capture_count={len(capture_records)}")
    print(f"metadata={metadata_path}")


main()
