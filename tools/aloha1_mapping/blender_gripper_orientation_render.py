#!/usr/bin/env python3
"""Depth-buffered Blender renderer for baked ALOHA gripper meshes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

COLORS = {
    "gripper_body": (0.23, 0.27, 0.31, 1.0),
    "gripper_bar": (0.12, 0.14, 0.17, 1.0),
    "gripper_prop": (0.38, 0.42, 0.47, 1.0),
    "physical_left": (0.035, 0.38, 0.95, 1.0),
    "physical_right": (1.0, 0.22, 0.025, 1.0),
}
VIEWS = {
    "closing_axis": Vector((0.30, 0.0, 0.0)),
    "top": Vector((0.0, 0.0, 0.30)),
    "isometric": Vector((0.24, -0.24, 0.19)),
}


def _arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args(argv)


def _material(name: str, rgba: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = rgba
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = rgba
    principled.inputs["Roughness"].default_value = 0.32
    principled.inputs["Metallic"].default_value = 0.24 if not name.startswith("physical_") else 0.02
    return material


def _reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)


def _look_at(camera: bpy.types.Object, target: Vector) -> None:
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _configure_scene() -> tuple[bpy.types.Object, bpy.types.Object]:
    scene = bpy.context.scene
    # Blender 5.2 reports the depth-buffered Eevee token as BLENDER_EEVEE.
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.008, 0.012, 0.020)
    scene.view_settings.exposure = -1.5

    camera_data = bpy.data.cameras.new("DiagnosticCamera")
    camera = bpy.data.objects.new("DiagnosticCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera_data.type = "ORTHO"
    camera_data.lens = 52.0
    scene.camera = camera

    key_data = bpy.data.lights.new("Key", type="AREA")
    key_data.energy = 8.0
    key_data.shape = "DISK"
    key_data.size = 0.18
    key = bpy.data.objects.new("Key", key_data)
    bpy.context.collection.objects.link(key)
    key.location = Vector((0.18, -0.20, 0.28))
    _look_at(key, Vector((0.05, 0.0, 0.0)))

    fill_data = bpy.data.lights.new("Fill", type="AREA")
    fill_data.energy = 4.0
    fill_data.size = 0.16
    fill = bpy.data.objects.new("Fill", fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = Vector((-0.12, 0.20, 0.16))
    _look_at(fill, Vector((0.05, 0.0, 0.0)))

    rim_data = bpy.data.lights.new("Rim", type="AREA")
    rim_data.energy = 6.0
    rim_data.size = 0.12
    rim = bpy.data.objects.new("Rim", rim_data)
    bpy.context.collection.objects.link(rim)
    rim.location = Vector((0.04, 0.0, -0.22))
    _look_at(rim, Vector((0.06, 0.0, 0.0)))
    return camera, key


def _load_state(state: dict, materials: dict[str, bpy.types.Material]) -> list[bpy.types.Object]:
    objects = []
    for label, record in state["meshes"].items():
        before = set(bpy.context.scene.objects)
        bpy.ops.wm.obj_import(filepath=record["obj_path"])
        imported = [item for item in bpy.context.scene.objects if item not in before]
        if len(imported) != 1:
            raise RuntimeError(f"Expected one object from {record['obj_path']}, got {len(imported)}")
        obj = imported[0]
        obj.name = label
        obj.data.materials.clear()
        obj.data.materials.append(materials[label])
        for polygon in obj.data.polygons:
            polygon.use_smooth = False
        objects.append(obj)
    return objects


def _shared_bounds(states: dict) -> tuple[Vector, float]:
    records = [mesh for state_name in ("closed", "open") for mesh in states[state_name]["meshes"].values()]
    lower = Vector(tuple(min(record["aabb_min_m"][index] for record in records) for index in range(3)))
    upper = Vector(tuple(max(record["aabb_max_m"][index] for record in records) for index in range(3)))
    center = 0.5 * (lower + upper)
    radius = (upper - lower).length * 0.5
    return center, max(radius, 0.06)


def _render_state(
    state_name: str,
    state: dict,
    output_root: Path,
    materials: dict[str, bpy.types.Material],
    shared_center: Vector,
    shared_radius: float,
) -> None:
    camera, _ = _configure_scene()
    _load_state(state, materials)
    camera.data.ortho_scale = shared_radius * 2.25
    for view_name, direction in VIEWS.items():
        unit = direction.normalized()
        camera.location = shared_center + unit * max(0.34, shared_radius * 4.5)
        if view_name == "top":
            camera.rotation_euler = (0.0, 0.0, -math.pi / 2.0)
            _look_at(camera, shared_center)
        else:
            _look_at(camera, shared_center)
        bpy.context.scene.render.filepath = str(output_root / f"{state_name}_{view_name}.png")
        bpy.ops.render.render(write_still=True)
    _reset_scene()


def main() -> None:
    args = _arguments()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _reset_scene()
    materials = {label: _material(label, rgba) for label, rgba in COLORS.items()}
    shared_center, shared_radius = _shared_bounds(manifest["states"])
    for state_name in ("closed", "open"):
        _render_state(
            state_name,
            manifest["states"][state_name],
            output_root,
            materials,
            shared_center,
            shared_radius,
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "renders": [
                    str(output_root / f"{state}_{view}.png")
                    for state in ("closed", "open")
                    for view in ("closing_axis", "top", "isometric")
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
