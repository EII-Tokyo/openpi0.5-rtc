"""Render paired four-direction screenshots from audited Viper CAD meshes.

Run through Blender:
blender --background --python this_file.py -- \
  --input states.json --output-root artifact_dir
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import bpy
from mathutils import Matrix
from mathutils import Vector

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.aloha1_mapping.cad_gripper_visual_states import MM_TO_M  # noqa: E402
from tools.aloha1_mapping.cad_gripper_visual_states import STATE_IDS  # noqa: E402
from tools.aloha1_mapping.cad_gripper_visual_states import VIEW_IDS  # noqa: E402
from tools.aloha1_mapping.cad_gripper_visual_states import capture_plan  # noqa: E402
from tools.aloha1_mapping.cad_gripper_visual_states import orthographic_frame  # noqa: E402
from tools.aloha1_mapping.cad_gripper_visual_states import points_mm_to_m  # noqa: E402

RESOLUTION = (1280, 900)
FRAME_MARGIN = 1.45
ROLE_COLORS = {
    "gripper_shell": (0.28, 0.31, 0.35, 1.0),
    "cad_positive_x_finger": (0.035, 0.30, 0.88, 1.0),
    "cad_negative_x_finger": (1.0, 0.24, 0.035, 1.0),
}


def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def _material(name: str, color: tuple[float, float, float, float]) -> Any:
    material = bpy.data.materials.new(name=name)
    material.diffuse_color = color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = 0.34
    principled.inputs["Metallic"].default_value = 0.08
    return material


def _mesh_object(
    *,
    name: str,
    mesh_record: dict[str, Any],
    material: Any,
) -> Any:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(
        points_mm_to_m(mesh_record["vertices_mm"]),
        [],
        mesh_record["triangles"],
    )
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    return obj


def _camera_matrix(
    *,
    location: tuple[float, float, float],
    forward: tuple[float, float, float],
    image_up: tuple[float, float, float],
) -> Matrix:
    local_y = Vector(image_up).normalized()
    local_z = -Vector(forward).normalized()
    local_x = local_y.cross(local_z).normalized()
    local_y = local_z.cross(local_x).normalized()
    return Matrix(
        (
            (local_x.x, local_y.x, local_z.x, location[0]),
            (local_x.y, local_y.y, local_z.y, location[1]),
            (local_x.z, local_y.z, local_z.z, location[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def _create_camera() -> Any:
    data = bpy.data.cameras.new("CADReviewCamera")
    data.type = "ORTHO"
    data.lens = 50
    data.clip_start = 0.1
    data.clip_end = 5000.0
    camera = bpy.data.objects.new("CADReviewCamera", data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def _create_area_light(name: str, energy: float, size: float) -> Any:
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    return light


def _point_light_at(
    light: Any,
    *,
    location: Vector,
    target: Vector,
) -> None:
    direction = target - location
    light.location = location
    light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _project_point(
    point: list[float],
    *,
    frame: dict[str, Any],
) -> list[float]:
    target = frame["target_mm"]
    right = frame["image_right"]
    up = frame["image_up"]
    relative = [point[index] - target[index] for index in range(3)]
    horizontal = sum(
        relative[index] * right[index] for index in range(3)
    )
    vertical = sum(relative[index] * up[index] for index in range(3))
    width, height = frame["resolution"]
    x = (0.5 + horizontal / frame["ortho_width_mm"]) * width
    y = (0.5 - vertical / frame["ortho_height_mm"]) * height
    return [x, y]


def _projected_bbox(
    vertices: list[list[float]],
    *,
    frame: dict[str, Any],
) -> dict[str, float]:
    projected = [_project_point(point, frame=frame) for point in vertices]
    return {
        "xmin": min(point[0] for point in projected),
        "ymin": min(point[1] for point in projected),
        "xmax": max(point[0] for point in projected),
        "ymax": max(point[1] for point in projected),
    }


def _inner_surface_point(
    vertices: list[list[float]],
    *,
    positive_x_side: bool,
) -> list[float]:
    """Sample the center-facing distal surface; this is not a contact point."""
    y_values = [point[1] for point in vertices]
    distal_y_limit = min(y_values) + 0.62 * (max(y_values) - min(y_values))
    distal = [point for point in vertices if point[1] <= distal_y_limit]
    x_values = sorted(point[0] for point in distal)
    quantile_index = max(0, int(len(x_values) * 0.08) - 1)
    inner_limit = (
        x_values[quantile_index]
        if positive_x_side
        else x_values[-quantile_index - 1]
    )
    tolerance = 0.8
    if positive_x_side:
        samples = [
            point
            for point in distal
            if point[0] <= inner_limit + tolerance
        ]
    else:
        samples = [
            point
            for point in distal
            if point[0] >= inner_limit - tolerance
        ]
    return [
        sum(point[axis] for point in samples) / len(samples)
        for axis in range(3)
    ]


def _scene_setup() -> tuple[Any, Any, Any]:
    scene = bpy.context.scene
    # Geometry review uses the host's depth-buffered Workbench studio
    # shading so visibility does not depend on photometric light calibration.
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = RESOLUTION[0]
    scene.render.resolution_y = RESOLUTION[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.image_settings.color_depth = "8"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.display.shading.background_type = "VIEWPORT"
    scene.display.shading.background_color = (0.025, 0.03, 0.04)
    scene.world.color = (0.035, 0.035, 0.035)
    world = scene.world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.025, 0.03, 0.04, 1.0)
    background.inputs["Strength"].default_value = 0.48
    camera = _create_camera()
    key_light = _create_area_light("KeyLight", 1050.0, 180.0)
    fill_light = _create_area_light("FillLight", 700.0, 160.0)
    return camera, key_light, fill_light


def main() -> None:
    args = _parse_args()
    source = args.input.resolve(strict=True)
    output_root = args.output_root.resolve()
    raw_root = output_root / "screenshots_raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    data = json.loads(source.read_text(encoding="utf-8"))
    if data["status"] != "PASS":
        raise RuntimeError("input CAD state report is not PASS")
    if tuple(data["states"]) != STATE_IDS:
        raise RuntimeError(
            f"unexpected state order: {tuple(data['states'])!r}"
        )

    _reset_scene()
    camera, key_light, fill_light = _scene_setup()
    materials = {
        role: _material(f"{role}_material", color)
        for role, color in ROLE_COLORS.items()
    }
    state_objects: dict[str, dict[str, Any]] = {}
    for state_id in STATE_IDS:
        meshes = data["states"][state_id]["meshes"]
        state_objects[state_id] = {
            role: _mesh_object(
                name=f"{state_id}_{role}",
                mesh_record=meshes[role],
                material=materials[role],
            )
            for role in ROLE_COLORS
        }

    union_points = [
        tuple(point)
        for state_id in STATE_IDS
        for role in ROLE_COLORS
        for point in data["states"][state_id]["meshes"][role]["vertices_mm"]
    ]
    frames = {
        view_id: orthographic_frame(
            points_mm=union_points,
            view_id=view_id,
            resolution=RESOLUTION,
            margin=FRAME_MARGIN,
        )
        for view_id in VIEW_IDS
    }
    plan_by_key = {
        (record["state_id"], record["view_id"]): record
        for record in capture_plan(output_root=output_root)
    }
    captures = []
    for view_id in VIEW_IDS:
        frame = frames[view_id]
        location = tuple(
            value * MM_TO_M for value in frame["camera_location_mm"]
        )
        camera.matrix_world = _camera_matrix(
            location=location,
            forward=tuple(frame["camera_forward"]),
            image_up=tuple(frame["image_up"]),
        )
        camera.data.ortho_scale = frame["ortho_height_mm"] * MM_TO_M
        target = Vector(frame["target_mm"]) * MM_TO_M
        location_vector = Vector(location)
        right = Vector(frame["image_right"])
        up = Vector(frame["image_up"])
        _point_light_at(
            key_light,
            location=(
                location_vector
                + right * (135.0 * MM_TO_M)
                + up * (95.0 * MM_TO_M)
            ),
            target=target,
        )
        _point_light_at(
            fill_light,
            location=(
                location_vector
                - right * (150.0 * MM_TO_M)
                - up * (45.0 * MM_TO_M)
            ),
            target=target,
        )
        for state_id in STATE_IDS:
            for candidate_state, objects in state_objects.items():
                hidden = candidate_state != state_id
                for obj in objects.values():
                    obj.hide_render = hidden
                    obj.hide_viewport = hidden
            plan_record = plan_by_key[(state_id, view_id)]
            raw_path = Path(plan_record["raw_path"])
            bpy.context.scene.render.filepath = str(raw_path)
            bpy.ops.render.render(write_still=True)
            meshes = data["states"][state_id]["meshes"]
            role_projection = {
                role: {
                    "bbox_px": _projected_bbox(
                        mesh["vertices_mm"],
                        frame=frame,
                    ),
                }
                for role, mesh in meshes.items()
            }
            for role, positive_x_side in (
                ("cad_positive_x_finger", True),
                ("cad_negative_x_finger", False),
            ):
                surface_point = _inner_surface_point(
                    meshes[role]["vertices_mm"],
                    positive_x_side=positive_x_side,
                )
                role_projection[role]["inner_surface_sample_mm"] = (
                    surface_point
                )
                role_projection[role]["inner_surface_sample_px"] = (
                    _project_point(surface_point, frame=frame)
                )
            captures.append(
                {
                    **plan_record,
                    "status": "NOT_RUN",
                    "visual_self_review": "NOT_RUN",
                    "retake_reasons": [],
                    "raw_sha256": _sha256(raw_path),
                    "source_state_report_path": str(source),
                    "source_state_report_sha256": _sha256(source),
                    "source_cad_path": data["source_path"],
                    "source_cad_sha256": data["source_sha256"],
                    "camera": frame,
                    "state_translation_mm": data["states"][state_id][
                        "translations_mm"
                    ],
                    "finger_minimum_distance_mm": data["states"][state_id][
                        "relationships"
                    ]["finger_to_finger"]["minimum_shape_distance_mm"],
                    "role_projection": role_projection,
                    "inner_surface_sample_method": (
                        "center of center-facing 8-percent X quantile over "
                        "the distal 62 percent of CAD vertices; annotation "
                        "only, not a physical contact point"
                    ),
                }
            )
    metadata = {
        "schema_version": 1,
        "status": "NOT_RUN",
        "renderer": {
            "blender_version": bpy.app.version_string,
            "render_engine": bpy.context.scene.render.engine,
            "resolution": list(RESOLUTION),
            "frame_margin": FRAME_MARGIN,
            "cad_length_unit": "millimetre",
            "blender_length_unit": "metre",
            "blender_metres_per_cad_millimetre": MM_TO_M,
        },
        "capture_count": len(captures),
        "captures": captures,
    }
    output = output_root / "render_metadata.json"
    output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"ALOHA_VIPER_RENDER_METADATA {output}")
    print(f"ALOHA_VIPER_RENDER_COUNT {len(captures)}")


if __name__ == "__main__":
    main()
