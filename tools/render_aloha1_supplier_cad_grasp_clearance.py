"""Render rejected/corrected complete-gripper CAD clearance evidence.

Run through Blender:
  blender --background --python tools/render_...py -- \
    --clearance-report REPORT --gripper-states STATES \
    --bottle-obj BOTTLE --output-root OUTPUT
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

RESOLUTION = (1600, 1000)
CAD_REFERENCE_OFFSETS_MM = {
    "x_from_global_y": -430.29999973392,
    "y_from_global_x": -0.0998902576627736,
    "z_from_global_z": -426.80133373174,
}
VIEWS = {
    "true_world_top": {
        "camera_forward_gripper": (1.0, 0.0, 0.0),
        "image_up_gripper": (0.0, 0.0, 1.0),
        "meaning": (
            "world top: gripper +X is world -Z vertical approach, "
            "so camera forward is gripper +X"
        ),
    },
    "world_side": {
        "camera_forward_gripper": (0.0, -1.0, 0.0),
        "image_up_gripper": (-1.0, 0.0, 0.0),
        "meaning": (
            "world side: image up is world +Z = gripper -X"
        ),
    },
}


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clearance-report", type=Path, required=True)
    parser.add_argument("--gripper-states", type=Path, required=True)
    parser.add_argument("--bottle-obj", type=Path, required=True)
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


def _material(
    name: str,
    color: tuple[float, float, float, float],
    *,
    alpha: float = 1.0,
) -> Any:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    rgba = (color[0], color[1], color[2], alpha)
    material.diffuse_color = rgba
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = rgba
    principled.inputs["Roughness"].default_value = 0.35
    principled.inputs["Metallic"].default_value = 0.05
    principled.inputs["Alpha"].default_value = alpha
    if alpha < 1.0:
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = "DITHERED"
        elif hasattr(material, "blend_method"):
            material.blend_method = "BLEND"
    return material


def _mesh_object(
    *,
    name: str,
    vertices_m: list[tuple[float, float, float]],
    triangles: list[list[int]] | list[tuple[int, int, int]],
    material: Any,
) -> Any:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices_m, [], triangles)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    return obj


def _cad_global_to_reference_m(
    point_mm: list[float] | tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        (-float(point_mm[1]) + CAD_REFERENCE_OFFSETS_MM["x_from_global_y"])
        * 0.001,
        (float(point_mm[0]) + CAD_REFERENCE_OFFSETS_MM["y_from_global_x"])
        * 0.001,
        (float(point_mm[2]) + CAD_REFERENCE_OFFSETS_MM["z_from_global_z"])
        * 0.001,
    )


def _parse_obj(path: Path) -> tuple[list[list[float]], list[list[int]]]:
    vertices = []
    triangles = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("f "):
            indices = [
                int(token.split("/")[0]) - 1
                for token in line.split()[1:]
            ]
            triangles.extend(
                [indices[0], indices[index], indices[index + 1]]
                for index in range(1, len(indices) - 1)
            )
    if not vertices or not triangles:
        raise RuntimeError(f"OBJ has no renderable geometry: {path}")
    return vertices, triangles


def _camera_matrix(
    *,
    location: Vector,
    forward: Vector,
    image_up: Vector,
) -> Matrix:
    local_y = image_up.normalized()
    local_z = -forward.normalized()
    local_x = local_y.cross(local_z).normalized()
    local_y = local_z.cross(local_x).normalized()
    return Matrix(
        (
            (local_x.x, local_y.x, local_z.x, location.x),
            (local_x.y, local_y.y, local_z.y, location.y),
            (local_x.z, local_y.z, local_z.z, location.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def _create_camera() -> Any:
    data = bpy.data.cameras.new("ClearanceEvidenceCamera")
    data.type = "ORTHO"
    data.clip_start = 0.01
    data.clip_end = 20.0
    camera = bpy.data.objects.new("ClearanceEvidenceCamera", data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def _configure_scene() -> Any:
    scene = bpy.context.scene
    # This project-local Blender 5.2 build exposes the Eevee engine under the
    # compatibility token BLENDER_EEVEE (verified by runtime enum readback).
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = RESOLUTION[0]
    scene.render.resolution_y = RESOLUTION[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -1.5
    scene.world.color = (0.018, 0.022, 0.03)
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.018, 0.022, 0.03, 1.0)
    background.inputs["Strength"].default_value = 0.08
    return _create_camera()


def _add_light(name: str, location: tuple[float, float, float]) -> None:
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = 900.0
    data.shape = "DISK"
    data.size = 1.2
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = location
    light.rotation_euler = (
        Vector((0.1, 0.0, 0.03)) - light.location
    ).to_track_quat("-Z", "Y").to_euler()


def _line_object(
    *,
    name: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
    material: Any,
) -> Any:
    start_vector = Vector(start)
    end_vector = Vector(end)
    direction = end_vector - start_vector
    midpoint = (start_vector + end_vector) / 2.0
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=24,
        radius=radius,
        depth=direction.length,
        location=midpoint,
    )
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(material)
    return obj


def _box_object(
    *,
    name: str,
    minimum: list[float],
    maximum: list[float],
    material: Any,
) -> Any:
    center = [
        (minimum[axis] + maximum[axis]) / 2.0
        for axis in range(3)
    ]
    scale = [
        maximum[axis] - minimum[axis]
        for axis in range(3)
    ]
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale
    obj.data.materials.append(material)
    modifier = obj.modifiers.new(name="AABBEdgeFrame", type="WIREFRAME")
    modifier.thickness = 0.0007
    modifier.use_replace = True
    bpy.context.view_layer.objects.active = obj
    obj.select_set(state=True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.select_set(state=False)
    return obj


def _project(
    point: list[float] | tuple[float, float, float],
    *,
    camera_record: dict[str, Any],
) -> list[float]:
    target = Vector(camera_record["target_m"])
    right = Vector(camera_record["image_right_gripper"])
    up = Vector(camera_record["image_up_gripper"])
    relative = Vector(point) - target
    horizontal = relative.dot(right)
    vertical = relative.dot(up)
    x = (
        0.5
        + horizontal / float(camera_record["ortho_width_m"])
    ) * RESOLUTION[0]
    y = (
        0.5
        - vertical / float(camera_record["ortho_height_m"])
    ) * RESOLUTION[1]
    return [float(x), float(y)]


def _geometry_signature(
    *,
    state: str,
    bottle_center_x_m: float,
    left_q_m: float,
    right_q_m: float,
) -> str:
    text = json.dumps(
        {
            "state": state,
            "bottle_center_x_m": bottle_center_x_m,
            "left_q_m": left_q_m,
            "right_q_m": right_q_m,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    args = _parse_args()
    report_path = args.clearance_report.resolve(strict=True)
    states_path = args.gripper_states.resolve(strict=True)
    bottle_path = args.bottle_obj.resolve(strict=True)
    output_root = args.output_root.resolve()
    raw_root = output_root / "screenshots_raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    states = json.loads(states_path.read_text(encoding="utf-8"))
    if report["status"] != "PASS" or states["status"] != "PASS":
        raise RuntimeError("CAD geometry inputs must both be PASS")

    bottle_vertices_mm, bottle_triangles = _parse_obj(bottle_path)
    closed_meshes = states["states"]["closed"]["meshes"]
    corrected_contact = report["contact_solution"]
    state_contracts = {
        "rejected_run13": {
            "bottle_center_x_m": report["station_selection"][
                "rejected_station"
            ]["station_m"],
            "left_q_m": 0.05498514324426651,
            "right_q_m": -0.06374162435531616,
            "grasp_x_m": report["station_selection"]["rejected_station"][
                "station_m"
            ],
            "status": "FAIL",
        },
        "corrected_cad": {
            "bottle_center_x_m": report["station_selection"][
                "selected_station_m"
            ],
            "left_q_m": corrected_contact["left_finger_q_m"],
            "right_q_m": corrected_contact["right_finger_q_m"],
            "grasp_x_m": report["grasp_frame"]["origin_reference_m"][0],
            "status": "PASS",
        },
    }

    _reset_scene()
    camera = _configure_scene()
    _add_light("KeyLight", (0.45, -0.55, 0.55))
    _add_light("FillLight", (-0.25, 0.45, 0.30))
    bpy.data.lights["KeyLight"].energy = 55.0
    bpy.data.lights["FillLight"].energy = 28.0
    materials = {
        "shell": _material("Shell", (0.16, 0.2, 0.27, 1.0), alpha=0.13),
        "left": _material("LeftFinger", (0.025, 0.16, 1.0, 1.0)),
        "right": _material("RightFinger", (1.0, 0.18, 0.015, 1.0)),
        "bottle": _material("Bottle", (0.015, 0.62, 0.78, 1.0)),
        "bar_box": _material("RuntimeBarEnvelope", (1.0, 0.015, 0.025, 1.0)),
        "x_axis": _material("XAxis", (0.95, 0.16, 0.16, 1.0)),
        "y_axis": _material("YAxis", (0.14, 0.9, 0.28, 1.0)),
        "z_axis": _material("ZAxis", (0.16, 0.45, 1.0, 1.0)),
    }
    objects_by_state: dict[str, list[Any]] = {}
    key_points_by_state: dict[str, dict[str, list[float]]] = {}
    all_points: list[tuple[float, float, float]] = []
    for state_name, contract in state_contracts.items():
        state_objects = []
        shell_mesh = closed_meshes["gripper_shell"]
        shell_vertices = [
            _cad_global_to_reference_m(point)
            for point in shell_mesh["vertices_mm"]
        ]
        state_objects.append(
            _mesh_object(
                name=f"{state_name}_supplier_shell",
                vertices_m=shell_vertices,
                triangles=shell_mesh["triangles"],
                material=materials["shell"],
            )
        )
        for role, q_key, material_key in (
            ("cad_positive_x_finger", "left_q_m", "left"),
            ("cad_negative_x_finger", "right_q_m", "right"),
        ):
            mesh = closed_meshes[role]
            delta_m = float(contract[q_key]) - (
                0.021 if q_key == "left_q_m" else -0.021
            )
            vertices = []
            for point in mesh["vertices_mm"]:
                transformed = list(_cad_global_to_reference_m(point))
                transformed[1] += delta_m
                vertices.append(tuple(transformed))
            state_objects.append(
                _mesh_object(
                    name=f"{state_name}_{role}",
                    vertices_m=vertices,
                    triangles=mesh["triangles"],
                    material=materials[material_key],
                )
            )
            all_points.extend(vertices)
        bottle_translation = (
            float(contract["bottle_center_x_m"]),
            0.0,
            -0.069,
        )
        bottle_vertices = [
            (
                vertex[0] * 0.001 + bottle_translation[0],
                vertex[1] * 0.001 + bottle_translation[1],
                vertex[2] * 0.001 + bottle_translation[2],
            )
            for vertex in bottle_vertices_mm
        ]
        state_objects.append(
            _mesh_object(
                name=f"{state_name}_Bottle500",
                vertices_m=bottle_vertices,
                triangles=bottle_triangles,
                material=materials["bottle"],
            )
        )
        bar_aabb = report["forbidden_envelopes"][
            "runtime_urdf_gripper_bar"
        ]["gripper_reference_aabb_m"]
        state_objects.append(
            _box_object(
                name=f"{state_name}_runtime_bar_AABB",
                minimum=bar_aabb["min"],
                maximum=bar_aabb["max"],
                material=materials["bar_box"],
            )
        )
        axis_origin = (0.0, 0.0, 0.0)
        for axis_name, end, material_key in (
            ("X", (0.055, 0.0, 0.0), "x_axis"),
            ("Y", (0.0, 0.055, 0.0), "y_axis"),
            ("Z", (0.0, 0.0, 0.055), "z_axis"),
        ):
            state_objects.append(
                _line_object(
                    name=f"{state_name}_{axis_name}_axis",
                    start=axis_origin,
                    end=end,
                    radius=0.0012,
                    material=materials[material_key],
                )
            )
        for obj in state_objects:
            obj.hide_render = True
            obj.hide_viewport = True
        objects_by_state[state_name] = state_objects
        key_points_by_state[state_name] = {
            "gripper_link_origin": [0.0, 0.0, 0.0],
            "official_ee_helper": [0.1072, 0.0, 0.0],
            "bottle_axis_center": [
                float(contract["bottle_center_x_m"]),
                0.0,
                0.0,
            ],
            "grasp_frame_origin": [
                float(contract["grasp_x_m"]),
                0.0,
                0.0,
            ],
            "bottle_axis_a": [
                float(contract["bottle_center_x_m"]),
                0.0,
                -0.069,
            ],
            "bottle_axis_b": [
                float(contract["bottle_center_x_m"]),
                0.0,
                0.137,
            ],
        }
        if state_name == "corrected_cad":
            key_points_by_state[state_name]["left_contact"] = list(
                corrected_contact["left_contact_reference_m"]
            )
            key_points_by_state[state_name]["right_contact"] = list(
                corrected_contact["right_contact_reference_m"]
            )
        all_points.extend(shell_vertices)
        all_points.extend(bottle_vertices)

    minimum = Vector(
        tuple(min(point[axis] for point in all_points) for axis in range(3))
    )
    maximum = Vector(
        tuple(max(point[axis] for point in all_points) for axis in range(3))
    )
    target = (minimum + maximum) / 2.0
    captures = []
    for view_name, view_contract in VIEWS.items():
        forward = Vector(view_contract["camera_forward_gripper"]).normalized()
        image_up = Vector(view_contract["image_up_gripper"]).normalized()
        image_right = image_up.cross(-forward).normalized()
        horizontal_extent = max(
            abs((Vector(point) - target).dot(image_right))
            for point in all_points
        )
        vertical_extent = max(
            abs((Vector(point) - target).dot(image_up))
            for point in all_points
        )
        # Use a generous evidence margin. Earlier 2.4x framing still clipped
        # Bottle500 and the proximal shell after Blender camera projection.
        ortho_width = max(horizontal_extent * 4.2, 0.42)
        ortho_height = max(vertical_extent * 4.2, 0.315)
        required_aspect = RESOLUTION[0] / RESOLUTION[1]
        if ortho_width / ortho_height < required_aspect:
            ortho_width = ortho_height * required_aspect
        else:
            ortho_height = ortho_width / required_aspect
        camera_location = target - forward * 1.4
        camera.matrix_world = _camera_matrix(
            location=camera_location,
            forward=forward,
            image_up=image_up,
        )
        camera.data.ortho_scale = ortho_height
        camera_record = {
            "projection": "ORTHOGRAPHIC",
            "location_m": list(camera_location),
            "target_m": list(target),
            "camera_forward_gripper": list(forward),
            "image_up_gripper": list(image_up),
            "image_right_gripper": list(image_right),
            "ortho_width_m": ortho_width,
            "ortho_height_m": ortho_height,
            "matrix_world": [
                list(row)
                for row in camera.matrix_world
            ],
            "meaning": view_contract["meaning"],
        }
        for state_name, contract in state_contracts.items():
            for candidate_state, state_objects in objects_by_state.items():
                hidden = candidate_state != state_name
                for obj in state_objects:
                    obj.hide_render = hidden
                    obj.hide_viewport = hidden
            raw_path = raw_root / f"{state_name}_{view_name}_raw.png"
            bpy.context.scene.render.filepath = str(raw_path)
            bpy.ops.render.render(write_still=True)
            projected = {
                name: _project(point, camera_record=camera_record)
                for name, point in key_points_by_state[state_name].items()
            }
            captures.append(
                {
                    "state": state_name,
                    "view": view_name,
                    "status": contract["status"],
                    "raw_absolute_path": str(raw_path),
                    "raw_sha256": _sha256(raw_path),
                    "width_px": RESOLUTION[0],
                    "height_px": RESOLUTION[1],
                    "camera": camera_record,
                    "key_points_reference_m": key_points_by_state[state_name],
                    "projected_points_px": projected,
                    "bottle_center_x_m": contract["bottle_center_x_m"],
                    "left_finger_q_m": contract["left_q_m"],
                    "right_finger_q_m": contract["right_q_m"],
                    "geometry_signature": _geometry_signature(
                        state=state_name,
                        bottle_center_x_m=float(contract["bottle_center_x_m"]),
                        left_q_m=float(contract["left_q_m"]),
                        right_q_m=float(contract["right_q_m"]),
                    ),
                }
            )
    metadata = {
        "schema_version": 1,
        "status": "PASS",
        "classification": "STATIC_COMPLETE_GRIPPER_CLEARANCE_RENDER",
        "inputs": {
            "clearance_report": {
                "absolute_path": str(report_path),
                "sha256": _sha256(report_path),
            },
            "gripper_states": {
                "absolute_path": str(states_path),
                "sha256": _sha256(states_path),
            },
            "bottle_obj": {
                "absolute_path": str(bottle_path),
                "sha256": _sha256(bottle_path),
            },
        },
        "blender_version": bpy.app.version_string,
        "captures": captures,
        "task8": "NOT_RUN",
    }
    metadata_path = output_root / "render_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "capture_count": len(captures),
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        )
    )


main()
