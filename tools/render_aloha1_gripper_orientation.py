#!/usr/bin/env python3
"""Render a read-only ALOHA 1 gripper orientation diagnostic.

The extraction child reads legal articulation states from the local Isaac Sim
5.1 PhysX tensor view.  Blender then renders the baked, gripper-frame meshes
with a real depth buffer.  The historical source USD is never saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

from aloha1_mapping.gripper_orientation import classify_orientation
from aloha1_mapping.gripper_orientation import expected_capture_names
from aloha1_mapping.gripper_orientation import finger_state_targets
from aloha1_mapping.gripper_orientation import inward_surface_normal_y
from aloha1_mapping.gripper_orientation import obj_text
from aloha1_mapping.gripper_orientation import physical_side_order
from aloha1_mapping.gripper_orientation import surface_normal_gate
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USD_PATH = PROJECT_ROOT / "local_eval_assets/aloha_isaac_assets/aloha_viperx.usd"
OUTPUT_ROOT = PROJECT_ROOT / ".codex/artifacts/20260728-aloha1-gripper-orientation"
STATE_MANIFEST = OUTPUT_ROOT / "runtime_mesh_states.json"
FINAL_MANIFEST = OUTPUT_ROOT / "orientation_diagnostic_manifest.json"
BLENDER_SCRIPT = PROJECT_ROOT / "tools/aloha1_mapping/blender_gripper_orientation_render.py"
BLENDER = Path("/home/eii/.local/bin/blender")

EXPECTED_USD_SHA256 = "b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e"
SOURCE_ASSET_ROOT = PROJECT_ROOT / ".venv/lib/python3.11/site-packages/gym_aloha/assets"
SOURCE_MESHES = {
    "physical_left": {
        "path": SOURCE_ASSET_ROOT / "vx300s_10_custom_finger_left.stl",
        "sha256": ("df73ae5b9058e5d50a6409ac2ab687dade75053a86591bb5e23ab051dbf2d659"),
    },
    "physical_right": {
        "path": SOURCE_ASSET_ROOT / "vx300s_10_custom_finger_right.stl",
        "sha256": ("56fb3cc1236d4193106038adf8e457c7252ae9e86c7cee6dabf0578c53666358"),
    },
    "rejected_generic_viperx_finger": {
        "path": SOURCE_ASSET_ROOT / "vx300s_10_gripper_finger.stl",
        "sha256": ("a4baacd9a64df1be60ea5e98f50f3c660e1b7a1fe9684aace6004c5058c09483"),
    },
}

ARTICULATION_PATH = "/World/candidate/vx300s_left/vx300s_left"
GRIPPER_LINK = "vx300s_left_gripper_link"
LEFT_DOF_SUFFIX = "left_finger"
RIGHT_DOF_SUFFIX = "right_finger"
MESH_SPECS = {
    "gripper_body": {
        "owner": "vx300s_left_gripper_link",
        "path": ("/World/candidate/vx300s_left/vx300s_left_gripper_link/visuals/vx300s_7_gripper/vx300s_7_gripper"),
    },
    "gripper_bar": {
        "owner": "vx300s_left_gripper_link",
        "path": (
            "/World/candidate/vx300s_left/vx300s_left_gripper_link/visuals/vx300s_9_gripper_bar/vx300s_9_gripper_bar"
        ),
    },
    "gripper_prop": {
        "owner": "vx300s_left_gripper_prop_link",
        "path": (
            "/World/candidate/vx300s_left/vx300s_left_gripper_prop_link/visuals/"
            "vx300s_8_gripper_prop/vx300s_8_gripper_prop"
        ),
    },
    "physical_left": {
        "owner": "vx300s_left_left_finger_link",
        "path": (
            "/World/candidate/vx300s_left/vx300s_left_left_finger_link/visuals/"
            "vx300s_10_gripper_finger_left/vx300s_10_gripper_finger_left"
        ),
    },
    "physical_right": {
        "owner": "vx300s_left_right_finger_link",
        "path": (
            "/World/candidate/vx300s_left/vx300s_left_right_finger_link/visuals/"
            "vx300s_10_gripper_finger_right/vx300s_10_gripper_finger_right"
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_png(path: Path) -> str:
    """Remove renderer metadata and return a deterministic pixel hash."""
    from PIL import Image

    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        pixel_hash = hashlib.sha256(rgba.tobytes()).hexdigest()
        rgba.save(path, format="PNG", compress_level=9, optimize=False)
    return pixel_hash


def _validate_frozen_inputs() -> dict[str, Any]:
    actual_usd_hash = sha256(USD_PATH)
    if actual_usd_hash != EXPECTED_USD_SHA256:
        raise RuntimeError(f"Frozen USD hash changed: {actual_usd_hash} != {EXPECTED_USD_SHA256}")
    source_readback: dict[str, Any] = {}
    for label, item in SOURCE_MESHES.items():
        path = Path(item["path"])
        actual_hash = sha256(path)
        if actual_hash != item["sha256"]:
            raise RuntimeError(f"Frozen source mesh hash changed for {label}: {actual_hash}")
        source_readback[label] = {
            "path": str(path),
            "sha256": actual_hash,
        }
    return {
        "usd": {"path": str(USD_PATH), "sha256": actual_usd_hash},
        "source_meshes": source_readback,
    }


def _pose_matrix(pose: np.ndarray, gf: Any) -> Any:
    """Build a local Isaac 5.1 position+xyzw link pose matrix."""
    position = pose[:3]
    x, y, z, w = pose[3:]
    matrix = gf.Matrix4d(1.0)
    matrix.SetRotate(gf.Quatd(float(w), gf.Vec3d(float(x), float(y), float(z))))
    matrix.SetTranslateOnly(gf.Vec3d(*map(float, position)))
    return matrix


def _triangles(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if not len(points) or not len(counts) or not np.all(counts == 3):
        raise RuntimeError(f"Expected non-empty triangle mesh: {mesh.GetPath()}")
    return points, indices.reshape((-1, 3))


def _unique_suffix_index(names: list[str], suffix: str) -> int:
    matches = [index for index, name in enumerate(names) if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one DOF ending in {suffix!r}, found {[names[index] for index in matches]}")
    return matches[0]


def _extract_only() -> None:
    frozen_inputs = _validate_frozen_inputs()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        import omni.usd
        from pxr import Gf
        from pxr import Usd
        from pxr import UsdGeom

        world = World(stage_units_in_meters=1.0)
        stage = omni.usd.get_context().get_stage()
        add_reference_to_stage(str(USD_PATH), "/World/candidate")
        articulation = world.scene.add(
            SingleArticulation(
                prim_path=ARTICULATION_PATH,
                name="historical_aloha_vx300s_left",
            )
        )
        world.reset()

        dof_names = list(articulation.dof_names)
        left_index = _unique_suffix_index(dof_names, LEFT_DOF_SUFFIX)
        right_index = _unique_suffix_index(dof_names, RIGHT_DOF_SUFFIX)
        limits_array = np.asarray(articulation.dof_properties["lower"], dtype=float)
        upper_array = np.asarray(articulation.dof_properties["upper"], dtype=float)
        limits = {
            "left": (
                float(limits_array[left_index]),
                float(upper_array[left_index]),
            ),
            "right": (
                float(limits_array[right_index]),
                float(upper_array[right_index]),
            ),
        }
        if limits["left"][0] <= 0.0 or limits["right"][1] >= 0.0:
            raise RuntimeError(f"Unexpected imported finger limits: {limits}")

        # Isaac Sim 5.1 exposes runtime link ordering and link poses through
        # this initialized PhysX tensor bridge; the readback is diagnostic-only.
        articulation_view = articulation._articulation_view  # noqa: SLF001
        body_names = list(articulation_view.body_names)
        body_indices = {name: body_names.index(name) for name in body_names}
        physics_view = articulation_view._physics_view  # noqa: SLF001
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        local_mesh_data: dict[str, Any] = {}
        for label, spec in MESH_SPECS.items():
            mesh_prim = stage.GetPrimAtPath(spec["path"])
            owner_path = f"/World/candidate/vx300s_left/{spec['owner']}"
            owner_prim = stage.GetPrimAtPath(owner_path)
            if not mesh_prim.IsA(UsdGeom.Mesh) or not owner_prim.IsValid():
                raise RuntimeError(f"Missing expected mesh/owner: {spec['path']} / {owner_path}")
            relative, reset_stack = xform_cache.ComputeRelativeTransform(mesh_prim, owner_prim)
            if reset_stack:
                raise RuntimeError(f"Unexpected resetXformStack between {spec['path']} and {owner_path}")
            points, faces = _triangles(UsdGeom.Mesh(mesh_prim))
            local_mesh_data[label] = {
                "relative": relative,
                "points": points,
                "faces": faces,
                "owner": spec["owner"],
                "prim_path": spec["path"],
            }

        state_records: dict[str, Any] = {}
        for state in ("closed", "open"):
            targets = finger_state_targets(limits, state)
            positions = np.asarray(articulation.get_joint_positions(), dtype=float)
            positions[left_index] = targets["left"]
            positions[right_index] = targets["right"]
            articulation.set_joint_positions(positions)
            articulation.set_joint_velocities(np.zeros_like(positions))

            readback = np.asarray(articulation.get_joint_positions(), dtype=float)
            if not np.allclose(
                readback[[left_index, right_index]],
                [targets["left"], targets["right"]],
                atol=1e-8,
                rtol=0.0,
            ):
                raise RuntimeError(f"Finger q readback mismatch for {state}: {readback[[left_index, right_index]]}")
            link_poses = np.asarray(physics_view.get_link_transforms(), dtype=np.float64)[0]
            gripper_world = _pose_matrix(link_poses[body_indices[GRIPPER_LINK]], Gf)
            gripper_world_inverse = gripper_world.GetInverse()

            state_dir = OUTPUT_ROOT / state
            state_dir.mkdir(parents=True, exist_ok=True)
            mesh_records: dict[str, Any] = {}
            baked_meshes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for label, mesh_data in local_mesh_data.items():
                owner_pose = link_poses[body_indices[mesh_data["owner"]]]
                owner_world = _pose_matrix(owner_pose, Gf)
                baked_points = np.asarray(
                    [
                        tuple(
                            gripper_world_inverse.Transform(
                                owner_world.Transform(mesh_data["relative"].Transform(Gf.Vec3d(*map(float, point))))
                            )
                        )
                        for point in mesh_data["points"]
                    ],
                    dtype=np.float64,
                )
                faces = mesh_data["faces"]
                obj_path = state_dir / f"{label}.obj"
                obj_path.write_text(
                    obj_text(label, baked_points, faces),
                    encoding="utf-8",
                )
                center = 0.5 * (baked_points.min(axis=0) + baked_points.max(axis=0))
                mesh_records[label] = {
                    "prim_path": mesh_data["prim_path"],
                    "owner_link": mesh_data["owner"],
                    "obj_path": str(obj_path),
                    "obj_sha256": sha256(obj_path),
                    "point_count": len(baked_points),
                    "face_count": len(faces),
                    "aabb_min_m": baked_points.min(axis=0).tolist(),
                    "aabb_max_m": baked_points.max(axis=0).tolist(),
                    "aabb_center_m": center.tolist(),
                    "runtime_owner_pose_position_xyzw": owner_pose.tolist(),
                }
                baked_meshes[label] = (baked_points, faces)

            left_points, left_faces = baked_meshes["physical_left"]
            right_points, right_faces = baked_meshes["physical_right"]
            left_center_y = mesh_records["physical_left"]["aabb_center_m"][1]
            right_center_y = mesh_records["physical_right"]["aabb_center_m"][1]
            left_normal_y = inward_surface_normal_y(left_points, left_faces, "left")
            right_normal_y = inward_surface_normal_y(right_points, right_faces, "right")
            state_records[state] = {
                "finger_targets_m": targets,
                "finger_readback_m": {
                    "left": float(readback[left_index]),
                    "right": float(readback[right_index]),
                },
                "gripper_link_pose_position_xyzw": link_poses[body_indices[GRIPPER_LINK]].tolist(),
                "meshes": mesh_records,
                "geometry_metrics": {
                    "left_aabb_center_y_m": left_center_y,
                    "right_aabb_center_y_m": right_center_y,
                    "center_separation_m": left_center_y - right_center_y,
                    "left_inward_normal_y": left_normal_y,
                    "right_inward_normal_y": right_normal_y,
                    "physical_side_order_ok": physical_side_order(left_center_y, right_center_y),
                    "inward_normals_ok": surface_normal_gate(left_normal_y, right_normal_y),
                    "crossed_centerline": not (left_center_y > 0.0 and right_center_y < 0.0),
                },
            }

        closed_metrics = state_records["closed"]["geometry_metrics"]
        open_metrics = state_records["open"]["geometry_metrics"]
        classification = classify_orientation(
            side_order_ok=(closed_metrics["physical_side_order_ok"] and open_metrics["physical_side_order_ok"]),
            inward_normals_ok=(closed_metrics["inward_normals_ok"] and open_metrics["inward_normals_ok"]),
            closed_aperture_m=closed_metrics["center_separation_m"],
            open_aperture_m=open_metrics["center_separation_m"],
            crossed_centerline=(closed_metrics["crossed_centerline"] or open_metrics["crossed_centerline"]),
        )
        manifest = {
            "schema_version": 1,
            "scope": ("read-only legal articulation-state extraction for orientation diagnosis; no source stage save"),
            "runtime": {
                "isaac_sim": "5.1.0.0",
                "kit": "107.3.3",
                "transform_source": ("SingleArticulation PhysX tensor view get_link_transforms; no physics step"),
                "coordinate_frame": ("vx300s_left_gripper_link; +X forward, +Y physical left, +Z up"),
                "dof_order": dof_names,
                "body_order": body_names,
                "finger_limits_m": {side: {"lower": values[0], "upper": values[1]} for side, values in limits.items()},
            },
            "frozen_inputs": frozen_inputs,
            "states": state_records,
            "classification": classification,
            "source_usd_modified": False,
            "source_usd_saved": False,
            "active_gui_stage_switched": False,
        }
        STATE_MANIFEST.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if sha256(USD_PATH) != EXPECTED_USD_SHA256:
            raise RuntimeError("Source USD changed during extraction")
        print(
            json.dumps(
                {
                    "status": classification["status"],
                    "state_manifest": str(STATE_MANIFEST),
                },
                indent=2,
            ),
            flush=True,
        )
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close()


def _render_and_finalize() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    previous_capture_hashes = None
    previous_user_confirmation = None
    if FINAL_MANIFEST.is_file():
        previous_manifest = json.loads(FINAL_MANIFEST.read_text(encoding="utf-8"))
        previous_capture_hashes = [capture["sha256"] for capture in previous_manifest.get("captures", [])]
        previous_user_confirmation = previous_manifest.get("user_visual_confirmation")
    extraction_log = OUTPUT_ROOT / "isaac_runtime_extraction.log"
    command = [sys.executable, str(Path(__file__).resolve()), "--extract-only"]
    with extraction_log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"Isaac extraction failed with {completed.returncode}; see {extraction_log}")
    if not BLENDER.is_file():
        raise RuntimeError(f"Blender executable missing: {BLENDER}")

    blender_log = OUTPUT_ROOT / "blender_render.log"
    blender_command = [
        str(BLENDER),
        "--background",
        "--factory-startup",
        "--python",
        str(BLENDER_SCRIPT),
        "--",
        "--manifest",
        str(STATE_MANIFEST),
        "--output-root",
        str(OUTPUT_ROOT),
    ]
    with blender_log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            blender_command,
            cwd=PROJECT_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"Blender render failed with {completed.returncode}; see {blender_log}")

    manifest = json.loads(STATE_MANIFEST.read_text(encoding="utf-8"))
    captures = []
    for name in expected_capture_names():
        path = OUTPUT_ROOT / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected non-empty capture missing: {path}")
        pixel_sha256 = _normalize_png(path)
        captures.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "pixel_sha256": pixel_sha256,
                "bytes": path.stat().st_size,
                "width": 1280,
                "height": 900,
            }
        )
    source_hash_after = sha256(USD_PATH)
    if source_hash_after != EXPECTED_USD_SHA256:
        raise RuntimeError("Frozen source USD changed during render")
    capture_hashes = [capture["sha256"] for capture in captures]
    deterministic_rerun = "PASS" if previous_capture_hashes == capture_hashes else "NOT_YET_VERIFIED"
    user_confirmed = isinstance(previous_user_confirmation, dict) and previous_user_confirmation.get("status") == "PASS"
    manifest.update(
        {
            "captures": captures,
            "logs": {
                "isaac_runtime_extraction": str(extraction_log),
                "blender_render": str(blender_log),
            },
            "source_usd_sha256_after": source_hash_after,
            "previous_screenshot_status": "INVALID_AUTHORED_ZERO_STATE",
            "screenshot_machine_gate": "PASS",
            "deterministic_rerun": deterministic_rerun,
            "user_visual_confirmation": previous_user_confirmation or {"status": "NOT_RECORDED"},
            "awaiting": (
                "CORRECT_FINGER_ASSET_INTEGRATION_BEFORE_TASK5_RERUN"
                if user_confirmed
                else "ASSISTANT_VISUAL_INSPECTION_THEN_USER_CONFIRMATION"
            ),
        }
    )
    FINAL_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": manifest["classification"]["status"],
                "manifest": str(FINAL_MANIFEST),
                "captures": [capture["path"] for capture in captures],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    if args.extract_only:
        _extract_only()
    else:
        _render_and_finalize()


if __name__ == "__main__":
    main()
