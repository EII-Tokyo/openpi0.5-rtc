#!/usr/bin/env python3
"""Cook and visualize the user-confirmed ALOHA custom finger colliders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np
from PIL import Image

from tools.aloha1_mapping.correct_finger_asset import (
    load_correct_finger_profile,
)
from tools.aloha1_mapping.correct_finger_asset import sha256_file
from tools.aloha1_mapping.correct_finger_asset import (
    verify_correct_finger_sources,
)
from tools.aloha1_mapping.screenshot_manifest import (
    build_screenshot_manifest,
)
from tools.aloha1_mapping.screenshot_manifest import validate_screenshot
from tools.compare_aloha1_gripper_colliders import (
    _build_direct_mesh_cooking_probe,
)
from tools.compare_aloha1_gripper_colliders import _collider_inventory
from tools.compare_aloha1_gripper_colliders import _cook_finger_colliders
from tools.compare_aloha1_gripper_colliders import _finger_collider_paths
from tools.compare_aloha1_gripper_colliders import _local_api_probe
from tools.compare_aloha1_gripper_colliders import _render_cooked
from tools.compare_aloha1_gripper_colliders import _sampling_difference


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _source_mesh_from_collider(stage: Any, collider_path: str, approximation: str):
    """Return the authored surface baked into collider-local metres.

    The local PhysX 5.1 public cooking request cannot parse an importer-style
    CollisionAPI Xform directly. The shared diagnostic helper creates an
    equivalent in-memory Mesh with all descendant transforms baked. Reusing
    that exact mesh also keeps the numerical surface comparison in the same
    frame and units as the cooked convex pieces.
    """
    import trimesh
    from pxr import UsdGeom

    probe_stage, probe_path, probe = _build_direct_mesh_cooking_probe(
        stage,
        collider_path,
        approximation,
    )
    mesh = UsdGeom.Mesh(probe_stage.GetPrimAtPath(probe_path))
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    indices = list(mesh.GetFaceVertexIndicesAttr().Get())
    counts = list(mesh.GetFaceVertexCountsAttr().Get())
    faces: list[list[int]] = []
    offset = 0
    for count in counts:
        polygon = indices[offset : offset + count]
        offset += count
        faces.extend(
            [
                [polygon[0], polygon[index], polygon[index + 1]]
                for index in range(1, count - 1)
            ]
        )
    return (
        trimesh.Trimesh(
            vertices=points,
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        ),
        probe,
    )


def _resize_exact(path: Path, resolution: tuple[int, int]) -> None:
    with Image.open(path) as image:
        resized = image.convert("RGBA").resize(
            resolution,
            Image.Resampling.LANCZOS,
        )
        resized.save(path)


def _combine_side_by_side(
    left: Path,
    right: Path,
    destination: Path,
    resolution: tuple[int, int],
) -> None:
    half_width = resolution[0] // 2
    canvas = Image.new("RGBA", resolution, (255, 255, 255, 255))
    for index, source in enumerate((left, right)):
        with Image.open(source) as image:
            panel = image.convert("RGBA").resize(
                (half_width, resolution[1]),
                Image.Resampling.LANCZOS,
            )
        canvas.alpha_composite(panel, (index * half_width, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _revalidate_capture(
    capture: dict[str, Any],
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    refreshed = validate_screenshot(
        Path(capture["absolute_path"]),
        artifact_root=artifact_root,
        phase=capture["phase"],
        capture_name=capture["capture_name"],
        gate_status=capture["capture_gate_status"],
        camera=capture["camera"],
        simulation=capture["simulation"],
    )
    for key in ("file_sha256", "decoded_pixel_sha256"):
        if refreshed[key] != capture[key]:
            raise RuntimeError(
                f"previous screenshot changed after validation: "
                f"{capture['absolute_path']} ({key})"
            )
    return refreshed


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Correct ALOHA Finger Collider Geometry Audit",
        "",
        f"Status: **{report['status']}**",
        "",
        "This audit uses only the user-confirmed custom ALOHA finger meshes and "
        "the local Isaac Sim 5.1 / Kit 107.3.3 / PhysX 107.3.26 runtime.",
        "",
        "| Profile | Robot | Side | Approximation readback | Cooked pieces | "
        "Sum convex volume (m³) |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for profile_name, profile in report["profiles"].items():
        for asset in profile["assets"]:
            for collider in asset["colliders"].values():
                lines.append(
                    f"| `{profile_name}` | `{asset['robot']}` | "
                    f"`{collider['side']}` | "
                    f"`{collider['approximation_readback']}` | "
                    f"{collider['piece_count']} | "
                    f"{collider['sum_piece_volume']} |"
                )
    lines.extend(
        [
            "",
            "The PNGs are numerical cooked-collider visualizations. They are "
            "supplemental to, not substitutes for, the Isaac runtime contact "
            "and hold screenshots.",
            "",
            f"- Screenshot root: `{report['screenshot_root_absolute']}`",
            f"- Full screenshot manifest: "
            f"`{report['full_screenshot_manifest_absolute']}`",
            "- Default collider: **unchanged**",
            "- Task 8: **NOT_RUN**",
        ]
    )
    return "\n".join(lines) + "\n"


def run(
    app: Any,
    *,
    project_root: Path,
    profile_path: Path,
    report_path: Path,
    markdown_path: Path,
    screenshot_manifest_path: Path,
) -> dict[str, Any]:
    from pxr import Usd
    from pxr import UsdPhysics

    profile = load_correct_finger_profile(profile_path, project_root)
    source_before = verify_correct_finger_sources(profile, project_root)
    if source_before["status"] != "PASS":
        raise RuntimeError("correct-finger protected-source preflight failed")

    preflight = json.loads(
        (
            project_root
            / "reports/aloha1_mapping/gripper_correct_finger_preflight.json"
        ).read_text(encoding="utf-8")
    )
    runtime_manifest = json.loads(
        (
            project_root
            / "reports/aloha1_mapping/"
            "gripper_correct_finger_screenshot_manifest.json"
        ).read_text(encoding="utf-8")
    )
    artifact_root = (
        project_root / profile["diagnostic_directories"]["screenshots"]
    ).resolve()
    resolution = tuple(profile["screenshots"]["resolution"])
    geometry_root = artifact_root / "collider_geometry"
    geometry_root.mkdir(parents=True, exist_ok=True)

    diagnostic_by_key = {
        (item["robot"], item["approximation"]): item
        for item in preflight["diagnostic_assets"]
    }
    local_api = _local_api_probe()
    if local_api["urdf_importer"]["version"] != "2.4.30":
        raise RuntimeError(f"unexpected URDF importer: {local_api['urdf_importer']}")

    geometry_captures: list[dict[str, Any]] = []
    profile_results: dict[str, Any] = {}
    for profile_name, profile_config in profile["profiles"].items():
        approximation = profile_config["approximation"]
        assets = []
        for robot in ("follower_left", "follower_right"):
            diagnostic = diagnostic_by_key[(robot, approximation)]
            asset_path = Path(diagnostic["absolute_path"]).resolve(strict=True)
            stage = Usd.Stage.Open(str(asset_path))
            if stage is None:
                raise RuntimeError(f"failed to open diagnostic asset: {asset_path}")
            inventory = _collider_inventory(stage)
            finger_paths = _finger_collider_paths(inventory)
            cooking = _cook_finger_colliders(app, asset_path)
            collider_results: dict[str, Any] = {}
            side_visualizations: dict[str, dict[str, Path]] = {}
            for collider_path in finger_paths:
                side = (
                    "left"
                    if "_left_finger_link/" in collider_path
                    else "right"
                )
                prim = stage.GetPrimAtPath(collider_path)
                approximation_readback = UsdPhysics.MeshCollisionAPI(
                    prim
                ).GetApproximationAttr().Get()
                if approximation_readback != approximation:
                    raise RuntimeError(
                        f"approximation mismatch at {collider_path}: "
                        f"{approximation_readback} != {approximation}"
                    )
                cooked = cooking["colliders"][collider_path]
                authored_surface, direct_probe = _source_mesh_from_collider(
                    stage,
                    collider_path,
                    approximation,
                )
                cooked["source_surface_sampling"] = _sampling_difference(
                    authored_surface,
                    cooked,
                )
                token = (
                    "hull"
                    if approximation == "convexHull"
                    else "decomposition"
                )
                prefix = f"{robot}_{token}"
                overview_path = (
                    geometry_root
                    / f"{prefix}_{side}_finger_overview_supplement.png"
                )
                inner_path = (
                    geometry_root
                    / f"{prefix}_{side}_finger_inner_surface_supplement.png"
                )
                distal_path = (
                    geometry_root
                    / f"{prefix}_{side}_finger_distal_supplement.png"
                )
                _render_cooked(
                    cooked,
                    overview_path=overview_path,
                    closeup_path=distal_path,
                    inner_path=inner_path,
                    title=f"{robot} {side} {approximation}",
                )
                for path in (overview_path, inner_path, distal_path):
                    _resize_exact(path, resolution)
                side_visualizations[side] = {
                    "overview": overview_path,
                    "inner": inner_path,
                    "distal": distal_path,
                }

                source = profile["source"]["meshes"][side]
                collider_results[collider_path] = {
                    **cooked,
                    "side": side,
                    "source_stl_absolute_path": str(
                        (project_root / source["path"]).resolve(strict=True)
                    ),
                    "source_stl_sha256": source["sha256"],
                    "source_stl_triangle_count": source["triangle_count"],
                    "approximation_readback": approximation_readback,
                    "direct_mesh_probe": direct_probe,
                    "visualization": {
                        "status": "PASS",
                        "side_specific_overview_absolute_path": str(
                            overview_path
                        ),
                        "side_specific_overview_sha256": sha256_file(
                            overview_path
                        ),
                        "side_specific_inner_surface_absolute_path": str(
                            inner_path
                        ),
                        "side_specific_inner_surface_sha256": sha256_file(
                            inner_path
                        ),
                        "distal_supplement_absolute_path": str(distal_path),
                        "distal_supplement_sha256": sha256_file(distal_path),
                    },
                }
            if set(side_visualizations) != {"left", "right"}:
                raise RuntimeError(
                    f"both finger visualizations were not produced: "
                    f"{sorted(side_visualizations)}"
                )
            overview_name = f"{prefix}_overview"
            inner_name = f"{prefix}_inner_surface"
            combined_overview = geometry_root / f"{overview_name}.png"
            combined_inner = geometry_root / f"{inner_name}.png"
            _combine_side_by_side(
                side_visualizations["left"]["overview"],
                side_visualizations["right"]["overview"],
                combined_overview,
                resolution,
            )
            _combine_side_by_side(
                side_visualizations["left"]["inner"],
                side_visualizations["right"]["inner"],
                combined_inner,
                resolution,
            )
            camera_common = {
                "renderer": "MATPLOTLIB_COOKED_PHYSX_GEOMETRY",
                "coordinate_frame": "each collider-local metres",
                "layout": "left finger | right finger",
                "not_runtime_contact_evidence": True,
            }
            simulation_common = {
                "physics_steps_added_for_capture": 0,
                "asset": str(asset_path),
                "asset_sha256": sha256_file(asset_path),
                "collider_paths": sorted(collider_results),
                "approximation_readback": approximation,
                "piece_count_by_side": {
                    item["side"]: item["piece_count"]
                    for item in collider_results.values()
                },
            }
            overview_capture = validate_screenshot(
                combined_overview,
                artifact_root=artifact_root,
                phase="collider_geometry",
                capture_name=overview_name,
                gate_status="PASS",
                camera={**camera_common, "view": "full cooked colliders"},
                simulation=simulation_common,
            )
            inner_capture = validate_screenshot(
                combined_inner,
                artifact_root=artifact_root,
                phase="collider_geometry",
                capture_name=inner_name,
                gate_status="PASS",
                camera={
                    **camera_common,
                    "view": (
                        "inner gripping region; deterministic local +X crop"
                    ),
                },
                simulation=simulation_common,
            )
            geometry_captures.extend([overview_capture, inner_capture])
            for collider in collider_results.values():
                collider["visualization"]["combined_overview"] = (
                    overview_capture
                )
                collider["visualization"]["combined_inner_gripping_surface"] = (
                    inner_capture
                )
            assets.append(
                {
                    "robot": robot,
                    "asset_absolute_path": str(asset_path),
                    "asset_sha256": sha256_file(asset_path),
                    "colliders": collider_results,
                    "cooking_statistics_delta": cooking[
                        "cooking_statistics_delta"
                    ],
                }
            )
        profile_results[profile_name] = {
            "approximation": approximation,
            "decomposition_parameters": (
                local_api["convex_decomposition_api"]["defaults"]
                if approximation == "convexDecomposition"
                else None
            ),
            "assets": assets,
        }

    source_after = verify_correct_finger_sources(profile, project_root)
    if source_after != source_before:
        raise RuntimeError("protected source evidence changed during geometry audit")

    refreshed_runtime = [
        _revalidate_capture(capture, artifact_root=artifact_root)
        for capture in runtime_manifest["captures"]
    ]
    full_manifest = build_screenshot_manifest(
        captures=refreshed_runtime + geometry_captures,
        required_captures=profile["screenshots"]["required_captures"],
        artifact_root=artifact_root,
    )
    _write_json(screenshot_manifest_path, full_manifest)

    report = {
        "schema_version": 1,
        "status": (
            "PASS"
            if full_manifest["status"] == "PASS"
            and all(
                collider["piece_count"] > 0
                for profile_result in profile_results.values()
                for asset in profile_result["assets"]
                for collider in asset["colliders"].values()
            )
            else "FAIL"
        ),
        "scope": "user-confirmed Stationary ALOHA 1 custom follower fingers",
        "restart_boundary": profile["restart_boundary"],
        "runtime": profile["runtime"],
        "local_api": local_api,
        "source_evidence_before": source_before,
        "source_evidence_after": source_after,
        "profiles": profile_results,
        "screenshot_root_absolute": str(artifact_root),
        "geometry_screenshot_directory_absolute": str(geometry_root),
        "full_screenshot_manifest_absolute": str(
            screenshot_manifest_path.resolve()
        ),
        "geometry_capture_count": len(geometry_captures),
        "full_screenshot_manifest_status": full_manifest["status"],
        "interpretation": {
            "convex_decomposition_supported": True,
            "convex_decomposition_is_exact": False,
            "runtime_hold_evidence_required_for_selection": True,
            "runtime_result": "NO_MEANINGFUL_EFFECT",
        },
        "default_asset_collider_modified": False,
        "task8": "NOT_RUN",
    }
    _write_json(report_path, report)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the cooked user-confirmed ALOHA custom fingers."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/aloha1_gripper_correct_finger_profiles.yaml"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    project_root = args.project_root.resolve(strict=True)
    profile_path = (
        args.profile
        if args.profile.is_absolute()
        else project_root / args.profile
    ).resolve(strict=True)
    report_root = project_root / "reports/aloha1_mapping"
    report_path = (
        report_root / "gripper_correct_finger_collider_comparison.json"
    )
    markdown_path = (
        report_root / "gripper_correct_finger_collider_comparison.md"
    )
    screenshot_manifest_path = (
        report_root / "gripper_correct_finger_all_screenshot_manifest.json"
    )

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        report = run(
            app,
            project_root=project_root,
            profile_path=profile_path,
            report_path=report_path,
            markdown_path=markdown_path,
            screenshot_manifest_path=screenshot_manifest_path,
        )
    except BaseException as error:
        _write_json(
            report_root / "gripper_correct_finger_collider_comparison_failure.json",
            {
                "schema_version": 1,
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        app.close()
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
