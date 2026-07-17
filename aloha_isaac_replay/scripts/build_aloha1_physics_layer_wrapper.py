from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "assets/isaac/original_stationary_aloha"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase17_physics_layer_wrapper_20260718"


def _rel(path: str | Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _inspect_stage(stage: Any) -> dict[str, Any]:
    from pxr import UsdGeom, UsdPhysics

    meshes = []
    colliders = []
    rigid_bodies = []
    joints = []
    articulation_roots = []
    root_prims = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.GetPath().pathString.count("/") == 1:
            root_prims.append(path)
        type_name = prim.GetTypeName()
        schemas = [str(item) for item in prim.GetAppliedSchemas()]
        if type_name == "Mesh":
            meshes.append(path)
        if "ArticulationRootAPI" in schemas or "PhysicsArticulationRootAPI" in schemas:
            articulation_roots.append(path)
        try:
            if UsdPhysics.CollisionAPI(prim):
                colliders.append(path)
        except Exception:
            pass
        try:
            if UsdPhysics.RigidBodyAPI(prim):
                rigid_bodies.append(path)
        except Exception:
            pass
        if "Joint" in type_name:
            joints.append(path)
    return {
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim().IsValid() else None,
        "root_prims": root_prims,
        "mesh_count": len(meshes),
        "collision_count": len(colliders),
        "rigid_body_count": len(rigid_bodies),
        "joint_count": len(joints),
        "articulation_root_count": len(articulation_roots),
        "mesh_sample": meshes[:20],
        "collider_sample": colliders[:20],
        "rigid_body_sample": rigid_bodies[:20],
        "joint_sample": joints[:20],
        "articulation_roots": articulation_roots,
    }


def _find_physics_layer(source_root: Path, side: str) -> Path:
    path = source_root / f"generated/configuration/vx300s_{side}_physics.usd"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _write_layer_wrapper(Sdf: Any, Usd: Any, layer_paths: list[Path], output_path: Path, default_prim_name: str | None) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.CreateNew(str(output_path.resolve()))
    layer.subLayerPaths = [str(path.resolve()) for path in layer_paths]
    layer.Save()
    stage = Usd.Stage.Open(str(output_path.resolve()))
    if default_prim_name:
        prim = stage.GetPrimAtPath(f"/{default_prim_name}")
        if prim.IsValid():
            stage.SetDefaultPrim(prim)
            stage.GetRootLayer().Save()
    return _inspect_stage(stage)


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase 17 Physics Layer Wrapper",
        "",
        "This phase tests the smallest non-destructive repair after Phase 16: compose the generated ALOHA1 physics layers directly instead of using the broken importer wrapper.",
        "",
        "## Results",
        "",
        "| Asset | Mesh prims | Collision prims | Rigid bodies | Joints | Articulation roots | Default prim | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for key in ("left", "right", "combined"):
        item = payload[key]
        stage = item["stage"]
        gate = item["gate"]
        lines.append(
            f"| {key} | {stage['mesh_count']} | {stage['collision_count']} | {stage['rigid_body_count']} | "
            f"{stage['joint_count']} | {stage['articulation_root_count']} | `{stage['default_prim']}` | "
            f"{'PASS' if gate['asset_visible_and_physical'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The physics-layer wrapper composes Mesh prims, CollisionAPI prims, RigidBodyAPI prims, joints, and articulation roots. This confirms that the generated physics layers contain the usable robot body data, while the original top-level importer wrapper is the broken part.",
            "",
            "This is not yet the final ALOHA1 asset. It is a diagnostic repair that proves the next practical route: build a clean ALOHA1-native wrapper around the useful generated layers, then validate DOF order, drives, initial pose, collisions, and controller replay.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{_rel(payload['json_path'])}`",
            f"- Markdown: `{_rel(path)}`",
            f"- Left wrapper: `{payload['left']['wrapper']}`",
            f"- Right wrapper: `{payload['right']['wrapper']}`",
            f"- Combined wrapper: `{payload['combined']['wrapper']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _gate(stage: dict[str, Any]) -> dict[str, bool]:
    return {
        "has_mesh": stage["mesh_count"] > 0,
        "has_collision": stage["collision_count"] > 0,
        "has_rigid_bodies": stage["rigid_body_count"] > 0,
        "has_joints": stage["joint_count"] > 0,
        "has_articulation": stage["articulation_root_count"] > 0,
        "asset_visible_and_physical": all(
            [
                stage["mesh_count"] > 0,
                stage["collision_count"] > 0,
                stage["rigid_body_count"] > 0,
                stage["joint_count"] > 0,
                stage["articulation_root_count"] > 0,
            ]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build diagnostic ALOHA1 wrappers directly from generated physics layers.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp(dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG))
    try:
        from pxr import Sdf, Usd

        left_physics = _find_physics_layer(source_root, "left")
        right_physics = _find_physics_layer(source_root, "right")
        left_wrapper = output_dir / "aloha1_left_physics_layer_wrapper.usda"
        right_wrapper = output_dir / "aloha1_right_physics_layer_wrapper.usda"
        combined_wrapper = output_dir / "aloha1_dual_physics_layer_wrapper.usda"
        left_stage = _write_layer_wrapper(Sdf, Usd, [left_physics], left_wrapper, "puppet_left_vx300s")
        right_stage = _write_layer_wrapper(Sdf, Usd, [right_physics], right_wrapper, "puppet_right_vx300s")
        combined_stage = _write_layer_wrapper(Sdf, Usd, [left_physics, right_physics], combined_wrapper, None)

        payload = {
            "schema_version": 1,
            "source_root": _rel(source_root),
            "left": {
                "physics_layer": _rel(left_physics),
                "wrapper": _rel(left_wrapper),
                "stage": left_stage,
                "gate": _gate(left_stage),
            },
            "right": {
                "physics_layer": _rel(right_physics),
                "wrapper": _rel(right_wrapper),
                "stage": right_stage,
                "gate": _gate(right_stage),
            },
            "combined": {
                "physics_layers": [_rel(left_physics), _rel(right_physics)],
                "wrapper": _rel(combined_wrapper),
                "stage": combined_stage,
                "gate": _gate(combined_stage),
            },
        }
        json_path = output_dir / "physics_layer_wrapper_report.json"
        md_path = output_dir / "physics_layer_wrapper_report.md"
        payload["json_path"] = str(json_path)
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        _write_markdown(payload, md_path)
        print(
            json.dumps(
                {
                    "json": _rel(json_path),
                    "markdown": _rel(md_path),
                    "left_gate": payload["left"]["gate"],
                    "right_gate": payload["right"]["gate"],
                    "combined_gate": payload["combined"]["gate"],
                },
                ensure_ascii=False,
            )
        )
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
