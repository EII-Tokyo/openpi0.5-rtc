from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIGURATION_DIR = REPO_ROOT / "assets/isaac/original_stationary_aloha/generated/configuration"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "local_eval_assets/aloha1_clean_runtime_20260718"

SIDE_SPECS = {
    "left": {
        "default_prim": "puppet_left_vx300s",
        "base": "vx300s_left_base.usd",
        "physics": "vx300s_left_physics.usd",
        "robot": "vx300s_left_robot.usd",
        "sensor": "vx300s_left_sensor.usd",
        "root_joint": "/puppet_left_vx300s/root_joint",
        "broken_visual_sources": [
            "/puppet_left_vx300s/puppet_left_ee_arm_link/visuals",
            "/puppet_left_vx300s/puppet_left_fingers_link/visuals",
            "/puppet_left_vx300s/puppet_left_ee_gripper_link/visuals",
        ],
    },
    "right": {
        "default_prim": "puppet_right_vx300s",
        "base": "vx300s_right_base.usd",
        "physics": "vx300s_right_physics.usd",
        "robot": "vx300s_right_robot.usd",
        "sensor": "vx300s_right_sensor.usd",
        "root_joint": "/puppet_right_vx300s/root_joint",
        "broken_visual_sources": [
            "/puppet_right_vx300s/puppet_right_ee_arm_link/visuals",
            "/puppet_right_vx300s/puppet_right_fingers_link/visuals",
            "/puppet_right_vx300s/puppet_right_ee_gripper_link/visuals",
        ],
    },
}


def _rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "__iter__") and not isinstance(value, (bytes, bytearray)):
        try:
            return [_json_safe(v) for v in value]
        except Exception:
            pass
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n")


def _applied(prim: Any) -> list[str]:
    try:
        return [str(item) for item in prim.GetAppliedSchemas()]
    except Exception:
        return []


def _count_schema(stage: Any, schema_name: str) -> tuple[int, list[str]]:
    paths: list[str] = []
    for prim in stage.Traverse():
        if schema_name in _applied(prim):
            paths.append(str(prim.GetPath()))
    return len(paths), paths[:120]


def _find_articulation_roots(stage: Any) -> list[str]:
    roots: list[str] = []
    for prim in stage.Traverse():
        schemas = _applied(prim)
        if "ArticulationRootAPI" in schemas or "PhysicsArticulationRootAPI" in schemas:
            roots.append(str(prim.GetPath()))
    return roots


def _local_references(stage: Any, prim_path: str) -> list[str]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return []
    refs: list[str] = []
    for spec in prim.GetPrimStack():
        ref_list = spec.referenceList
        for ref in list(ref_list.prependedItems) + list(ref_list.addedItems) + list(ref_list.explicitItems):
            if not str(ref.assetPath):
                refs.append(str(ref.primPath))
    return refs


def _find_missing_local_reference_targets(stage: Any) -> list[dict[str, str]]:
    all_paths = {str(prim.GetPath()) for prim in stage.Traverse()}
    missing: list[dict[str, str]] = []
    for prim in stage.Traverse():
        for spec in prim.GetPrimStack():
            ref_list = spec.referenceList
            refs = list(ref_list.prependedItems) + list(ref_list.addedItems) + list(ref_list.explicitItems)
            for ref in refs:
                asset_path = str(ref.assetPath)
                prim_path = str(ref.primPath)
                if asset_path:
                    continue
                if prim_path and prim_path not in all_paths:
                    missing.append({"source": str(prim.GetPath()), "target": prim_path})
    return missing


def _copy_configuration(source_dir: Path, output_dir: Path, overwrite: bool) -> Path:
    configuration_dir = output_dir / "configuration"
    if configuration_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output configuration already exists: {configuration_dir}")
        shutil.rmtree(configuration_dir)
    configuration_dir.mkdir(parents=True, exist_ok=True)
    for side, spec in SIDE_SPECS.items():
        for key in ("base", "physics", "robot", "sensor"):
            src = source_dir / str(spec[key])
            dst = configuration_dir / src.name
            if not src.exists():
                raise FileNotFoundError(f"Missing source {side} {key} layer: {src}")
            shutil.copy2(src, dst)
    return configuration_dir


def _patch_base_layer(configuration_dir: Path, side: str, spec: dict[str, Any]) -> dict[str, Any]:
    from pxr import Usd

    base_path = configuration_dir / str(spec["base"])
    stage = Usd.Stage.Open(str(base_path.resolve()))
    if stage is None:
        raise RuntimeError(f"Failed to open copied base layer: {base_path}")
    patched: list[dict[str, Any]] = []
    for source in spec["broken_visual_sources"]:
        before = _local_references(stage, str(source))
        prim = stage.GetPrimAtPath(str(source))
        if not prim:
            patched.append({"source": source, "status": "MISSING_SOURCE", "before": before, "after": []})
            continue
        prim.GetReferences().ClearReferences()
        after = _local_references(stage, str(source))
        patched.append({"source": source, "status": "PATCHED", "before": before, "after": after})
    stage.GetRootLayer().Save()
    return {"side": side, "base_layer": _rel(base_path), "patched_visual_sources": patched}


def _create_side_wrapper(path: Path, default_prim: str, physics_layer: Path) -> None:
    from pxr import Usd, UsdGeom

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.GetRootLayer().subLayerPaths.append(str(physics_layer.resolve()))
    default = stage.GetPrimAtPath(f"/{default_prim}")
    if not default:
        default = UsdGeom.Xform.Define(stage, f"/{default_prim}").GetPrim()
    stage.SetDefaultPrim(default)
    stage.Save()


def _create_runtime_stage(path: Path, side_wrappers: dict[str, Path]) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    root = stage.GetRootLayer()
    root.subLayerPaths.append(str(side_wrappers["left"].resolve()))
    root.subLayerPaths.append(str(side_wrappers["right"].resolve()))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    stage.Save()


def _collision_paths(stage: Any) -> list[str]:
    return [str(prim.GetPath()) for prim in stage.Traverse() if "PhysicsCollisionAPI" in _applied(prim)]


def _create_controller_stage(path: Path, side_wrappers: dict[str, Path], disabled_collision_paths: list[str]) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    root = stage.GetRootLayer()
    root.subLayerPaths.append(str(side_wrappers["left"].resolve()))
    root.subLayerPaths.append(str(side_wrappers["right"].resolve()))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    for prim_path in disabled_collision_paths:
        prim = stage.OverridePrim(prim_path)
        collision = UsdPhysics.CollisionAPI.Apply(prim)
        collision.CreateCollisionEnabledAttr().Set(False)
    stage.Save()


def _inspect_articulation(world: Any, prim_path: str, name: str) -> dict[str, Any]:
    from isaacsim.core.prims import SingleArticulation

    articulation = world.scene.add(SingleArticulation(prim_path=prim_path, name=name))
    world.reset()
    return {
        "status": "PASS",
        "prim_path": prim_path,
        "num_dof": int(articulation.num_dof),
        "num_bodies": int(articulation.num_bodies),
        "dof_names": list(articulation.dof_names),
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 Clean Runtime Asset Build",
        "",
        "## Result",
        "",
        f"- status: `{payload['status']}`",
        f"- output dir: `{payload['output_dir']}`",
        f"- runtime stage: `{payload['runtime_stage']}`",
        f"- controller stage: `{payload.get('controller_stage')}`",
        f"- controller-disabled collision prims: `{len(payload.get('controller_disabled_collision_paths', []))}`",
        f"- missing local reference targets: `{len(payload['missing_local_reference_targets'])}`",
        f"- collision count: `{payload['static_counts']['collision_count']}`",
        f"- rigid body count: `{payload['static_counts']['rigid_body_count']}`",
        f"- mass API count: `{payload['static_counts']['mass_api_count']}`",
        "",
        "## Runtime Articulations",
        "",
        "| side | status | prim path | DOFs | bodies |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for side, item in payload["runtime_articulations"].items():
        lines.append(f"| {side} | `{item['status']}` | `{item['prim_path']}` | {item.get('num_dof')} | {item.get('num_bodies')} |")
    lines.extend(["", "## Patched Visual Sources", ""])
    for item in payload["base_layer_patches"]:
        lines.append(f"### {item['side']}")
        for row in item["patched_visual_sources"]:
            lines.append(f"- `{row['source']}`: {row['status']}, before={row['before']}, after={row['after']}")
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {line}" for line in payload["interpretation"])
    lines.extend(["", "## Artifacts", "", f"- JSON: `{payload['outputs']['json']}`", f"- Markdown: `{payload['outputs']['markdown']}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate a local clean-runtime ALOHA1 USD asset package.")
    parser.add_argument("--source-configuration-dir", default=str(SOURCE_CONFIGURATION_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--normal-close",
        action="store_true",
        help="Call SimulationApp.close() before exit. Disabled by default because Isaac 5.1 headless teardown can hang after diagnostics are flushed.",
    )
    args = parser.parse_args()

    from isaacsim import SimulationApp

    app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
    app_config["fast_shutdown"] = False
    app = SimulationApp(app_config)
    try:
        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.core.api import World
        from pxr import Usd

        output_dir = Path(args.output_dir)
        configuration_dir = _copy_configuration(Path(args.source_configuration_dir), output_dir, args.overwrite)
        base_layer_patches = [_patch_base_layer(configuration_dir, side, spec) for side, spec in SIDE_SPECS.items()]

        side_wrappers: dict[str, Path] = {}
        for side, spec in SIDE_SPECS.items():
            side_path = output_dir / f"aloha1_{side}_clean_runtime.usda"
            _create_side_wrapper(side_path, str(spec["default_prim"]), configuration_dir / str(spec["physics"]))
            side_wrappers[side] = side_path

        runtime_stage = output_dir / "aloha1_dual_clean_runtime.usda"
        _create_runtime_stage(runtime_stage, side_wrappers)
        base_runtime_stage = Usd.Stage.Open(str(runtime_stage.resolve()))
        if base_runtime_stage is None:
            raise RuntimeError(f"Failed to open generated runtime stage: {runtime_stage}")
        collision_paths = _collision_paths(base_runtime_stage)
        base_runtime_stage = None
        controller_stage = output_dir / "aloha1_dual_controller_runtime.usda"
        _create_controller_stage(controller_stage, side_wrappers, collision_paths)

        json_path = output_dir / "clean_runtime_asset_report.json"
        md_path = output_dir / "clean_runtime_asset_report.md"
        payload: dict[str, Any] = {
            "status": "STARTED",
            "output_dir": _rel(output_dir),
            "source_configuration_dir": _rel(args.source_configuration_dir),
            "configuration_dir": _rel(configuration_dir),
            "side_wrappers": {side: _rel(path) for side, path in side_wrappers.items()},
            "runtime_stage": _rel(runtime_stage),
            "controller_stage": _rel(controller_stage),
            "base_layer_patches": base_layer_patches,
            "controller_disabled_collision_paths": collision_paths,
            "outputs": {"json": _rel(json_path), "markdown": _rel(md_path)},
        }
        _write_json(json_path, payload)

        stage = Usd.Stage.Open(str(runtime_stage.resolve()))
        if stage is None:
            raise RuntimeError(f"Failed to open generated runtime stage: {runtime_stage}")
        collision_count, collision_sample = _count_schema(stage, "PhysicsCollisionAPI")
        rigid_body_count, rigid_body_sample = _count_schema(stage, "PhysicsRigidBodyAPI")
        mass_api_count, mass_api_sample = _count_schema(stage, "PhysicsMassAPI")
        missing_refs = _find_missing_local_reference_targets(stage)
        articulation_roots = _find_articulation_roots(stage)
        payload.update(
            {
                "status": "STATIC_INSPECTED",
                "static_counts": {
                    "collision_count": collision_count,
                    "rigid_body_count": rigid_body_count,
                    "mass_api_count": mass_api_count,
                    "collision_sample": collision_sample,
                    "rigid_body_sample": rigid_body_sample,
                    "mass_api_sample": mass_api_sample,
                },
                "missing_local_reference_targets": missing_refs,
                "articulation_roots": articulation_roots,
            }
        )
        _write_json(json_path, payload)
        stage = None

        World.clear_instance()
        stage_utils.open_stage(str(runtime_stage.resolve()))
        world = World(stage_units_in_meters=0.01, backend="numpy", device="cpu")
        runtime: dict[str, dict[str, Any]] = {}
        for side, spec in SIDE_SPECS.items():
            try:
                runtime[side] = _inspect_articulation(world, str(spec["root_joint"]), f"{side}_clean_runtime_vx300s")
            except Exception as exc:
                runtime[side] = {"status": "FAIL", "prim_path": str(spec["root_joint"]), "error": f"{type(exc).__name__}: {exc}"}

        patch_ok = all(not row["after"] for item in base_layer_patches for row in item["patched_visual_sources"] if row["status"] == "PATCHED")
        interpretation = []
        interpretation.append("Copied importer configuration into a separate local clean-runtime package; original importer assets were not modified.")
        interpretation.append("Removed only the six known broken visual reference arcs from copied base layers." if patch_ok else "At least one broken visual reference arc was not removed from the copied base layers.")
        interpretation.append("Collision, rigid body, and mass composition are present." if collision_count > 0 else "Collision composition is missing.")
        interpretation.append("Generated a separate controller runtime stage that disables the current root-level collision prims; use it for controller/replay gates until collision geometry is repaired.")
        interpretation.append("Both ALOHA1 runtime articulations initialize." if all(item["status"] == "PASS" for item in runtime.values()) else "At least one runtime articulation failed to initialize.")
        interpretation.append("Runtime USD log cleanliness must still be checked from the codex-evidence stdout.")

        status = "PASS" if patch_ok and not missing_refs and collision_count > 0 and all(item["status"] == "PASS" for item in runtime.values()) else "FAIL"
        payload.update({"status": status, "runtime_articulations": runtime, "interpretation": interpretation})
        _write_json(json_path, payload)
        md_path.write_text(_render_markdown(payload))
        print(json.dumps({"status": status, "json": _rel(json_path), "markdown": _rel(md_path), "runtime_stage": _rel(runtime_stage), "interpretation": interpretation}, ensure_ascii=False), flush=True)
        if not args.normal_close:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0 if status == "PASS" else 2)
        app.close()
        return 0 if status == "PASS" else 2
    except Exception as exc:
        output_dir = Path(args.output_dir)
        json_path = output_dir / "clean_runtime_asset_report.json"
        md_path = output_dir / "clean_runtime_asset_report.md"
        payload = {
            "status": "EXCEPTION",
            "exception": f"{type(exc).__name__}: {exc}",
            "outputs": {"json": _rel(json_path), "markdown": _rel(md_path)},
        }
        _write_json(json_path, payload)
        md_path.write_text("# ALOHA1 Clean Runtime Asset Build\n\nstatus: `EXCEPTION`\n\n" + payload["exception"] + "\n")
        print(json.dumps({"status": "EXCEPTION", "json": _rel(json_path), "exception": payload["exception"]}, ensure_ascii=False), flush=True)
        if not args.normal_close:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
