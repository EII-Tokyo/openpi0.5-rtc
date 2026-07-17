from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEFT_PHYSICS = REPO_ROOT / "assets/isaac/original_stationary_aloha/generated/configuration/vx300s_left_physics.usd"
DEFAULT_RIGHT_PHYSICS = REPO_ROOT / "assets/isaac/original_stationary_aloha/generated/configuration/vx300s_right_physics.usd"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase32_clean_side_wrappers_20260718"

SIDE_TARGETS = {
    "left": {
        "default_prim": "puppet_left_vx300s",
        "physics_layer": DEFAULT_LEFT_PHYSICS,
        "broken_visual_sources": [
            "/puppet_left_vx300s/puppet_left_ee_arm_link/visuals",
            "/puppet_left_vx300s/puppet_left_fingers_link/visuals",
            "/puppet_left_vx300s/puppet_left_ee_gripper_link/visuals",
        ],
        "root_joint": "/puppet_left_vx300s/root_joint",
    },
    "right": {
        "default_prim": "puppet_right_vx300s",
        "physics_layer": DEFAULT_RIGHT_PHYSICS,
        "broken_visual_sources": [
            "/puppet_right_vx300s/puppet_right_ee_arm_link/visuals",
            "/puppet_right_vx300s/puppet_right_fingers_link/visuals",
            "/puppet_right_vx300s/puppet_right_ee_gripper_link/visuals",
        ],
        "root_joint": "/puppet_right_vx300s/root_joint",
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


def _create_side_wrapper(path: Path, default_prim: str, physics_layer: Path, broken_visual_sources: list[str]) -> None:
    from pxr import Sdf, Usd, UsdGeom

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.GetRootLayer().subLayerPaths.append(str(physics_layer.resolve()))
    # The imported URDF authoring creates local references from these visual
    # child prims to non-existent root-level /visuals targets. These links have
    # no useful visible mesh in the source data, so the least invasive clean
    # wrapper is to clear only those broken reference lists in a stronger layer.
    for source in broken_visual_sources:
        prim = stage.OverridePrim(source)
        prim.SetMetadata("references", Sdf.ReferenceListOp.CreateExplicit([]))
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
        "# Phase 32 Clean Side Wrapper Diagnostic",
        "",
        "## Question",
        "",
        "Can the missing local `/visuals/...` reference warnings be repaired by clearing only the broken visual reference arcs inside each side wrapper?",
        "",
        "This is a diagnostic stage and wrapper generator. It does not overwrite the promoted production wrapper assets.",
        "",
        "## Result",
        "",
        f"- status: `{payload['status']}`",
        f"- runtime stage: `{payload['runtime_stage']}`",
        f"- missing local reference targets: `{len(payload['missing_local_reference_targets'])}`",
        f"- collision count: `{payload['static_counts']['collision_count']}`",
        f"- rigid body count: `{payload['static_counts']['rigid_body_count']}`",
        f"- mass API count: `{payload['static_counts']['mass_api_count']}`",
        f"- articulation roots: `{payload['articulation_roots']}`",
        "",
        "## Generated Wrappers",
        "",
    ]
    for side, path in payload["side_wrappers"].items():
        lines.append(f"- {side}: `{path}`")
    lines.extend(
        [
            "",
            "## Runtime Articulations",
            "",
            "| side | status | prim path | DOFs | bodies |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for side, item in payload["runtime_articulations"].items():
        lines.append(f"| {side} | `{item['status']}` | `{item['prim_path']}` | {item.get('num_dof')} | {item.get('num_bodies')} |")
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {line}" for line in payload["interpretation"])
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- JSON: `{payload['outputs']['json']}`",
            f"- Markdown: `{payload['outputs']['markdown']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ALOHA1 clean side wrappers with missing local visual targets patched inside the side layer stacks.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
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
        wrappers_dir = output_dir / "generated_wrappers"
        side_wrappers: dict[str, Path] = {}
        for side, spec in SIDE_TARGETS.items():
            side_path = wrappers_dir / f"aloha1_{side}_clean_visual_targets.usda"
            _create_side_wrapper(side_path, str(spec["default_prim"]), Path(spec["physics_layer"]), list(spec["broken_visual_sources"]))
            side_wrappers[side] = side_path

        runtime_stage = output_dir / "aloha1_dual_clean_side_wrappers.usda"
        _create_runtime_stage(runtime_stage, side_wrappers)

        json_path = output_dir / "clean_side_wrappers.json"
        md_path = output_dir / "clean_side_wrappers.md"
        payload: dict[str, Any] = {
            "status": "STARTED",
            "side_wrappers": {side: _rel(path) for side, path in side_wrappers.items()},
            "runtime_stage": _rel(runtime_stage),
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
        for side, spec in SIDE_TARGETS.items():
            root_path = str(spec["root_joint"])
            try:
                runtime[side] = _inspect_articulation(world, root_path, f"{side}_clean_side_wrapper_vx300s")
            except Exception as exc:
                runtime[side] = {"status": "FAIL", "prim_path": root_path, "error": f"{type(exc).__name__}: {exc}"}

        interpretation = []
        if not missing_refs:
            interpretation.append("Side-wrapper patching clears the broken local visual reference targets at static composition time.")
        else:
            interpretation.append("Some local visual targets remain missing after side-wrapper patching.")
        if collision_count > 0:
            interpretation.append("Collision composition remains present in the runtime stage.")
        else:
            interpretation.append("Collision composition is missing in the runtime stage.")
        if all(item["status"] == "PASS" for item in runtime.values()):
            interpretation.append("Both ALOHA1 articulations initialize from the clean side-wrapper runtime stage.")
        else:
            interpretation.append("At least one ALOHA1 articulation fails to initialize from the clean side-wrapper runtime stage.")

        status = "PASS" if not missing_refs and collision_count > 0 and all(item["status"] == "PASS" for item in runtime.values()) else "FAIL"
        payload.update({"status": status, "runtime_articulations": runtime, "interpretation": interpretation})
        _write_json(json_path, payload)
        md_path.write_text(_render_markdown(payload))
        print(json.dumps({"status": status, "json": _rel(json_path), "markdown": _rel(md_path), "interpretation": interpretation}, ensure_ascii=False), flush=True)
        if not args.normal_close:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0 if status == "PASS" else 2)
        app.close()
        return 0 if status == "PASS" else 2
    except Exception as exc:
        output_dir = Path(args.output_dir)
        json_path = output_dir / "clean_side_wrappers.json"
        md_path = output_dir / "clean_side_wrappers.md"
        payload = {
            "status": "EXCEPTION",
            "exception": f"{type(exc).__name__}: {exc}",
            "outputs": {"json": _rel(json_path), "markdown": _rel(md_path)},
        }
        _write_json(json_path, payload)
        md_path.write_text("# Phase 32 Clean Side Wrapper Diagnostic\n\nstatus: `EXCEPTION`\n\n" + payload["exception"] + "\n")
        print(json.dumps({"status": "EXCEPTION", "json": _rel(json_path), "exception": payload["exception"]}, ensure_ascii=False), flush=True)
        if not args.normal_close:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
