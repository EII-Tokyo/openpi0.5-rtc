from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEFT_USD = REPO_ROOT / "assets/isaac/aloha1_native_physics_wrapper/aloha1_left.usda"
DEFAULT_RIGHT_USD = REPO_ROOT / "assets/isaac/aloha1_native_physics_wrapper/aloha1_right.usda"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase30_sublayer_runtime_composition_20260718"


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


def _find_articulation_roots(stage: Any) -> list[str]:
    roots: list[str] = []
    for prim in stage.Traverse():
        schemas = _applied(prim)
        if "ArticulationRootAPI" in schemas or "PhysicsArticulationRootAPI" in schemas:
            roots.append(str(prim.GetPath()))
    return roots


def _count_schema(stage: Any, schema_name: str) -> tuple[int, list[str]]:
    paths: list[str] = []
    for prim in stage.Traverse():
        if schema_name in _applied(prim):
            paths.append(str(prim.GetPath()))
    return len(paths), paths[:120]


def _create_sublayer_stage(path: Path, left_usd: Path, right_usd: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    root = stage.GetRootLayer()
    root.subLayerPaths.append(str(left_usd.resolve()))
    root.subLayerPaths.append(str(right_usd.resolve()))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    stage.Save()


def _inspect_articulation(world: Any, prim_path: str, name: str) -> dict[str, Any]:
    from isaacsim.core.prims import SingleArticulation

    art = world.scene.add(SingleArticulation(prim_path=prim_path, name=name))
    world.reset()
    return {
        "status": "PASS",
        "prim_path": prim_path,
        "num_dof": int(art.num_dof),
        "num_bodies": int(art.num_bodies),
        "dof_names": list(art.dof_names),
        "body_names_sample": list(art._articulation_view.body_names)[:40],
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 30 Sublayer Runtime Composition",
        "",
        "## Question",
        "",
        "If ALOHA1 left/right wrappers are loaded as whole-stage sublayers instead of defaultPrim references, do the root-level collider scopes compose into the runtime stage?",
        "",
        "This is a diagnostic gate, not a final asset format.",
        "",
        "## Result",
        "",
        f"- status: `{payload['status']}`",
        f"- generated stage: `{payload['generated_stage']}`",
        f"- collision count: `{payload['static_counts']['collision_count']}`",
        f"- rigid body count: `{payload['static_counts']['rigid_body_count']}`",
        f"- mass API count: `{payload['static_counts']['mass_api_count']}`",
        f"- articulation roots: `{payload['articulation_roots']}`",
        "",
        "## Runtime Articulations",
        "",
        "| side | status | prim path | DOFs | bodies |",
        "| --- | --- | --- | ---: | ---: |",
    ]
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
    parser = argparse.ArgumentParser(description="Validate whole-stage sublayer composition for ALOHA1 diagnostic runtime stage.")
    parser.add_argument("--left-usd", default=str(DEFAULT_LEFT_USD))
    parser.add_argument("--right-usd", default=str(DEFAULT_RIGHT_USD))
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
        stage_path = output_dir / "aloha1_dual_sublayer_diagnostic.usda"
        json_path = output_dir / "sublayer_runtime_composition.json"
        md_path = output_dir / "sublayer_runtime_composition.md"
        _create_sublayer_stage(stage_path, Path(args.left_usd), Path(args.right_usd))

        payload: dict[str, Any] = {
            "status": "STARTED",
            "left_usd": _rel(args.left_usd),
            "right_usd": _rel(args.right_usd),
            "generated_stage": _rel(stage_path),
            "outputs": {"json": _rel(json_path), "markdown": _rel(md_path)},
        }
        _write_json(json_path, payload)

        stage = Usd.Stage.Open(str(stage_path.resolve()))
        if stage is None:
            raise RuntimeError(f"Failed to reopen generated stage: {stage_path}")
        collision_count, collision_sample = _count_schema(stage, "PhysicsCollisionAPI")
        rigid_body_count, rigid_body_sample = _count_schema(stage, "PhysicsRigidBodyAPI")
        mass_api_count, mass_api_sample = _count_schema(stage, "PhysicsMassAPI")
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
                "articulation_roots": articulation_roots,
            }
        )
        _write_json(json_path, payload)
        stage = None

        World.clear_instance()
        stage_utils.open_stage(str(stage_path.resolve()))
        world = World(stage_units_in_meters=0.01, backend="numpy", device="cpu")
        runtime: dict[str, dict[str, Any]] = {}
        for side, root in {
            "left": "/puppet_left_vx300s/root_joint",
            "right": "/puppet_right_vx300s/root_joint",
        }.items():
            try:
                runtime[side] = _inspect_articulation(world, root, f"{side}_sublayer_vx300s")
            except Exception as exc:
                runtime[side] = {"status": "FAIL", "prim_path": root, "error": f"{type(exc).__name__}: {exc}"}

        interpretation = []
        if collision_count > 0:
            interpretation.append("Whole-stage sublayer composition brings root-level collision prims into the stage, supporting the defaultPrim-reference-loss hypothesis.")
        else:
            interpretation.append("Whole-stage sublayer composition still has zero collision prims; the issue is not only defaultPrim reference scoping.")
        if all(item["status"] == "PASS" for item in runtime.values()):
            interpretation.append("Both ALOHA1 articulations can still initialize from the sublayer-composed diagnostic stage.")
        else:
            interpretation.append("At least one articulation failed to initialize from the sublayer-composed diagnostic stage.")

        status = "PASS" if collision_count > 0 and all(item["status"] == "PASS" for item in runtime.values()) else "FAIL"
        payload.update(
            {
            "status": status,
            "runtime_articulations": runtime,
            "interpretation": interpretation,
            }
        )
        _write_json(json_path, payload)
        md_path.write_text(_render_markdown(payload))
        print(json.dumps({"status": status, "json": _rel(json_path), "markdown": _rel(md_path), "interpretation": interpretation}, ensure_ascii=False), flush=True)
        if not args.normal_close:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        return 0 if status == "PASS" else 2
    except Exception as exc:
        output_dir = Path(args.output_dir)
        json_path = output_dir / "sublayer_runtime_composition.json"
        md_path = output_dir / "sublayer_runtime_composition.md"
        payload = {
            "status": "EXCEPTION",
            "left_usd": _rel(args.left_usd),
            "right_usd": _rel(args.right_usd),
            "generated_stage": _rel(output_dir / "aloha1_dual_sublayer_diagnostic.usda"),
            "exception": f"{type(exc).__name__}: {exc}",
            "outputs": {"json": _rel(json_path), "markdown": _rel(md_path)},
        }
        _write_json(json_path, payload)
        md_path.write_text("# Phase 30 Sublayer Runtime Composition\n\nstatus: `EXCEPTION`\n\n" + payload["exception"] + "\n")
        print(json.dumps({"status": "EXCEPTION", "json": _rel(json_path), "exception": payload["exception"]}, ensure_ascii=False), flush=True)
        if not args.normal_close:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        return 1
    finally:
        if args.normal_close:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
