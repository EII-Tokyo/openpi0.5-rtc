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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase31_clean_visual_targets_20260718"

MISSING_VISUAL_TARGETS = [
    "/visuals/puppet_left_ee_arm_link",
    "/visuals/puppet_left_fingers_link",
    "/visuals/puppet_left_ee_gripper_link",
    "/visuals/puppet_right_ee_arm_link",
    "/visuals/puppet_right_fingers_link",
    "/visuals/puppet_right_ee_gripper_link",
]


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


def _create_clean_visual_target_stage(path: Path, left_usd: Path, right_usd: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    root = stage.GetRootLayer()
    root.subLayerPaths.append(str(left_usd.resolve()))
    root.subLayerPaths.append(str(right_usd.resolve()))
    for target in MISSING_VISUAL_TARGETS:
        UsdGeom.Xform.Define(stage, target)
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
    runtime = payload.get("runtime_articulations", {})
    lines = [
        "# Phase 31 Clean Visual Target Diagnostic",
        "",
        "## Question",
        "",
        "Can the known missing ALOHA1 local `/visuals/...` reference targets be satisfied by explicit top-layer placeholder Xforms while preserving collision composition and runtime articulation initialization?",
        "",
        "This is still a diagnostic asset, not the final production robot asset.",
        "",
        "## Result",
        "",
        f"- status: `{payload['status']}`",
        f"- generated stage: `{payload['generated_stage']}`",
        f"- missing local reference targets: `{len(payload['missing_local_reference_targets'])}`",
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
    for side, item in runtime.items():
        lines.append(f"| {side} | `{item['status']}` | `{item['prim_path']}` | {item.get('num_dof')} | {item.get('num_bodies')} |")
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {line}" for line in payload.get("interpretation", []))
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
    parser = argparse.ArgumentParser(description="Validate ALOHA1 runtime composition after adding explicit missing visual target placeholders.")
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
        stage_path = output_dir / "aloha1_dual_clean_visual_targets.usda"
        json_path = output_dir / "clean_visual_targets.json"
        md_path = output_dir / "clean_visual_targets.md"
        _create_clean_visual_target_stage(stage_path, Path(args.left_usd), Path(args.right_usd))

        payload: dict[str, Any] = {
            "status": "STARTED",
            "left_usd": _rel(args.left_usd),
            "right_usd": _rel(args.right_usd),
            "generated_stage": _rel(stage_path),
            "known_placeholder_targets": list(MISSING_VISUAL_TARGETS),
            "outputs": {"json": _rel(json_path), "markdown": _rel(md_path)},
        }
        _write_json(json_path, payload)

        stage = Usd.Stage.Open(str(stage_path.resolve()))
        if stage is None:
            raise RuntimeError(f"Failed to reopen generated stage: {stage_path}")
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
        stage_utils.open_stage(str(stage_path.resolve()))
        world = World(stage_units_in_meters=0.01, backend="numpy", device="cpu")
        runtime: dict[str, dict[str, Any]] = {}
        for side, root_path in {
            "left": "/puppet_left_vx300s/root_joint",
            "right": "/puppet_right_vx300s/root_joint",
        }.items():
            try:
                runtime[side] = _inspect_articulation(world, root_path, f"{side}_clean_visual_target_vx300s")
            except Exception as exc:
                runtime[side] = {"status": "FAIL", "prim_path": root_path, "error": f"{type(exc).__name__}: {exc}"}

        interpretation = []
        if not missing_refs:
            interpretation.append("Explicit placeholder Xforms satisfy all local missing `/visuals/...` reference targets in composed-stage static inspection.")
        else:
            interpretation.append("Some local reference targets remain missing; placeholder repair is incomplete.")
        if collision_count > 0:
            interpretation.append("Collision composition is preserved after adding the visual target placeholders.")
        else:
            interpretation.append("Collision composition is lost; this approach is not sufficient.")
        if all(item["status"] == "PASS" for item in runtime.values()):
            interpretation.append("Both ALOHA1 articulations still initialize in Isaac Sim runtime.")
        else:
            interpretation.append("At least one ALOHA1 articulation failed to initialize in Isaac Sim runtime.")

        status = "STATIC_PASS_RUNTIME_LOG_AUDIT_REQUIRED" if not missing_refs and collision_count > 0 and all(item["status"] == "PASS" for item in runtime.values()) else "FAIL"
        if status == "STATIC_PASS_RUNTIME_LOG_AUDIT_REQUIRED":
            interpretation.append("This script cannot prove the Isaac runtime USD log is clean; inspect the codex-evidence stdout for unresolved reference warnings.")
        payload.update({"status": status, "runtime_articulations": runtime, "interpretation": interpretation})
        _write_json(json_path, payload)
        md_path.write_text(_render_markdown(payload))
        print(json.dumps({"status": status, "json": _rel(json_path), "markdown": _rel(md_path), "interpretation": interpretation}, ensure_ascii=False), flush=True)
        exit_code = 0 if status in {"PASS", "STATIC_PASS_RUNTIME_LOG_AUDIT_REQUIRED"} else 2
        if not args.normal_close:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
        app.close()
        return exit_code
    except Exception as exc:
        output_dir = Path(args.output_dir)
        json_path = output_dir / "clean_visual_targets.json"
        md_path = output_dir / "clean_visual_targets.md"
        payload = {
            "status": "EXCEPTION",
            "left_usd": _rel(args.left_usd),
            "right_usd": _rel(args.right_usd),
            "generated_stage": _rel(output_dir / "aloha1_dual_clean_visual_targets.usda"),
            "exception": f"{type(exc).__name__}: {exc}",
            "outputs": {"json": _rel(json_path), "markdown": _rel(md_path)},
        }
        _write_json(json_path, payload)
        md_path.write_text("# Phase 31 Clean Visual Target Diagnostic\n\nstatus: `EXCEPTION`\n\n" + payload["exception"] + "\n")
        print(json.dumps({"status": "EXCEPTION", "json": _rel(json_path), "exception": payload["exception"]}, ensure_ascii=False), flush=True)
        if not args.normal_close:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
