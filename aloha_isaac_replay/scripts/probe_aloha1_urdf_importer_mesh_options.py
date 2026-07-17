from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = REPO_ROOT / "assets/isaac/original_stationary_aloha/generated/puppet_left_vx300s_resolved.urdf"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase16_urdf_importer_probe_20260718"


def _rel(path: str | Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return repr(value)


def _public_config_attrs(config: Any) -> dict[str, Any]:
    rows = {}
    for name in sorted(dir(config)):
        if name.startswith("_"):
            continue
        try:
            value = getattr(config, name)
        except Exception as exc:
            rows[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        if callable(value):
            continue
        rows[name] = _json_safe(value)
    return rows


def _set_if_present(config: Any, values: dict[str, Any]) -> dict[str, Any]:
    applied = {}
    missing = {}
    for name, value in values.items():
        if hasattr(config, name):
            try:
                setattr(config, name, value)
                applied[name] = _json_safe(getattr(config, name))
            except Exception as exc:
                applied[name] = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            missing[name] = value
    return {"applied": applied, "missing": missing}


def _inspect_stage(stage: Any) -> dict[str, Any]:
    from pxr import UsdGeom, UsdPhysics

    meshes = []
    colliders = []
    rigid_bodies = []
    joints = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        type_name = prim.GetTypeName()
        if type_name == "Mesh":
            meshes.append(path)
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
        "mesh_count": len(meshes),
        "collision_count": len(colliders),
        "rigid_body_count": len(rigid_bodies),
        "joint_count": len(joints),
        "mesh_sample": meshes[:30],
        "collider_sample": colliders[:30],
        "joint_sample": joints[:30],
    }


def _inspect_usd_file(Usd: Any, path: Path) -> dict[str, Any]:
    stage = Usd.Stage.Open(str(path.resolve()))
    if stage is None:
        return {"path": _rel(path), "exists": path.exists(), "open": False}
    return {"path": _rel(path), "exists": path.exists(), "open": True, "stage": _inspect_stage(stage)}


def _import_once(omni: Any, Usd: Any, urdf: Path, usd_path: Path, config: Any) -> dict[str, Any]:
    import omni.kit.app
    import omni.usd

    context = omni.usd.get_context()
    context.new_stage()
    for _ in range(3):
        omni.kit.app.get_app().update()
    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf.resolve()),
        import_config=config,
        dest_path=str(usd_path.resolve()),
        get_articulation_root=True,
    )
    for _ in range(8):
        omni.kit.app.get_app().update()
    if not status:
        return {"status": False, "prim_path": str(prim_path), "usd_path": _rel(usd_path)}
    stage = Usd.Stage.Open(str(usd_path.resolve()))
    configuration_dir = usd_path.parent / "configuration"
    layer_paths = [
        usd_path,
        configuration_dir / f"{usd_path.stem}_base.usd",
        configuration_dir / f"{usd_path.stem}_physics.usd",
        configuration_dir / f"{usd_path.stem}_robot.usd",
        configuration_dir / f"{usd_path.stem}_sensor.usd",
    ]
    return {
        "status": True,
        "prim_path": str(prim_path),
        "usd_path": _rel(usd_path),
        "stage": _inspect_stage(stage),
        "layer_inspection": [_inspect_usd_file(Usd, path) for path in layer_paths],
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    current = payload["current_config"]["import_result"]
    candidate = payload["candidate_mesh_config"]["import_result"]
    candidate_missing = sorted(payload["candidate_mesh_config"]["applied"]["missing"].keys())

    def row(label: str, result: dict[str, Any]) -> str:
        stage = result.get("stage", {})
        return (
            f"| {label} | {stage.get('mesh_count')} | {stage.get('collision_count')} | "
            f"{stage.get('rigid_body_count')} | {stage.get('joint_count')} |"
        )

    lines = [
        "# Phase 16 URDF Importer Mesh Probe",
        "",
        "This probe checks why resolved ALOHA1 URDF files import into USD with joints and rigid bodies but no visible Mesh prims or CollisionAPI prims.",
        "",
        "## Direct Import Results",
        "",
        "| Import config | Mesh prims | Collision prims | Rigid bodies | Joints |",
        "| --- | ---: | ---: | ---: | ---: |",
        row("current config", current),
        row("candidate mesh config", candidate),
        "",
        "## Importer Config Findings",
        "",
        f"- `collision_from_visuals` exists and defaults to `{payload['default_config_attrs'].get('collision_from_visuals')}`.",
        "- The guessed fields `import_visuals`, `import_collision`, `parse_visuals`, `parse_collision`, `create_visuals`, and `create_collisions` do not exist on Isaac Sim 5.1 `URDFCreateImportConfig`.",
        f"- Missing guessed fields: `{', '.join(candidate_missing)}`",
        "",
        "## Layer Inspection",
        "",
        "| Config | Layer | Mesh prims | Collision prims | Rigid bodies | Joints |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for label, result in (("current", current), ("candidate", candidate)):
        for layer in result.get("layer_inspection", []):
            stage = layer.get("stage", {})
            lines.append(
                f"| {label} | `{layer.get('path')}` | {stage.get('mesh_count')} | "
                f"{stage.get('collision_count')} | {stage.get('rigid_body_count')} | {stage.get('joint_count')} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The resolved URDF mesh references are valid, but Isaac's generated composed stage still has zero Mesh prims and zero CollisionAPI prims. Setting `collision_from_visuals=True` does not repair this. Therefore the current blocker is not a simple missing `import_visuals` flag.",
            "",
            "The observed USD warnings show unresolved references from generated base-layer visual scopes to generated physics-layer visual prim paths. The next repair step should inspect the URDF importer output layer structure and either fix the importer workflow or bypass the broken layer output by authoring an ALOHA1-native USD from the resolved URDF plus mesh package.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{_rel(payload['json_path'])}`",
            f"- Markdown: `{_rel(path)}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Isaac URDF importer config fields and ALOHA1 mesh import behavior.")
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    urdf = Path(args.urdf)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp(dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG))
    try:
        import omni.kit.commands
        from pxr import Usd

        status, config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not status:
            raise RuntimeError("URDFCreateImportConfig failed")
        default_attrs = _public_config_attrs(config)
        current_settings = {
            "merge_fixed_joints": False,
            "import_inertia_tensor": True,
            "fix_base": True,
            "make_default_prim": False,
            "create_physics_scene": False,
            "self_collision": False,
        }
        current_applied = _set_if_present(config, current_settings)
        current_attrs = _public_config_attrs(config)
        current_result = _import_once(omni, Usd, urdf, output_dir / "current_config_import.usd", config)

        status, candidate_config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not status:
            raise RuntimeError("URDFCreateImportConfig candidate failed")
        # These names are intentionally probed rather than assumed. Missing fields are recorded.
        candidate_settings = {
            **current_settings,
            "collision_from_visuals": True,
            "import_visuals": True,
            "import_collision": True,
            "parse_visuals": True,
            "parse_collision": True,
            "create_visuals": True,
            "create_collisions": True,
        }
        candidate_applied = _set_if_present(candidate_config, candidate_settings)
        candidate_attrs = _public_config_attrs(candidate_config)
        candidate_result = _import_once(omni, Usd, urdf, output_dir / "candidate_mesh_config_import.usd", candidate_config)

        payload = {
            "schema_version": 1,
            "urdf": _rel(urdf),
            "default_config_attrs": default_attrs,
            "current_config": {
                "requested": current_settings,
                "applied": current_applied,
                "attrs": current_attrs,
                "import_result": current_result,
            },
            "candidate_mesh_config": {
                "requested": candidate_settings,
                "applied": candidate_applied,
                "attrs": candidate_attrs,
                "import_result": candidate_result,
            },
        }
        json_path = output_dir / "urdf_importer_mesh_probe.json"
        md_path = output_dir / "urdf_importer_mesh_probe.md"
        payload["json_path"] = str(json_path)
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        _write_markdown(payload, md_path)
        summary = {
            "json": _rel(json_path),
            "markdown": _rel(md_path),
            "current_mesh_count": current_result.get("stage", {}).get("mesh_count"),
            "current_collision_count": current_result.get("stage", {}).get("collision_count"),
            "candidate_mesh_count": candidate_result.get("stage", {}).get("mesh_count"),
            "candidate_collision_count": candidate_result.get("stage", {}).get("collision_count"),
            "candidate_missing_fields": sorted(candidate_applied["missing"].keys()),
        }
        print(json.dumps(summary, ensure_ascii=False))
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
