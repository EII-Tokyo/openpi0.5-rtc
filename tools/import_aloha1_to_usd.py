#!/usr/bin/env python3
"""Headless Isaac Sim 5.1 URDF import for Stationary ALOHA 1."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
import xml.etree.ElementTree as ET

from tools.aloha1_mapping.urdf_audit import audit_urdf

IMPORT_SETTINGS: dict[str, Any] = {
    "merge_fixed_joints": False,
    "replace_cylinders_with_capsules": False,
    "convex_decomp": False,
    "import_inertia_tensor": True,
    "fix_base": True,
    "self_collision": False,
    "density": 0.0,
    "distance_scale": 1.0,
    "default_drive_type": "JOINT_DRIVE_POSITION",
    "default_drive_strength": 1000.0,
    "default_position_drive_damping": 100.0,
    "make_default_prim": True,
    "parse_mimic": True,
    "create_physics_scene": False,
    "collision_from_visuals": False,
    "override_joint_dynamics": False,
    "mesh_merge_requested": False,
    "requires_complete_urdf_dynamics": True,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _robot_plan(project_root: Path, name: str, model: str) -> dict[str, Any]:
    family = "follower_vx300s" if model == "aloha_vx300s" else "leader_wx250s"
    output_dir = (
        project_root / "assets/Trossen/ALOHA1/1.0" / family / name
    ).resolve()
    return {
        "name": name,
        "model": model,
        "urdf": str(
            (project_root / "generated/urdf" / f"{name}.urdf").resolve()
        ),
        "output_dir": str(output_dir),
        "source_urdf": str((output_dir / "source" / f"{name}.urdf").resolve()),
        "imported_usd": str(
            (output_dir / "source" / f"{name}_imported.usd").resolve()
        ),
        "final_usd": str((output_dir / f"{name}.usd").resolve()),
    }


def build_import_plan(
    *,
    project_root: Path,
    enable_leaders: bool,
) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    robots = [
        _robot_plan(root, "follower_left", "aloha_vx300s"),
        _robot_plan(root, "follower_right", "aloha_vx300s"),
    ]
    if enable_leaders:
        robots.extend(
            [
                _robot_plan(root, "leader_left", "aloha_wx250s"),
                _robot_plan(root, "leader_right", "aloha_wx250s"),
            ]
        )
    for robot in robots:
        urdf_path = Path(robot["urdf"])
        if not urdf_path.is_file():
            raise FileNotFoundError(f"generated URDF is unavailable: {urdf_path}")
    return {
        "schema_version": 1,
        "isaac_sim_required": "5.1.0.0",
        "output_strategy": "direct_to_stable_destination",
        "post_import_dependency_check": True,
        "enable_leaders": enable_leaders,
        "settings": dict(IMPORT_SETTINGS),
        "robots": robots,
    }


def _file_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _asset_path_text(value: Any) -> str:
    path = getattr(value, "path", None)
    return str(path) if path is not None else str(value)


def _urdf_links_without_visuals(urdf_path: Path) -> list[str]:
    root = ET.parse(urdf_path).getroot()
    return sorted(
        link.attrib["name"]
        for link in root.findall("link")
        if link.find("visual") is None
    )


def _existing_import_matches(
    robot: Mapping[str, Any],
    *,
    urdf_sha256: str,
    settings: Mapping[str, Any],
) -> bool:
    manifest_path = Path(robot["output_dir"]) / "source/import_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        manifest.get("urdf_sha256") == urdf_sha256
        and manifest.get("settings") == settings
        and Path(robot["imported_usd"]).is_file()
        and Path(robot["final_usd"]).is_file()
    )


def _configure_import(import_config: Any, urdf_module: Any) -> None:
    import_config.merge_fixed_joints = IMPORT_SETTINGS["merge_fixed_joints"]
    import_config.replace_cylinders_with_capsules = IMPORT_SETTINGS[
        "replace_cylinders_with_capsules"
    ]
    import_config.convex_decomp = IMPORT_SETTINGS["convex_decomp"]
    import_config.import_inertia_tensor = IMPORT_SETTINGS[
        "import_inertia_tensor"
    ]
    import_config.fix_base = IMPORT_SETTINGS["fix_base"]
    import_config.self_collision = IMPORT_SETTINGS["self_collision"]
    import_config.density = IMPORT_SETTINGS["density"]
    import_config.distance_scale = IMPORT_SETTINGS["distance_scale"]
    import_config.default_drive_type = (
        urdf_module.UrdfJointTargetType.JOINT_DRIVE_POSITION
    )
    import_config.default_drive_strength = IMPORT_SETTINGS[
        "default_drive_strength"
    ]
    import_config.default_position_drive_damping = IMPORT_SETTINGS[
        "default_position_drive_damping"
    ]
    import_config.make_default_prim = IMPORT_SETTINGS["make_default_prim"]
    import_config.parse_mimic = IMPORT_SETTINGS["parse_mimic"]
    import_config.create_physics_scene = IMPORT_SETTINGS[
        "create_physics_scene"
    ]
    import_config.collision_from_visuals = IMPORT_SETTINGS[
        "collision_from_visuals"
    ]
    import_config.override_joint_dynamics = IMPORT_SETTINGS[
        "override_joint_dynamics"
    ]


def _inspect_usd(usd_path: Path) -> dict[str, Any]:
    from pxr import PhysxSchema
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdPhysics
    from pxr import UsdUtils

    diagnostic_delegate = UsdUtils.CoalescingDiagnosticDelegate()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"unable to open imported USD: {usd_path}")
    dependency_layers, dependency_assets, unresolved_paths = (
        UsdUtils.ComputeAllDependencies(stage.GetRootLayer().identifier)
    )
    diagnostics = diagnostic_delegate.TakeUncoalescedDiagnostics()
    non_status_diagnostics = [
        {
            "code": item.diagnosticCodeString,
            "commentary": item.commentary,
            "source_file": item.sourceFileName,
            "source_function": item.sourceFunction,
            "source_line": item.sourceLineNumber,
        }
        for item in diagnostics
        if item.diagnosticCodeString != "TF_DIAGNOSTIC_STATUS_TYPE"
    ]
    default_prim = stage.GetDefaultPrim()
    articulation_roots: list[str] = []
    joints: list[str] = []
    mimic_joints: list[str] = []
    invalid_names: list[str] = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(str(prim.GetPath()))
        if prim.IsA(UsdPhysics.Joint):
            joints.append(str(prim.GetPath()))
            schemas = set(prim.GetAppliedSchemas())
            if any(
                schema.startswith("PhysxMimicJointAPI")
                for schema in schemas
            ) or prim.HasAPI(PhysxSchema.PhysxMimicJointAPI):
                mimic_joints.append(str(prim.GetPath()))
        if prim.GetPath() != Sdf.Path.absoluteRootPath and not Sdf.Path.IsValidIdentifier(
            prim.GetName()
        ):
            invalid_names.append(str(prim.GetPath()))
    return {
        "default_prim": str(default_prim.GetPath()) if default_prim else None,
        "articulation_roots": articulation_roots,
        "joint_paths": joints,
        "mimic_joint_paths": mimic_joints,
        "invalid_prim_names": invalid_names,
        "dependency_layers": sorted(
            str(layer.identifier) for layer in dependency_layers
        ),
        "dependency_assets": sorted(
            _asset_path_text(asset) for asset in dependency_assets
        ),
        "unresolved_paths": sorted(
            _asset_path_text(path) for path in unresolved_paths
        ),
        "composition_diagnostics": non_status_diagnostics,
    }


def _create_reference_wrapper(
    *,
    final_usd: Path,
    imported_usd: Path,
    imported_root: str,
    robot_name: str,
) -> None:
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom

    stage = Usd.Stage.CreateNew(str(final_usd))
    root = UsdGeom.Xform.Define(stage, Sdf.Path(f"/{robot_name}")).GetPrim()
    stage.SetDefaultPrim(root)
    relative_asset = os.path.relpath(imported_usd, final_usd.parent)
    if not root.GetReferences().AddReference(relative_asset, imported_root):
        raise RuntimeError(
            f"unable to author explicit reference to {imported_usd}"
        )
    stage.GetRootLayer().Save()


def _remove_importer_empty_visual_specs(
    *,
    imported_usd: Path,
    imported_root: str,
    empty_visual_links: Sequence[str],
) -> list[str]:
    """Remove Isaac 5.1 importer arcs that target nonexistent visual prims."""
    from pxr import Sdf

    base_layer_path = (
        imported_usd.parent
        / "configuration"
        / f"{imported_usd.stem}_base.usd"
    )
    layer = Sdf.Layer.FindOrOpen(str(base_layer_path))
    if layer is None:
        raise RuntimeError(f"importer base layer is unavailable: {base_layer_path}")
    removed: list[str] = []
    edit = Sdf.BatchNamespaceEdit()
    for link_name in empty_visual_links:
        spec_path = Sdf.Path(imported_root).AppendChild(
            link_name
        ).AppendChild("visuals")
        if layer.GetPrimAtPath(spec_path) is not None:
            edit.Add(spec_path, Sdf.Path.emptyPath)
            removed.append(str(spec_path))
    if removed:
        if not layer.Apply(edit):
            raise RuntimeError(
                f"failed to remove invalid empty-visual specs from {base_layer_path}"
            )
        layer.Save()
    return removed


def _validate_inspection(
    inspection: Mapping[str, Any],
    *,
    robot_name: str,
) -> None:
    if len(inspection["articulation_roots"]) != 1:
        raise RuntimeError(
            f"expected one articulation root for {robot_name}, "
            f"found {inspection['articulation_roots']}"
        )
    if inspection["invalid_prim_names"]:
        raise RuntimeError(
            f"invalid USD names after import for {robot_name}: "
            f"{inspection['invalid_prim_names']}"
        )
    if inspection["unresolved_paths"]:
        raise RuntimeError(
            f"unresolved USD dependencies for {robot_name}: "
            f"{inspection['unresolved_paths']}"
        )
    if inspection["composition_diagnostics"]:
        raise RuntimeError(
            f"USD composition diagnostics for {robot_name}: "
            f"{inspection['composition_diagnostics']}"
        )


def run_import(
    plan: Mapping[str, Any],
    *,
    report_path: Path | None = None,
) -> dict[str, Any]:
    for robot in plan["robots"]:
        audit = audit_urdf(Path(robot["urdf"]), package_map={})
        if audit["status"] != "PASS" or audit["missing_dynamics"]:
            raise RuntimeError(
                f"URDF preflight failed for {robot['name']}: "
                f"{audit['issues']}"
            )

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    results: list[dict[str, Any]] = []
    try:
        from isaacsim.asset.importer.urdf import _urdf
        import omni.kit.app
        import omni.kit.commands

        manager = omni.kit.app.get_app().get_extension_manager()
        extension_id = "isaacsim.asset.importer.urdf"
        if not manager.is_extension_enabled(extension_id):
            manager.set_extension_enabled_immediate(
                extension_id,
                True,  # noqa: FBT003
            )
        if not manager.is_extension_enabled(extension_id):
            raise RuntimeError(f"required extension is disabled: {extension_id}")

        for robot in plan["robots"]:
            urdf_path = Path(robot["urdf"])
            urdf_sha256 = _sha256(urdf_path)
            output_dir = Path(robot["output_dir"])
            if output_dir.exists():
                if _existing_import_matches(
                    robot,
                    urdf_sha256=urdf_sha256,
                    settings=plan["settings"],
                ):
                    inspection = _inspect_usd(Path(robot["final_usd"]))
                    _validate_inspection(
                        inspection,
                        robot_name=robot["name"],
                    )
                    results.append(
                        {
                            "name": robot["name"],
                            "status": "REUSED_IDENTICAL",
                            "urdf_sha256": urdf_sha256,
                            "inspection": inspection,
                            "file_inventory": _file_inventory(output_dir),
                        }
                    )
                    continue
                raise RuntimeError(
                    f"existing import differs or is incomplete: {output_dir}"
                )

            output_dir.parent.mkdir(parents=True, exist_ok=True)
            source_dir = output_dir / "source"
            source_dir.mkdir(parents=True)
            source_urdf = Path(robot["source_urdf"])
            imported_usd = Path(robot["imported_usd"])
            final_usd = Path(robot["final_usd"])
            shutil.copy2(urdf_path, source_urdf)
            status, import_config = omni.kit.commands.execute(
                "URDFCreateImportConfig"
            )
            if not status:
                raise RuntimeError("URDFCreateImportConfig failed")
            _configure_import(import_config, _urdf)
            status, imported_root = omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=str(source_urdf),
                import_config=import_config,
                dest_path=str(imported_usd),
                get_articulation_root=False,
            )
            if not status or not imported_root:
                raise RuntimeError(
                    f"URDFParseAndImportFile failed for {robot['name']}"
                )
            removed_empty_visual_specs = _remove_importer_empty_visual_specs(
                imported_usd=imported_usd,
                imported_root=str(imported_root),
                empty_visual_links=_urdf_links_without_visuals(source_urdf),
            )
            imported_inspection = _inspect_usd(imported_usd)
            _validate_inspection(
                imported_inspection,
                robot_name=robot["name"],
            )
            _create_reference_wrapper(
                final_usd=final_usd,
                imported_usd=imported_usd,
                imported_root=str(imported_root),
                robot_name=robot["name"],
            )
            wrapper_inspection = _inspect_usd(final_usd)
            _validate_inspection(
                wrapper_inspection,
                robot_name=robot["name"],
            )
            import_manifest = {
                "schema_version": 1,
                "robot": robot["name"],
                "model": robot["model"],
                "urdf_sha256": urdf_sha256,
                "settings": plan["settings"],
                "output_strategy": plan["output_strategy"],
                "isaac_5_1_empty_visual_spec_fix": {
                    "basis": "URDF links without visual elements",
                    "removed_specs": removed_empty_visual_specs,
                },
                "imported_root": str(imported_root),
                "imported_inspection": imported_inspection,
                "wrapper_inspection": wrapper_inspection,
            }
            _write_json(source_dir / "import_manifest.json", import_manifest)
            results.append(
                {
                    "name": robot["name"],
                    "status": "IMPORTED",
                    "urdf_sha256": urdf_sha256,
                    "inspection": wrapper_inspection,
                    "file_inventory": _file_inventory(output_dir),
                }
            )
        report = {
            "schema_version": 1,
            "status": "PASS",
            "isaac_sim": "5.1.0.0",
            "output_strategy": plan["output_strategy"],
            "settings": plan["settings"],
            "enable_leaders": plan["enable_leaders"],
            "robots": results,
        }
        if report_path is not None:
            _write_json(report_path, report)
    finally:
        app.close()
    return report


def _environment_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--enable-leaders", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/aloha1_mapping/import_manifest.json"),
    )
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)
    enable_leaders = arguments.enable_leaders or _environment_flag(
        "ENABLE_LEADERS"
    )
    plan = build_import_plan(
        project_root=arguments.project_root,
        enable_leaders=enable_leaders,
    )
    report_path = (
        arguments.report
        if arguments.report.is_absolute()
        else arguments.project_root / arguments.report
    )
    run_import(plan, report_path=report_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
