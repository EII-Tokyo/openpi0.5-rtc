from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase28_physics_property_comparison_20260718"
DEFAULT_ALOHA_LEFT_USD = REPO_ROOT / "assets/isaac/aloha1_native_physics_wrapper/aloha1_left.usda"
DEFAULT_ALOHA_RIGHT_USD = REPO_ROOT / "assets/isaac/aloha1_native_physics_wrapper/aloha1_right.usda"
DEFAULT_TROSSEN_USD = REPO_ROOT / "external/trossen_ai_isaac/assets/robots/stationary_ai/stationary_ai.usd"


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


def _attr_value(attr: Any) -> Any:
    try:
        value = attr.Get()
    except Exception:
        return None
    return _json_safe(value)


def _targets(rel: Any) -> list[str]:
    try:
        return [str(target) for target in rel.GetTargets()]
    except Exception:
        return []


def _applied(prim: Any) -> list[str]:
    try:
        return [str(item) for item in prim.GetAppliedSchemas()]
    except Exception:
        return []


def _drive_instances(applied_schemas: list[str]) -> list[str]:
    instances: list[str] = []
    for schema in applied_schemas:
        if schema.startswith("PhysicsDriveAPI:"):
            instances.append(schema.split(":", 1)[1])
    return sorted(set(instances))


def _drive_values(prim: Any, instance: str) -> dict[str, Any]:
    from pxr import UsdPhysics

    drive = UsdPhysics.DriveAPI(prim, instance)
    return {
        "instance": instance,
        "stiffness": _attr_value(drive.GetStiffnessAttr()),
        "damping": _attr_value(drive.GetDampingAttr()),
        "max_force": _attr_value(drive.GetMaxForceAttr()),
        "target_position": _attr_value(drive.GetTargetPositionAttr()),
        "target_velocity": _attr_value(drive.GetTargetVelocityAttr()),
    }


def _joint_row(prim: Any) -> dict[str, Any] | None:
    from pxr import UsdPhysics

    type_name = prim.GetTypeName()
    if "Joint" not in type_name:
        return None
    applied_schemas = _applied(prim)
    row: dict[str, Any] = {
        "path": str(prim.GetPath()),
        "name": prim.GetName(),
        "type": type_name,
        "applied_schemas": applied_schemas,
        "body0": [],
        "body1": [],
        "axis": None,
        "lower": None,
        "upper": None,
        "local_pos0": None,
        "local_pos1": None,
        "local_rot0": None,
        "local_rot1": None,
        "drives": [],
        "is_mimic": any(schema.startswith("PhysxMimicJointAPI") for schema in applied_schemas),
    }
    try:
        joint = UsdPhysics.Joint(prim)
        row["body0"] = _targets(joint.GetBody0Rel())
        row["body1"] = _targets(joint.GetBody1Rel())
        row["local_pos0"] = _attr_value(joint.GetLocalPos0Attr())
        row["local_pos1"] = _attr_value(joint.GetLocalPos1Attr())
        row["local_rot0"] = _attr_value(joint.GetLocalRot0Attr())
        row["local_rot1"] = _attr_value(joint.GetLocalRot1Attr())
    except Exception as exc:
        row["joint_error"] = f"{type(exc).__name__}: {exc}"
    try:
        if prim.IsA(UsdPhysics.RevoluteJoint):
            typed = UsdPhysics.RevoluteJoint(prim)
            row["axis"] = _attr_value(typed.GetAxisAttr())
            row["lower"] = _attr_value(typed.GetLowerLimitAttr())
            row["upper"] = _attr_value(typed.GetUpperLimitAttr())
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            typed = UsdPhysics.PrismaticJoint(prim)
            row["axis"] = _attr_value(typed.GetAxisAttr())
            row["lower"] = _attr_value(typed.GetLowerLimitAttr())
            row["upper"] = _attr_value(typed.GetUpperLimitAttr())
    except Exception as exc:
        row["typed_joint_error"] = f"{type(exc).__name__}: {exc}"
    for instance in _drive_instances(applied_schemas):
        row["drives"].append(_drive_values(prim, instance))
    return row


def _mass_row(prim: Any) -> dict[str, Any] | None:
    from pxr import UsdPhysics

    applied_schemas = _applied(prim)
    if "PhysicsMassAPI" not in applied_schemas and not bool(UsdPhysics.MassAPI(prim)):
        return None
    mass = UsdPhysics.MassAPI(prim)
    return {
        "path": str(prim.GetPath()),
        "name": prim.GetName(),
        "type": prim.GetTypeName(),
        "mass": _attr_value(mass.GetMassAttr()),
        "density": _attr_value(mass.GetDensityAttr()),
        "center_of_mass": _attr_value(mass.GetCenterOfMassAttr()),
        "diagonal_inertia": _attr_value(mass.GetDiagonalInertiaAttr()),
        "principal_axes": _attr_value(mass.GetPrincipalAxesAttr()),
    }


def _has_api(prim: Any, schema_name: str) -> bool:
    return schema_name in _applied(prim)


def _stage_layer_summary(stage: Any) -> dict[str, Any]:
    root = stage.GetRootLayer()
    return {
        "root_identifier": root.identifier,
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim().IsValid() else None,
        "sub_layers": list(root.subLayerPaths),
    }


def _analyze_stage(path: Path, focus_terms: tuple[str, ...]) -> dict[str, Any]:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path.resolve()))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {path}")

    joints: list[dict[str, Any]] = []
    focus_joints: list[dict[str, Any]] = []
    mass_rows: list[dict[str, Any]] = []
    colliders: list[str] = []
    rigid_bodies: list[str] = []
    articulation_roots: list[str] = []
    unresolved_like_refs: list[dict[str, Any]] = []

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        applied_schemas = _applied(prim)
        if "ArticulationRootAPI" in applied_schemas or "PhysicsArticulationRootAPI" in applied_schemas:
            articulation_roots.append(prim_path)
        if _has_api(prim, "PhysicsCollisionAPI"):
            colliders.append(prim_path)
        if _has_api(prim, "PhysicsRigidBodyAPI"):
            rigid_bodies.append(prim_path)
        joint = _joint_row(prim)
        if joint is not None:
            joints.append(joint)
            lowered = joint["path"].lower()
            if any(term.lower() in lowered for term in focus_terms):
                focus_joints.append(joint)
        mass = _mass_row(prim)
        if mass is not None:
            mass_rows.append(mass)
        try:
            refs = prim.GetMetadata("references")
            if refs and not prim.IsValid():
                unresolved_like_refs.append({"path": prim_path, "references": str(refs)})
        except Exception:
            pass

    driven_joints = [row for row in joints if row["drives"]]
    mimic_joints = [row for row in joints if row["is_mimic"]]
    return {
        "path": _rel(path),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "layers": _stage_layer_summary(stage),
        "prim_count": sum(1 for _ in stage.Traverse()),
        "articulation_roots": articulation_roots,
        "joint_count": len(joints),
        "driven_joint_count": len(driven_joints),
        "mimic_joint_count": len(mimic_joints),
        "collision_count": len(colliders),
        "rigid_body_count": len(rigid_bodies),
        "mass_api_count": len(mass_rows),
        "focus_terms": list(focus_terms),
        "focus_joints": focus_joints,
        "joints_sample": joints[:80],
        "mass_rows_sample": mass_rows[:120],
        "colliders_sample": colliders[:120],
        "rigid_bodies_sample": rigid_bodies[:120],
        "unresolved_like_refs": unresolved_like_refs[:40],
    }


def _compact_asset_summary(asset: dict[str, Any]) -> dict[str, Any]:
    drive_max_forces: list[float] = []
    drive_stiffness: list[float] = []
    drive_damping: list[float] = []
    for row in asset["joints_sample"]:
        for drive in row.get("drives", []):
            for key, target in (
                ("max_force", drive_max_forces),
                ("stiffness", drive_stiffness),
                ("damping", drive_damping),
            ):
                value = drive.get(key)
                if isinstance(value, (int, float)):
                    target.append(float(value))
    return {
        "path": asset["path"],
        "meters_per_unit": asset["meters_per_unit"],
        "up_axis": asset["up_axis"],
        "articulation_root_count": len(asset["articulation_roots"]),
        "joint_count": asset["joint_count"],
        "driven_joint_count": asset["driven_joint_count"],
        "mimic_joint_count": asset["mimic_joint_count"],
        "collision_count": asset["collision_count"],
        "rigid_body_count": asset["rigid_body_count"],
        "mass_api_count": asset["mass_api_count"],
        "drive_max_force_values": sorted(set(drive_max_forces)),
        "drive_stiffness_values": sorted(set(drive_stiffness)),
        "drive_damping_values": sorted(set(drive_damping)),
        "focus_joint_names": [row["name"] for row in asset["focus_joints"]],
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 28 Physics Property Comparison",
        "",
        "## Question",
        "",
        "Phase 26 ruled out a generic `SingleArticulation` API failure and showed that the native ALOHA1 drift remains inside the imported articulation chain. This report compares the static USD physics properties of the current ALOHA1-native assets against the known-working Trossen Stationary AI asset.",
        "",
        "## Official Isaac Basis",
        "",
        "Isaac robot setup guidance says a controller-ready robot needs clean composition, active joint drives, meaningful limits/effort/damping, rigid bodies, colliders, mass/inertia, and a clear articulation root. Initializing an articulation is not enough.",
        "",
        "## Inputs",
        "",
    ]
    for name, item in payload["assets"].items():
        lines.append(f"- {name}: `{item['path']}`")
    lines.extend(["", "## Compact Summary", "", "| asset | roots | joints | driven | mimic | rigid bodies | colliders | mass APIs | max force values | stiffness values | damping values |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |"])
    for name, item in payload["compact"].items():
        lines.append(
            f"| {name} | {item['articulation_root_count']} | {item['joint_count']} | "
            f"{item['driven_joint_count']} | {item['mimic_joint_count']} | {item['rigid_body_count']} | "
            f"{item['collision_count']} | {item['mass_api_count']} | "
            f"`{item['drive_max_force_values']}` | `{item['drive_stiffness_values']}` | `{item['drive_damping_values']}` |"
        )
    lines.extend(["", "## Key Findings", ""])
    lines.extend(f"- {finding}" for finding in payload["findings"])
    lines.extend(
        [
            "",
            "## Decision",
            "",
        "Do not continue to bottle contact or grasp simulation from the current ALOHA1-native asset until the missing or inconsistent physics properties identified here are repaired and revalidated with the dynamic hold gate.",
        "",
        "Important nuance: this static source-layer inspection sees ALOHA1 mass and collider APIs in the wrapper files. Phase 27 separately showed that the current runtime reference paths compose zero collision prims under `/World`. Therefore the current problem is not simply that the source files contain no collider data; it is that source-layer physics, runtime composition, units/up-axis, and dynamic drive behavior are not yet a clean Isaac-ready robot asset.",
        "",
        "## Artifacts",
            "",
            f"- JSON: `{payload['outputs']['json']}`",
            f"- Markdown: `{payload['outputs']['markdown']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ALOHA1-native and Trossen Stationary AI USD physics properties.")
    parser.add_argument("--aloha-left-usd", default=str(DEFAULT_ALOHA_LEFT_USD))
    parser.add_argument("--aloha-right-usd", default=str(DEFAULT_ALOHA_RIGHT_USD))
    parser.add_argument("--trossen-usd", default=str(DEFAULT_TROSSEN_USD))
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
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        assets = {
            "aloha_left_native": _analyze_stage(
                Path(args.aloha_left_usd),
                ("waist", "shoulder", "elbow", "forearm", "wrist", "gripper", "finger"),
            ),
            "aloha_right_native": _analyze_stage(
                Path(args.aloha_right_usd),
                ("waist", "shoulder", "elbow", "forearm", "wrist", "gripper", "finger"),
            ),
            "trossen_stationary_ai": _analyze_stage(
                Path(args.trossen_usd),
                ("follower_left", "follower_right", "leader_left", "leader_right"),
            ),
        }
        compact = {name: _compact_asset_summary(asset) for name, asset in assets.items()}

        findings: list[str] = []
        trossen = compact["trossen_stationary_ai"]
        for name in ("aloha_left_native", "aloha_right_native"):
            item = compact[name]
            if item["meters_per_unit"] != trossen["meters_per_unit"]:
                findings.append(
                    f"{name} uses meters_per_unit={item['meters_per_unit']} while Trossen uses {trossen['meters_per_unit']}; unit conversion must be audited before copying gains or inertias."
                )
            if item["up_axis"] != trossen["up_axis"]:
                findings.append(f"{name} uses up_axis={item['up_axis']} while Trossen uses {trossen['up_axis']}; frame conversion must be explicit.")
            if item["drive_damping_values"] == [0.0]:
                findings.append(f"{name} arm drives have zero authored damping in this static inspection, unlike Trossen's positive damping values.")
            if item["collision_count"] > 0:
                findings.append(
                    f"{name} contains {item['collision_count']} static collision prims, but Phase 27 showed the current runtime /World reference path composes zero collision prims."
                )
            else:
                findings.append(f"{name} has zero composed `PhysicsCollisionAPI` prims in this static stage view.")
            if item["mass_api_count"] == 0:
                findings.append(f"{name} has zero composed `PhysicsMassAPI` prims in this static stage view.")
            if item["driven_joint_count"] == 0:
                findings.append(f"{name} has no driven joints in static USD inspection.")
        if trossen["collision_count"] > 0:
            findings.append("Trossen Stationary AI composes collision prims, unlike the current ALOHA1 runtime entry points.")
        if trossen["mass_api_count"] > 0:
            findings.append("Trossen Stationary AI authors mass/inertia data, giving a concrete reference for rebuilding ALOHA1 physics.")
        if not findings:
            findings.append("No obvious static-count difference was detected; inspect the JSON focus joint rows next.")

        payload = {
            "status": "PASS",
            "assets": assets,
            "compact": compact,
            "findings": findings,
            "outputs": {
                "json": _rel(output_dir / "physics_property_comparison.json"),
                "markdown": _rel(output_dir / "physics_property_comparison.md"),
            },
        }
        json_path = output_dir / "physics_property_comparison.json"
        md_path = output_dir / "physics_property_comparison.md"
        json_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n")
        md_path.write_text(_render_markdown(payload))
        print(
            json.dumps(
                {"status": "PASS", "json": str(json_path), "markdown": str(md_path), "findings": findings},
                ensure_ascii=False,
            ),
            flush=True,
        )
        if not args.normal_close:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        return 0
    finally:
        if args.normal_close:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
