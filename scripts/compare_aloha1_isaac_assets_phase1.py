#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import math
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
from typing import Any


REPO = Path(__file__).resolve().parents[1]


DEFAULT_ALOHA1_USD = REPO / "assets/isaac/original_stationary_aloha/generated/original_stationary_aloha.usd"
DEFAULT_ALOHA1_IMPORT_REPORT = REPO / "assets/isaac/original_stationary_aloha/reports/import_report.json"
DEFAULT_TROSSEN_USD = REPO / "external/trossen_ai_isaac/assets/robots/stationary_ai/stationary_ai.usd"
DEFAULT_TROSSEN_PICK_PLACE = REPO / "external/trossen_ai_isaac/scripts/stationary_ai_pick_place.py"
DEFAULT_TROSSEN_ASSET_GENERATION = REPO / "external/trossen_ai_isaac/assets/robots/asset_generation.md"
DEFAULT_MENAGERIE_ALOHA = REPO / "external/mujoco_menagerie/aloha/aloha.xml"
DEFAULT_MENAGERIE_SCENE = REPO / "external/mujoco_menagerie/aloha/scene.xml"
DEFAULT_OUTPUT_DIR = REPO / "reports/aloha1_isaac_adaptation/phase1_asset_comparison_20260717"


USD_KEYWORDS = (
    "defaultPrim",
    "metersPerUnit",
    "upAxis",
    "ArticulationRoot",
    "PhysicsArticulationRoot",
    "PhysicsRevoluteJoint",
    "PhysicsPrismaticJoint",
    "PhysicsFixedJoint",
    "CollisionAPI",
    "Mesh",
    "Camera",
    "Material",
    "vx300s",
    "viperx",
    "aloha",
    "aloha2",
    "wxai",
    "stationary_ai",
    "follower_left",
    "follower_right",
    "left_finger",
    "right_finger",
    "carriage",
)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _file_type(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return subprocess.run(["file", str(path)], check=False, capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return "file command unavailable"


def _strings_text(path: Path, max_bytes: int = 8_000_000) -> str:
    if not path.exists():
        return ""
    if path.suffix.lower() == ".usda":
        return path.read_text(errors="replace")
    data = path.read_bytes()[:max_bytes]
    return "\n".join(token.decode("utf-8", errors="ignore") for token in data.split(b"\x00"))


def _pxr_available() -> bool:
    try:
        __import__("pxr")
        return True
    except Exception:
        return False


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        if math.isnan(value):
            return "NaN"
    return value


def inspect_usd_static(path: Path) -> dict[str, Any]:
    text = _strings_text(path)
    lowered = text.lower()
    return {
        "path": _rel(path),
        "exists": path.exists(),
        "file_type": _file_type(path),
        "pxr_available": _pxr_available(),
        "inspection_mode": "bounded_static_text_or_strings",
        "keyword_hits": {key: lowered.count(key.lower()) for key in USD_KEYWORDS},
        "blocked_fields": [
            "full prim tree",
            "composed stage layers",
            "articulation runtime DOF order",
            "true collider count",
            "world transforms",
        ],
        "blocked_reason": "BLOCKED_PXR_OR_ISAAC_REQUIRED",
    }


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _all(root: ET.Element, name: str) -> list[ET.Element]:
    return [elem for elem in root.iter() if _strip_namespace(elem.tag) == name]


def _attrs(elem: ET.Element, keys: tuple[str, ...]) -> dict[str, str | None]:
    return {key: elem.attrib.get(key) for key in keys}


def inspect_mjcf(aloha_xml: Path, scene_xml: Path) -> dict[str, Any]:
    root = ET.parse(aloha_xml).getroot()
    scene_root = ET.parse(scene_xml).getroot()
    base_dir = aloha_xml.parent
    includes = [elem.attrib.get("file") for elem in _all(root, "include") if elem.attrib.get("file")]
    include_details: dict[str, Any] = {}
    for include in includes:
        include_path = base_dir / include
        if not include_path.exists():
            include_details[include] = {"exists": False}
            continue
        inc_root = ET.parse(include_path).getroot()
        include_details[include] = {
            "exists": True,
            "actuators": [_attrs(elem, ("name", "joint", "class", "ctrlrange", "forcelimited", "forcerange")) for elem in _all(inc_root, "position") + _all(inc_root, "general")],
            "keyframes": [_attrs(elem, ("name", "qpos", "ctrl")) for elem in _all(inc_root, "key")],
        }

    joints = [_attrs(elem, ("name", "type", "axis", "range", "class", "limited")) for elem in _all(root, "joint")]
    cameras = [_attrs(elem, ("name", "mode", "pos", "xyaxes", "fovy")) for elem in _all(root, "camera")]
    scene_cameras = [_attrs(elem, ("name", "mode", "pos", "xyaxes", "fovy")) for elem in _all(scene_root, "camera")]
    geoms = [_attrs(elem, ("name", "type", "class", "mesh", "material", "pos", "size")) for elem in _all(root, "geom")]
    scene_geoms = [_attrs(elem, ("name", "type", "class", "mesh", "material", "pos", "size")) for elem in _all(scene_root, "geom")]
    meshes = [_attrs(elem, ("name", "file")) for elem in _all(root, "mesh") + _all(scene_root, "mesh")]
    equality = [_attrs(elem, ("joint1", "joint2", "polycoef")) for elem in _all(root, "joint") if "joint1" in elem.attrib or "joint2" in elem.attrib]

    return {
        "aloha_xml": _rel(aloha_xml),
        "scene_xml": _rel(scene_xml),
        "exists": aloha_xml.exists() and scene_xml.exists(),
        "model": root.attrib.get("model"),
        "compiler": [_attrs(elem, ("angle", "meshdir", "autolimits")) for elem in _all(root, "compiler")],
        "includes": includes,
        "include_details": include_details,
        "joint_count": len(joints),
        "joints": joints,
        "camera_count": len(cameras) + len(scene_cameras),
        "cameras": cameras,
        "scene_cameras": scene_cameras,
        "geom_count": len(geoms) + len(scene_geoms),
        "mesh_count": len(meshes),
        "mesh_names_sample": meshes[:40],
        "geoms_sample": (geoms + scene_geoms)[:60],
        "equality_constraints": equality,
        "markers": {
            "has_left_right_arm_prefixes": any((j.get("name") or "").startswith("left/") for j in joints) and any((j.get("name") or "").startswith("right/") for j in joints),
            "has_aloha2_custom_finger_meshes": any("custom_finger" in (mesh.get("file") or "") for mesh in meshes),
            "has_d405_mesh": any("d405" in (mesh.get("file") or "").lower() for mesh in meshes),
        },
    }


def _safe_eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_safe_eval_node(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return [_safe_eval_node(item) for item in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _safe_eval_node(node.operand)
        if isinstance(value, int | float):
            return -value
    if isinstance(node, ast.BinOp):
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        if isinstance(node.op, ast.Add) and isinstance(left, list) and isinstance(right, list):
            return left + right
        if isinstance(node.op, ast.Mult):
            if isinstance(left, list) and isinstance(right, int):
                return left * right
            if isinstance(right, list) and isinstance(left, int):
                return right * left
    if isinstance(node, ast.Call):
        func = ast.unparse(node.func)
        if func in {"np.array", "numpy.array"} and node.args:
            return _safe_eval_node(node.args[0])
    return ast.literal_eval(node)


def _literal_assignments(py_path: Path, names: set[str]) -> dict[str, Any]:
    tree = ast.parse(py_path.read_text())
    found: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                try:
                    found[target.id] = _safe_eval_node(node.value)
                except Exception:
                    found[target.id] = ast.unparse(node.value)
    return found


def inspect_trossen_demo(py_path: Path, asset_generation_md: Path) -> dict[str, Any]:
    names = {
        "ROBOT_USD_PATH",
        "ROBOT_SCENE_PATH",
        "LEFT_ARM_DOF_INDICES",
        "LEFT_GRIPPER_DOF_INDEX",
        "RIGHT_ARM_DOF_INDICES",
        "RIGHT_GRIPPER_DOF_INDEX",
        "STATIONARY_AI_DEFAULT_DOF_POSITIONS",
        "LEFT_ARM_HOME_POSITION",
        "RIGHT_ARM_HOME_POSITION",
        "LEFT_ARM_DOWNWARD_ORIENTATION",
        "RIGHT_ARM_DOWNWARD_ORIENTATION",
        "LEFT_ARM_HANDOFF_ORIENTATION",
        "RIGHT_ARM_HANDOFF_ORIENTATION",
    }
    assignments = _literal_assignments(py_path, names)
    md_text = asset_generation_md.read_text(errors="replace") if asset_generation_md.exists() else ""
    return {
        "script": _rel(py_path),
        "asset_generation_doc": _rel(asset_generation_md),
        "exists": py_path.exists(),
        "literal_assignments": assignments,
        "asset_generation_claims": {
            "mentions_trossen_arm_description": "trossen_arm_description" in md_text,
            "mentions_distance_scale_1": "Distance Scale: 1.0" in md_text,
            "mentions_self_collision_true": "Self Collision: True" in md_text,
            "mentions_convex_decomposition": "Convex Decomposition" in md_text,
            "mentions_isaac_sim_5_1": "Isaac Sim 5.1" in md_text or "Isaac Sim 5.1.0" in md_text,
        },
    }


def load_import_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": _rel(path), "exists": False}
    return {"path": _rel(path), "exists": True, "report": json.loads(path.read_text())}


def _limits_from_import_report(report: dict[str, Any]) -> dict[str, Any]:
    if not report.get("exists"):
        return {"status": "UNKNOWN", "reason": "missing import report"}
    side_reports = report["report"].get("side_reports", {})
    result: dict[str, Any] = {}
    for side, payload in side_reports.items():
        result[side] = {
            "stage_meters_per_unit": payload.get("stage_meters_per_unit"),
            "articulation_roots": payload.get("articulation_roots"),
            "rigid_body_count": payload.get("rigid_body_count"),
            "collision_count": payload.get("collision_count"),
            "mesh_count": payload.get("mesh_count"),
            "joint_count": payload.get("joint_count"),
            "joints": payload.get("joints", []),
            "default_prim": payload.get("default_prim"),
        }
    return result


def make_payload(args: argparse.Namespace) -> dict[str, Any]:
    aloha1_usd = Path(args.aloha1_usd)
    aloha1_import_report = Path(args.aloha1_import_report)
    trossen_usd = Path(args.trossen_usd)
    trossen_pick_place = Path(args.trossen_pick_place)
    trossen_asset_generation = Path(args.trossen_asset_generation)
    menagerie_aloha = Path(args.menagerie_aloha_xml)
    menagerie_scene = Path(args.menagerie_scene_xml)
    confirmed_startup = Path(args.confirmed_startup_usd)
    required_inputs = {
        "aloha1_usd": aloha1_usd,
        "aloha1_import_report": aloha1_import_report,
        "trossen_usd": trossen_usd,
        "trossen_pick_place": trossen_pick_place,
        "trossen_asset_generation": trossen_asset_generation,
        "menagerie_aloha_xml": menagerie_aloha,
        "menagerie_scene_xml": menagerie_scene,
        "confirmed_startup_usd": confirmed_startup,
    }
    missing_inputs = {name: _rel(path) for name, path in required_inputs.items() if not path.exists()}

    aloha1_import = load_import_report(aloha1_import_report)
    payload = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "policy": {
            "read_only_inputs": True,
            "usd_runtime_not_started": True,
            "real_robot_not_touched": True,
            "unknowns_must_not_be_filled_by_visual_guess": True,
        },
        "inputs": {
            "aloha1_usd": _rel(aloha1_usd),
            "aloha1_import_report": _rel(aloha1_import_report),
            "trossen_usd": _rel(trossen_usd),
            "trossen_pick_place": _rel(trossen_pick_place),
            "trossen_asset_generation": _rel(trossen_asset_generation),
            "menagerie_aloha_xml": _rel(menagerie_aloha),
            "menagerie_scene_xml": _rel(menagerie_scene),
            "confirmed_startup_usd_context": _rel(confirmed_startup),
        },
        "tool_capabilities": {
            "pxr_available": _pxr_available(),
            "usd_inspection_level": "static_text_or_strings_only",
            "note": "This script never opens USD stages with pxr or Isaac runtime. pxr availability is recorded only to explain why runtime USD fields are blocked.",
        },
        "input_gate": {
            "status": "FAIL_MISSING_INPUT" if missing_inputs else "PASS",
            "missing_inputs": missing_inputs,
        },
        "assets": {
            "aloha1_original_stationary": {
                "usd_static": inspect_usd_static(aloha1_usd),
                "import_report": aloha1_import,
                "derived_from_import_report": _limits_from_import_report(aloha1_import),
            },
            "trossen_stationary_ai": {
                "usd_static": inspect_usd_static(trossen_usd),
                "demo_constants": inspect_trossen_demo(trossen_pick_place, trossen_asset_generation),
            },
            "menagerie_aloha2_mjcf": inspect_mjcf(menagerie_aloha, menagerie_scene),
            "confirmed_project_startup_context": {
                "usd_static": inspect_usd_static(confirmed_startup),
                "note": "Recorded as context only. This confirmed startup stage is not treated as ALOHA1 truth.",
            },
        },
    }
    payload["comparison"] = build_comparison(payload)
    return _json_safe(payload)


def build_comparison(payload: dict[str, Any]) -> dict[str, Any]:
    aloha1 = payload["assets"]["aloha1_original_stationary"]
    trossen = payload["assets"]["trossen_stationary_ai"]
    menagerie = payload["assets"]["menagerie_aloha2_mjcf"]
    aloha1_sides = aloha1["derived_from_import_report"]
    t_constants = trossen["demo_constants"]["literal_assignments"]

    return {
        "identity": {
            "aloha1": "original_stationary_aloha generated from puppet_left/puppet_right vx300s robot descriptions",
            "trossen": "stationary_ai USD from Trossen AI Isaac / trossen_arm_description",
            "menagerie": f"MJCF model={menagerie.get('model')!r}; documented as ALOHA2 reference",
            "gate": "PASS_AS_DISTINCT_SOURCES",
        },
        "unit_system": {
            "aloha1_stage_meters_per_unit_from_import_report": {
                side: data.get("stage_meters_per_unit") for side, data in aloha1_sides.items()
            } if isinstance(aloha1_sides, dict) else {},
            "trossen_distance_scale_claim_from_doc": trossen["demo_constants"]["asset_generation_claims"].get("mentions_distance_scale_1"),
            "menagerie_compiler": menagerie.get("compiler"),
            "gate": "PARTIAL_PASS_RUNTIME_USD_METADATA_STILL_REQUIRED",
        },
        "joint_names_and_order": {
            "aloha1_joints_from_import_report": {
                side: [joint.get("path", "").split("/")[-1] for joint in data.get("joints", [])]
                for side, data in aloha1_sides.items()
            } if isinstance(aloha1_sides, dict) else {},
            "trossen_demo_indices": {
                key: t_constants.get(key)
                for key in ("LEFT_ARM_DOF_INDICES", "LEFT_GRIPPER_DOF_INDEX", "RIGHT_ARM_DOF_INDICES", "RIGHT_GRIPPER_DOF_INDEX")
            },
            "menagerie_joint_names": [joint.get("name") for joint in menagerie.get("joints", [])],
            "gate": "MAPPING_REQUIRED_DO_NOT_ALIGN_BY_INDEX",
        },
        "gripper": {
            "aloha1_joint_names_from_import_report": {
                side: [joint for joint in data.get("joints", []) if "finger" in joint.get("path", "") or joint.get("path", "").endswith("/gripper")]
                for side, data in aloha1_sides.items()
            } if isinstance(aloha1_sides, dict) else {},
            "trossen_default_positions": t_constants.get("STATIONARY_AI_DEFAULT_DOF_POSITIONS"),
            "menagerie_finger_joints": [joint for joint in menagerie.get("joints", []) if "finger" in str(joint.get("name"))],
            "gate": "BLOCK_GRIPPER_MAPPING_UNTIL_CALIBRATED",
        },
        "colliders_and_meshes": {
            "aloha1_from_import_report": {
                side: {
                    "collision_count": data.get("collision_count"),
                    "mesh_count": data.get("mesh_count"),
                    "rigid_body_count": data.get("rigid_body_count"),
                }
                for side, data in aloha1_sides.items()
            } if isinstance(aloha1_sides, dict) else {},
            "trossen_static_keyword_hits": trossen["usd_static"]["keyword_hits"],
            "menagerie_geom_count": menagerie.get("geom_count"),
            "menagerie_mesh_count": menagerie.get("mesh_count"),
            "gate": "BLOCK_CONTACT_RL_UNTIL_RUNTIME_COLLIDER_REPORT",
        },
        "camera": {
            "aloha1_runtime_camera_expectation": "cam_high/cam_low/cam_left_wrist/cam_right_wrist from project runtime, not validated in this USD comparison",
            "trossen_static_camera_keyword_hits": trossen["usd_static"]["keyword_hits"].get("Camera"),
            "menagerie_cameras": menagerie.get("cameras", []) + menagerie.get("scene_cameras", []),
            "gate": "CAMERA_EXTRINSICS_REQUIRED",
        },
        "phase1_result": {
            "status": "FAIL_MISSING_INPUT" if payload.get("input_gate", {}).get("missing_inputs") else "PASS_WITH_BLOCKED_RUNTIME_FIELDS",
            "blocking_fields_before_implementation": [
                "composed USD prim tree for binary crate USD",
                "runtime articulation DOF order for Trossen stationary_ai",
                "collider shapes and contact material for both USD assets",
                "camera prim world transforms",
            ],
        },
    }


def markdown_report(payload: dict[str, Any]) -> str:
    comp = payload["comparison"]
    lines = [
        "# Phase 1 ALOHA1 Isaac Asset Comparison",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        "- Mode: read-only static comparison plus existing import reports",
        "- Isaac runtime started: `false`",
        "- Real robot touched: `false`",
        f"- PXR available: `{payload['tool_capabilities']['pxr_available']}`",
        f"- USD inspection level: `{payload['tool_capabilities']['usd_inspection_level']}`",
        f"- Input gate: `{payload['input_gate']['status']}`",
        "",
        "## Inputs",
        "",
    ]
    for key, value in payload["inputs"].items():
        lines.append(f"- {key}: `{value}`")
    if payload["input_gate"]["missing_inputs"]:
        lines.extend(["", "## Missing Inputs", ""])
        for key, value in payload["input_gate"]["missing_inputs"].items():
            lines.append(f"- {key}: `{value}`")

    lines.extend(
        [
            "",
            "## Result",
            "",
            f"- Phase 1 status: `{comp['phase1_result']['status']}`",
            "- Main conclusion: sources are distinct; Trossen/Menagerie are references, not ALOHA1 truth.",
            "",
            "## Identity",
            "",
        ]
    )
    for key, value in comp["identity"].items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Joint Names And Order", ""])
    lines.append(f"- Gate: `{comp['joint_names_and_order']['gate']}`")
    lines.append("- ALOHA1 side joint names from import report:")
    for side, names in comp["joint_names_and_order"]["aloha1_joints_from_import_report"].items():
        lines.append(f"  - {side}: `{names}`")
    lines.append(f"- Trossen demo indices: `{comp['joint_names_and_order']['trossen_demo_indices']}`")
    lines.append(f"- Menagerie joint count: `{len(comp['joint_names_and_order']['menagerie_joint_names'])}`")

    lines.extend(["", "## Gripper", ""])
    lines.append(f"- Gate: `{comp['gripper']['gate']}`")
    lines.append(f"- Trossen default DOF positions: `{comp['gripper']['trossen_default_positions']}`")
    lines.append(f"- Menagerie finger joints: `{comp['gripper']['menagerie_finger_joints']}`")

    lines.extend(["", "## Colliders And Meshes", ""])
    lines.append(f"- Gate: `{comp['colliders_and_meshes']['gate']}`")
    lines.append(f"- ALOHA1 import report counts: `{comp['colliders_and_meshes']['aloha1_from_import_report']}`")
    lines.append(f"- Menagerie geom count: `{comp['colliders_and_meshes']['menagerie_geom_count']}`")
    lines.append(f"- Menagerie mesh count: `{comp['colliders_and_meshes']['menagerie_mesh_count']}`")

    lines.extend(["", "## Camera", ""])
    lines.append(f"- Gate: `{comp['camera']['gate']}`")
    lines.append(f"- Menagerie cameras: `{comp['camera']['menagerie_cameras']}`")

    lines.extend(["", "## Blocked Runtime Fields", ""])
    for item in comp["phase1_result"]["blocking_fields_before_implementation"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Quality Gate",
            "",
            "Do not implement controller, grasp, contact, or RL changes from this report alone.",
            "The next safe step is a runtime USD/articulation inspection using Isaac tooling, still read-only.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 read-only comparison of ALOHA1, Trossen AI Isaac, and Menagerie ALOHA assets.")
    parser.add_argument("--aloha1-usd", default=str(DEFAULT_ALOHA1_USD))
    parser.add_argument("--aloha1-import-report", default=str(DEFAULT_ALOHA1_IMPORT_REPORT))
    parser.add_argument("--trossen-usd", default=str(DEFAULT_TROSSEN_USD))
    parser.add_argument("--trossen-pick-place", default=str(DEFAULT_TROSSEN_PICK_PLACE))
    parser.add_argument("--trossen-asset-generation", default=str(DEFAULT_TROSSEN_ASSET_GENERATION))
    parser.add_argument("--menagerie-aloha-xml", default=str(DEFAULT_MENAGERIE_ALOHA))
    parser.add_argument("--menagerie-scene-xml", default=str(DEFAULT_MENAGERIE_SCENE))
    parser.add_argument("--confirmed-startup-usd", default=str(REPO / "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/aloha2_menagerie_scene_deep_black_real_start_pose.usd"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = make_payload(args)

    json_path = output_dir / "phase1_asset_comparison.json"
    md_path = output_dir / "phase1_asset_comparison.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    md_path.write_text(markdown_report(payload))
    print(json.dumps({"json": _rel(json_path), "markdown": _rel(md_path), "status": payload["comparison"]["phase1_result"]["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
