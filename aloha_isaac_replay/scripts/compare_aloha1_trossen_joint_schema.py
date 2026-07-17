from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import DEFAULT_LEFT_URDF
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import DEFAULT_SCAFFOLD_USD
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import _rel


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TROSSEN_USD = REPO_ROOT / "external/trossen_ai_isaac/assets/robots/stationary_ai/stationary_ai.usd"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase13_joint_schema_20260718"

ALOHA_LEFT_JOINTS = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
    "ee_gripper",
)

TROSSEN_LEFT_JOINT_HINTS = (
    "follower_left_joint_0",
    "follower_left_joint_1",
    "follower_left_joint_2",
    "follower_left_joint_3",
    "follower_left_joint_4",
    "follower_left_joint_5",
)


def _float_list(text: str | None, default: tuple[float, ...]) -> list[float]:
    if text is None or not text.strip():
        return list(default)
    return [float(part) for part in text.split()]


def _rpy_to_matrix(rpy: list[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _normed(vec: list[float] | np.ndarray) -> list[float]:
    array = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return array.tolist()
    return (array / norm).tolist()


def _parse_aloha_urdf_joints(path: Path) -> list[dict[str, Any]]:
    root = ET.parse(path).getroot()
    rows = []
    for semantic in ALOHA_LEFT_JOINTS:
        joint = root.find(f"./joint[@name='{semantic}']")
        if joint is None:
            joint = root.find(f"./joint[@name='puppet_left/{semantic}']")
        if joint is None:
            rows.append({"semantic": semantic, "status": "MISSING"})
            continue
        origin = joint.find("origin")
        axis_el = joint.find("axis")
        limit = joint.find("limit")
        xyz = _float_list(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
        rpy = _float_list(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
        axis = _float_list(axis_el.get("xyz") if axis_el is not None else None, (0.0, 0.0, 0.0))
        parent_axis = (_rpy_to_matrix(rpy) @ np.asarray(axis, dtype=np.float64)).tolist()
        rows.append(
            {
                "semantic": semantic,
                "status": "OK",
                "name": joint.get("name"),
                "type": joint.get("type"),
                "parent": joint.find("parent").get("link") if joint.find("parent") is not None else None,
                "child": joint.find("child").get("link") if joint.find("child") is not None else None,
                "origin_xyz_m": xyz,
                "origin_rpy_rad": rpy,
                "axis_joint_frame": _normed(axis),
                "axis_parent_frame": _normed(parent_axis),
                "lower_rad": float(limit.get("lower")) if limit is not None and limit.get("lower") else None,
                "upper_rad": float(limit.get("upper")) if limit is not None and limit.get("upper") else None,
            }
        )
    return rows


def _vec3(value: Any) -> list[float] | None:
    if value is None:
        return None
    return [float(value[0]), float(value[1]), float(value[2])]


def _quat_wxyz(value: Any) -> list[float] | None:
    if value is None:
        return None
    return [float(value.GetReal()), float(value.GetImaginary()[0]), float(value.GetImaginary()[1]), float(value.GetImaginary()[2])]


def _axis_token_to_vec(token: str | None) -> list[float] | None:
    if token == "X":
        return [1.0, 0.0, 0.0]
    if token == "Y":
        return [0.0, 1.0, 0.0]
    if token == "Z":
        return [0.0, 0.0, 1.0]
    return None


def _rotate_axis_with_quat(axis: list[float] | None, quat: Any) -> list[float] | None:
    if axis is None or quat is None:
        return None
    from pxr import Gf

    rotation = Gf.Rotation(quat)
    transformed = rotation.TransformDir(Gf.Vec3d(*axis))
    return _normed([float(transformed[0]), float(transformed[1]), float(transformed[2])])


def _targets(rel: Any) -> list[str]:
    return [str(target) for target in rel.GetTargets()]


def _joint_row(prim: Any) -> dict[str, Any] | None:
    from pxr import UsdPhysics

    if prim.IsA(UsdPhysics.RevoluteJoint):
        joint = UsdPhysics.RevoluteJoint(prim)
        axis_token = joint.GetAxisAttr().Get()
        lower = joint.GetLowerLimitAttr().Get()
        upper = joint.GetUpperLimitAttr().Get()
    elif prim.IsA(UsdPhysics.PrismaticJoint):
        joint = UsdPhysics.PrismaticJoint(prim)
        axis_token = joint.GetAxisAttr().Get()
        lower = joint.GetLowerLimitAttr().Get()
        upper = joint.GetUpperLimitAttr().Get()
    elif prim.IsA(UsdPhysics.FixedJoint):
        joint = UsdPhysics.FixedJoint(prim)
        axis_token = None
        lower = None
        upper = None
    else:
        return None

    base_joint = UsdPhysics.Joint(prim)
    local_rot0 = base_joint.GetLocalRot0Attr().Get()
    local_rot1 = base_joint.GetLocalRot1Attr().Get()
    axis = _axis_token_to_vec(axis_token)
    return {
        "path": str(prim.GetPath()),
        "name": prim.GetName(),
        "type": prim.GetTypeName(),
        "body0": _targets(base_joint.GetBody0Rel()),
        "body1": _targets(base_joint.GetBody1Rel()),
        "axis_token": axis_token,
        "axis_joint_frame": axis,
        "axis_body0_frame": _rotate_axis_with_quat(axis, local_rot0),
        "axis_body1_frame": _rotate_axis_with_quat(axis, local_rot1),
        "local_pos0": _vec3(base_joint.GetLocalPos0Attr().Get()),
        "local_rot0_wxyz": _quat_wxyz(local_rot0),
        "local_pos1": _vec3(base_joint.GetLocalPos1Attr().Get()),
        "local_rot1_wxyz": _quat_wxyz(local_rot1),
        "lower": float(lower) if lower is not None else None,
        "upper": float(upper) if upper is not None else None,
    }


def _extract_usd_joints(path: Path, focus_terms: tuple[str, ...]) -> dict[str, Any]:
    from pxr import Usd

    stage = Usd.Stage.Open(str(path.resolve()))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {path}")
    rows = []
    focus_rows = []
    for prim in stage.Traverse():
        row = _joint_row(prim)
        if row is None:
            continue
        rows.append(row)
        lowered = row["path"].lower()
        if any(term.lower() in lowered for term in focus_terms):
            focus_rows.append(row)
    return {
        "path": _rel(path),
        "joint_count": len(rows),
        "focus_joint_count": len(focus_rows),
        "focus_terms": list(focus_terms),
        "focus_joints": focus_rows,
        "all_joint_names": [row["name"] for row in rows],
    }


def _semantic_comparison(aloha_rows: list[dict[str, Any]], trossen_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {row["name"]: row for row in trossen_rows}
    output = []
    for idx, aloha in enumerate(aloha_rows[:6]):
        trossen_name = TROSSEN_LEFT_JOINT_HINTS[idx]
        trossen = by_name.get(trossen_name)
        output.append(
            {
                "semantic": aloha["semantic"],
                "aloha_name": aloha.get("name"),
                "aloha_axis_parent_frame": aloha.get("axis_parent_frame"),
                "trossen_name": trossen_name,
                "trossen_present": trossen is not None,
                "trossen_axis_token": trossen.get("axis_token") if trossen else None,
                "trossen_axis_body0_frame": trossen.get("axis_body0_frame") if trossen else None,
                "trossen_axis_body1_frame": trossen.get("axis_body1_frame") if trossen else None,
                "note": "Axis rows are not directly equivalent unless the parent/body frames have already been aligned.",
            }
        )
    return output


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 13 - Joint Axis Schema Comparison - 2026-07-18",
        "",
        "## Scope",
        "",
        "This diagnostic compares static joint schema facts from the trusted ALOHA1 URDF and the Trossen-backed Isaac USD assets.",
        "",
        "It is read-only: no real robot command, no stage save, no controller execution.",
        "",
        "## Important Limitation",
        "",
        "URDF and USD joint axes are local-frame quantities. A raw `X/Y/Z` axis token is not a world-frame direction. This report records transformed body-frame axes where available, but it does not by itself prove the correct runtime joint mapping.",
        "",
        "## Inputs",
        "",
        f"- ALOHA1 left URDF: `{payload['inputs']['aloha_left_urdf']}`",
        f"- Trossen USD: `{payload['inputs']['trossen_usd']}`",
        f"- Scaffold USD: `{payload['inputs']['scaffold_usd']}`",
        "",
        "## Gates",
        "",
    ]
    for key, value in payload["gates"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- ALOHA1 parsed joints: `{payload['summary']['aloha_joint_count']}`",
            f"- Trossen raw focus joints: `{payload['summary']['trossen_focus_joint_count']}`",
            f"- Scaffold focus joints: `{payload['summary']['scaffold_focus_joint_count']}`",
            "",
            "## ALOHA1 Left Arm URDF Axes",
            "",
            "| semantic | type | parent | child | axis in parent frame | limit rad |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in payload["aloha_left_joints"]:
        limit = f"{row.get('lower_rad')} .. {row.get('upper_rad')}"
        lines.append(
            "| "
            f"{row['semantic']} | {row.get('type')} | `{row.get('parent')}` | `{row.get('child')}` | "
            f"`{row.get('axis_parent_frame')}` | `{limit}` |"
        )
    lines.extend(
        [
            "",
            "## Expected Trossen Semantic Joint Rows",
            "",
            "| semantic | Trossen joint | present | token | body0 axis | body1 axis |",
            "|---|---|---:|---|---|---|",
        ]
    )
    for row in payload["semantic_comparison"]:
        lines.append(
            "| "
            f"{row['semantic']} | `{row['trossen_name']}` | {row['trossen_present']} | "
            f"`{row['trossen_axis_token']}` | `{row['trossen_axis_body0_frame']}` | `{row['trossen_axis_body1_frame']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This phase is an evidence-gathering gate. It explains why Phase 11/12 cannot be fixed by only picking another terminal body.",
            "",
            "If the Trossen semantic joint rows are missing or the axis/body-frame facts do not match the assumed ALOHA1 semantics after frame alignment, the next phase must search wrist/forearm sign and offset with orientation included in the objective.",
            "",
            "## Status",
            "",
            f"`{payload['status']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ALOHA1 URDF joint axes against Trossen-backed Isaac USD joint schemas.")
    parser.add_argument("--aloha-left-urdf", type=Path, default=DEFAULT_LEFT_URDF)
    parser.add_argument("--trossen-usd", type=Path, default=DEFAULT_TROSSEN_USD)
    parser.add_argument("--scaffold-usd", type=Path, default=DEFAULT_SCAFFOLD_USD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--normal-close", action="store_true")
    args = parser.parse_args()

    aloha_rows = _parse_aloha_urdf_joints(args.aloha_left_urdf)

    from isaacsim import SimulationApp

    app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
    if args.normal_close:
        app_config["fast_shutdown"] = False
    app = SimulationApp(app_config)
    try:
        trossen = _extract_usd_joints(args.trossen_usd, ("follower_left", "left_follower"))
        scaffold = _extract_usd_joints(args.scaffold_usd, ("follower_left", "left_follower"))
        focus_rows = scaffold["focus_joints"] if scaffold["focus_joint_count"] else trossen["focus_joints"]
        semantic = _semantic_comparison(aloha_rows, focus_rows)
        all_present = all(row["trossen_present"] for row in semantic)
        payload = {
            "inputs": {
                "aloha_left_urdf": _rel(args.aloha_left_urdf),
                "trossen_usd": _rel(args.trossen_usd),
                "scaffold_usd": _rel(args.scaffold_usd),
            },
            "aloha_left_joints": aloha_rows,
            "trossen_raw_usd": trossen,
            "scaffold_usd": scaffold,
            "semantic_comparison": semantic,
            "summary": {
                "aloha_joint_count": sum(1 for row in aloha_rows if row["status"] == "OK"),
                "trossen_focus_joint_count": trossen["focus_joint_count"],
                "scaffold_focus_joint_count": scaffold["focus_joint_count"],
                "semantic_rows_present": all_present,
            },
            "gates": {
                "real_robot_touched": "PASS_FALSE",
                "stage_saved": "PASS_FALSE",
                "isaac_runtime_started": "PASS",
                "aloha_urdf_loaded": "PASS",
                "trossen_usd_loaded": "PASS" if trossen["joint_count"] else "FAIL_NO_JOINTS_FOUND",
                "scaffold_usd_loaded": "PASS" if scaffold["joint_count"] else "FAIL_NO_JOINTS_FOUND",
                "semantic_trossen_rows_present": "PASS" if all_present else "FAIL_MISSING_EXPECTED_ROWS",
                "controller": "BLOCKED_NOT_ATTEMPTED",
            },
            "status": "BLOCKED_REQUIRES_ORIENTATION_AWARE_MAPPING_SEARCH",
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "joint_schema_comparison.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "joint_schema_comparison.md").write_text(_render_markdown(payload), encoding="utf-8")
        print(
            json.dumps(
                {
                    "output_dir": _rel(args.output_dir),
                    "aloha_joint_count": payload["summary"]["aloha_joint_count"],
                    "trossen_focus_joint_count": payload["summary"]["trossen_focus_joint_count"],
                    "scaffold_focus_joint_count": payload["summary"]["scaffold_focus_joint_count"],
                    "semantic_rows_present": all_present,
                    "status": payload["status"],
                },
                ensure_ascii=False,
            )
        )
    finally:
        app.close(skip_cleanup=not args.normal_close)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
