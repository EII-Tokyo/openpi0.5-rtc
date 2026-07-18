from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE_USD = (
    REPO_ROOT
    / "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose"
    / "aloha2_menagerie_scene_deep_black_real_start_pose_proxy_runtime.usda"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase109_workcell_semantic_audit_20260719"
DEFAULT_CONTACT_REPORT = (
    REPO_ROOT
    / "reports/aloha1_isaac_adaptation/phase108_bottleusd_hdf5_diagnostic_table_gate_20260719"
    / "gripper_passive_contact_metrics.json"
)


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
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        return list(value)
    except Exception:
        return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _applied(prim: Any) -> list[str]:
    return [str(item) for item in prim.GetAppliedSchemas()]


def _bbox_row(cache: Any, prim: Any) -> dict[str, Any]:
    # ComputeWorldBound returns an oriented GfBBox3d. GetBox() alone can expose
    # the unaligned local range for scaled/rotated prims, so use the aligned
    # world range for semantic geometry classification.
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if box.IsEmpty():
        return {
            "bbox_valid": False,
            "bbox_min": None,
            "bbox_max": None,
            "bbox_center": None,
            "bbox_size": None,
        }
    min_pt = box.GetMin()
    max_pt = box.GetMax()
    return {
        "bbox_valid": True,
        "bbox_min": [float(min_pt[i]) for i in range(3)],
        "bbox_max": [float(max_pt[i]) for i in range(3)],
        "bbox_center": [float((min_pt[i] + max_pt[i]) * 0.5) for i in range(3)],
        "bbox_size": [float(max_pt[i] - min_pt[i]) for i in range(3)],
    }


def _semantic_guess(path: str, type_name: str, bbox_size: list[float] | None) -> str:
    leaf = path.rsplit("/", 1)[-1].lower()
    if "floor" in path.lower() or type_name == "Plane":
        return "floor"
    if "table" in path.lower():
        return "table_named_prim"
    if not bbox_size:
        return "unknown_no_bbox"
    sorted_dims = sorted(float(item) for item in bbox_size)
    max_dim = sorted_dims[-1]
    mid_dim = sorted_dims[1]
    min_dim = sorted_dims[0]
    if ("extrusion" in leaf or "angled_extrusion" in leaf) and max_dim >= 0.5 and mid_dim <= 0.06:
        return "frame_or_rail_extrusion"
    if max_dim >= 0.5 and mid_dim <= 0.06 and min_dim <= 0.06:
        return "long_thin_frame_member"
    if bbox_size[0] >= 0.2 and bbox_size[1] >= 0.2 and bbox_size[2] <= 0.08:
        return "tabletop_or_plate_candidate"
    if max_dim <= 0.12:
        return "small_fixture_or_bracket"
    return "unknown_workcell_geometry"


def _load_contact_paths(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "paths": {}, "object_contact_paths": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    counters: Counter[str] = Counter()
    object_counters: Counter[str] = Counter()
    for row in data.get("contact_pairs_sample", []) + data.get("contact_pair_rows", []):
        for key in ("collider0", "collider1"):
            value = row.get(key)
            if isinstance(value, str):
                counters[value] += 1
    for category in (data.get("object_contact_categories") or {}).values():
        for pair in category.get("unique_contact_pairs", []):
            for value in pair:
                if isinstance(value, str):
                    object_counters[value] += 1
    return {
        "exists": True,
        "report": _rel(path),
        "paths": dict(counters),
        "object_contact_paths": dict(object_counters),
    }


def _is_path_or_ancestor(candidate: str, observed: str) -> bool:
    return observed == candidate or observed.startswith(candidate + "/") or candidate.startswith(observed + "/")


def _contact_evidence_for(path: str, contact_data: dict[str, Any]) -> dict[str, Any]:
    all_matches = {observed: count for observed, count in contact_data.get("paths", {}).items() if _is_path_or_ancestor(path, observed)}
    object_matches = {
        observed: count
        for observed, count in contact_data.get("object_contact_paths", {}).items()
        if _is_path_or_ancestor(path, observed)
    }
    return {
        "appears_in_contact_sample": bool(all_matches),
        "appears_in_object_contact_pairs": bool(object_matches),
        "contact_sample_matches": all_matches,
        "object_contact_matches": object_matches,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 109 Workcell Semantic Audit",
        "",
        f"- status: `{payload['status']}`",
        f"- stage: `{payload['inputs']['stage_usd']}`",
        f"- contact report: `{payload['inputs'].get('contact_report')}`",
        f"- row count: `{len(payload.get('rows', []))}`",
        "",
        "## Semantic Counts",
        "",
        "| semantic | count |",
        "| --- | ---: |",
    ]
    for name, count in sorted(payload.get("semantic_counts", {}).items()):
        lines.append(f"| `{name}` | {count} |")
    lines.extend(
        [
            "",
            "## Contact-Relevant Rows",
            "",
            "| path | semantic guess | bbox center | bbox size | object contact |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload.get("rows", []):
        evidence = row.get("contact_evidence") or {}
        if not evidence.get("appears_in_object_contact_pairs"):
            continue
        lines.append(
            "| "
            f"`{row['path']}` | "
            f"`{row['semantic_guess']}` | "
            f"`{row.get('bbox_center')}` | "
            f"`{row.get('bbox_size')}` | "
            f"{evidence.get('appears_in_object_contact_pairs')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a semantic audit, not a calibration claim. A prim guessed as a frame or rail must not be treated as a tabletop support until the real workcell frame is measured and mapped.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit /scene/worldBody composed prims and classify likely workcell semantics.")
    parser.add_argument("--stage-usd", default=str(DEFAULT_STAGE_USD))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--contact-report", default=str(DEFAULT_CONTACT_REPORT))
    parser.add_argument("--root", default="/scene/worldBody")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    json_path = output_dir / "workcell_semantic_audit.json"
    md_path = output_dir / "workcell_semantic_audit.md"
    payload: dict[str, Any] = {
        "status": "STARTED",
        "real_robot_touched": False,
        "stage_saved": False,
        "inputs": {
            "stage_usd": _rel(args.stage_usd),
            "root": args.root,
            "contact_report": _rel(args.contact_report),
        },
    }
    _write_json(json_path, payload)

    try:
        from isaacsim import SimulationApp

        app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
        app_config["fast_shutdown"] = False
        _app = SimulationApp(app_config)

        import isaacsim.core.utils.stage as stage_utils
        from pxr import Usd
        from pxr import UsdGeom

        stage_utils.open_stage(str(Path(args.stage_usd).resolve()))
        stage = stage_utils.get_current_stage()
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=False,
        )
        contact_data = _load_contact_paths(Path(args.contact_report))
        rows: list[dict[str, Any]] = []
        root = args.root.rstrip("/")
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path != root and not path.startswith(root + "/"):
                continue
            bbox = _bbox_row(cache, prim)
            row = {
                "path": path,
                "type_name": prim.GetTypeName(),
                "is_instance": bool(prim.IsInstance()),
                "is_instanceable": bool(prim.IsInstanceable()),
                "applied_schemas": _applied(prim),
            }
            row.update(bbox)
            row["semantic_guess"] = _semantic_guess(path, str(row["type_name"]), row.get("bbox_size"))
            row["contact_evidence"] = _contact_evidence_for(path, contact_data)
            rows.append(row)
        semantic_counts = Counter(row["semantic_guess"] for row in rows)
        payload.update(
            {
                "status": "PASS",
                "contact_data": contact_data,
                "semantic_counts": dict(semantic_counts),
                "rows": rows,
                "outputs": {"json": _rel(json_path), "markdown": _rel(md_path)},
            }
        )
        _write_json(json_path, payload)
        md_path.write_text(_render_markdown(_json_safe(payload)), encoding="utf-8")
        print(json.dumps({"status": "PASS", "json": _rel(json_path), "markdown": _rel(md_path)}, ensure_ascii=False), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except BaseException as exc:
        payload.update({"status": "EXCEPTION", "exception": repr(exc), "traceback": traceback.format_exc(limit=20)})
        _write_json(json_path, payload)
        md_path.write_text(_render_markdown(_json_safe(payload)), encoding="utf-8")
        print(json.dumps({"status": "EXCEPTION", "json": _rel(json_path), "error": repr(exc)}, ensure_ascii=False), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
