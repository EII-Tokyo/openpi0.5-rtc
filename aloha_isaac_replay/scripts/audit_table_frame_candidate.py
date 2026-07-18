from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase64_table_frame_static_audit_20260718"


def _rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    frame_status = payload["frame_status"]
    corner_lines = [
        f"- `{name}`: `{coords}`" for name, coords in payload["table_geometry"]["top_corners_world"].items()
    ]
    lines = [
        "# Table Frame Candidate Static Audit",
        "",
        f"- status: `{payload['status']}`",
        f"- config: `{payload['config']}`",
        f"- support center: `{payload['support_plane']['center']}`",
        f"- support size: `{payload['support_plane']['size']}`",
        f"- table top center: `{payload['table_geometry']['top_center_world']}`",
        "",
        "## Frame Status",
        "",
        f"- T_world_table: `{frame_status['T_world_table']}`",
        f"- T_table_left_base: `{frame_status['T_table_left_base']}`",
        f"- T_table_right_base: `{frame_status['T_table_right_base']}`",
        "",
        "## Table Top Corners",
        "",
        *corner_lines,
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _as_float3(value: Any, *, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-value list")
    return [float(v) for v in value]


def _table_top_corners(center: list[float], size: list[float]) -> dict[str, list[float]]:
    cx, cy, cz = center
    sx, sy, sz = size
    top_z = cz + sz * 0.5
    return {
        "xmin_ymin": [cx - sx * 0.5, cy - sy * 0.5, top_z],
        "xmax_ymin": [cx + sx * 0.5, cy - sy * 0.5, top_z],
        "xmax_ymax": [cx + sx * 0.5, cy + sy * 0.5, top_z],
        "xmin_ymax": [cx - sx * 0.5, cy + sy * 0.5, top_z],
    }


def audit_table_frame(config_path: Path) -> dict[str, Any]:
    cfg = _load_yaml(config_path)
    support = cfg.get("support_plane")
    table_frame = cfg.get("table_frame")
    if not isinstance(support, dict):
        raise ValueError("config must contain support_plane")
    if not isinstance(table_frame, dict):
        raise ValueError("config must contain table_frame")

    center = _as_float3(support.get("center"), name="support_plane.center")
    size = _as_float3(support.get("size"), name="support_plane.size")
    t_world_table = table_frame.get("T_world_table") or {}
    t_left = table_frame.get("T_table_left_base") or {}
    t_right = table_frame.get("T_table_right_base") or {}
    frame_status = {
        "T_world_table": str(t_world_table.get("status", "unknown")),
        "T_table_left_base": str(t_left.get("status", "unknown")),
        "T_table_right_base": str(t_right.get("status", "unknown")),
    }
    missing_base_transform = any(
        frame_status[key] in {"unknown", "not_calibrated"}
        for key in ("T_table_left_base", "T_table_right_base")
    )
    status = (
        "BLOCKED_REQUIRES_MEASURED_TABLE_TO_BASE_TRANSFORM"
        if missing_base_transform
        else "PASS_TABLE_FRAME_STATIC_AUDIT"
    )
    top_center = [center[0], center[1], center[2] + size[2] * 0.5]
    interpretation = (
        "The table candidate is explicit and auditable, but the robot base transforms are not calibrated. "
        "Do not use this candidate as a real workcell pose until T_table_left_base and T_table_right_base are measured."
        if missing_base_transform
        else "The table candidate includes calibrated table-to-base transforms and is ready for replay contact validation."
    )
    return {
        "status": status,
        "config": _rel(config_path),
        "support_plane": {
            "mode": support.get("mode"),
            "center": center,
            "size": size,
            "provenance": support.get("provenance"),
        },
        "frame_status": frame_status,
        "table_frame": table_frame,
        "table_geometry": {
            "top_center_world": top_center,
            "top_corners_world": _table_top_corners(center, size),
        },
        "interpretation": interpretation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a diagnostic table-frame candidate without running Isaac.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    json_path = output_dir / "table_frame_static_audit.json"
    md_path = output_dir / "table_frame_static_audit.md"
    payload = audit_table_frame(Path(args.config))
    payload["outputs"] = {"json": _rel(json_path), "markdown": _rel(md_path)}
    _write_json(json_path, payload)
    _write_markdown(md_path, payload)
    print(json.dumps({"status": payload["status"], "json": _rel(json_path), "markdown": _rel(md_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
