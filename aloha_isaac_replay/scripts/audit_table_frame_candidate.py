from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase64_table_frame_static_audit_20260718"
CALIBRATED_STATUSES = {"measured", "calibrated", "read_from_103", "read_from_usd"}
BLOCKING_STATUSES = {"unknown", "not_calibrated", "diagnostic_candidate"}


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
    blocking_lines = [f"- {reason}" for reason in payload.get("blocking_reasons", [])] or ["- none"]
    world_base_lines = [
        f"- `{name}`: translation `{value['translation']}`, rotation_quat_wxyz `{value['rotation_quat_wxyz']}`"
        for name, value in payload.get("world_base_transforms", {}).items()
    ] or ["- unavailable"]
    lines = [
        "# Table Frame Candidate Static Audit",
        "",
        f"- status: `{payload['status']}`",
        f"- calibration ready: `{payload['calibration_ready']}`",
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
        "## Blocking Reasons",
        "",
        *blocking_lines,
        "",
        "## Derived World Base Transforms",
        "",
        *world_base_lines,
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


def _as_float4(value: Any, *, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name} must be a 4-value list")
    return [float(v) for v in value]


def _quat_norm(quat_wxyz: list[float]) -> float:
    return math.sqrt(sum(v * v for v in quat_wxyz))


def _quat_conjugate(quat_wxyz: list[float]) -> list[float]:
    w, x, y, z = quat_wxyz
    return [w, -x, -y, -z]


def _quat_multiply(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _quat_rotate_vector(quat_wxyz: list[float], vector: list[float]) -> list[float]:
    rotated = _quat_multiply(_quat_multiply(quat_wxyz, [0.0, *vector]), _quat_conjugate(quat_wxyz))
    return rotated[1:]


def _compose_transform(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, list[float]]:
    parent_t = _as_float3(parent.get("translation"), name="parent.translation")
    parent_q = _as_float4(parent.get("rotation_quat_wxyz"), name="parent.rotation_quat_wxyz")
    child_t = _as_float3(child.get("translation"), name="child.translation")
    child_q = _as_float4(child.get("rotation_quat_wxyz"), name="child.rotation_quat_wxyz")
    rotated_child_t = _quat_rotate_vector(parent_q, child_t)
    return {
        "translation": [parent_t[i] + rotated_child_t[i] for i in range(3)],
        "rotation_quat_wxyz": _quat_multiply(parent_q, child_q),
    }


def _transform_status(transform: Any) -> str:
    if not isinstance(transform, dict):
        return "unknown"
    return str(transform.get("status", "unknown"))


def _validate_transform(
    name: str,
    transform: Any,
    *,
    require_calibrated: bool,
    issues: list[str],
) -> dict[str, Any] | None:
    if not isinstance(transform, dict):
        issues.append(f"{name} missing")
        return None
    status = _transform_status(transform)
    if require_calibrated and status not in CALIBRATED_STATUSES:
        issues.append(f"{name} status is {status}, expected one of {sorted(CALIBRATED_STATUSES)}")
    source = str(transform.get("source", "unknown"))
    if require_calibrated and source == "unknown":
        issues.append(f"{name} source is unknown")
    try:
        translation = _as_float3(transform.get("translation"), name=f"{name}.translation")
        quat = _as_float4(transform.get("rotation_quat_wxyz"), name=f"{name}.rotation_quat_wxyz")
    except ValueError as exc:
        if require_calibrated:
            issues.append(str(exc))
        return None
    norm = _quat_norm(quat)
    if abs(norm - 1.0) > 1e-3:
        issues.append(f"{name} rotation_quat_wxyz norm is {norm:.6f}, expected 1.0")
    return {
        "status": status,
        "source": source,
        "translation": translation,
        "rotation_quat_wxyz": quat,
        "quat_norm": norm,
    }


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
        "T_world_table": _transform_status(t_world_table),
        "T_table_left_base": _transform_status(t_left),
        "T_table_right_base": _transform_status(t_right),
    }
    blocking_reasons: list[str] = []
    missing_base_transform = any(frame_status[key] in BLOCKING_STATUSES for key in ("T_table_left_base", "T_table_right_base"))
    diagnostic_world_table = frame_status["T_world_table"] in BLOCKING_STATUSES

    world_table = _validate_transform(
        "T_world_table",
        t_world_table,
        require_calibrated=not diagnostic_world_table,
        issues=blocking_reasons,
    )
    left_base = _validate_transform(
        "T_table_left_base",
        t_left,
        require_calibrated=not missing_base_transform,
        issues=blocking_reasons,
    )
    right_base = _validate_transform(
        "T_table_right_base",
        t_right,
        require_calibrated=not missing_base_transform,
        issues=blocking_reasons,
    )
    top_center = [center[0], center[1], center[2] + size[2] * 0.5]
    if diagnostic_world_table:
        blocking_reasons.append("T_world_table is diagnostic, not calibrated")
    if missing_base_transform:
        blocking_reasons.append("T_table_left_base or T_table_right_base is missing/not calibrated")
    if world_table is not None:
        world_table_translation = world_table["translation"]
        mismatch = max(abs(world_table_translation[i] - top_center[i]) for i in range(3))
        if mismatch > 1e-6:
            blocking_reasons.append(
                f"T_world_table.translation does not match support_plane top center; max mismatch {mismatch:.6g} m"
            )

    calibration_ready = not blocking_reasons
    status = (
        "PASS_TABLE_TO_BASE_CALIBRATION_READY"
        if calibration_ready
        else "BLOCKED_REQUIRES_MEASURED_TABLE_TO_BASE_TRANSFORM"
    )
    world_base_transforms: dict[str, dict[str, list[float]]] = {}
    if world_table is not None and left_base is not None:
        world_base_transforms["T_world_left_base"] = _compose_transform(t_world_table, t_left)
    if world_table is not None and right_base is not None:
        world_base_transforms["T_world_right_base"] = _compose_transform(t_world_table, t_right)

    interpretation = (
        "The table candidate includes calibrated table-to-base transforms and is ready for replay contact validation."
        if calibration_ready
        else "The table candidate is explicit and auditable, but the real table-to-robot base transforms are not calibrated. "
        "Do not use this candidate as a real workcell pose until T_world_table, T_table_left_base, and "
        "T_table_right_base are measured or read from a trusted source."
    )
    return {
        "status": status,
        "calibration_ready": calibration_ready,
        "blocking_reasons": blocking_reasons,
        "config": _rel(config_path),
        "support_plane": {
            "mode": support.get("mode"),
            "center": center,
            "size": size,
            "provenance": support.get("provenance"),
        },
        "frame_status": frame_status,
        "frame_validation": {
            "T_world_table": world_table,
            "T_table_left_base": left_base,
            "T_table_right_base": right_base,
        },
        "table_frame": table_frame,
        "world_base_transforms": world_base_transforms,
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
