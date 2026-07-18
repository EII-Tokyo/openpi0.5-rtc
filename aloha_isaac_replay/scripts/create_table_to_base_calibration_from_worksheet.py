from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from aloha_isaac_replay.calibration.table_measurement_guidance import forbidden_table_base_source_reason
from aloha_isaac_replay.calibration.table_measurement_guidance import missing_field_details
from aloha_isaac_replay.scripts.audit_table_frame_candidate import audit_table_frame
from aloha_isaac_replay.scripts.create_table_to_base_calibration import build_calibration_config
from aloha_isaac_replay.scripts.create_table_to_base_calibration import build_evidence_record

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSHEET = REPO_ROOT / "examples/aloha_isaac/config/phase68_table_to_base_measurement_worksheet.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase68_table_to_base_measurement_worksheet_20260718"


def _rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _get_nested(data: dict[str, Any], path: str) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _require_float3(data: dict[str, Any], path: str, missing: list[str]) -> list[float] | None:
    value = _get_nested(data, path)
    if value is None:
        missing.append(path)
        return None
    if not isinstance(value, list) or len(value) != 3:
        missing.append(f"{path} must be a 3-value list")
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        missing.append(f"{path} must contain numeric values")
        return None


def _require_float(data: dict[str, Any], path: str, missing: list[str]) -> float | None:
    value = _get_nested(data, path)
    if value is None:
        missing.append(path)
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        missing.append(f"{path} must be numeric")
        return None


def _require_string(data: dict[str, Any], path: str, missing: list[str]) -> str | None:
    value = _get_nested(data, path)
    if value is None or str(value).strip() == "":
        missing.append(path)
        return None
    return str(value)


def _require_bool_or_readonly(data: dict[str, Any], path: str, missing: list[str]) -> bool | str | None:
    value = _get_nested(data, path)
    if value is None:
        missing.append(path)
        return None
    if value is False:
        return False
    if value in {"readonly", "read_only"}:
        return str(value)
    missing.append(f"{path} must be false or readonly")
    return None


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    missing_lines = [f"- `{item}`" for item in payload["missing_fields"]] or ["- none"]
    detail_lines: list[str] = []
    for item in payload["missing_fields"]:
        detail = payload.get("missing_field_details", {}).get(item)
        if not detail:
            continue
        detail_lines.extend(
            [
                f"### `{item}`",
                "",
                f"- description: {detail.get('description', 'n/a')}",
                f"- unit: `{detail.get('unit', 'n/a')}`",
                f"- shape: `{detail.get('shape', 'n/a')}`",
                f"- example: `{detail.get('example', 'n/a')}`",
                f"- how to measure: {detail.get('how_to_measure', 'n/a')}",
                "",
            ]
        )
    if not detail_lines:
        detail_lines = ["- none", ""]
    lines = [
        "# Table-To-Base Measurement Worksheet Audit",
        "",
        f"- status: `{payload['status']}`",
        f"- worksheet: `{payload['worksheet']}`",
        f"- calibration output: `{payload.get('calibration_output')}`",
        "",
        "## Missing Or Invalid Fields",
        "",
        *missing_lines,
        "",
        "## Measurement Guidance",
        "",
        *detail_lines,
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_from_worksheet(
    worksheet_path: Path,
    *,
    output_calibration: Path | None = None,
) -> dict[str, Any]:
    data = _load_yaml(worksheet_path)
    missing: list[str] = []
    source = _require_string(data, "measurement.source", missing)
    status = _require_string(data, "measurement.status", missing)
    _require_string(data, "measurement.measured_at", missing)
    _require_string(data, "measurement.measured_by", missing)
    _require_string(data, "measurement.units", missing)
    _require_string(data, "measurement.coordinate_frame", missing)
    _require_string(data, "measurement.tool", missing)
    _require_float(data, "measurement.uncertainty_m", missing)
    real_robot_touched = _require_bool_or_readonly(data, "measurement.real_robot_touched", missing)
    remote_103_touched = _require_bool_or_readonly(data, "measurement.remote_103_touched", missing)
    table_top_center = _require_float3(data, "table.top_center_world_m", missing)
    table_size = _require_float3(data, "table.size_m", missing)
    table_yaw = _require_float(data, "table.yaw_deg", missing)
    left_base = _require_float3(data, "left_base.translation_table_m", missing)
    left_yaw = _require_float(data, "left_base.yaw_deg", missing)
    right_base = _require_float3(data, "right_base.translation_table_m", missing)
    right_yaw = _require_float(data, "right_base.yaw_deg", missing)
    output_from_sheet = _get_nested(data, "output.calibration_path")
    calibration_output = output_calibration or (REPO_ROOT / str(output_from_sheet) if output_from_sheet else None)
    source_reason = forbidden_table_base_source_reason(source)
    if source_reason:
        missing.append(source_reason)
    if source == "read_from_103" and remote_103_touched not in {"readonly", "read_only"}:
        missing.append("measurement.remote_103_touched must be readonly when source is read_from_103")

    if missing:
        return {
            "status": "BLOCKED_REQUIRES_MEASUREMENT_FIELDS",
            "worksheet": _rel(worksheet_path),
            "calibration_output": _rel(calibration_output) if calibration_output else None,
            "missing_fields": missing,
            "missing_field_details": missing_field_details(missing),
            "interpretation": "The worksheet is only a template or incomplete measurement record. Fill all required fields first.",
        }
    if calibration_output is None:
        return {
            "status": "BLOCKED_REQUIRES_MEASUREMENT_FIELDS",
            "worksheet": _rel(worksheet_path),
            "calibration_output": None,
            "missing_fields": ["output.calibration_path"],
            "missing_field_details": missing_field_details(["output.calibration_path"]),
            "interpretation": "No calibration output path was provided.",
        }

    evidence = build_evidence_record(
        worksheet_path,
        evidence_type="table_to_base_measurement_worksheet",
        real_robot_touched=bool(real_robot_touched),
        remote_103_touched=remote_103_touched if remote_103_touched is not None else False,
    )
    cfg = build_calibration_config(
        table_top_center=table_top_center or [],
        table_size=table_size or [],
        table_yaw_deg=table_yaw if table_yaw is not None else 0.0,
        left_base_in_table=left_base or [],
        left_yaw_deg=left_yaw if left_yaw is not None else 0.0,
        right_base_in_table=right_base or [],
        right_yaw_deg=right_yaw if right_yaw is not None else 0.0,
        source=source or "",
        status=status or "",
        calibration_evidence=evidence,
    )
    calibration_output.parent.mkdir(parents=True, exist_ok=True)
    calibration_output.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    audit = audit_table_frame(calibration_output)
    return {
        "status": "PASS_MEASUREMENT_WORKSHEET_TO_CALIBRATION" if audit["calibration_ready"] else audit["status"],
        "worksheet": _rel(worksheet_path),
        "calibration_output": _rel(calibration_output),
        "missing_fields": [],
        "missing_field_details": {},
        "calibration_audit": audit,
        "interpretation": "The worksheet is complete and generated a calibrated table-to-base config.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a measured table/base worksheet into calibration YAML.")
    parser.add_argument("--worksheet", default=str(DEFAULT_WORKSHEET))
    parser.add_argument("--output-calibration", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--fail-on-incomplete", action="store_true")
    args = parser.parse_args()

    output_calibration = Path(args.output_calibration) if args.output_calibration else None
    payload = build_from_worksheet(Path(args.worksheet), output_calibration=output_calibration)
    output_dir = Path(args.output_dir)
    json_path = output_dir / "measurement_worksheet_audit.json"
    md_path = output_dir / "measurement_worksheet_audit.md"
    payload["outputs"] = {"json": _rel(json_path), "markdown": _rel(md_path)}
    _write_report(json_path, payload)
    _write_markdown(md_path, payload)
    print(json.dumps({"status": payload["status"], "json": _rel(json_path), "markdown": _rel(md_path)}))
    if args.fail_on_incomplete and payload["status"] != "PASS_MEASUREMENT_WORKSHEET_TO_CALIBRATION":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
