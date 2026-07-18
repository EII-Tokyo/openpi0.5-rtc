from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aloha_isaac_replay.calibration.table_measurement_guidance import REQUIRED_WORKSHEET_FIELDS
from aloha_isaac_replay.calibration.table_measurement_guidance import forbidden_table_base_source_reason
from aloha_isaac_replay.calibration.table_measurement_guidance import missing_field_details
from aloha_isaac_replay.scripts.audit_table_frame_candidate import audit_table_frame
from aloha_isaac_replay.scripts.create_calibrated_table_base_overlay import DEFAULT_BASE_USD
from aloha_isaac_replay.scripts.create_calibrated_table_base_overlay import build_overlay
from aloha_isaac_replay.scripts.create_table_to_base_calibration_from_worksheet import DEFAULT_WORKSHEET
from aloha_isaac_replay.scripts.create_table_to_base_calibration_from_worksheet import _get_nested
from aloha_isaac_replay.scripts.create_table_to_base_calibration_from_worksheet import _load_yaml
from aloha_isaac_replay.scripts.create_table_to_base_calibration_from_worksheet import build_from_worksheet

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION = REPO_ROOT / "local_eval_assets/aloha1_calibration/table_to_base_calibration.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase71_table_calibration_readiness_20260718"


def _rel(path: str | Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    missing = payload.get("worksheet", {}).get("missing_fields") or []
    details = payload.get("worksheet", {}).get("missing_field_details") or {}
    missing_lines = [f"- `{item}`" for item in missing] or ["- none"]
    detail_lines: list[str] = []
    for item in missing:
        detail = details.get(item)
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
    next_steps = payload.get("next_steps") or []
    next_lines = [f"- {item}" for item in next_steps] or ["- none"]
    lines = [
        "# ALOHA Table Calibration Readiness",
        "",
        f"- status: `{payload['status']}`",
        f"- calibration: `{payload['calibration']['path']}`",
        f"- calibration exists: `{payload['calibration']['exists']}`",
        f"- worksheet: `{payload['worksheet']['path']}`",
        f"- worksheet status: `{payload['worksheet']['status']}`",
        f"- overlay status: `{payload.get('overlay', {}).get('status')}`",
        "",
        "## Missing Measurement Fields",
        "",
        *missing_lines,
        "",
        "## Measurement Guidance",
        "",
        *detail_lines,
        "## Next Steps",
        "",
        *next_lines,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _static_worksheet_status(worksheet: Path) -> dict[str, Any]:
    if not worksheet.exists():
        return {
            "path": _rel(worksheet),
            "exists": False,
            "status": "BLOCKED_WORKSHEET_MISSING",
            "missing_fields": REQUIRED_WORKSHEET_FIELDS,
            "missing_field_details": missing_field_details(REQUIRED_WORKSHEET_FIELDS),
        }
    data = _load_yaml(worksheet)
    missing = [field for field in REQUIRED_WORKSHEET_FIELDS if _get_nested(data, field) is None]
    source = _get_nested(data, "measurement.source")
    remote = _get_nested(data, "measurement.remote_103_touched")
    real_robot = _get_nested(data, "measurement.real_robot_touched")
    source_reason = forbidden_table_base_source_reason(source)
    if source_reason:
        missing.append(source_reason)
    if source == "read_from_103" and remote not in {"readonly", "read_only"}:
        missing.append("measurement.remote_103_touched must be readonly when source is read_from_103")
    if real_robot is not False:
        missing.append("measurement.real_robot_touched must be false for simulation-only calibration")
    return {
        "path": _rel(worksheet),
        "exists": True,
        "status": "BLOCKED_REQUIRES_MEASUREMENT_FIELDS" if missing else "READY_TO_GENERATE_CALIBRATION",
        "missing_fields": missing,
        "missing_field_details": missing_field_details(missing),
        "calibration_path_from_worksheet": _get_nested(data, "output.calibration_path"),
    }


def summarize_readiness(
    *,
    worksheet: Path = DEFAULT_WORKSHEET,
    calibration: Path = DEFAULT_CALIBRATION,
    base_usd: Path = DEFAULT_BASE_USD,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    try_generate_calibration: bool = False,
    try_generate_overlay: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "table_calibration_readiness.json"
    summary_md = output_dir / "table_calibration_readiness.md"
    worksheet_payload: dict[str, Any] = _static_worksheet_status(worksheet)
    calibration_payload: dict[str, Any] = {
        "path": _rel(calibration),
        "exists": calibration.exists(),
        "audit_status": None,
        "blocking_reasons": [],
    }
    overlay_payload: dict[str, Any] = {"status": "NOT_REQUESTED"}
    next_steps: list[str] = []

    if (
        not calibration.exists()
        and try_generate_calibration
        and worksheet_payload["status"] == "READY_TO_GENERATE_CALIBRATION"
    ):
        generated = build_from_worksheet(worksheet, output_calibration=calibration)
        worksheet_payload.update(
            {
                "status": generated["status"],
                "missing_fields": generated.get("missing_fields", []),
                "generated_calibration": generated.get("calibration_output"),
            }
        )
        calibration_payload["exists"] = calibration.exists()

    if calibration.exists():
        audit = audit_table_frame(calibration)
        calibration_payload.update(
            {
                "exists": True,
                "audit_status": audit["status"],
                "calibration_ready": audit["calibration_ready"],
                "blocking_reasons": audit.get("blocking_reasons", []),
                "world_base_transforms": audit.get("world_base_transforms", {}),
                "calibration_evidence": audit.get("calibration_evidence"),
            }
        )
        if try_generate_overlay:
            overlay = build_overlay(calibration_path=calibration, base_usd=base_usd, output_dir=output_dir / "overlay")
            overlay_payload = {
                "status": overlay["status"],
                "overlay_usd": overlay.get("overlay_usd"),
                "command_manifest": overlay.get("command_manifest"),
                "blocking_reasons": overlay.get("blocking_reasons", []),
            }
        else:
            overlay_payload = {
                "status": "READY_TO_GENERATE" if audit["calibration_ready"] else "BLOCKED_BY_CALIBRATION_AUDIT"
            }

    if calibration_payload["audit_status"] == "PASS_TABLE_TO_BASE_CALIBRATION_READY":
        status = "READY_FOR_CALIBRATED_OVERLAY"
        next_steps.append(
            "Run the calibrated overlay generator, then inspect the overlay in Isaac before contact validation."
        )
    else:
        status = "BLOCKED_REQUIRES_TABLE_BASE_MEASUREMENT"
        if worksheet_payload["missing_fields"]:
            next_steps.append("Fill every missing worksheet field listed above.")
        if not calibration.exists():
            next_steps.append("After the worksheet is complete, rerun this script with --try-generate-calibration.")
        next_steps.append("Do not run final replay/contact validation until the calibration audit passes.")

    payload = {
        "status": status,
        "real_robot_touched": False,
        "remote_103_touched": False,
        "worksheet": worksheet_payload,
        "calibration": calibration_payload,
        "overlay": overlay_payload,
        "next_steps": next_steps,
        "outputs": {"json": _rel(summary_json), "markdown": _rel(summary_md)},
    }
    _write_json(summary_json, payload)
    _write_markdown(summary_md, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize whether ALOHA table-to-base calibration is ready for overlay/replay."
    )
    parser.add_argument("--worksheet", default=str(DEFAULT_WORKSHEET))
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--base-usd", default=str(DEFAULT_BASE_USD))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--try-generate-calibration", action="store_true")
    parser.add_argument("--try-generate-overlay", action="store_true")
    args = parser.parse_args()
    payload = summarize_readiness(
        worksheet=Path(args.worksheet),
        calibration=Path(args.calibration),
        base_usd=Path(args.base_usd),
        output_dir=Path(args.output_dir),
        try_generate_calibration=args.try_generate_calibration,
        try_generate_overlay=args.try_generate_overlay,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "worksheet_status": payload["worksheet"]["status"],
                "calibration_audit_status": payload["calibration"]["audit_status"],
                "json": payload["outputs"]["json"],
                "markdown": payload["outputs"]["markdown"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["status"].startswith("READY") else 2


if __name__ == "__main__":
    raise SystemExit(main())
