from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aloha_isaac_replay.scripts.audit_table_frame_candidate import audit_table_frame


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_USD = (
    REPO_ROOT
    / "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose"
    / "aloha2_menagerie_scene_deep_black_real_start_pose.usd"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "local_eval_assets/aloha1_calibrated_table_base_overlay"
DEFAULT_LEFT_TARGET = "/scene/left_base_link"
DEFAULT_RIGHT_TARGET = "/scene/right_base_link"


def _rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _quote_usd_asset(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace("@", "\\@")


def _quote_usd_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _prim_over_block(prim_path: str, transform: dict[str, Any], side: str) -> str:
    if not prim_path.startswith("/"):
        raise ValueError(f"{side} target prim path must be absolute: {prim_path}")
    parts = [part for part in prim_path.split("/") if part]
    if not parts:
        raise ValueError(f"{side} target prim path must not be the pseudo-root")
    indent = ""
    lines: list[str] = []
    for part in parts[:-1]:
        lines.append(f'{indent}over "{_quote_usd_string(part)}"')
        lines.append(f"{indent}{{")
        indent += "    "
    leaf = parts[-1]
    translation = transform["translation"]
    quat = transform["rotation_quat_wxyz"]
    lines.extend(
        [
            f'{indent}over "{_quote_usd_string(leaf)}" (',
            f"{indent}    customData = {{",
            f'{indent}        string aloha1_calibration_side = "{side}"',
            f'{indent}        string aloha1_calibration_note = "Simulation-only table-to-base transform authored from Phase68 audited calibration."',
            f"{indent}    }}",
            f"{indent})",
            f"{indent}{{",
            f"{indent}    double3 xformOp:translate = ({translation[0]:.12g}, {translation[1]:.12g}, {translation[2]:.12g})",
            f"{indent}    quatd xformOp:orient = ({quat[0]:.12g}, {quat[1]:.12g}, {quat[2]:.12g}, {quat[3]:.12g})",
            f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]',
            f"{indent}}}",
        ]
    )
    for part in reversed(parts[:-1]):
        indent = indent[:-4]
        lines.append(f"{indent}}}")
    return "\n".join(lines)


def _render_overlay_usda(
    *,
    base_usd: Path,
    calibration_path: Path,
    audit: dict[str, Any],
    left_target_prim: str,
    right_target_prim: str,
) -> str:
    world_base = audit["world_base_transforms"]
    left_block = _prim_over_block(left_target_prim, world_base["T_world_left_base"], "left")
    right_block = _prim_over_block(right_target_prim, world_base["T_world_right_base"], "right")
    evidence = audit.get("calibration_evidence") or {}
    return "\n".join(
        [
            "#usda 1.0",
            "(",
            "    metersPerUnit = 1",
            "    upAxis = \"Z\"",
            "    subLayers = [",
            f"        @{_quote_usd_asset(base_usd)}@",
            "    ]",
            ")",
            "",
            "def Xform \"World\" (",
            "    customData = {",
            f'        string overlay_scope = "phase69_calibrated_table_base_overlay"',
            f'        string overlay_base_usd = "{_quote_usd_string(str(base_usd.resolve()))}"',
            f'        string calibration_config = "{_quote_usd_string(str(calibration_path.resolve()))}"',
            f'        string calibration_audit_status = "{_quote_usd_string(str(audit["status"]))}"',
            f'        string calibration_evidence_sha256 = "{_quote_usd_string(str(evidence.get("sha256", "")))}"',
            "    }",
            ")",
            "{",
            "}",
            "",
            left_block,
            "",
            right_block,
            "",
        ]
    )


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Calibrated Table-to-Base Overlay",
        "",
        f"- status: `{payload['status']}`",
        f"- calibration: `{payload['calibration']}`",
        f"- base USD: `{payload['base_usd']}`",
        f"- overlay USD: `{payload.get('overlay_usd')}`",
        f"- command manifest: `{payload.get('command_manifest')}`",
        f"- left target prim: `{payload['target_prims']['left']}`",
        f"- right target prim: `{payload['target_prims']['right']}`",
        "",
        "## Gate",
        "",
        f"- calibration audit status: `{payload['calibration_audit']['status']}`",
        f"- calibration ready: `{payload['calibration_audit']['calibration_ready']}`",
    ]
    if payload.get("blocking_reasons"):
        lines.extend(["", "## Blocking Reasons", ""])
        lines.extend([f"- {reason}" for reason in payload["blocking_reasons"]])
    lines.extend(
        [
            "",
            "## Replay Command",
            "",
            "The command manifest is intentionally separate from execution. Review it before opening Isaac.",
            "",
            "```bash",
            payload.get("open_command", ""),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def build_overlay(
    *,
    calibration_path: Path,
    base_usd: Path = DEFAULT_BASE_USD,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    left_target_prim: str = DEFAULT_LEFT_TARGET,
    right_target_prim: str = DEFAULT_RIGHT_TARGET,
) -> dict[str, Any]:
    audit = audit_table_frame(calibration_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "calibrated_overlay_summary.json"
    markdown_path = output_dir / "calibrated_overlay_summary.md"
    payload: dict[str, Any] = {
        "status": "BLOCKED_CALIBRATION_AUDIT_NOT_READY",
        "calibration": _rel(calibration_path),
        "base_usd": _rel(base_usd),
        "target_prims": {"left": left_target_prim, "right": right_target_prim},
        "calibration_audit": audit,
        "blocking_reasons": list(audit.get("blocking_reasons", [])),
        "outputs": {"summary_json": _rel(summary_path), "summary_markdown": _rel(markdown_path)},
    }
    if audit["status"] != "PASS_TABLE_TO_BASE_CALIBRATION_READY":
        _write_json(summary_path, payload)
        markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
        return payload
    if not base_usd.exists():
        payload["blocking_reasons"].append(f"base USD does not exist: {base_usd}")
        _write_json(summary_path, payload)
        markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
        return payload
    overlay_path = output_dir / "aloha1_calibrated_table_base_overlay.usda"
    manifest_path = output_dir / "replay_command_manifest.json"
    overlay_text = _render_overlay_usda(
        base_usd=base_usd,
        calibration_path=calibration_path,
        audit=audit,
        left_target_prim=left_target_prim,
        right_target_prim=right_target_prim,
    )
    overlay_path.write_text(overlay_text, encoding="utf-8")
    open_command = (
        "OMNI_KIT_ACCEPT_EULA=YES .venv_issac/bin/python "
        "examples/aloha_isaac/scripts/open_workcell_gui.py "
        f"--usd {overlay_path} --allow-noncanonical-usd"
    )
    manifest = {
        "schema": "aloha1_phase69_calibrated_table_base_overlay.v1",
        "status": "READY_FOR_REVIEW_NOT_EXECUTED",
        "overlay_usd": _rel(overlay_path),
        "base_usd": _rel(base_usd),
        "calibration": _rel(calibration_path),
        "target_prims": {"left": left_target_prim, "right": right_target_prim},
        "world_base_transforms": audit["world_base_transforms"],
        "calibration_evidence": audit.get("calibration_evidence"),
        "open_command": open_command,
        "safety": {
            "simulation_only": True,
            "real_robot_touched": False,
            "remote_103_touched": False,
            "timeline_starts_paused": True,
            "requires_human_review_before_replay": True,
        },
    }
    _write_json(manifest_path, manifest)
    payload.update(
        {
            "status": "PASS_CALIBRATED_OVERLAY_READY_FOR_REVIEW",
            "overlay_usd": _rel(overlay_path),
            "command_manifest": _rel(manifest_path),
            "open_command": open_command,
            "blocking_reasons": [],
        }
    )
    _write_json(summary_path, payload)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a simulation-only USD overlay from a Phase68 audited table-to-base calibration."
    )
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--base-usd", default=str(DEFAULT_BASE_USD))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--left-target-prim", default=DEFAULT_LEFT_TARGET)
    parser.add_argument("--right-target-prim", default=DEFAULT_RIGHT_TARGET)
    args = parser.parse_args()

    payload = build_overlay(
        calibration_path=Path(args.calibration),
        base_usd=Path(args.base_usd),
        output_dir=Path(args.output_dir),
        left_target_prim=args.left_target_prim,
        right_target_prim=args.right_target_prim,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "summary": payload["outputs"]["summary_json"],
                "overlay_usd": payload.get("overlay_usd"),
                "command_manifest": payload.get("command_manifest"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
