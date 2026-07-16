from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from aloha_isaac_replay.assets.urdf_audit import audit_urdf
from aloha_isaac_replay.assets.usd_static_inspector import inspect_usd_static


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _markdown(payload: dict[str, Any]) -> str:
    lines = ["# Isaac ALOHA Asset Static Inspection", ""]
    lines.append(f"- Asset: `{payload['asset']}`")
    lines.append(f"- Mode: `{payload['mode']}`")
    lines.append("")
    if payload["mode"] == "usd_static":
        usd = payload["usd_static"]
        lines.extend(
            [
                f"- File type: `{usd['file_type']}`",
                f"- PXR available: `{usd['pxr_available']}`",
                f"- Candidate score: `{usd['candidate_score']}`",
                f"- Likely rejections: {usd['likely_rejections']}",
                "",
                "## Keyword Hits",
                "",
                "| keyword | count |",
                "|---|---:|",
            ]
        )
        for key, count in usd["keyword_hits"].items():
            lines.append(f"| `{key}` | {count} |")
    elif payload["mode"] == "urdf":
        urdf = payload["urdf"]
        lines.extend(
            [
                f"- Robot name: `{urdf['robot_name']}`",
                f"- VX300S-like: `{urdf['is_vx300s_like']}`",
                f"- Root links: `{urdf['root_links']}`",
                f"- Arm joints: `{urdf['arm_joint_names_present']}`",
                f"- Finger joints: `{urdf['finger_joint_names_present']}`",
                f"- EE links: `{urdf['ee_links']}`",
                f"- Identity errors: {urdf['identity_errors']}",
            ]
        )
    return "\n".join(lines) + "\n"


def inspect_asset(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() in {".urdf", ".xacro"}:
        return {"asset": str(path), "mode": "urdf", "urdf": _jsonable(audit_urdf(path))}
    return {"asset": str(path), "mode": "usd_static", "usd_static": _jsonable(inspect_usd_static(path))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Statically inspect an ALOHA USD/URDF asset candidate.")
    parser.add_argument("--asset", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    payload = inspect_asset(args.asset)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(payload))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

