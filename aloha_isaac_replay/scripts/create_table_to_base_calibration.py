from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import yaml

from aloha_isaac_replay.scripts.audit_table_frame_candidate import audit_table_frame


DEFAULT_TABLE_SIZE = [1.22, 0.625, 0.04]


def _parse_float_list(value: str, *, name: str, length: int) -> list[float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != length:
        raise argparse.ArgumentTypeError(f"{name} must contain {length} comma-separated values")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must contain only numeric values") from exc


def _yaw_quat_wxyz(yaw_deg: float) -> list[float]:
    half = math.radians(yaw_deg) * 0.5
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


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


def build_calibration_config(
    *,
    table_top_center: list[float],
    table_size: list[float],
    table_yaw_deg: float,
    left_base_in_table: list[float],
    left_yaw_deg: float,
    right_base_in_table: list[float],
    right_yaw_deg: float,
    source: str,
    status: str,
) -> dict[str, Any]:
    table_quat = _yaw_quat_wxyz(table_yaw_deg)
    half_thickness_world = _quat_rotate_vector(table_quat, [0.0, 0.0, table_size[2] * 0.5])
    support_center = [table_top_center[i] - half_thickness_world[i] for i in range(3)]
    return {
        "stage": {"units": "meters", "up_axis": "Z"},
        "support_plane": {
            "mode": "fixed_box",
            "prim_path": "/World/phase66_measured_static_table",
            "center": support_center,
            "size": table_size,
            "provenance": {
                "table_size": {"source": source, "note": "Measured table collision size."},
                "center": {"source": source, "note": "Derived from measured table top center and thickness."},
            },
        },
        "table_frame": {
            "T_world_table": {
                "source": source,
                "translation": table_top_center,
                "rotation_quat_wxyz": table_quat,
                "convention": "Isaac world frame, +Z up, translation at table top center.",
                "status": status,
            },
            "T_table_left_base": {
                "source": source,
                "translation": left_base_in_table,
                "rotation_quat_wxyz": _yaw_quat_wxyz(left_yaw_deg),
                "convention": "Left ALOHA base origin expressed in table frame, +Z up.",
                "status": status,
            },
            "T_table_right_base": {
                "source": source,
                "translation": right_base_in_table,
                "rotation_quat_wxyz": _yaw_quat_wxyz(right_yaw_deg),
                "convention": "Right ALOHA base origin expressed in table frame, +Z up.",
                "status": status,
            },
        },
        "validation": {
            "generated_by": "aloha_isaac_replay/scripts/create_table_to_base_calibration.py",
            "next_replay_gate": "aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a Phase65-compatible table-to-base calibration YAML from measured values."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="user_measured")
    parser.add_argument("--status", default="measured", choices=["measured", "calibrated", "read_from_103", "read_from_usd"])
    parser.add_argument("--table-top-center", required=True, help="x,y,z in Isaac world meters")
    parser.add_argument("--table-size", default="1.22,0.625,0.04", help="x,y,z size in meters")
    parser.add_argument("--table-yaw-deg", type=float, default=0.0)
    parser.add_argument("--left-base", required=True, help="x,y,z left base origin in table frame meters")
    parser.add_argument("--left-yaw-deg", type=float, default=0.0)
    parser.add_argument("--right-base", required=True, help="x,y,z right base origin in table frame meters")
    parser.add_argument("--right-yaw-deg", type=float, default=180.0)
    args = parser.parse_args()

    output = Path(args.output)
    cfg = build_calibration_config(
        table_top_center=_parse_float_list(args.table_top_center, name="--table-top-center", length=3),
        table_size=_parse_float_list(args.table_size, name="--table-size", length=3),
        table_yaw_deg=args.table_yaw_deg,
        left_base_in_table=_parse_float_list(args.left_base, name="--left-base", length=3),
        left_yaw_deg=args.left_yaw_deg,
        right_base_in_table=_parse_float_list(args.right_base, name="--right-base", length=3),
        right_yaw_deg=args.right_yaw_deg,
        source=args.source,
        status=args.status,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    audit = audit_table_frame(output)
    if audit["status"] != "PASS_TABLE_TO_BASE_CALIBRATION_READY":
        raise SystemExit(f"generated calibration did not pass audit: {audit['blocking_reasons']}")
    print(f"wrote {output}")
    print(f"status {audit['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
