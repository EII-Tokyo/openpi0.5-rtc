#!/usr/bin/env python3
"""Build the authorized follower-left read-only runtime evidence report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from tools.aloha1_mapping.home_sleep_live_runtime_report import build_live_runtime_report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "absolute_path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _parse_named_paths(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        parsed[name] = Path(raw_path)
    return parsed


def _joint_state_summary(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    position_columns = [
        name
        for name in (rows[0].keys() if rows else [])
        if name.startswith("field.position")
    ]
    velocity_columns = [
        name
        for name in (rows[0].keys() if rows else [])
        if name.startswith("field.velocity")
    ]
    spans = {
        name: max(float(row[name]) for row in rows)
        - min(float(row[name]) for row in rows)
        for name in position_columns
    }
    max_abs_velocity = max(
        (abs(float(row[name])) for row in rows for name in velocity_columns),
        default=0.0,
    )
    return {
        "source": _file_record(path),
        "rows": len(rows),
        "position_columns": position_columns,
        "velocity_columns": velocity_columns,
        "position_span_rad_by_joint": spans,
        "max_position_span_rad": max(spans.values(), default=0.0),
        "max_abs_reported_velocity": max_abs_velocity,
    }


def _markdown(report: dict[str, object]) -> str:
    driver = report["driver"]
    camera_pre = report["camera_pre_driver"]
    camera_post = report["camera_post_driver"]
    ros = report["ros_readback"]
    joints = report["joint_state_samples"]
    assert isinstance(driver, dict)
    assert isinstance(camera_pre, dict)
    assert isinstance(camera_post, dict)
    assert isinstance(ros, dict)
    assert isinstance(joints, dict)
    lines = [
        "# ALOHA1 follower_left live read-only runtime",
        "",
        f"- Overall: `{report['status']}`",
        f"- Real motion: `{report['REAL_MOTION']}`",
        f"- Real/digital correspondence: `{report['REAL_DIGITAL_CORRESPONDENCE']}`",
        f"- Workspace motion gate: `{report['WORKSPACE_CLEAR_FOR_MOTION']}`",
        f"- Stop/hold gate: `{report['STOP_HOLD_PATH']}`",
        "",
        "## Verified runtime evidence",
        "",
        f"- Driver running at final readback: `{driver['left_driver_running']}`.",
        f"- Arm/gripper modes: `{driver['arm_operating_mode']}` / "
        f"`{driver['gripper_operating_mode']}`.",
        f"- ROS status: `{ros['status']}`.",
        f"- Explicit arm order: `{', '.join(ros['arm_joint_order'])}`.",
        f"- Command messages published by diagnostics: `{report['commands_published']}`.",
        f"- Joint-state samples: `{joints['rows']}`; max position span "
        f"`{joints['max_position_span_rad']:.9f} rad`; maximum reported velocity "
        f"`{joints['max_abs_reported_velocity']:.9f} rad/s`.",
        f"- cam_high pre/post frames: `{camera_pre['frames_captured']}` / "
        f"`{camera_post['frames_captured']}`; hardware resets: "
        f"`{camera_pre['hardware_resets'] + camera_post['hardware_resets']}`.",
        "",
        "## Safety boundary",
        "",
        "This phase enabled the existing follower-left arm and gripper torque "
        "through the isolated driver configuration, but did not construct a "
        "robot command publisher and did not send Home, Sleep, or any other "
        "motion command. Torque enable is inferred from the deployed mode "
        "configuration and driver startup log; it is not a direct register "
        "readback.",
        "",
        "The cam_high images show a cluttered tabletop. They are auxiliary "
        "workspace-safety evidence and do not prove signal correspondence. "
        "Real motion remains blocked pending a cleared workspace, an "
        "operator-tested stop/hold path, and fresh explicit authorization.",
        "",
        "## Remaining gates",
        "",
    ]
    lines.extend(f"- `{gate}`" for gate in report["remaining_gates"])
    lines.extend(["", "## Evidence files", ""])
    for name, record in report["artifact_files"].items():
        lines.append(
            f"- `{name}`: `{record['absolute_path']}` "
            f"(SHA-256 `{record['sha256']}`)"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ros-report", type=Path, required=True)
    parser.add_argument("--camera-pre", type=Path, required=True)
    parser.add_argument("--camera-post", type=Path, required=True)
    parser.add_argument("--driver-log", type=Path, required=True)
    parser.add_argument("--joint-csv", type=Path, required=True)
    parser.add_argument("--deployment-file", action="append", default=[])
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--driver-running", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    deployment_paths = _parse_named_paths(args.deployment_file)
    artifact_paths = _parse_named_paths(args.artifact)
    deployment_files = {
        name: _file_record(path) for name, path in deployment_paths.items()
    }
    artifact_files = {
        name: _file_record(path) for name, path in artifact_paths.items()
    }
    artifact_files.update(
        {
            "ros_report": _file_record(args.ros_report),
            "camera_pre_report": _file_record(args.camera_pre),
            "camera_post_report": _file_record(args.camera_post),
            "driver_log": _file_record(args.driver_log),
            "joint_state_csv": _file_record(args.joint_csv),
        }
    )
    report = build_live_runtime_report(
        ros_report=json.loads(args.ros_report.read_text(encoding="utf-8")),
        camera_pre=json.loads(args.camera_pre.read_text(encoding="utf-8")),
        camera_post=json.loads(args.camera_post.read_text(encoding="utf-8")),
        driver_log=args.driver_log.read_text(encoding="utf-8", errors="replace"),
        deployment_hashes={
            name: str(record["sha256"])
            for name, record in deployment_files.items()
        },
        artifact_paths={
            name: str(record["absolute_path"])
            for name, record in artifact_files.items()
        },
        driver_running=args.driver_running,
        joint_state_samples=_joint_state_summary(args.joint_csv),
    )
    report["deployment_files"] = deployment_files
    report["artifact_files"] = artifact_files
    report["remote_runtime"] = {
        "host": "192.168.1.103",
        "project_root": "/home/eii/openpi0.5-rtc-reward-learning",
        "deployment_root": (
            "/home/eii/openpi0.5-rtc-reward-learning/.codex/runtime/"
            "aloha1_home_sleep_live_readback"
        ),
        "project_head": "ea818494bf9ee7756c955864ba3b0d62be6ce649",
        "project_branch": "paper_actor_sample",
        "dirty_entries_preserved": 45,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(args.output_json), "status": report["status"]}))
    return 0 if report["status"] == "PASS_READ_ONLY_RUNTIME_MOTION_NOT_RUN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
