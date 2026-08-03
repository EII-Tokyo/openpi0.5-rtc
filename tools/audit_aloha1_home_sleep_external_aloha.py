#!/usr/bin/env python3
"""Audit the explicitly authorized external ALOHA source without starting ROS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from tools.aloha1_mapping.home_sleep_external_aloha_audit import REMOTE_EXTERNAL_READ_ONLY_SCRIPT
from tools.aloha1_mapping.home_sleep_external_aloha_audit import build_external_audit_report
from tools.aloha1_mapping.home_sleep_external_aloha_audit import parse_snapshot

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = (
    ROOT / "configs/aloha1_home_sleep_puppet_left_only_candidate.launch"
)
DEFAULT_OUTPUT = (
    ROOT
    / "reports/aloha1_mapping/aloha1_home_sleep_sync_external_aloha_audit.json"
)
DEFAULT_MARKDOWN = DEFAULT_OUTPUT.with_suffix(".md")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect_snapshot(host: str) -> tuple[dict[str, str], dict[str, Any]]:
    command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "LogLevel=ERROR",
        host,
        "bash -s",
    ]
    completed = subprocess.run(
        command,
        input=REMOTE_EXTERNAL_READ_ONLY_SCRIPT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "external ALOHA read-only audit failed with exit "
            f"{completed.returncode}: {completed.stderr[-1000:]}"
        )
    return parse_snapshot(completed.stdout), {
        "argv": command,
        "exit_code": completed.returncode,
        "stdout_sha256": _sha256(completed.stdout.encode()),
        "stderr_sha256": _sha256(completed.stderr.encode()),
        "remote_script_sha256": _sha256(
            REMOTE_EXTERNAL_READ_ONLY_SCRIPT.encode()
        ),
        "operation": "READ_ONLY_SOURCE_INSPECTION",
    }


def build_markdown(report: dict[str, Any]) -> str:
    repo = report["external_repository"]
    existing = report["existing_deployment"]
    candidate = report["left_only_candidate"]
    remaining = "\n".join(f"- `{item}`" for item in report["remaining_gates"])
    files = "\n".join(
        f"- `{item['path']}` — `{item['sha256']}`"
        for item in report["file_manifest"]
    )
    return f"""# ALOHA1 external ROS1 source audit

Status: **{report['status']}**

This audit was read-only. It started no ROS node or robot driver, constructed
no publisher, sent no command, and changed no torque state.

## Frozen source

- Root: `{repo['local_path']}`
- Git top-level: `{repo['git_toplevel']}`
- Origin: `{repo['origin']}`
- Branch/commit: `{repo['branch']}` / `{repo['commit']}`
- Dirty entries: `{repo['dirty_entry_count']}` (preserved)
- License: `{repo['license']}`

{files}

## Existing deployment boundary

Existing `ros_nodes.launch`: **{existing['status']}**.
It includes `{existing['driver_scope']['include_count']}` robot drivers:
`{', '.join(existing['driver_scope']['robot_names'])}`. The left follower arm
and gripper mode configuration both have torque enabled. The bundled camera
publisher requires four camera serials and calls `hardware_reset()`. The
bundled `sleep.py` constructs and commands both puppet arms. None of these
entry points is accepted for the left-only supervised replay.

## Isolated launch candidate

Candidate static status: **{candidate['status']}**. It includes only
`puppet_left` / `vx300s`, uses the deployed left mode configuration, keeps
`load_configs=false`, and contains no camera node. This is an inert source
file, not runtime evidence. Starting it would touch real hardware and remains
**{candidate['real_execution']}**.

## Remaining gates

{remaining}

Real execution remains **{report['authorization']['real_execution']}**.
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.1.103")
    parser.add_argument("--execute-read-only-ssh", action="store_true")
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.execute_read_only_ssh:
        print(
            json.dumps(
                {
                    "status": "NOT_RUN_EXPLICIT_READ_ONLY_FLAG_REQUIRED",
                    "network_access_performed": False,
                    "driver_started": False,
                    "commands_published": 0,
                },
                sort_keys=True,
            )
        )
        return 2

    candidate = args.candidate.resolve()
    snapshot, execution = collect_snapshot(args.host)
    candidate_bytes = candidate.read_bytes()
    report = build_external_audit_report(
        snapshot, candidate_text=candidate_bytes.decode("utf-8")
    )
    report["host"] = args.host
    report["execution"] = execution
    report["left_only_candidate"]["path"] = str(candidate)
    report["left_only_candidate"]["sha256"] = _sha256(candidate_bytes)

    output = args.output.resolve()
    markdown = args.markdown.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown.write_text(build_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "real_execution": report["authorization"]["real_execution"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "READY_FOR_MINIMAL_START_AUTHORIZATION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
