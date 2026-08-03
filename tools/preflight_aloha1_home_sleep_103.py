#!/usr/bin/env python3
"""Run the bounded read-only 103 static preflight and write local evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from tools.aloha1_mapping.home_sleep_103_preflight import REMOTE_READ_ONLY_SCRIPT
from tools.aloha1_mapping.home_sleep_103_preflight import classify_remote_snapshot
from tools.aloha1_mapping.home_sleep_103_preflight import parse_remote_snapshot

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "reports/aloha1_mapping/aloha1_home_sleep_sync_103_read_only_preflight.json"
)
DEFAULT_MARKDOWN = (
    ROOT / "reports/aloha1_mapping/aloha1_home_sleep_sync_103_read_only_preflight.md"
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_snapshot(host: str) -> tuple[dict[str, str], dict[str, Any]]:
    """Execute only the frozen read-only shell fragment over SSH."""

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
        input=REMOTE_READ_ONLY_SCRIPT,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"read-only SSH preflight failed with exit {completed.returncode}: "
            f"{completed.stderr[-1000:]}"
        )
    return parse_remote_snapshot(completed.stdout), {
        "argv": command,
        "exit_code": completed.returncode,
        "stdout_sha256": _sha256_text(completed.stdout),
        "stderr_sha256": _sha256_text(completed.stderr),
        "remote_script_sha256": _sha256_text(REMOTE_READ_ONLY_SCRIPT),
    }


def _markdown(report: dict[str, Any]) -> str:
    checks = "\n".join(
        f"- `{name}`: {'PASS' if passed else 'FAIL'}"
        for name, passed in report["static_checks"].items()
    )
    remaining = "\n".join(f"- `{item}`" for item in report["remaining_gates"])
    return f"""# ALOHA1 103 read-only preflight

Status: **{report['status']}**

The inspection was restricted to
`/home/eii/openpi0.5-rtc-reward-learning`. It created no ROS publisher, sent
no command and changed no torque state.

## Static checks

{checks}

Remote HEAD is `{report['remote_git']['head']}` on
`{report['remote_git']['branch']}` with
`{report['remote_git']['dirty_entry_count']}` dirty/untracked entries. They
must be preserved. The robot stack and ROS master are stopped, so static
source declarations are not treated as runtime readback.

The compose file declares an external ALOHA mount at
`{report['external_mount']['path']}`. It was not inspected because it lies
outside the approved remote project boundary.

## Remaining gates

{remaining}

The next operation would start the real robot driver. It requires explicit
user authorization before execution.
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.1.103")
    parser.add_argument("--execute-read-only-ssh", action="store_true")
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
                },
                sort_keys=True,
            )
        )
        return 2
    snapshot, execution = collect_snapshot(args.host)
    report = classify_remote_snapshot(snapshot)
    report["host"] = args.host
    report["execution"] = execution
    report["snapshot"] = snapshot
    output = args.output.resolve()
    markdown = args.markdown.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "real_execution": report["real_execution"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0 if report["static_project_evidence"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
