#!/usr/bin/env python3
"""Poll Isaac Sim health and ask Codex for a read-only diagnosis on changes.

The collector is deliberately deterministic and lightweight.  It does not use
inotify, modify the Stage, or restart any process.  Codex is invoked only for a
baseline, a health-state transition, or newly observed error lines.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path("/home/eii/openpi0.5-rtc-reward-learning")
REPORT_DIR = PROJECT_ROOT / "remote_isaac_assets/aloha1_bottle_server/attempt1/reports/codex_monitor"
STATE_PATH = REPORT_DIR / "monitor_state.json"
LATEST_PATH = REPORT_DIR / "latest_status.json"
CODEX_BIN = Path("/home/eii/.local/bin/codex")
KIT_LOG_GLOB = "/home/eii/.nvidia-omniverse/logs/Kit/Isaac-Sim Streaming/5.1/kit_*.log"
WATCH_LIMIT_PATH = Path("/proc/sys/fs/inotify/max_user_watches")
ERROR_RE = re.compile(
    r"\[Error\]|Traceback|Failed to create change watch|Segmentation fault|"
    r"CUDA[^\n]*(?:error|failed)|PhysX[^\n]*(?:error|failed)",
    re.IGNORECASE,
)
MAX_LOG_READ_BYTES = 2 * 1024 * 1024
CODEX_COOLDOWN_SECONDS = 15 * 60


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path, default: dict) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def atomic_write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def find_kit_processes() -> list[dict]:
    found: list[dict] = []
    for proc_dir in glob.glob("/proc/[0-9]*"):
        pid = Path(proc_dir).name
        try:
            cmdline = (Path(proc_dir) / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "/kit/kit " in cmdline and "isaacsim" in cmdline.lower():
            found.append({"pid": int(pid), "command": cmdline[:1000]})
    return sorted(found, key=lambda item: item["pid"])


def tcp_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.5):
            return True
    except OSError:
        return False


def http_probe(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            body = response.read(4096).decode(errors="replace")
            return {"ok": 200 <= response.status < 300, "status": response.status, "body": body}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "error": str(exc)}
    except Exception as exc:  # Network diagnostics must not terminate the collector.
        return {"ok": False, "status": None, "error": f"{type(exc).__name__}: {exc}"}


def inotify_usage() -> dict:
    instances = 0
    watches = 0
    by_pid: dict[str, int] = {}
    for fdinfo in glob.glob("/proc/[0-9]*/fdinfo/*"):
        try:
            text = Path(fdinfo).read_text(errors="ignore")
        except OSError:
            continue
        count = text.count("inotify wd:")
        if not count:
            continue
        instances += 1
        watches += count
        pid = fdinfo.split("/")[2]
        by_pid[pid] = by_pid.get(pid, 0) + count
    try:
        limit = int(WATCH_LIMIT_PATH.read_text().strip())
    except (OSError, ValueError):
        limit = 0
    top = []
    for pid, count in sorted(by_pid.items(), key=lambda pair: pair[1], reverse=True)[:8]:
        try:
            command = Path(f"/proc/{pid}/comm").read_text().strip()
        except OSError:
            command = "unknown"
        top.append({"pid": int(pid), "command": command, "watches": count})
    return {
        "instances": instances,
        "watches": watches,
        "limit": limit,
        "ratio": (watches / limit) if limit else None,
        "top_processes": top,
    }


def newest_kit_log() -> Path | None:
    candidates = [Path(path) for path in glob.glob(KIT_LOG_GLOB)]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def read_new_log_errors(previous: dict) -> tuple[dict, list[str]]:
    path = newest_kit_log()
    if path is None:
        return {"path": None, "offset": 0, "size": 0}, []
    size = path.stat().st_size
    old_path = previous.get("log_path")
    old_offset = int(previous.get("log_offset", 0))
    if old_path != str(path) or old_offset > size:
        old_offset = max(0, size - MAX_LOG_READ_BYTES)
    start = max(old_offset, size - MAX_LOG_READ_BYTES)
    with path.open("rb") as stream:
        stream.seek(start)
        text = stream.read().decode(errors="replace")
    errors = [line[-2000:] for line in text.splitlines() if ERROR_RE.search(line)]
    return {"path": str(path), "offset": size, "size": size}, errors[-40:]


def collect(previous: dict) -> dict:
    kit = find_kit_processes()
    health = http_probe("http://127.0.0.1:8006/health")
    streaming = http_probe("http://127.0.0.1:8006/v1/streaming/ready")
    watches = inotify_usage()
    log_state, new_errors = read_new_log_errors(previous)
    required_ok = bool(kit) and tcp_open(49100) and tcp_open(8006) and health.get("ok", False)
    watch_degraded = bool(watches["ratio"] is not None and watches["ratio"] >= 0.90)
    status = "healthy" if required_ok and not watch_degraded else "degraded" if required_ok else "unhealthy"
    return {
        "timestamp_utc": utc_now(),
        "host": socket.gethostname(),
        "status": status,
        "isaac_sim": {"processes": kit, "kit_running": bool(kit)},
        "ports": {"49100_webrtc": tcp_open(49100), "8006_services": tcp_open(8006)},
        "http": {"health": health, "streaming_ready": streaming},
        "inotify": watches,
        "log": {**log_state, "new_error_count": len(new_errors), "new_errors": new_errors},
    }


def should_run_codex(snapshot: dict, previous: dict, force: bool) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if force:
        reasons.append("forced baseline")
    if not previous.get("initialized"):
        reasons.append("first monitor run")
    if previous.get("last_status") not in (None, snapshot["status"]):
        reasons.append(f"status changed: {previous.get('last_status')} -> {snapshot['status']}")
    if snapshot["log"]["new_error_count"]:
        reasons.append(f"{snapshot['log']['new_error_count']} new matching log lines")
    if snapshot["status"] != "healthy":
        reasons.append(f"current status is {snapshot['status']}")
    if not reasons:
        return False, []
    last_run = float(previous.get("last_codex_epoch", 0))
    urgent = snapshot["status"] == "unhealthy" or force or not previous.get("initialized")
    if not urgent and time.time() - last_run < CODEX_COOLDOWN_SECONDS:
        return False, reasons + ["Codex cooldown active"]
    return True, reasons


def run_codex(snapshot: dict, reasons: list[str]) -> dict:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"codex_diagnosis_{timestamp}.md"
    prompt = f"""You are monitoring an Isaac Sim server on the host aloha.
Analyze the health snapshot below. This is a READ-ONLY monitoring run.
Do not edit files, do not restart or signal processes, do not control the robot,
do not save the USD Stage, and do not expose credentials.

Return a concise Markdown report with:
1. health verdict (healthy/degraded/unhealthy),
2. evidence from the snapshot,
3. likely cause of each anomaly,
4. safe next checks that require no mutation,
5. actions requiring explicit human approval.

Trigger reasons: {json.dumps(reasons, ensure_ascii=False)}
Snapshot:
{json.dumps(snapshot, indent=2, ensure_ascii=False)}
"""
    command = [
        str(CODEX_BIN),
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "-C",
        str(PROJECT_ROOT),
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=240,
            check=False,
            env={**os.environ, "HOME": "/home/eii", "PATH": f"/home/eii/.local/bin:{os.environ.get('PATH', '')}"},
        )
        report_path.write_text(completed.stdout or completed.stderr or "Codex produced no output.\n")
        return {
            "attempted": True,
            "exit_code": completed.returncode,
            "report": str(report_path),
            "stderr_tail": completed.stderr[-4000:],
        }
    except Exception as exc:
        report_path.write_text(f"Codex monitor invocation failed: {type(exc).__name__}: {exc}\n")
        return {"attempted": True, "exit_code": None, "report": str(report_path), "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-codex", action="store_true", help="Generate a baseline Codex report now")
    parser.add_argument("--collect-only", action="store_true", help="Never invoke Codex")
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    previous = load_json(STATE_PATH, {})
    snapshot = collect(previous)
    run, reasons = should_run_codex(snapshot, previous, args.force_codex)
    codex_result = {"attempted": False}
    if run and not args.collect_only:
        codex_result = run_codex(snapshot, reasons)
    snapshot["codex"] = {"trigger_reasons": reasons, **codex_result}
    atomic_write_json(LATEST_PATH, snapshot)
    state = {
        "initialized": True,
        "last_status": snapshot["status"],
        "log_path": snapshot["log"]["path"],
        "log_offset": snapshot["log"]["offset"],
        "last_run_utc": snapshot["timestamp_utc"],
        "last_codex_epoch": time.time() if codex_result.get("attempted") else previous.get("last_codex_epoch", 0),
        "last_codex_report": codex_result.get("report", previous.get("last_codex_report")),
    }
    atomic_write_json(STATE_PATH, state)
    print(json.dumps({"status": snapshot["status"], "codex": snapshot["codex"]}, ensure_ascii=False))
    return 0 if snapshot["status"] != "unhealthy" else 2


if __name__ == "__main__":
    sys.exit(main())
