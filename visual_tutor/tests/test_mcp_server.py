from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "visual_tutor/my_gui_teacher/server.py"


def _rpc(process: subprocess.Popen[str], payload: dict) -> dict:
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(json.dumps(payload) + "\n")
    process.stdin.flush()
    return json.loads(process.stdout.readline())


def test_server_exposes_only_high_level_tools() -> None:
    process = subprocess.Popen(
        [sys.executable, str(SERVER)],
        cwd=ROOT,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        init = _rpc(process, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert init["result"]["serverInfo"]["name"] == "my-gui-teacher"
        listed = _rpc(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        names = {tool["name"] for tool in listed["result"]["tools"]}
        forbidden = {"click", "shell", "sudo", "python", "delete", "ros_publish", "control_robot"}
        assert forbidden.isdisjoint(names)
        assert {"probe_app", "start_lesson", "next_step", "abort_lesson"}.issubset(names)
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_server_runs_isaac_dry_run_lesson() -> None:
    process = subprocess.Popen(
        [sys.executable, str(SERVER)],
        cwd=ROOT,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        lesson_path = "visual_tutor/lessons/isaac_cube_dry_run.yaml"
        started = _rpc(
            process,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "start_lesson", "arguments": {"lesson_path": lesson_path}},
            },
        )
        assert "error" not in started
        stepped = _rpc(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "next_step", "arguments": {}}})
        text = stepped["result"]["content"][0]["text"]
        assert "CHECKPOINTED" in text or "COMPLETED" in text
    finally:
        process.terminate()
        process.wait(timeout=5)
