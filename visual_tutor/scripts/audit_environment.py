#!/usr/bin/env python3
"""Read-only environment audit for the Visual Tutor system."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "visual_tutor"
AUDIT = OUT / "audit"
REPORTS = OUT / "reports"


SECRET_PATTERNS = [
    re.compile(r"sntryu_[A-Za-z0-9_]+"),
    re.compile(r"ctx7sk-[A-Za-z0-9-]+"),
    re.compile(r"(api-key\s+)[^\s]+", re.IGNORECASE),
    re.compile(r"(access-token=)[^\s]+", re.IGNORECASE),
    re.compile(r"([A-Z0-9_]*(?:TOKEN|KEY|SECRET|PASSWORD)[A-Z0-9_]*=)[^\s,]+"),
]


def sanitize(text: str) -> str:
    cleaned = text
    for pattern in SECRET_PATTERNS:
        cleaned = pattern.sub(lambda match: match.group(1) + "*****" if match.lastindex else "*****", cleaned)
    return cleaned


def run(command: list[str], timeout: int = 10) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout": sanitize(proc.stdout.strip()),
            "stderr": sanitize(proc.stderr.strip()),
        }
    except Exception as exc:
        return {"command": command, "error": repr(exc)}


def command_info(name: str, version_args: list[str] | None = None) -> dict[str, Any]:
    path = shutil.which(name)
    info: dict[str, Any] = {"name": name, "path": path, "available": path is not None}
    if path and version_args:
        info["version_probe"] = run([path, *version_args])
    return info


def safe_find(root: Path, patterns: tuple[str, ...], max_depth: int = 4, limit: int = 200) -> list[str]:
    if not root.exists():
        return []
    results: list[str] = []
    root = root.resolve()
    for current, dirs, files in os.walk(root):
        cur = Path(current)
        rel_depth = len(cur.relative_to(root).parts)
        dirs[:] = [d for d in dirs if d not in {".git", ".cache", "node_modules", "__pycache__", ".venv", ".venv_issac"}]
        if rel_depth >= max_depth:
            dirs[:] = []
        for file_name in files:
            if any(cur.joinpath(file_name).match(pattern) for pattern in patterns):
                results.append(str(cur.joinpath(file_name)))
                if len(results) >= limit:
                    return results
    return results


def read_text(path: Path, max_chars: int = 2000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception as exc:
        return f"ERROR: {exc!r}"


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    commands = {
        "hostname": run(["hostname"]),
        "whoami": run(["whoami"]),
        "pwd": run(["pwd"]),
        "git_root": run(["git", "rev-parse", "--show-toplevel"]),
        "git_status_short": run(["git", "status", "--short"]),
        "os_release": read_text(Path("/etc/os-release")),
        "uname": run(["uname", "-a"]),
        "arch": run(["uname", "-m"]),
        "python3": run(["python3", "--version"]),
        "nvidia_smi": run(["nvidia-smi", "-q"], timeout=15),
        "sudo_noninteractive": run(["sudo", "-n", "true"]),
        "codex_version": run(["codex", "--version"]),
        "codex_mcp_list": run(["codex", "mcp", "list"]),
        "codex_mcp_help": run(["codex", "mcp", "--help"]),
    }

    gui_tools = [
        "xdotool",
        "wmctrl",
        "scrot",
        "xwininfo",
        "xprop",
        "import",
        "gnome-screenshot",
        "ffmpeg",
        "dogtail-detect",
        "dogtail-run-headless",
        "ydotool",
        "Xephyr",
        "openbox",
    ]
    cad_tools = ["FreeCAD", "freecad", "freecadcmd", "FreeCADCmd", "blender", "openscad"]
    isaac_tools = ["isaac-sim.sh", "isaaclab.sh", "usdcat", "usdview"]

    skills_root = Path("/home/eii/.codex/skills")
    my_skills = sorted(str(p) for p in skills_root.glob("my-*/SKILL.md")) if skills_root.exists() else []

    controlled_roots = [
        REPO_ROOT,
        Path("/home/eii/isaac_mcp_setup"),
        Path("/home/eii/IsaacLab"),
        Path("/home/eii/isaacsim"),
        Path("/home/eii/Documents"),
        Path("/home/eii/Projects"),
        Path("/home/eii/workspace"),
        Path("/home/eii/ws"),
    ]
    usd_files: list[str] = []
    extensions: list[str] = []
    for root in controlled_roots:
        usd_files.extend(safe_find(root, ("*.usd", "*.usda"), max_depth=5, limit=80))
        extensions.extend(safe_find(root, ("extension.toml",), max_depth=7, limit=80))

    env = {
        "hostname": platform.node(),
        "cwd": str(REPO_ROOT),
        "is_103": "192.168.1.103" in commands.get("hostname_i", {}).get("stdout", ""),
        "DISPLAY": os.environ.get("DISPLAY"),
        "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY"),
        "XDG_SESSION_TYPE": os.environ.get("XDG_SESSION_TYPE"),
        "DESKTOP_SESSION": os.environ.get("DESKTOP_SESSION"),
        "codex_home": os.environ.get("CODEX_HOME"),
        "python_executable": os.sys.executable,
    }

    python_modules: dict[str, str] = {}
    for module in ["yaml", "PIL", "cv2", "mcp", "fastmcp", "pyautogui", "dogtail", "gi"]:
        probe = run(["python3", "-c", f"import {module}; print('OK')"])
        python_modules[module] = "OK" if probe["returncode"] == 0 else probe.get("stderr", probe.get("stdout", "MISSING"))[:300]

    freecad_paths = {
        "config": str(Path.home() / ".config/FreeCAD"),
        "user_data": str(Path.home() / ".local/share/FreeCAD"),
        "config_exists": (Path.home() / ".config/FreeCAD").exists(),
        "user_data_exists": (Path.home() / ".local/share/FreeCAD").exists(),
    }

    isaac_probe = {
        "repo_venv_issac_exists": (REPO_ROOT / ".venv_issac").exists(),
        "isaac_mcp_setup_exists": Path("/home/eii/isaac_mcp_setup").exists(),
        "isaac_setup_status": run(["/home/eii/isaac_mcp_setup/scripts/status_all.sh"]) if Path("/home/eii/isaac_mcp_setup/scripts/status_all.sh").exists() else None,
        "aloha_isaac_files": safe_find(REPO_ROOT / "examples/aloha_isaac", ("*.py", "*.md", "*.yaml"), max_depth=3, limit=120),
    }

    data = {
        "environment": env,
        "commands": commands,
        "tools": {
            "gui": [command_info(name, ["--version"] if name in {"xdotool", "wmctrl", "ffmpeg", "openbox"} else None) for name in gui_tools],
            "cad": [command_info(name, ["--version"] if name.lower().startswith("freecad") else None) for name in cad_tools],
            "isaac": [command_info(name, ["--version"]) for name in isaac_tools],
        },
        "python_modules": python_modules,
        "skills": {"root": str(skills_root), "my_skills": my_skills},
        "freecad": freecad_paths,
        "isaac": isaac_probe,
        "bounded_search": {"usd_files": usd_files[:120], "extension_toml": extensions[:120]},
        "adaptive_contract": read_text(Path("/tmp/my_visual_tutor_contract.json"), max_chars=6000),
    }
    (AUDIT / "environment_audit.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    def status_for_tool(group: str, name: str) -> str:
        tool = next((t for t in data["tools"][group] if t["name"] == name), None)
        return "available" if tool and tool["available"] else "missing"

    matrix_rows = [
        ("Display", "DISPLAY", env["DISPLAY"] or "unset", "required", "Use current X11 display if present; Wayland fallback needs non-global approach."),
        ("Display", "WAYLAND_DISPLAY", env["WAYLAND_DISPLAY"] or "unset", "context", "Do not default to whole-desktop ydotool."),
        ("GUI automation", "xdotool", status_for_tool("gui", "xdotool"), "recommended", "Visible mouse movement on X11/Xephyr."),
        ("GUI automation", "wmctrl", status_for_tool("gui", "wmctrl"), "recommended", "Window discovery/move/activate."),
        ("GUI automation", "scrot", status_for_tool("gui", "scrot"), "recommended", "Screenshot capture."),
        ("GUI automation", "dogtail", python_modules["dogtail"], "recommended", "AT-SPI semantic UI control for FreeCAD when available."),
        ("FreeCAD", "FreeCAD", status_for_tool("cad", "FreeCAD"), "required for FreeCAD adapter", "If missing, keep adapter probe-only and do not install without approval."),
        ("FreeCAD", "FreeCADCmd", status_for_tool("cad", "FreeCADCmd"), "recommended", "Checkpoint/verification via FreeCAD Python."),
        ("Isaac", ".venv_issac", str(isaac_probe["repo_venv_issac_exists"]), "required for Isaac adapter", "Use existing Isaac environment."),
        ("Isaac MCP", "nvidia-isaac-docs", "documented in docs/agents", "required before Isaac modifications", "Already used before implementation."),
        ("MCP", "Codex mcp list", str(commands["codex_mcp_list"]["returncode"]), "required", "Use high-level server only; no arbitrary shell tools."),
    ]
    md = [
        "# Visual Tutor Environment Audit",
        "",
        "## Localization Decision",
        "",
        "- `my-core-adaptive` selected `/home/eii/isaac_mcp_setup/repos/isaacsim-mcp`, but project AGENTS and the current git root identify this repository as the source of truth.",
        "- Implementation workspace: `/home/eii/project/openpi0.5-rtc-reward-learning/visual_tutor`.",
        "- Existing unrelated untracked paths are preserved: `scene_reconstruction/cad/.tmp/`, `scene_reconstruction/cad/aloha_incremental/`.",
        "",
        "## System",
        "",
        f"- Hostname: `{commands['hostname'].get('stdout')}`",
        f"- User: `{commands['whoami'].get('stdout')}`",
        f"- Git root: `{commands['git_root'].get('stdout')}`",
        f"- Python: `{commands['python3'].get('stdout')}`",
        f"- DISPLAY: `{env['DISPLAY']}`",
        f"- WAYLAND_DISPLAY: `{env['WAYLAND_DISPLAY']}`",
        f"- XDG_SESSION_TYPE: `{env['XDG_SESSION_TYPE']}`",
        f"- Noninteractive sudo exit: `{commands['sudo_noninteractive'].get('returncode')}`",
        "",
        "## Capability Matrix",
        "",
        "| Area | Capability | Current status | Need | Handling |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in matrix_rows:
        md.append("| " + " | ".join(str(x).replace("\n", " ")[:160] for x in row) + " |")
    md.extend(
        [
            "",
            "## my- Skills",
            "",
            *[f"- `{path}`" for path in my_skills],
            "",
            "## Bounded Search Summary",
            "",
            f"- USD/USDAs found in controlled roots: `{len(usd_files)}`",
            f"- Extension manifests found in controlled roots: `{len(extensions)}`",
            "",
            "## Route Decision",
            "",
            "- Implement a project-local minimal Visual Tutor core and high-level MCP server first.",
            "- FreeCAD adapter starts as probe/checkpoint skeleton unless FreeCAD is available.",
            "- Isaac adapter uses project-local extension skeleton and OpenUSD/Kit-compatible APIs; no robot or ROS control.",
            "- No package installation in this phase because noninteractive sudo is unavailable and a minimal implementation can be tested without it.",
        ]
    )
    (REPORTS / "environment_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(AUDIT / "environment_audit.json"), "report": str(REPORTS / "environment_audit.md")}, indent=2))


if __name__ == "__main__":
    main()
