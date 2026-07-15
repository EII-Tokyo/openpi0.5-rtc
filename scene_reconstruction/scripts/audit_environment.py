#!/usr/bin/env python3
"""Read-only environment audit for the ALOHA scene reconstruction task."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "scene_reconstruction"
AUDIT = OUT / "audit"
REPORTS = OUT / "reports"
PHOTO_DIR = Path("/home/eii/Downloads/iphone")
BOUNDED_SEARCH_ROOTS = [
    REPO_ROOT,
    Path("/home/eii/IsaacLab"),
    Path("/home/eii/isaacsim"),
    Path("/home/eii/Documents"),
    Path("/home/eii/Projects"),
    Path("/home/eii/workspace"),
]


def run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 20) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001 - audit must not abort on missing tools.
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": repr(exc)}


def command_info(command: str, version_args: list[str] | None = None) -> dict[str, Any]:
    path = shutil.which(command)
    info: dict[str, Any] = {"command": command, "path": path, "available": path is not None}
    if path and version_args:
        info["version_probe"] = run([command, *version_args], timeout=15)
    return info


def bounded_find(root: Path, names: tuple[str, ...], max_results: int = 200) -> list[str]:
    if not root.exists():
        return []
    results: list[str] = []
    excluded = {".git", ".cache", "node_modules", "__pycache__", "logs", "log", "large_data", "datasets"}
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in excluded]
        path = Path(current)
        for file_name in files:
            if file_name.lower().endswith(names):
                results.append(str(path / file_name))
                if len(results) >= max_results:
                    return results
    return results


def inspect_python_modules(python: str = "python3") -> dict[str, str]:
    code = """
mods = ["FreeCAD", "Part", "numpy", "PIL", "cv2", "yaml", "pxr", "isaacsim"]
for mod in mods:
    try:
        m = __import__(mod)
        version = getattr(m, "__version__", "")
        print(f"{mod}: OK {version}")
    except Exception as exc:
        print(f"{mod}: MISSING ({type(exc).__name__}: {exc})")
"""
    result = run([python, "-c", code], timeout=20)
    modules: dict[str, str] = {}
    for line in result["stdout"].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            modules[key.strip()] = value.strip()
    return modules


def photo_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {"path": str(PHOTO_DIR), "exists": PHOTO_DIR.exists(), "counts": {}, "sample_files": []}
    if not PHOTO_DIR.exists():
        return summary
    suffix_counts: dict[str, int] = {}
    files = [p for p in PHOTO_DIR.iterdir() if p.is_file()]
    for p in files:
        suffix = p.suffix.lower() or "<none>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    summary["counts"] = dict(sorted(suffix_counts.items()))
    summary["file_count"] = len(files)
    summary["sample_files"] = [str(p) for p in sorted(files)[:25]]
    return summary


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    system = {
        "hostname": run(["hostname"]),
        "whoami": run(["whoami"]),
        "pwd": run(["pwd"], cwd=REPO_ROOT),
        "git_root": run(["git", "rev-parse", "--show-toplevel"], cwd=REPO_ROOT),
        "git_status": run(["git", "status", "--short"], cwd=REPO_ROOT),
        "os_release": Path("/etc/os-release").read_text(encoding="utf-8", errors="replace")
        if Path("/etc/os-release").exists()
        else "",
        "uname": run(["uname", "-a"]),
        "machine": platform.machine(),
        "python3": run(["python3", "--version"]),
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], timeout=15),
        "sudo_noninteractive": run(["sudo", "-n", "true"], timeout=5),
    }

    commands = {
        name: command_info(name, args)
        for name, args in {
            "codex": ["--version"],
            "magick": ["--version"],
            "convert": ["--version"],
            "heif-convert": ["--version"],
            "exiftool": ["-ver"],
            "ffmpeg": ["-version"],
            "isaac-sim.sh": ["--help"],
            "isaaclab.sh": ["--help"],
            "usdcat": ["--version"],
            "usdview": ["--help"],
            "FreeCAD": ["--version"],
            "freecadcmd": ["--version"],
            "FreeCADCmd": ["--version"],
            "blender": ["--version"],
            "openscad": ["--version"],
            "colmap": ["--version"],
            "meshlabserver": ["-h"],
            "cloudcompare": ["-h"],
            "uv": ["--version"],
        }.items()
    }
    commands["codex_mcp_list"] = run(["codex", "mcp", "list"], timeout=20) if commands["codex"]["available"] else {}
    commands["codex_mcp_help"] = run(["codex", "mcp", "--help"], timeout=20) if commands["codex"]["available"] else {}

    project_python = REPO_ROOT / ".venv/bin/python"
    isaac_python = REPO_ROOT / ".venv_issac/bin/python"
    python_modules = {
        "system_python3": inspect_python_modules("python3"),
    }
    if project_python.exists():
        python_modules["project_venv"] = inspect_python_modules(str(project_python))
    if isaac_python.exists():
        python_modules["isaac_venv"] = inspect_python_modules(str(isaac_python))

    searches = {
        "roots": [str(p) for p in BOUNDED_SEARCH_ROOTS],
        "usd_files": [],
        "camera_related_python": [],
        "isaac_readmes": [],
    }
    for root in BOUNDED_SEARCH_ROOTS:
        searches["usd_files"].extend(bounded_find(root, (".usd", ".usda"), max_results=80))
        searches["camera_related_python"].extend(
            [p for p in bounded_find(root, (".py",), max_results=200) if "camera" in Path(p).name.lower()]
        )
        searches["isaac_readmes"].extend(
            [p for p in bounded_find(root, (".md",), max_results=200) if "isaac" in str(p).lower()]
        )

    audit = {
        "adaptive_contract": json.loads(Path("/tmp/scene_reconstruction_contract.json").read_text())
        if Path("/tmp/scene_reconstruction_contract.json").exists()
        else {},
        "system": system,
        "commands": commands,
        "python_modules": python_modules,
        "photos": photo_summary(),
        "bounded_searches": searches,
        "route_decision": {
            "selected_workspace": str(OUT),
            "reason": (
                "The adaptive script preferred the Isaac MCP scratch workspace, but the user explicitly required "
                "the current repository scene_reconstruction directory and project docs mark this repository as source of truth."
            ),
        },
    }
    (AUDIT / "environment_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    def status(name: str, ok: bool) -> str:
        return "available" if ok else "missing"

    matrix_rows = [
        ("Isaac Sim", status("Isaac Sim", isaac_python.exists()), "from .venv_issac / project docs", "required", "use existing .venv_issac"),
        ("ALOHA USD", status("ALOHA USD", bool(searches["usd_files"])), "bounded search", "required", "reuse read-only base USD"),
        ("OpenUSD Python", status("OpenUSD", python_modules.get("isaac_venv", {}).get("pxr", "").startswith("OK")), ".venv_issac", "required", "use Isaac Python"),
        ("FreeCAD", status("FreeCAD", commands["FreeCAD"]["available"] or commands["FreeCADCmd"]["available"] or commands["freecadcmd"]["available"]), "PATH/module probe", "recommended", "fallback to proxy CAD/USDA"),
        ("FreeCADCmd", status("FreeCADCmd", commands["FreeCADCmd"]["available"] or commands["freecadcmd"]["available"]), "PATH probe", "recommended", "not required for first pass"),
        ("HEIC conversion", status("HEIC", commands["heif-convert"]["available"] or commands["magick"]["available"] or commands["ffmpeg"]["available"]), "PATH probe", "conditional", "use JPEG/MOV extracts if needed"),
        ("EXIF read", status("EXIF", commands["exiftool"]["available"]), "PATH probe", "recommended", "fallback to PIL metadata"),
        ("COLMAP", status("COLMAP", commands["colmap"]["available"]), "PATH probe", "optional", "not selected unless photos support SfM"),
        ("Context7 MCP", "check codex mcp list", "codex mcp list", "installed per user", "do not reinstall"),
        ("USD Code MCP", "not installed by default", "not required", "recommended", "skip; use official Isaac MCP and OpenUSD Python"),
        ("FreeCAD MCP", "not installed by default", "not required", "optional", "skip; use reproducible scripts"),
    ]

    md = [
        "# Environment Audit",
        "",
        "## Workspace Decision",
        "",
        f"- Selected output workspace: `{OUT}`",
        "- The adaptive-localization helper suggested `/home/eii/isaac_mcp_setup/aloha_project`, but that is only an Isaac MCP scratch workspace. The user explicitly required the current repository `scene_reconstruction/` directory, so outputs stay here.",
        "",
        "## System",
        "",
        f"- Hostname: `{system['hostname']['stdout']}`",
        f"- User: `{system['whoami']['stdout']}`",
        f"- Git root: `{system['git_root']['stdout']}`",
        f"- CPU architecture: `{system['machine']}`",
        f"- Python: `{system['python3']['stdout']}`",
        f"- GPU: `{system['nvidia_smi']['stdout'] or system['nvidia_smi']['stderr']}`",
        f"- Noninteractive sudo exit: `{system['sudo_noninteractive']['returncode']}`",
        "",
        "## Capability Matrix",
        "",
        "| Capability | Current status | Version/evidence | Needed | Plan |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in matrix_rows:
        md.append("| " + " | ".join(row) + " |")
    md.extend(
        [
            "",
            "## Route Decision",
            "",
            "- Selected route: Route C first, OpenUSD/Isaac proxy geometry.",
            "- Reason: the task needs a credible spatial model now, while FreeCAD is not guaranteed and first-pass dimensions include estimates. Proxy geometry keeps all parameters centralized and can later be exported or migrated to FreeCAD/OpenSCAD.",
            "- COLMAP is not selected for this pass. Thin metal frame, repeated profiles, reflective surfaces, and limited scale constraints make SfM unreliable as a source of truth.",
            "",
            "## Raw Audit Files",
            "",
            "- `scene_reconstruction/audit/environment_audit.json`",
        ]
    )
    (REPORTS / "environment_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"audit_json": str(AUDIT / "environment_audit.json"), "report": str(REPORTS / "environment_audit.md")}, indent=2))


if __name__ == "__main__":
    main()
