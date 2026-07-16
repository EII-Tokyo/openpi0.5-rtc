#!/usr/bin/env python3
"""Run the complete Iteration 000 generation pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "scene_reconstruction" / "cad" / "aloha_incremental" / "scripts"


def _run(name: str) -> None:
    cmd = [sys.executable, str(SCRIPTS / name)]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    _run("apply_parameters.py")
    _run("render_standard_views.py")
    _run("export_step.py")
    _run("audit_model.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
