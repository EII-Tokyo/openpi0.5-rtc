#!/usr/bin/env python3
"""Export editable Iteration 000 solids to STEP.

The ALOHA reference is mesh geometry, so this exporter intentionally exports
only editable/reference solids such as the desktop plane and axes. It does not
pretend to convert ALOHA meshes into parametric STEP CAD.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
WORKDIR = ROOT / "scene_reconstruction" / "cad" / "aloha_incremental"
SCRIPT = WORKDIR / "scripts" / "_freecad_export_step_iter000.py"


def main() -> int:
    cmd = ["/snap/bin/freecad.cmd", "-c", f"exec(open('{SCRIPT}').read())"]
    env = os.environ.copy()
    result = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
