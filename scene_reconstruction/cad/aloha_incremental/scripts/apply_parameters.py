#!/usr/bin/env python3
"""Create the Iteration 000 FreeCAD reference file."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
WORKDIR = ROOT / "scene_reconstruction" / "cad" / "aloha_incremental"
ITER_DIR = WORKDIR / "iterations" / "iter_000_reference"
FREECAD_SCRIPT = WORKDIR / "scripts" / "_freecad_build_iter000.py"


def main() -> int:
    ITER_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "/snap/bin/freecad.cmd",
        "-c",
        f"exec(open('{FREECAD_SCRIPT}').read())",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKDIR / "scripts") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    (ITER_DIR / "freecad_build.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (ITER_DIR / "freecad_build.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        return result.returncode
    manifest = {
        "iteration": "iter_000_reference",
        "fcstd": str((ITER_DIR / "iter_000_reference.FCStd").relative_to(ROOT)),
        "metadata": str((ITER_DIR / "bbox_and_dimensions.json").relative_to(ROOT)),
    }
    (ITER_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
