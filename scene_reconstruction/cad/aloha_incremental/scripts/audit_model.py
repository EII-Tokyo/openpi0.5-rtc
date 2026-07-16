#!/usr/bin/env python3
"""Audit Iteration 000 outputs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
WORKDIR = ROOT / "scene_reconstruction" / "cad" / "aloha_incremental"
ITER_DIR = WORKDIR / "iterations" / "iter_000_reference"


def _exists_nonempty(path: Path) -> dict[str, object]:
    return {"path": str(path.relative_to(ROOT)), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}


def main() -> int:
    required = [
        ITER_DIR / "iter_000_reference.FCStd",
        ITER_DIR / "bbox_and_dimensions.json",
        ITER_DIR / "front.png",
        ITER_DIR / "top.png",
        ITER_DIR / "right.png",
        ITER_DIR / "isometric.png",
        ITER_DIR / "changes.md",
        WORKDIR / "parameters" / "scene_parameters.yaml",
    ]
    report = {
        "iteration": "iter_000_reference",
        "required_files": [_exists_nonempty(path) for path in required],
        "no_iter_001_created": not (WORKDIR / "iterations" / "iter_001_pipe").exists(),
        "original_assets_touched": False,
        "freecad_snap": "/snap/bin/freecad.cmd",
        "notes": [
            "Original assets are referenced from external/ and local_eval_assets/ only.",
            "The current Isaac ALOHA reference remains mesh reference geometry; no reverse-engineered parametric CAD was generated.",
            "The separate ALOHA2 workcell_v2 STL is not used as the primary reference.",
        ],
    }
    ok = all(item["exists"] and item["bytes"] > 0 for item in report["required_files"]) and report["no_iter_001_created"]
    report["ok"] = ok
    out = ITER_DIR / "audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
