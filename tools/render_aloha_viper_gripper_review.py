#!/usr/bin/env python3
"""Run the audited CAD gripper Blender renderer reproducibly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from PIL import Image


def validate_render_outputs(output_root: Path) -> dict[str, object]:
    """Reject Blender's zero exit code unless all expected products exist."""
    metadata_path = output_root.resolve() / "render_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"render metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    captures = metadata.get("captures", [])
    if metadata.get("capture_count") != 8 or len(captures) != 8:
        raise RuntimeError(
            "expected 8 captures, found "
            f"{metadata.get('capture_count')} / {len(captures)}"
        )
    raw_paths = [Path(record["raw_path"]) for record in captures]
    if len(set(raw_paths)) != 8:
        raise RuntimeError("expected 8 unique raw screenshot paths")
    for path in raw_paths:
        if not path.is_file():
            raise RuntimeError(f"raw screenshot is missing: {path}")
        with Image.open(path) as image:
            if image.size != (1280, 900):
                raise RuntimeError(
                    f"unexpected screenshot resolution for {path}: "
                    f"{image.size}"
                )
            image.verify()
    return {"capture_count": len(captures), "resolution": [1280, 900]}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    blender_script = (
        repository_root
        / "tools"
        / "aloha1_mapping"
        / "render_aloha_viper_gripper_blender.py"
    )
    command = [
        str(args.blender.resolve(strict=True)),
        "--background",
        "--python",
        str(blender_script.resolve(strict=True)),
        "--",
        "--input",
        str(args.input.resolve(strict=True)),
        "--output-root",
        str(args.output_root.resolve()),
    ]
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=repository_root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        print(f"Blender render failed; inspect {args.log.resolve()}")
        return completed.returncode
    try:
        validation = validate_render_outputs(args.output_root)
    except RuntimeError as exc:
        print(f"Blender output validation failed: {exc}")
        print(f"Inspect Blender log: {args.log.resolve()}")
        return 1
    print(f"Blender render log: {args.log.resolve()}")
    print(
        "Blender render metadata: "
        f"{(args.output_root / 'render_metadata.json').resolve()}"
    )
    print(f"Validated render outputs: {validation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
