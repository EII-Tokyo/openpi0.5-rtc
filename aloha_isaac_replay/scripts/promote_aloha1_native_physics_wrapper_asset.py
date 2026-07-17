from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "assets/isaac/original_stationary_aloha"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "assets/isaac/aloha1_native_physics_wrapper"


def _rel(path: str | Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _relative_sublayer(output_path: Path, source_path: Path) -> str:
    return os.path.relpath(source_path.resolve(), start=output_path.parent.resolve())


def _write_wrapper(output_path: Path, source_layer: Path, default_prim_name: str) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sublayer = _relative_sublayer(output_path, source_layer)
    output_path.write_text(
        "\n".join(
            [
                "#usda 1.0",
                "(",
                f'    defaultPrim = "{default_prim_name}"',
                "    subLayers = [",
                f"        @{sublayer}@",
                "    ]",
                ")",
                "",
            ]
        )
    )
    return {
        "path": _rel(output_path),
        "source_layer": _rel(source_layer),
        "sublayer": sublayer,
        "default_prim_name": default_prim_name,
        "authoring_gate": {
            "source_layer_exists": source_layer.exists(),
            "wrapper_written": output_path.exists(),
            "uses_relative_sublayer": not os.path.isabs(sublayer),
        },
    }


def _write_readme(output_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# ALOHA1 Native Physics Wrapper Candidate",
        "",
        "This directory contains thin USD wrapper stages around the generated ALOHA1 physics layers.",
        "",
        "The source importer output under `../original_stationary_aloha/` remains unchanged. These wrappers are a candidate asset entry point for validation, replay, and future controller work.",
        "",
        "## Files",
        "",
        "- `aloha1_left.usda`: left ALOHA1 physics-layer wrapper.",
        "- `aloha1_right.usda`: right ALOHA1 physics-layer wrapper.",
        "- `manifest.json`: generated provenance and validation summary.",
        "",
        "## Known Limits",
        "",
        "- This is not yet a final production robot asset.",
        "- It intentionally preserves the generated physics layers instead of flattening or rewriting them.",
        "- Some visual reference warnings may still appear in Isaac logs; runtime articulation validation is the current acceptance gate.",
        "- The next required gates are DOF limits/drives, pose replay, collision/contact behavior, and controller integration.",
        "",
        "## Current Authoring Gate",
        "",
        "| Asset | Source exists | Wrapper written | Relative sublayer | Gate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key in ("left", "right"):
        item = manifest[key]
        gate = item["authoring_gate"]
        gate_pass = all(gate.values())
        lines.append(
            f"| {key} | {gate['source_layer_exists']} | {gate['wrapper_written']} | {gate['uses_relative_sublayer']} | "
            f"{'PASS' if gate_pass else 'FAIL'} |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote verified ALOHA1 physics-layer wrappers into a tracked candidate asset directory.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    left_physics = source_root / "generated/configuration/vx300s_left_physics.usd"
    right_physics = source_root / "generated/configuration/vx300s_right_physics.usd"
    missing = [path for path in (left_physics, right_physics) if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))

    manifest: dict[str, object] = {
        "schema_version": 1,
        "asset_status": "candidate",
        "source_root": _rel(source_root),
        "source_layers": {
            "left_physics": _rel(left_physics),
            "right_physics": _rel(right_physics),
        },
        "principles": [
            "source importer output remains read-only",
            "wrapper sublayers are relative paths",
            "runtime articulation validation is required before controller use",
        ],
    }
    manifest["left"] = _write_wrapper(output_dir / "aloha1_left.usda", left_physics, "puppet_left_vx300s")
    manifest["right"] = _write_wrapper(output_dir / "aloha1_right.usda", right_physics, "puppet_right_vx300s")
    manifest["overall_authoring_gate"] = all(
        all(manifest[key]["authoring_gate"].values())  # type: ignore[index]
        for key in ("left", "right")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    _write_readme(output_dir, manifest)
    print(json.dumps({"manifest": _rel(manifest_path), "overall_authoring_gate": manifest["overall_authoring_gate"]}, ensure_ascii=False))
    return 0 if manifest["overall_authoring_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
