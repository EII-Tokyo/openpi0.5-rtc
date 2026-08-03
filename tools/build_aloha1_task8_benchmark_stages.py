#!/usr/bin/env python3
"""Build deterministic isolated multi-environment Task 8 benchmark Stages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_REPORT = ROOT / "reports/aloha1_mapping/aloha1_task8_collider_lod_candidate.json"
BOUNDS_REPORT = ROOT / (
    ".codex/artifacts/20260803-aloha1-task8-lightweight/collider_lod_validation/"
    "static_fidelity_01_coverage.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / (
    "assets/Trossen/ALOHA1/1.0/diagnostics/task8_collider_lod/1.0/"
    "benchmark_stages"
)
DEFAULT_MANIFEST = (
    ROOT / "reports/aloha1_mapping/aloha1_task8_benchmark_stage_manifest.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _derive_spacing(bounds: dict[str, float]) -> float:
    width = float(bounds["xmax"]) - float(bounds["xmin"])
    if width <= 0.0:
        raise ValueError("invalid frozen collider x bounds")
    return 2.0 * width


def _stage_text(source: Path, environment_count: int, spacing_m: float) -> str:
    source = source.resolve(strict=True)
    environments = []
    for index in range(environment_count):
        x = index * spacing_m
        children = [
                f"""        def Xform "{name}" (
            prepend references = @{source}@</World/{name}>
        )
        {{
        }}"""
            for name in ("follower_left", "follower_right", "environment")
        ]
        environments.append(
            f"""    def Xform "env_{index:03d}"
    {{
        double3 xformOp:translate = ({x:.17g}, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
{chr(10).join(children)}
    }}"""
        )
    return f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def PhysicsScene "PhysicsScene" (
        prepend references = @{source}@</World/PhysicsScene>
    )
    {{
    }}
{chr(10).join(environments)}
}}
"""


def _write_set(
    output_root: Path,
    profiles: dict[str, Path],
    environment_counts: list[int],
    spacing_m: float,
) -> dict[str, dict[str, Path]]:
    outputs: dict[str, dict[str, Path]] = {}
    for profile, source in profiles.items():
        outputs[profile] = {"1": source.resolve(strict=True)}
        for count in environment_counts:
            if count == 1:
                continue
            destination = output_root / profile / f"environments_{count}.usda"
            destination.parent.mkdir(parents=True, exist_ok=True)
            text = _stage_text(source, count, spacing_m)
            if destination.exists() and destination.read_text(encoding="utf-8") != text:
                raise RuntimeError(f"refusing to overwrite changed benchmark Stage: {destination}")
            destination.write_text(text, encoding="utf-8")
            outputs[profile][str(count)] = destination
    return outputs


def build(output_root: Path, manifest_path: Path) -> dict[str, Any]:
    candidate = json.loads(CANDIDATE_REPORT.read_text(encoding="utf-8"))
    coverage = json.loads(BOUNDS_REPORT.read_text(encoding="utf-8"))
    x_min = min(record["aabb_min_world_m"][0] for record in coverage["colliders"])
    x_max = max(record["aabb_max_world_m"][0] for record in coverage["colliders"])
    bounds = {"xmin": x_min, "xmax": x_max}
    spacing_m = _derive_spacing(bounds)
    profiles = {
        profile: Path(candidate["layers"][profile]["absolute_path"])
        for profile in ("fidelity_profile", "throughput_profile")
    }
    environment_counts = [1, 2, 4]
    artifact_root = ROOT / (
        ".codex/artifacts/20260803-aloha1-task8-lightweight/"
        "benchmark_stage_determinism"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory(prefix="run_a_", dir=artifact_root) as run_a_dir,
        tempfile.TemporaryDirectory(prefix="run_b_", dir=artifact_root) as run_b_dir,
    ):
        run_a = _write_set(Path(run_a_dir), profiles, environment_counts, spacing_m)
        run_b = _write_set(Path(run_b_dir), profiles, environment_counts, spacing_m)
        deterministic = all(
            _sha256(run_a[profile][str(count)])
            == _sha256(run_b[profile][str(count)])
            for profile in profiles
            for count in (2, 4)
        )
        if not deterministic:
            raise RuntimeError("multi-environment Stage authoring is not deterministic")
    outputs = _write_set(output_root.resolve(), profiles, environment_counts, spacing_m)
    stage_records = {
        profile: {
            count: {
                "absolute_path": str(path.resolve()),
                "sha256": _sha256(path),
                "environment_count": int(count),
            }
            for count, path in records.items()
        }
        for profile, records in outputs.items()
    }
    manifest = {
        "schema_version": 1,
        "status": "PASS_STATIC_AUTHORING",
        "classification": "TASK8_ISOLATED_BENCHMARK_STAGES_NOT_PROMOTED",
        "source_candidate_report": {
            "absolute_path": str(CANDIDATE_REPORT.resolve()),
            "sha256": _sha256(CANDIDATE_REPORT),
        },
        "spacing": {
            "x_bounds_world_m": bounds,
            "x_extent_m": x_max - x_min,
            "spacing_m": spacing_m,
            "derivation": "2 * frozen full-robot collider x extent",
            "source_coverage_report": str(BOUNDS_REPORT.resolve()),
            "source_coverage_sha256": _sha256(BOUNDS_REPORT),
        },
        "stages": stage_records,
        "two_fresh_directory_determinism": "PASS",
        "physics_scene_count_authored_per_stage": 1,
        "environment_members": ["follower_left", "follower_right", "environment"],
        "candidate_promoted": False,
        "final_or_default_asset_modified": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = build(args.output_root, args.manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "spacing_m": manifest["spacing"]["spacing_m"],
                "manifest": str(args.manifest.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
