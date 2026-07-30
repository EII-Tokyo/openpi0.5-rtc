#!/usr/bin/env python3
"""Validate the isolated ALOHA tabletop/support alignment composition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pxr import Usd
from pxr import UsdGeom
from pxr import UsdPhysics

from tools.aloha1_mapping.table_support_alignment import follower_articulation_roots
from tools.aloha1_mapping.table_support_alignment import support_stack_metrics

ROOT = Path(__file__).resolve().parents[1]
SOURCE_STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/"
    "1.0/aloha1_signal_correspondence_workcell.usda"
)
DIAGNOSTIC_STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "table_support_alignment/1.0/"
    "aloha1_table_support_aligned_workcell.usda"
)
REPORT_PATH = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_table_support_alignment_validation.json"
)
TABLE_PATH = "/World/environment/worldBody/user_confirmed_table"
WORLD_BODY_PATH = "/World/environment/worldBody"
LEFT_BASE_PATH = (
    "/World/follower_left/vx300s_left/follower_left_base_link"
)
RIGHT_BASE_PATH = (
    "/World/follower_right/vx300s_right/follower_right_base_link"
)
SUPPORT_PATHS = {
    "follower_left": (
        "/World/environment/worldBody/__14",
        "/World/environment/worldBody/__18",
    ),
    "follower_right": (
        "/World/environment/worldBody/__20",
        "/World/environment/worldBody/__23",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aabb(cache: UsdGeom.BBoxCache, prim: Usd.Prim) -> dict[str, list[float]]:
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return {
        "minimum": [float(value) for value in aligned.GetMin()],
        "maximum": [float(value) for value in aligned.GetMax()],
    }


def _articulation_roots(stage: Usd.Stage) -> list[str]:
    return sorted(
        str(prim.GetPath())
        for prim in Usd.PrimRange.Stage(stage)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    )


def _world_body_children(stage: Usd.Stage) -> list[dict[str, Any]]:
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
        useExtentsHint=False,
    )
    parent = stage.GetPrimAtPath(WORLD_BODY_PATH)
    records = []
    for prim in parent.GetChildren():
        if not prim.IsActive():
            continue
        bounds = _aabb(cache, prim)
        records.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "aabb_world_m": bounds,
            }
        )
    return records


def _support_stack(
    *,
    stage: Usd.Stage,
    cache: UsdGeom.BBoxCache,
    table_top_z_m: float,
    base_path: str,
    support_paths: tuple[str, ...],
) -> dict[str, Any]:
    support_bounds = [
        {"path": path, "aabb_world_m": _aabb(cache, stage.GetPrimAtPath(path))}
        for path in support_paths
    ]
    support_bottom = min(
        record["aabb_world_m"]["minimum"][2]
        for record in support_bounds
    )
    support_top = max(
        record["aabb_world_m"]["maximum"][2]
        for record in support_bounds
    )
    base_bounds = _aabb(cache, stage.GetPrimAtPath(base_path))
    return {
        "base_path": base_path,
        "base_aabb_world_m": base_bounds,
        "support_members": support_bounds,
        "metrics": support_stack_metrics(
            table_top_z_m=table_top_z_m,
            support_bottom_z_m=support_bottom,
            support_top_z_m=support_top,
            robot_base_bottom_z_m=base_bounds["minimum"][2],
        ),
    }


def validate(
    *,
    source_path: Path,
    diagnostic_path: Path,
) -> dict[str, Any]:
    source_path = source_path.resolve(strict=True)
    diagnostic_path = diagnostic_path.resolve(strict=True)
    source_hash_before = sha256_file(source_path)
    source = Usd.Stage.Open(str(source_path))
    diagnostic = Usd.Stage.Open(str(diagnostic_path))
    if source is None or diagnostic is None:
        raise RuntimeError("failed to open source or diagnostic USD Stage")

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
        useExtentsHint=False,
    )
    source_table = _aabb(cache, source.GetPrimAtPath(TABLE_PATH))
    cache.Clear()
    diagnostic_table = _aabb(
        cache,
        diagnostic.GetPrimAtPath(TABLE_PATH),
    )
    source_roots = _articulation_roots(source)
    diagnostic_roots = _articulation_roots(diagnostic)
    source_follower_roots = follower_articulation_roots(source_roots)
    diagnostic_follower_roots = follower_articulation_roots(
        diagnostic_roots
    )
    source_hash_after = sha256_file(source_path)
    table_top = diagnostic_table["maximum"][2]
    table_center = (
        diagnostic_table["minimum"][2] + diagnostic_table["maximum"][2]
    ) / 2.0

    world_body_records = _world_body_children(diagnostic)
    zero_plane_members = [
        record
        for record in world_body_records
        if record["path"] != TABLE_PATH
        and min(
            abs(record["aabb_world_m"]["minimum"][2]),
            abs(record["aabb_world_m"]["maximum"][2]),
        )
        <= 1.0e-6
    ]
    support_stacks = {
        "follower_left": _support_stack(
            stage=diagnostic,
            cache=cache,
            table_top_z_m=table_top,
            base_path=LEFT_BASE_PATH,
            support_paths=SUPPORT_PATHS["follower_left"],
        ),
        "follower_right": _support_stack(
            stage=diagnostic,
            cache=cache,
            table_top_z_m=table_top,
            base_path=RIGHT_BASE_PATH,
            support_paths=SUPPORT_PATHS["follower_right"],
        ),
    }
    gates = {
        "source_stage_hash_unchanged": source_hash_before
        == source_hash_after,
        "default_prim_is_world": str(
            diagnostic.GetDefaultPrim().GetPath()
        )
        == "/World",
        "table_top_is_world_zero": abs(table_top) <= 1.0e-9,
        "table_center_matches_thickness": abs(table_center + 0.0075)
        <= 1.0e-9,
        "support_has_zero_plane_members": bool(zero_plane_members),
        "articulation_roots_unchanged": source_roots == diagnostic_roots,
        "two_follower_articulations": len(diagnostic_follower_roots) == 2,
        "left_table_support_base_stack_aligned": support_stacks[
            "follower_left"
        ]["metrics"]["classification"]
        == "STACK_ALIGNED",
        "right_table_support_base_stack_aligned": support_stacks[
            "follower_right"
        ]["metrics"]["classification"]
        == "STACK_ALIGNED",
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "DIAGNOSTIC_ONLY_NOT_FINAL_ASSET",
        "runtime": {
            "usd_version": ".".join(str(value) for value in Usd.GetVersion()),
            "method": "LOCAL_ISAAC_SIM_5_1_USD_24_05_COMPOSITION_READBACK",
        },
        "source_stage": {
            "path": str(source_path),
            "sha256_before": source_hash_before,
            "sha256_after": source_hash_after,
            "table_aabb_world_m": source_table,
            "articulation_roots": source_roots,
            "follower_articulation_roots": source_follower_roots,
        },
        "diagnostic_stage": {
            "path": str(diagnostic_path),
            "sha256": sha256_file(diagnostic_path),
            "default_prim": str(diagnostic.GetDefaultPrim().GetPath()),
            "sublayers": list(
                diagnostic.GetRootLayer().subLayerPaths
            ),
            "table_aabb_world_m": diagnostic_table,
            "articulation_roots": diagnostic_roots,
            "follower_articulation_roots": diagnostic_follower_roots,
        },
        "alignment": {
            "old_table_top_z_m": source_table["maximum"][2],
            "new_table_top_z_m": table_top,
            "removed_positive_gap_m": -source_table["maximum"][2],
            "zero_plane_support_members": zero_plane_members,
            "support_stacks": support_stacks,
        },
        "gates": gates,
        "boundaries": {
            "table_translation_only": True,
            "source_stage_modified": False,
            "support_geometry_modified": False,
            "robot_geometry_modified": False,
            "collider_or_physics_parameters_modified": False,
            "task8": "NOT_RUN",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-stage", type=Path, default=SOURCE_STAGE)
    parser.add_argument(
        "--diagnostic-stage",
        type=Path,
        default=DIAGNOSTIC_STAGE,
    )
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate(
        source_path=args.source_stage,
        diagnostic_path=args.diagnostic_stage,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
