"""Geometry helpers for the isolated ALOHA table/support alignment gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table_center_from_top(*, top_z_m: float, thickness_m: float) -> float:
    """Return a table center Z for a requested top plane and thickness."""
    if thickness_m <= 0.0:
        raise ValueError("table thickness must be finite and positive")
    return float(top_z_m) - float(thickness_m) / 2.0


def alignment_metrics(
    *,
    table_top_z_m: float,
    support_contact_z_m: float,
    tolerance_m: float = 1.0e-6,
) -> dict[str, float | str]:
    """Classify the signed support-contact-plane to tabletop gap."""
    if tolerance_m < 0.0:
        raise ValueError("tolerance must be non-negative")
    signed_gap = float(support_contact_z_m) - float(table_top_z_m)
    if abs(signed_gap) <= tolerance_m:
        classification = "ALIGNED_WITHIN_TOLERANCE"
    elif signed_gap > 0.0:
        classification = "FLOATING_SUPPORT_ABOVE_TABLE"
    else:
        classification = "SUPPORT_PENETRATES_TABLE"
    return {
        "table_top_z_m": float(table_top_z_m),
        "support_contact_z_m": float(support_contact_z_m),
        "signed_gap_m": signed_gap,
        "tolerance_m": float(tolerance_m),
        "classification": classification,
    }


def follower_articulation_roots(paths: list[str]) -> list[str]:
    """Return only the two follower robot articulation-root paths."""
    prefixes = ("/World/follower_left/", "/World/follower_right/")
    return sorted(path for path in paths if path.startswith(prefixes))


def support_stack_metrics(
    *,
    table_top_z_m: float,
    support_bottom_z_m: float,
    support_top_z_m: float,
    robot_base_bottom_z_m: float,
    tolerance_m: float = 1.0e-6,
) -> dict[str, float | str]:
    """Measure the table-to-rail-to-robot-base vertical stack."""
    table_gap = float(support_bottom_z_m) - float(table_top_z_m)
    base_gap = float(robot_base_bottom_z_m) - float(support_top_z_m)
    aligned = abs(table_gap) <= tolerance_m and abs(base_gap) <= tolerance_m
    return {
        "table_top_z_m": float(table_top_z_m),
        "support_bottom_z_m": float(support_bottom_z_m),
        "support_top_z_m": float(support_top_z_m),
        "robot_base_bottom_z_m": float(robot_base_bottom_z_m),
        "table_to_support_gap_m": table_gap,
        "support_to_robot_base_gap_m": base_gap,
        "tolerance_m": float(tolerance_m),
        "classification": (
            "STACK_ALIGNED" if aligned else "STACK_NOT_ALIGNED"
        ),
    }


def _relative_asset_path(path: Path, owner_layer: Path) -> str:
    return os.path.relpath(
        path.resolve(),
        owner_layer.resolve().parent,
    ).replace(os.sep, "/")


def _nested_over_block(prim_path: str, body: str) -> str:
    names = [name for name in prim_path.split("/") if name]
    if not names:
        raise ValueError("prim path must not be empty")
    lines: list[str] = []
    for depth, name in enumerate(names):
        indent = "    " * depth
        lines.extend((f'{indent}over "{name}"', f"{indent}{{"))
    body_indent = "    " * len(names)
    lines.extend(f"{body_indent}{line}" for line in body.splitlines())
    lines.extend(
        f'{"    " * depth}}}' for depth in reversed(range(len(names)))
    )
    return "\n".join(lines)


def build_alignment_diagnostic(
    *,
    source_stage: Path,
    output_dir: Path,
    table_prim_path: str,
    table_dimensions_m: tuple[float, float, float],
    target_table_top_z_m: float,
    support_contact_z_m: float,
) -> dict[str, object]:
    """Build a root layer plus stronger table-alignment diagnostic layer."""
    source_stage = source_stage.resolve(strict=True)
    output_dir = output_dir.resolve()
    config_dir = output_dir / "configuration"
    config_dir.mkdir(parents=True, exist_ok=True)
    root_path = output_dir / "aloha1_table_support_aligned_workcell.usda"
    config_path = config_dir / "aloha1_tabletop_world_zero.usda"
    report_path = output_dir / "aloha1_table_support_alignment_manifest.json"

    source_hash_before = sha256_file(source_stage)
    thickness_m = float(table_dimensions_m[2])
    center_z_m = table_center_from_top(
        top_z_m=target_table_top_z_m,
        thickness_m=thickness_m,
    )
    alignment = alignment_metrics(
        table_top_z_m=target_table_top_z_m,
        support_contact_z_m=support_contact_z_m,
    )

    config_body = "\n".join(
        (
            f"double3 xformOp:scale = ({table_dimensions_m[0]}, "
            f"{table_dimensions_m[1]}, {thickness_m})",
            f"double3 xformOp:translate = (0, 0, {center_z_m})",
            'uniform token[] xformOpOrder = ["xformOp:translate", '
            '"xformOp:scale"]',
        )
    )
    config_text = "\n".join(
        (
            "#usda 1.0",
            "(",
            "    metersPerUnit = 1",
            '    upAxis = "Z"',
            ")",
            "",
            _nested_over_block(table_prim_path, config_body),
            "",
        )
    )
    config_path.write_text(config_text, encoding="utf-8")

    source_asset = _relative_asset_path(source_stage, root_path)
    config_asset = _relative_asset_path(config_path, root_path)
    root_text = "\n".join(
        (
            "#usda 1.0",
            "(",
            '    defaultPrim = "World"',
            "    metersPerUnit = 1",
            "    subLayers = [",
            f"        @{config_asset}@,",
            f"        @{source_asset}@",
            "    ]",
            '    upAxis = "Z"',
            ")",
            "",
        )
    )
    root_path.write_text(root_text, encoding="utf-8")

    source_hash_after = sha256_file(source_stage)
    report: dict[str, object] = {
        "schema_version": 1,
        "status": (
            "PASS"
            if source_hash_before == source_hash_after
            and alignment["classification"] == "ALIGNED_WITHIN_TOLERANCE"
            else "FAIL"
        ),
        "scope": "DIAGNOSTIC_ONLY_NOT_FINAL_ASSET",
        "source_stage": {
            "path": str(source_stage),
            "sha256_before": source_hash_before,
            "sha256_after": source_hash_after,
        },
        "diagnostic_stage": {
            "path": str(root_path),
            "sha256": sha256_file(root_path),
            "default_prim": "/World",
            "sublayers": [str(config_path), str(source_stage)],
        },
        "configuration_layer": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "modified_prim": table_prim_path,
        },
        "table": {
            "prim_path": table_prim_path,
            "dimensions_m": list(table_dimensions_m),
            "target_top_z_m": float(target_table_top_z_m),
            "target_center_z_m": center_z_m,
        },
        "alignment": alignment,
        "boundaries": {
            "source_stage_modified": source_hash_before
            != source_hash_after,
            "support_geometry_modified": False,
            "robot_geometry_modified": False,
            "physics_or_collider_modified": False,
            "task8": "NOT_RUN",
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["manifest_path"] = str(report_path)
    return report
