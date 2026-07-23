#!/usr/bin/env python3
"""Create an A17 source-joint to clean-link articulation mapping plan.

This is a preflight artifact only.  It does not author joints, articulation
roots, drives, physics schemas, or runtime controllers into the clean /aloha
stage.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

from pxr import Usd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aloha_isaac_rebuild.scripts.create_aloha_stationary_style_rebuild_stages import _source_components  # noqa: E402

DEFAULT_CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml")
DEFAULT_OUTPUT = Path("aloha_isaac_rebuild/artifacts/validation/a17_clean_articulation_mapping_plan.json")


def _load_mapping_config(path: Path) -> dict:
    if not path.exists():
        return {"joint_mappings": []}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _source_to_clean_link_map(stage: Usd.Stage) -> dict[str, str]:
    mapping = {}
    for component in _source_components(stage):
        if component.assembly_link_path is not None:
            mapping[component.source_component_path] = component.assembly_link_path
    return mapping


def _canonical_joint_map(mapping_yaml: Path) -> dict[str, dict]:
    data = _load_mapping_config(mapping_yaml)
    result = {}
    joint_items = data.get("dof_mapping") or data.get("joint_mappings", [])
    for item in joint_items:
        canonical_name = str(item.get("canonical_name", ""))
        if canonical_name:
            result[canonical_name] = item
        isaac_name = str(item.get("isaac_dof_name", ""))
        source_joint = isaac_name.rsplit("/", maxsplit=1)[-1] if isaac_name else str(item.get("canonical_name", ""))
        if source_joint:
            result[source_joint] = item
        side = isaac_name.split("/", maxsplit=1)[0] if "/" in isaac_name else ""
        if side and source_joint in {"left_finger", "right_finger"}:
            result[f"{side}_{source_joint}"] = item
    return result


def _finite_override_number(override: dict, field: str) -> float:
    value = override[field]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"clean runtime mapping override requires finite numeric {field}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"clean runtime mapping override requires finite numeric {field}")
    return result


def apply_clean_runtime_mapping_override(record: dict, overrides: dict[str, dict]) -> dict:
    """Apply a validated clean-coordinate mapping while retaining source provenance."""
    canonical_mapping = record.get("canonical_mapping")
    if canonical_mapping is None:
        return record
    if not isinstance(canonical_mapping, dict):
        raise ValueError("canonical_mapping must be a mapping when present")

    result = deepcopy(record)
    source_mapping = deepcopy(canonical_mapping)
    result["source_canonical_mapping"] = source_mapping
    result["clean_runtime_mapping_override"] = None

    path = result.get("proposed_clean_joint_path")
    override = overrides.get(path)
    if override is None:
        return result
    if not isinstance(override, dict):
        raise ValueError(f"clean runtime mapping override for {path} must be a mapping")

    required_fields = ("sign", "offset", "scale", "unit", "rationale", "source")
    for field in required_fields:
        if field not in override:
            raise ValueError(f"clean runtime mapping override for {path} requires {field}")
    for field in ("rationale", "source"):
        value = override[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"clean runtime mapping override for {path} requires nonempty {field}")

    sign = _finite_override_number(override, "sign")
    offset = _finite_override_number(override, "offset")
    scale = _finite_override_number(override, "scale")
    if scale <= 0.0:
        raise ValueError(f"clean runtime mapping override for {path} requires positive scale")
    if override["unit"] != source_mapping.get("unit"):
        raise ValueError(f"clean runtime mapping override unit mismatch for {path}")

    try:
        lower_limit = float(result["lower_limit"])
        upper_limit = float(result["upper_limit"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"clean runtime mapping override for {path} requires numeric joint limits") from error
    if not math.isfinite(lower_limit) or not math.isfinite(upper_limit):
        raise ValueError(f"clean runtime mapping override for {path} requires finite joint limits")
    endpoints = (offset, offset + scale)
    tolerance = 1e-9
    if any(endpoint < lower_limit - tolerance or endpoint > upper_limit + tolerance for endpoint in endpoints):
        raise ValueError(f"clean runtime mapping override endpoints outside clean joint limits for {path}")

    effective_mapping = deepcopy(source_mapping)
    effective_mapping.update(
        {
            "sign": sign,
            "offset": offset,
            "scale": scale,
            "unit": override["unit"],
            "source": override["source"],
        }
    )
    result["canonical_mapping"] = effective_mapping
    result["clean_runtime_mapping_override"] = deepcopy(override)
    return result


def validate_clean_runtime_mapping_override_paths(
    records: list[dict], overrides: dict[str, dict]
) -> list[dict]:
    """Apply overrides and reject configured paths that do not match exactly one mapping record."""
    if not isinstance(overrides, dict):
        raise ValueError("clean_runtime_mapping_overrides must be a mapping")
    raw_match_counts = dict.fromkeys(overrides, 0)
    applied_override_counts = dict.fromkeys(overrides, 0)
    transformed_records = []
    for record in records:
        path = record.get("proposed_clean_joint_path")
        if path in raw_match_counts:
            raw_match_counts[path] += 1
        transformed = apply_clean_runtime_mapping_override(record, overrides)
        if path in applied_override_counts and transformed.get("clean_runtime_mapping_override") is not None:
            applied_override_counts[path] += 1
        transformed_records.append(transformed)
    invalid_counts = [
        (
            path,
            raw_match_counts[path],
            applied_override_counts[path],
        )
        for path in overrides
        if raw_match_counts[path] != 1 or applied_override_counts[path] != 1
    ]
    if invalid_counts:
        details = ", ".join(
            f"{path} (raw match count={raw_count}, applied override count={applied_count})"
            for path, raw_count, applied_count in sorted(invalid_counts)
        )
        raise ValueError(
            "clean runtime mapping overrides were not consumed exactly once; " + details
        )
    return transformed_records


def _joint_record(prim: Usd.Prim, source_to_clean: dict[str, str], canonical: dict[str, dict]) -> dict:
    joint_name = prim.GetName()
    relationships = {}
    for rel in prim.GetRelationships():
        targets = [str(target) for target in rel.GetTargets()]
        if targets:
            relationships[rel.GetName()] = targets
    attrs = {}
    for attr in prim.GetAuthoredAttributes():
        name = attr.GetName()
        if (
            name.startswith(("physics:", "drive:", "physxLimit:", "isaac:physics:"))
            or "limit" in name.lower()
            or "axis" in name.lower()
        ):
            attrs[name] = str(attr.Get())
    body0 = relationships.get("physics:body0", [])
    body1 = relationships.get("physics:body1", [])
    clean_body0 = [source_to_clean.get(path) for path in body0]
    clean_body1 = [source_to_clean.get(path) for path in body1]
    unmapped = [
        path
        for path in [*body0, *body1]
        if path not in source_to_clean
    ]
    joint_type = prim.GetTypeName()
    is_dof_joint = joint_type in {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}
    canonical_mapping = canonical.get(joint_name)
    limit_unit = None
    if joint_type == "PhysicsRevoluteJoint":
        limit_unit = "degrees_in_usd_authored_attributes; runtime_control_uses_radians_after_conversion"
    elif joint_type == "PhysicsPrismaticJoint":
        limit_unit = "meters"
    return {
        "source_joint_path": str(prim.GetPath()),
        "source_joint_name": joint_name,
        "proposed_clean_joint_path": f"/aloha/joints/{joint_name}",
        "joint_type": joint_type,
        "is_dof_joint": is_dof_joint,
        "source_body0": body0,
        "source_body1": body1,
        "clean_body0": clean_body0,
        "clean_body1": clean_body1,
        "unmapped_source_bodies": unmapped,
        "axis": attrs.get("physics:axis"),
        "lower_limit": attrs.get("physics:lowerLimit"),
        "upper_limit": attrs.get("physics:upperLimit"),
        "drive_attrs": {key: value for key, value in attrs.items() if key.startswith("drive:")},
        "other_joint_attrs": {key: value for key, value in attrs.items() if not key.startswith("drive:")},
        "axis_frame_semantics": "USD PhysicsJoint local joint frame; preserve source localPos/localRot evidence before authoring clean articulation",
        "limit_unit": limit_unit,
        "canonical_mapping": canonical_mapping,
        "canonical_dof_name": canonical_mapping.get("canonical_name") if canonical_mapping else None,
        "canonical_dataset_index": canonical_mapping.get("dataset_index") if canonical_mapping else None,
        "canonical_openpi_index": canonical_mapping.get("openpi_index") if canonical_mapping else None,
        "mapping_status": "MAPPED" if not unmapped and (body0 or body1) else "MAPPING_NEEDS_REVIEW",
    }


def create_mapping_plan(config_path: Path, output_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    clean_runtime_mapping_overrides = config.get("clean_runtime_mapping_overrides", {})
    source_usd = REPO_ROOT / config["source_aloha1_usd"]
    source_stage = Usd.Stage.Open(str(source_usd), load=Usd.Stage.LoadAll)
    if source_stage is None:
        raise RuntimeError(f"Could not open source stage: {source_usd}")

    source_to_clean = _source_to_clean_link_map(source_stage)
    canonical = _canonical_joint_map(REPO_ROOT / "configs/aloha/original_stationary_aloha_mapping.yaml")

    joint_records = []
    for prim in source_stage.Traverse():
        type_name = prim.GetTypeName() or ""
        if type_name not in {"PhysicsFixedJoint", "PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}:
            continue
        joint_records.append(_joint_record(prim, source_to_clean, canonical))
    joint_records = validate_clean_runtime_mapping_override_paths(
        joint_records, clean_runtime_mapping_overrides
    )

    dof_records = [record for record in joint_records if record["is_dof_joint"]]
    unmapped_records = [
        record
        for record in joint_records
        if record["unmapped_source_bodies"]
    ]
    no_canonical_dof_records = [
        record
        for record in dof_records
        if record["canonical_mapping"] is None
    ]
    dof_order_records = sorted(
        dof_records,
        key=lambda record: (
            10_000 if record["canonical_openpi_index"] is None else record["canonical_openpi_index"],
            record["source_joint_name"],
        ),
    )
    clean_joint_type_counts: dict[str, int] = {}
    for record in joint_records:
        clean_joint_type_counts[record["joint_type"]] = clean_joint_type_counts.get(record["joint_type"], 0) + 1

    ok = (
        len(joint_records) == 20
        and len(dof_records) == 16
        and not unmapped_records
        and not no_canonical_dof_records
    )
    plan = {
        "ok": ok,
        "status": "PASS_CLEAN_ARTICULATION_MAPPING_PLAN_NOT_AUTHORED" if ok else "FAIL_CLEAN_ARTICULATION_MAPPING_PLAN",
        "source_usd": str(source_usd),
        "clean_root": config["root_prim"],
        "author_articulation": False,
        "physics_ready": False,
        "training_eligible": False,
        "clean_joint_root_proposal": "/aloha/joints",
        "source_to_clean_link_count": len(source_to_clean),
        "source_to_clean_link_map": source_to_clean,
        "joint_count": len(joint_records),
        "dof_joint_count": len(dof_records),
        "joint_type_counts": clean_joint_type_counts,
        "unmapped_joint_count": len(unmapped_records),
        "unmapped_joints": unmapped_records,
        "no_canonical_dof_joint_count": len(no_canonical_dof_records),
        "no_canonical_dof_joints": no_canonical_dof_records,
        "proposed_canonical_dof_order": [
            {
                "openpi_index": record["canonical_openpi_index"],
                "dataset_index": record["canonical_dataset_index"],
                "canonical_dof_name": record["canonical_dof_name"],
                "source_joint_name": record["source_joint_name"],
                "source_joint_path": record["source_joint_path"],
                "clean_joint_path": record["proposed_clean_joint_path"],
                "clean_body0": record["clean_body0"],
                "clean_body1": record["clean_body1"],
                "axis": record["axis"],
                "limit_unit": record["limit_unit"],
            }
            for record in dof_order_records
        ],
        "joint_records": joint_records,
        "next_required_gates": [
            "author clean links with correct parent transforms, not only baked world visual transforms",
            "author clean joints under /aloha/joints from this mapping",
            "run Isaac Asset Validator RobotRules and PhysicsRules",
            "run set-target/readback gate for all DOFs",
            "run gravity-off hold, gravity-on hold, and 50Hz qpos replay gates",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = create_mapping_plan(args.config, args.json_output)
    summary = {key: value for key, value in result.items() if key not in {"joint_records", "source_to_clean_link_map"}}
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
