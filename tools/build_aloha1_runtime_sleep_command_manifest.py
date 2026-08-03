#!/usr/bin/env python3
"""Build an isolated Sleep-Home-Sleep manifest from frozen real readback."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

import yaml

from tools.aloha1_mapping.home_sleep_correspondence import ARM_JOINT_ORDER
from tools.aloha1_mapping.home_sleep_correspondence import build_sleep_home_samples
from tools.aloha1_mapping.home_sleep_correspondence import command_signature
from tools.aloha1_mapping.home_sleep_correspondence import expand_joint_limits_to_reference

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/aloha1_runtime_measured_sleep_correspondence.yaml"
DEFAULT_MANIFEST = ROOT / "reports/aloha1_mapping/aloha1_runtime_measured_sleep_command_manifest.json"
DEFAULT_AUDIT = ROOT / "reports/aloha1_mapping/aloha1_runtime_measured_sleep_alignment.json"
DEFAULT_AUDIT_MD = ROOT / "reports/aloha1_mapping/aloha1_runtime_measured_sleep_alignment.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verified_path(record: Mapping[str, Any], project_root: Path) -> Path:
    path = (project_root / str(record["path"])).resolve(strict=True)
    actual = _sha256(path)
    expected = str(record["sha256"])
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")
    return path


def _runtime_median(record: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    path = _verified_path(record, project_root)
    columns = [f"field.position{index}" for index in range(6)]
    values = {column: [] for column in columns}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            for column in columns:
                values[column].append(float(row[column]))
    row_count = len(values[columns[0]])
    if row_count != int(record["expected_rows"]):
        raise ValueError(f"runtime readback rows changed: {row_count}")
    medians = [statistics.median(values[column]) for column in columns]
    spans = [max(values[column]) - min(values[column]) for column in columns]
    return {
        "absolute_path": str(path),
        "sha256": _sha256(path),
        "sample_count": row_count,
        "median_arm_rad": medians,
        "span_rad_by_joint": dict(zip(ARM_JOINT_ORDER, spans, strict=True)),
        "maximum_span_rad": max(spans),
        "host": str(record["host"]),
        "namespace": str(record["namespace"]),
        "access_mode": str(record["access_mode"]),
        "classification": "RUNTIME_MEASURED_SLEEP_REFERENCE_NOT_OFFICIAL_SLEEP",
    }


def build_runtime_sleep_manifest(
    config: Mapping[str, Any], *, project_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a deterministic diagnostic manifest and its source audit."""

    if list(config["joint_order"]) != list(ARM_JOINT_ORDER):
        raise ValueError("runtime Sleep joint order mismatch")
    sources = config["sources"]
    runtime_reference = _runtime_median(sources["runtime_joint_states"], project_root)
    verified_sources = {
        name: {
            "absolute_path": str(_verified_path(record, project_root)),
            "sha256": str(record["sha256"]),
        }
        for name, record in sources.items()
        if name != "runtime_joint_states"
    }
    limit_report = json.loads(
        Path(verified_sources["isaac_limit_readback"]["absolute_path"]).read_text(encoding="utf-8")
    )
    lower, upper = limit_report["preflight"]["limits"]["follower_left"]
    runtime_sleep = runtime_reference["median_arm_rad"]
    expanded_lower, expanded_upper, changes = expand_joint_limits_to_reference(
        runtime_sleep,
        lower[:6],
        upper[:6],
        joint_names=ARM_JOINT_ORDER,
    )

    command = config["command"]
    home = [float(value) for value in command["home_rad"]]
    samples = build_sleep_home_samples(
        sleep=runtime_sleep,
        home=home,
        command_hz=int(command["command_rate_hz"]),
        move_seconds=int(command["move_seconds"]),
        hold_seconds=int(command["hold_seconds"]),
        cycles=int(command["cycles"]),
    )
    digital = config["digital"]
    stage = _verified_path(digital["stage"], project_root)
    finger_layer = _verified_path(digital["finger_limit_layer"], project_root)
    boundaries = config["boundaries"]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": str(config["experiment_id"]),
        "sequence_kind": "SLEEP_HOME_SLEEP",
        "robot": "follower_left",
        "joint_order": list(ARM_JOINT_ORDER),
        "home_rad": home,
        "sleep_rad": runtime_sleep,
        "initial_pose_label": "runtime_measured_sleep",
        "terminal_pose_label": "runtime_measured_sleep",
        "initial_arm_rad": runtime_sleep,
        "terminal_arm_rad": runtime_sleep,
        "command_rate_hz": int(command["command_rate_hz"]),
        "physics_rate_hz": int(command["physics_rate_hz"]),
        "move_seconds": int(command["move_seconds"]),
        "hold_seconds": int(command["hold_seconds"]),
        "cycles": int(command["cycles"]),
        "sample_count": len(samples),
        "samples": [asdict(sample) for sample in samples],
        "command_signature": command_signature(samples),
        "diagnostic_limit_override": {
            "policy": "MINIMAL_EXPANSION_TO_RUNTIME_REFERENCE",
            "classification": "DIAGNOSTIC_ONLY_RUNTIME_ALIGNMENT_NOT_FINAL_ASSET",
            "source_lower_rad": list(lower[:6]),
            "source_upper_rad": list(upper[:6]),
            "diagnostic_lower_rad": list(expanded_lower),
            "diagnostic_upper_rad": list(expanded_upper),
            "changes": list(changes),
        },
        "digital": {
            "isaac_sim": str(digital["isaac_sim"]),
            "kit": str(digital["kit"]),
            "physx": str(digital["physx"]),
            "stage_absolute_path": str(stage),
            "stage_sha256": _sha256(stage),
            "stage_root_prim": str(digital["stage"]["root_prim"]),
            "finger_limit_layer_absolute_path": str(finger_layer),
            "finger_limit_layer_sha256": _sha256(finger_layer),
        },
        "stationary_scope": {
            "follower_right": True,
            "follower_left_gripper": True,
            "follower_right_gripper": True,
        },
        "real_execution_authorized": bool(boundaries["real_execution_authorized"]),
        "real_motion_commands": int(boundaries["real_motion_commands"]),
        "candidate_promoted": bool(boundaries["candidate_promoted"]),
        "final_default_asset_modified": bool(boundaries["final_default_asset_modified"]),
        "source_records": {
            "runtime_joint_states": runtime_reference,
            **verified_sources,
        },
    }
    manifest["manifest_signature"] = _canonical_signature(manifest)
    audit = {
        "schema_version": 1,
        "status": "READY_FOR_ISOLATED_DIGITAL_VALIDATION",
        "classification": "RUNTIME_MEASURED_SLEEP_DIAGNOSTIC_ALIGNMENT",
        "runtime_reference": runtime_reference,
        "diagnostic_limit_policy": "DIAGNOSTIC_ONLY_RUNTIME_ALIGNMENT_NOT_FINAL_ASSET",
        "diagnostic_limit_changes": list(changes),
        "source_limits_rad": {"lower": list(lower[:6]), "upper": list(upper[:6])},
        "diagnostic_limits_rad": {"lower": list(expanded_lower), "upper": list(expanded_upper)},
        "sequence": {
            "kind": "SLEEP_HOME_SLEEP",
            "cycles": int(command["cycles"]),
            "sample_count": len(samples),
            "command_signature": manifest["command_signature"],
            "manifest_signature": manifest["manifest_signature"],
        },
        "historical_manifest_unchanged": verified_sources["historical_manifest"],
        "source_or_final_asset_modified": False,
        "candidate_promoted": False,
        "real_execution_authorized": False,
        "real_motion_commands": 0,
    }
    return manifest, audit


def _markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# ALOHA1 runtime-measured Sleep alignment",
        "",
        f"- Status: `{audit['status']}`",
        f"- Classification: `{audit['classification']}`",
        f"- Samples: `{audit['runtime_reference']['sample_count']}`",
        f"- Sequence: `{audit['sequence']['kind']}` x `{audit['sequence']['cycles']}`",
        f"- Diagnostic limit policy: `{audit['diagnostic_limit_policy']}`",
        "- Real motion commands: `0`",
        "- Final/default asset modified: `false`",
        "",
        "## Runtime Sleep reference",
        "",
        f"`{audit['runtime_reference']['median_arm_rad']}` rad",
        "",
        "## Diagnostic-only limit changes",
        "",
        "| Joint | Bound | Source (rad) | Diagnostic (rad) | Delta (rad) |",
        "|---|---|---:|---:|---:|",
    ]
    lines.extend(
        (
            "| `{joint_name}` | `{bound}` | {source_value_rad:.9f} | "
            "{diagnostic_value_rad:.9f} | {delta_rad:.9f} |".format(**item)
        )
        for item in audit["diagnostic_limit_changes"]
    )
    lines.extend(
        [
            "",
            "These changes exist only in the isolated runtime session layer so the digital arm can "
            "start from the frozen real readback. They are not hardware calibration and are not "
            "eligible for final/default asset promotion.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--audit-md", type=Path, default=DEFAULT_AUDIT_MD)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest, audit = build_runtime_sleep_manifest(config, project_root=ROOT)
    for path in (args.manifest, args.audit, args.audit_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    args.audit_md.write_text(_markdown(audit), encoding="utf-8")
    print(json.dumps({"status": audit["status"], "manifest": str(args.manifest.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
