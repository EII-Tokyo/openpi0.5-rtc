#!/usr/bin/env python3
"""Build the sole official-source-bound ALOHA Home/Sleep command manifest."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml

from tools.aloha1_mapping.home_sleep_correspondence import ARM_JOINT_ORDER
from tools.aloha1_mapping.home_sleep_correspondence import build_home_sleep_samples
from tools.aloha1_mapping.home_sleep_correspondence import command_signature
from tools.aloha1_mapping.home_sleep_correspondence import evaluate_interbotix_group_limit_gate
from tools.audit_aloha1_sleep_limit_correspondence import _xacro_limits

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/aloha1_home_sleep_correspondence.yaml"
REPORT_ROOT = ROOT / "reports/aloha1_mapping"
DEFAULT_MANIFEST = REPORT_ROOT / "aloha1_home_sleep_command_manifest.json"
DEFAULT_AUDIT = REPORT_ROOT / "aloha1_home_sleep_official_source_audit.json"
DEFAULT_AUDIT_MD = REPORT_ROOT / "aloha1_home_sleep_official_source_audit.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
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


def _numeric_literal(node: ast.AST) -> float:
    value = ast.literal_eval(node)
    if not isinstance(value, int | float):
        raise ValueError("expected numeric official-source literal")
    return float(value)


def _extract_robot_utils(path: Path) -> tuple[list[float], float]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    sleep_function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "sleep_arms"
        ),
        None,
    )
    if sleep_function is None:
        raise ValueError("official sleep_arms function is missing")
    home_candidates: list[list[float]] = []
    for node in ast.walk(sleep_function):
        if isinstance(node, ast.List) and len(node.elts) == 6:
            try:
                home_candidates.append([_numeric_literal(item) for item in node.elts])
            except (ValueError, TypeError, SyntaxError):
                continue
    if len(home_candidates) != 1:
        raise ValueError(f"expected one official Home vector, got {home_candidates}")

    positional = list(sleep_function.args.args)
    defaults = [None] * (len(positional) - len(sleep_function.args.defaults)) + list(
        sleep_function.args.defaults
    )
    default_by_name = {
        argument.arg: default
        for argument, default in zip(positional, defaults, strict=True)
    }
    moving_default = default_by_name.get("moving_time")
    if moving_default is None:
        raise ValueError("official moving_time default is missing")
    return home_candidates[0], _numeric_literal(moving_default)


def _extract_dt(path: Path) -> float:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DT"
            for target in node.targets
        ):
            return _numeric_literal(node.value)
    raise ValueError("official DT assignment is missing")


def _verified_source(
    name: str, record: Mapping[str, Any], *, project_root: Path
) -> dict[str, Any]:
    source_type = str(record.get("source_type", "file"))
    if source_type == "git_blob":
        repository_path = (
            project_root / str(record["local_repository_path"])
        ).resolve(strict=True)
        commit = str(record["commit"])
        object_path = str(record["git_object_path"])
        subprocess.run(
            ["git", "-C", str(repository_path), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=True,
            capture_output=True,
        )
        content = subprocess.run(
            ["git", "-C", str(repository_path), "show", f"{commit}:{object_path}"],
            check=True,
            capture_output=True,
        ).stdout
        actual = hashlib.sha256(content).hexdigest()
        expected = str(record["sha256"])
        if actual != expected:
            raise ValueError(f"{name} SHA-256 mismatch: {actual} != {expected}")
        return {
            "id": name,
            "source_type": source_type,
            "repository": str(record["repository"]),
            "branch": str(record["branch"]),
            "commit": commit,
            "license": str(record["license"]),
            "absolute_repository_path": str(repository_path),
            "git_object_path": object_path,
            "sha256": actual,
            "selection_status": str(record["selection_status"]),
            "_content_text": content.decode("utf-8"),
        }
    if source_type != "file":
        raise ValueError(f"unsupported source_type for {name}: {source_type}")
    path = (project_root / str(record["local_path"])).resolve(strict=True)
    actual = _sha256(path)
    expected = str(record["sha256"])
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch: {actual} != {expected}")
    return {
        "id": name,
        "source_type": source_type,
        "repository": str(record["repository"]),
        "branch": str(record["branch"]),
        "commit": str(record["commit"]),
        "license": str(record["license"]),
        "absolute_path": str(path),
        "sha256": actual,
    }


def _verified_file(record: Mapping[str, Any], *, project_root: Path) -> dict[str, Any]:
    path = (project_root / str(record["path"])).resolve(strict=True)
    actual = _sha256(path)
    expected = str(record["sha256"])
    if actual != expected:
        raise ValueError(f"frozen file SHA-256 mismatch: {actual} != {expected}")
    return {"absolute_path": str(path), "sha256": actual}


def build_manifest(
    config: Mapping[str, Any], *, project_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify exact official sources and return deterministic manifest/audit."""

    sources = {
        name: _verified_source(name, record, project_root=project_root)
        for name, record in config["sources"].items()
    }
    home_source, moving_time = _extract_robot_utils(
        Path(sources["robot_utils"]["absolute_path"])
    )
    dt = _extract_dt(Path(sources["constants"]["absolute_path"]))
    current_humble_motor_config = yaml.safe_load(
        Path(sources["aloha_vx300s"]["absolute_path"]).read_text(encoding="utf-8")
    )
    motor_config = yaml.safe_load(sources["selected_sleep"]["_content_text"])
    if motor_config["joint_order"][:6] != list(ARM_JOINT_ORDER):
        raise ValueError("official aloha_vx300s joint order mismatch")
    sleep_source = [float(value) for value in motor_config["sleep_positions"][:6]]
    command = config["command"]
    home = [float(value) for value in command["home_rad"]]
    sleep = [float(value) for value in command["sleep_rad"]]
    if home_source != home:
        raise ValueError(f"configured Home differs from official source: {home_source}")
    if sleep_source != sleep:
        raise ValueError(f"configured Sleep differs from official source: {sleep_source}")
    if dt != 1.0 / int(command["command_rate_hz"]):
        raise ValueError("configured command rate differs from official DT")
    if moving_time != float(command["move_seconds"]):
        raise ValueError("configured movement time differs from official default")

    samples = build_home_sleep_samples(
        home=home,
        sleep=sleep,
        command_hz=int(command["command_rate_hz"]),
        move_seconds=int(command["move_seconds"]),
        hold_seconds=int(command["hold_seconds"]),
        cycles=int(command["cycles"]),
    )
    lower_rad, upper_rad = _xacro_limits(
        Path(sources["aloha_vx300s_xacro"]["absolute_path"])
    )
    group_gate_detail = evaluate_interbotix_group_limit_gate(
        samples,
        lower_rad=lower_rad,
        upper_rad=upper_rad,
        moving_time_s=2.0,
        velocity_limits_rad_s=[3.141592653589793] * len(ARM_JOINT_ORDER),
    )
    group_limit_gate = {
        "status": (
            "PASS" if group_gate_detail["rejected_sample_count"] == 0 else "FAIL"
        ),
        "command_semantics": group_gate_detail["command_semantics"],
        "sample_count": group_gate_detail["sample_count"],
        "accepted_sample_count": group_gate_detail["accepted_sample_count"],
        "rejected_sample_count": group_gate_detail["rejected_sample_count"],
        "first_rejected_sample_index": group_gate_detail[
            "first_rejected_sample_index"
        ],
        "first_rejected_joint_names": group_gate_detail[
            "first_rejected_joint_names"
        ],
        "all_samples_publishable": group_gate_detail["rejected_sample_count"] == 0,
    }
    source_audit: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "classification": "OFFICIAL_HISTORICAL_LEGAL_ALOHA_SLEEP_EXPLICITLY_SELECTED_BY_USER",
        "product": str(config["product"]["model"]),
        "home": {"value_rad": home_source, "source_id": "robot_utils"},
        "sleep": {
            "value_rad": sleep_source,
            "source_id": "selected_sleep",
        },
        "command_dt_s": dt,
        "moving_time_s": moving_time,
        "joint_order": list(ARM_JOINT_ORDER),
        "sources": [
            {key: value for key, value in source.items() if not key.startswith("_")}
            for source in sources.values()
        ],
        "current_humble_comparison": {
            "source_id": "aloha_vx300s",
            "sleep_rad": [
                float(value)
                for value in current_humble_motor_config["sleep_positions"][:6]
            ],
            "used_as_command_authority": False,
            "classification": "CURRENT_HUMBLE_OUT_OF_LIMIT_COMPARISON_ONLY",
        },
        "version_selection": {
            "status": "EXPLICIT_CROSS_VERSION_COMMAND_SELECTION",
            "selected_command_source": "official historical ROS2 PR #189",
            "current_urdf_and_driver_limits_preserved": True,
        },
        "joint_limit_authority": {
            "source_id": "aloha_vx300s_xacro",
            "lower_rad": lower_rad,
            "upper_rad": upper_rad,
            "velocity_rad_s": [3.141592653589793] * len(ARM_JOINT_ORDER),
        },
        "group_limit_gate": group_limit_gate,
        "generic_vx300s_substituted": False,
    }
    source_audit["deterministic_signature"] = _canonical_signature(source_audit)

    digital = config["digital"]
    boundaries = config["boundaries"]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": str(config["experiment_id"]),
        "product": dict(config["product"]),
        "robot": str(config["scope"]["active_robot"]),
        "joint_order": list(ARM_JOINT_ORDER),
        "unit": "rad",
        "home_rad": home,
        "sleep_rad": sleep,
        "command_rate_hz": int(command["command_rate_hz"]),
        "physics_rate_hz": int(command["physics_rate_hz"]),
        "move_seconds": int(command["move_seconds"]),
        "hold_seconds": int(command["hold_seconds"]),
        "cycles": int(command["cycles"]),
        "sample_count": len(samples),
        "samples": [asdict(sample) for sample in samples],
        "command_signature": command_signature(samples),
        "official_source_audit_signature": source_audit[
            "deterministic_signature"
        ],
        "stationary_scope": dict(config["scope"]["stationary"]),
        "digital_runtime": {
            "isaac_sim": str(digital["isaac_sim"]),
            "kit": str(digital["kit"]),
            "physx": str(digital["physx"]),
            "stage": _verified_file(digital["stage"], project_root=project_root),
            "stage_root_prim": str(digital["stage"]["root_prim"]),
            "finger_limit_layer": _verified_file(
                digital["finger_limit_layer"], project_root=project_root
            ),
        },
        "candidate_promoted": bool(boundaries["candidate_promoted"]),
        "final_default_asset_modified": bool(
            boundaries["final_default_asset_modified"]
        ),
        "real_execution_authorized": bool(
            boundaries["real_execution_authorized"]
        ),
        "forbidden_mechanisms": {
            "surface_gripper": bool(boundaries["surface_gripper"]),
            "fixed_joint_attachment": bool(boundaries["fixed_joint_attachment"]),
            "parent_attachment": bool(boundaries["parent_attachment"]),
        },
    }
    if manifest["candidate_promoted"] or manifest["final_default_asset_modified"]:
        raise ValueError("diagnostic command manifest may not promote or modify assets")
    if manifest["real_execution_authorized"]:
        raise ValueError("command manifest may not authorize real execution")
    manifest["manifest_signature"] = _canonical_signature(manifest)
    return manifest, source_audit


def _source_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# ALOHA1 Home/Sleep official source audit",
        "",
        f"- Status: `{audit['status']}`",
        f"- Product: `{audit['product']}`",
        f"- Home: `{audit['home']['value_rad']}` rad",
        f"- Sleep: `{audit['sleep']['value_rad']}` rad",
        f"- Command DT: `{audit['command_dt_s']}` s",
        f"- Moving time: `{audit['moving_time_s']}` s",
        "- Generic vx300s substituted: `false`",
        f"- Deterministic signature: `{audit['deterministic_signature']}`",
        "",
        "| Source | Repository | Commit | License | SHA-256 |",
        "|---|---|---|---|---|",
    ]
    lines.extend(
        "| `{id}` | `{repository}` | `{commit}` | `{license}` | `{sha256}` |".format(
            **source
        )
        for source in audit["sources"]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--audit-markdown", type=Path, default=DEFAULT_AUDIT_MD)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.resolve(strict=True).read_text())
    manifest, audit = build_manifest(config, project_root=ROOT)
    for output in (args.manifest, args.audit, args.audit_markdown):
        output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.audit.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.audit_markdown.write_text(_source_markdown(audit), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": audit["status"],
                "sample_count": manifest["sample_count"],
                "command_signature": manifest["command_signature"],
                "manifest_signature": manifest["manifest_signature"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
