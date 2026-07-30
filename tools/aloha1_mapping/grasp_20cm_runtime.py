"""Frozen-input helpers for the Isaac Sim 5.1 ALOHA 20 cm grasp runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
from typing import Any

import yaml

EXPECTED_DOF_ORDER = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
    "gripper",
    "left_finger",
    "right_finger",
]


class FrozenInputError(RuntimeError):
    """Raised before runtime mutation when a frozen input contract fails."""


def sha256_file(path: Path) -> str:
    """Hash one file without following any guessed alternative path."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_file(
    path: Path,
    expected_sha256: str,
) -> dict[str, str]:
    """Resolve and verify one exact frozen file."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise FrozenInputError(f"missing frozen input: {resolved}")
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise FrozenInputError(
            f"sha256 mismatch for {resolved}: "
            f"{actual} != {expected_sha256}"
        )
    return {"absolute_path": str(resolved), "sha256": actual}


def load_and_verify_config(
    config_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Load the diagnostic profile and verify every referenced source."""
    config_record = verify_frozen_file(
        config_path,
        sha256_file(config_path.resolve()),
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise FrozenInputError("config root must be a mapping")
    if config.get("schema_version") != 1:
        raise FrozenInputError("unsupported config schema_version")
    if config.get("classification") != (
        "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
    ):
        raise FrozenInputError("unexpected diagnostic classification")

    records: dict[str, dict[str, str]] = {}
    records["stage"] = _verify_record(
        config.get("stage"),
        project_root=project_root,
        label="stage",
    )
    records["bottle"] = _verify_record(
        config.get("bottle"),
        project_root=project_root,
        label="bottle",
    )
    frozen = config.get("frozen_inputs")
    if not isinstance(frozen, Mapping):
        raise FrozenInputError("frozen_inputs must be a mapping")
    for name, record in frozen.items():
        records[str(name)] = _verify_record(
            record,
            project_root=project_root,
            label=str(name),
        )

    dof_order = config.get("robot", {}).get("dof_order")
    if dof_order != EXPECTED_DOF_ORDER:
        raise FrozenInputError(
            f"unexpected DOF order: {dof_order!r}"
        )
    if config.get("boundaries", {}).get("task8") != "NOT_RUN":
        raise FrozenInputError("Task 8 boundary must remain NOT_RUN")
    return {
        "config": dict(config),
        "config_path": config_record["absolute_path"],
        "config_sha256": config_record["sha256"],
        "frozen_inputs": records,
    }


def _verify_record(
    record: Any,
    *,
    project_root: Path,
    label: str,
) -> dict[str, str]:
    if not isinstance(record, Mapping):
        raise FrozenInputError(f"{label} record must be a mapping")
    path_value = record.get("path")
    hash_value = record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(hash_value, str):
        raise FrozenInputError(
            f"{label} requires string path and sha256"
        )
    path = Path(path_value)
    if not path.is_absolute():
        path = project_root / path
    return verify_frozen_file(path, hash_value)


def validate_composed_stage(
    *,
    stage: Any,
    expected_root_prim: str,
    required_prims: Sequence[str],
) -> dict[str, Any]:
    """Validate required composed prims without editing the USD Stage."""
    root_prim = stage.GetPrimAtPath(expected_root_prim)
    if not root_prim.IsValid():
        raise FrozenInputError(
            f"missing expected root prim: {expected_root_prim}"
        )
    missing = [
        path
        for path in required_prims
        if not stage.GetPrimAtPath(path).IsValid()
    ]
    if missing:
        raise FrozenInputError(f"missing required prims: {missing}")
    return {
        "root_prim": expected_root_prim,
        "sublayers": list(stage.GetRootLayer().subLayerPaths),
        "required_prims": list(required_prims),
    }
