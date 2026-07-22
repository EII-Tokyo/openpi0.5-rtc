from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class CheckpointSchema:
    path: Path
    action_dim: int
    action_horizon: int
    sharded: bool
    safetensor_files: tuple[str, ...]
    has_trossen_norm_stats: bool
    has_robotwin_norm_stats: bool
    total_size: int | None


def inspect_checkpoint_schema(path: str | Path) -> CheckpointSchema:
    root = Path(path)
    cfg_path = root / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    index_path = root / "model.safetensors.index.json"
    safetensor_files = tuple(sorted(p.name for p in root.glob("*.safetensors")))
    total_size = None
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        metadata = index.get("metadata") or {}
        total_size = metadata.get("total_size")
    return CheckpointSchema(
        path=root,
        action_dim=int(cfg["action_dim"]),
        action_horizon=int(cfg["action_horizon"]),
        sharded=index_path.exists(),
        safetensor_files=safetensor_files,
        has_trossen_norm_stats=(root / "assets" / "trossen" / "norm_stats.json").exists(),
        has_robotwin_norm_stats=(root / "physical-intelligence" / "robotwin" / "norm_stats.json").exists(),
        total_size=total_size,
    )


def validate_strict_openpi_native_compatibility(schema: CheckpointSchema) -> list[str]:
    """Return blockers for this repo's current OpenPI native loader."""

    blockers: list[str] = []
    if schema.sharded and "model.safetensors" not in schema.safetensor_files:
        blockers.append("local OpenPI create_trained_policy only detects single model.safetensors")
    if schema.action_dim != 32:
        blockers.append(f"unexpected model action_dim {schema.action_dim}; expected padded 32")
    if not schema.has_trossen_norm_stats and not schema.has_robotwin_norm_stats:
        blockers.append("no usable norm_stats found")
    return blockers

