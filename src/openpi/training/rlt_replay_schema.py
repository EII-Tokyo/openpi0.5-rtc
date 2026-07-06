from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np


FORMAL_REPLAY_STATE_GRAIN = "paper_subsampled_anchor"
RUNTIME_REPLAY_STATE_GRAIN = "runtime_action_cache_block"
EXPECTED_FORMAL_Z_DIM = 2048


@dataclasses.dataclass(frozen=True)
class ReplayConversionStatus:
    status: str
    trainable: bool
    reason: str


def load_manifest_from_npz(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "manifest" not in data.files:
        return {}
    raw = data["manifest"]
    if isinstance(raw, np.ndarray):
        raw = raw.item() if raw.shape == () else raw.tolist()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not raw:
        return {}
    parsed = json.loads(str(raw)) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("manifest is not a JSON object")
    return parsed


def classify_replay_manifest(
    manifest: dict[str, Any] | None,
    *,
    z_dim: int | None = None,
    expected_z_dim: int | None = EXPECTED_FORMAL_Z_DIM,
) -> ReplayConversionStatus:
    manifest = dict(manifest or {})
    if manifest.get("voided") is True:
        return ReplayConversionStatus("voided", False, "manifest voided")
    if manifest.get("train_eligible") is False:
        if manifest.get("requires_offline_reencode") is True or manifest.get("replay_state_grain") == RUNTIME_REPLAY_STATE_GRAIN:
            return ReplayConversionStatus("requires_offline_reencode", False, RUNTIME_REPLAY_STATE_GRAIN)
        return ReplayConversionStatus("not_train_eligible", False, "train_eligible is false")

    replay_state_grain = manifest.get("replay_state_grain")
    if manifest.get("requires_offline_reencode") is True or replay_state_grain == RUNTIME_REPLAY_STATE_GRAIN:
        return ReplayConversionStatus("requires_offline_reencode", False, RUNTIME_REPLAY_STATE_GRAIN)
    if not replay_state_grain:
        return ReplayConversionStatus("legacy_unmarked_requires_audit", False, "missing replay_state_grain")
    if replay_state_grain != FORMAL_REPLAY_STATE_GRAIN:
        return ReplayConversionStatus("unsupported_replay_state_grain", False, f"unsupported replay_state_grain={replay_state_grain}")
    if manifest.get("formal_replay_ready") is False:
        return ReplayConversionStatus("formal_replay_not_ready", False, "formal_replay_ready is false")

    manifest_z_dim = _optional_int(
        manifest.get("z_rl_dim")
        or manifest.get("z_dim")
        or manifest.get("replay_array_shapes", {}).get("z_rl", [None, None])[-1]
    )
    actual_z_dim = z_dim if z_dim is not None else manifest_z_dim
    if expected_z_dim is not None and actual_z_dim is not None and int(actual_z_dim) != int(expected_z_dim):
        return ReplayConversionStatus("z_dim_mismatch", False, f"z_dim={actual_z_dim}, expected {expected_z_dim}")
    return ReplayConversionStatus("formal_replay_ready", True, FORMAL_REPLAY_STATE_GRAIN)


def require_formal_trainable_manifest(
    manifest: dict[str, Any] | None,
    *,
    z_dim: int | None = None,
    expected_z_dim: int | None = EXPECTED_FORMAL_Z_DIM,
) -> None:
    status = classify_replay_manifest(manifest, z_dim=z_dim, expected_z_dim=expected_z_dim)
    if not status.trainable:
        raise ValueError(f"not formal replay trainable: {status.status}: {status.reason}")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
