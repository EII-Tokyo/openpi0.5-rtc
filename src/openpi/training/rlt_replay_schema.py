from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np

FORMAL_REPLAY_STATE_GRAIN = "paper_subsampled_anchor"
TRUNK_SHARED_FORMAL_REPLAY_STATE_GRAIN = "trunk_shared_z_subsampled_anchor"
RUNTIME_REPLAY_STATE_GRAIN = "runtime_action_cache_block"
EXPECTED_FORMAL_Z_DIM = 2048
FORMAL_REPLAY_STATE_GRAINS = {FORMAL_REPLAY_STATE_GRAIN, TRUNK_SHARED_FORMAL_REPLAY_STATE_GRAIN}


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
        return ReplayConversionStatus(status="voided", trainable=False, reason="manifest voided")
    if manifest.get("train_eligible") is False:
        if manifest.get("requires_offline_reencode") is True or manifest.get("replay_state_grain") == RUNTIME_REPLAY_STATE_GRAIN:
            return ReplayConversionStatus(
                status="requires_offline_reencode",
                trainable=False,
                reason=RUNTIME_REPLAY_STATE_GRAIN,
            )
        return ReplayConversionStatus(status="not_train_eligible", trainable=False, reason="train_eligible is false")

    replay_state_grain = manifest.get("replay_state_grain")
    if manifest.get("requires_offline_reencode") is True or replay_state_grain == RUNTIME_REPLAY_STATE_GRAIN:
        return ReplayConversionStatus(
            status="requires_offline_reencode",
            trainable=False,
            reason=RUNTIME_REPLAY_STATE_GRAIN,
        )
    if not replay_state_grain:
        return ReplayConversionStatus(
            status="legacy_unmarked_requires_audit",
            trainable=False,
            reason="missing replay_state_grain",
        )
    if replay_state_grain not in FORMAL_REPLAY_STATE_GRAINS:
        return ReplayConversionStatus(
            status="unsupported_replay_state_grain",
            trainable=False,
            reason=f"unsupported replay_state_grain={replay_state_grain}",
        )
    if manifest.get("formal_replay_ready") is False:
        return ReplayConversionStatus(
            status="formal_replay_not_ready",
            trainable=False,
            reason="formal_replay_ready is false",
        )

    manifest_z_dim = _optional_int(
        manifest.get("z_rl_dim")
        or manifest.get("z_dim")
        or manifest.get("replay_array_shapes", {}).get("z_rl", [None, None])[-1]
    )
    actual_z_dim = z_dim if z_dim is not None else manifest_z_dim
    if expected_z_dim is not None and actual_z_dim is not None and int(actual_z_dim) != int(expected_z_dim):
        return ReplayConversionStatus(
            status="z_dim_mismatch",
            trainable=False,
            reason=f"z_dim={actual_z_dim}, expected {expected_z_dim}",
        )
    provenance_error = _timeline_provenance_error(manifest)
    if provenance_error is not None:
        return ReplayConversionStatus(
            status="missing_replay_provenance",
            trainable=False,
            reason=provenance_error,
        )
    return ReplayConversionStatus(
        status="formal_replay_ready",
        trainable=True,
        reason=str(replay_state_grain),
    )


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


def _timeline_provenance_error(manifest: dict[str, Any]) -> str | None:
    if manifest.get("source_format") != "rlt_timeline_hdf5":
        return None
    required = [
        "z_rl_source",
        "proprio_alignment",
        "behavior_policy",
        "action_source",
        "reference_action_source",
        "rl_token_checkpoint_path",
    ]
    missing = [key for key in required if not manifest.get(key)]
    behavior_policy = str(manifest.get("behavior_policy") or "")
    actor_applied_ratio = _optional_float(manifest.get("actor_applied_ratio"))
    if behavior_policy in {"rlt_actor", "mixed"} or (actor_applied_ratio is not None and actor_applied_ratio > 0.0):
        missing.extend(
            key for key in ("actor_checkpoint_path", "actor_checkpoint_step") if manifest.get(key) in (None, "")
        )
    z_rl_source = str(manifest.get("z_rl_source") or "")
    if z_rl_source and not z_rl_source.startswith("vla_same_forward"):
        return f"z_rl_source={z_rl_source!r} is not a vla_same_forward source"
    if missing:
        return f"missing required timeline replay provenance fields: {sorted(set(missing))}"
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
