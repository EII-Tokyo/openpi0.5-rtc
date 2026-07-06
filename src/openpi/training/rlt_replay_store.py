from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import sqlite3

import jax.numpy as jnp
import numpy as np

from openpi.training import rlt_training
from openpi.training import rlt_replay_schema
from openpi.training import rlt_trainable_manifest

REQUIRED_REPLAY_KEYS: tuple[str, ...] = (
    "z_rl",
    "proprio",
    "action",
    "reference_action",
    "reward_seq",
    "next_z_rl",
    "next_proprio",
    "next_reference_action",
    "done",
)


@dataclasses.dataclass(frozen=True)
class ReplayShape:
    z_dim: int
    proprio_dim: int
    action_horizon: int
    action_dim: int


@dataclasses.dataclass(frozen=True)
class ReplayShardInfo:
    path: pathlib.Path
    num_transitions: int
    num_terminal: int
    num_success: int
    num_failure: int


@dataclasses.dataclass(frozen=True)
class ReplayStats:
    replay_size: int
    num_shards: int
    success_episodes: int
    failure_episodes: int
    bad_shards: int


@dataclasses.dataclass(frozen=True)
class ZRLNormalization:
    mean: np.ndarray
    std: np.ndarray


@dataclasses.dataclass
class _LoadedShard:
    info: ReplayShardInfo
    arrays: dict[str, np.ndarray]


class ReplayShardError(ValueError):
    """Raised when a replay shard is complete but cannot be used for RLT training."""


class RLTReplayStore:
    """Online replay store backed by atomically committed RLT NPZ shards."""

    def __init__(
        self,
        replay_dir: pathlib.Path | str,
        *,
        max_replay_samples: int | None = None,
        recursive: bool = False,
        sample_action_horizon: int | None = None,
        segment_db_path: pathlib.Path | str | None = None,
        manifest_path: pathlib.Path | str | None = None,
        normalize_z_rl: bool = False,
        require_formal_replay: bool = False,
        expected_formal_z_dim: int | None = None,
    ):
        if sample_action_horizon is not None and sample_action_horizon <= 0:
            raise ValueError("sample_action_horizon must be positive when provided")
        self._replay_dir = pathlib.Path(replay_dir)
        self._max_replay_samples = max_replay_samples
        self._recursive = recursive
        self._sample_action_horizon = sample_action_horizon
        self._segment_db_path = None if segment_db_path is None else pathlib.Path(segment_db_path)
        self._manifest_path = None if manifest_path is None else pathlib.Path(manifest_path)
        self._normalize_z_rl = normalize_z_rl
        self._require_formal_replay = require_formal_replay
        self._expected_formal_z_dim = expected_formal_z_dim
        self._z_rl_normalization: ZRLNormalization | None = None
        self._shards: list[_LoadedShard] = []
        self._loaded_paths: set[pathlib.Path] = set()
        self._bad_paths: dict[pathlib.Path, str] = {}
        self._shape: ReplayShape | None = None
        self._replay_size = 0

    @property
    def shape(self) -> ReplayShape | None:
        return self._shape

    @property
    def sample_shape(self) -> ReplayShape | None:
        if self._shape is None:
            return None
        return dataclasses.replace(
            self._shape,
            action_horizon=self._sample_action_horizon or self._shape.action_horizon,
        )

    @property
    def stats(self) -> ReplayStats:
        return ReplayStats(
            replay_size=self._replay_size,
            num_shards=len(self._shards),
            success_episodes=sum(shard.info.num_success for shard in self._shards),
            failure_episodes=sum(shard.info.num_failure for shard in self._shards),
            bad_shards=len(self._bad_paths),
        )

    @property
    def loaded_paths(self) -> tuple[pathlib.Path, ...]:
        return tuple(shard.info.path for shard in self._shards)

    @property
    def z_rl_normalization(self) -> ZRLNormalization | None:
        return self._z_rl_normalization

    def scan(self) -> list[ReplayShardInfo]:
        """Discover and load newly committed shards.

        Only files ending in `.npz` are considered. Recorder temp files such as
        `.npz.tmp` are intentionally ignored by the glob.
        """

        self._reconcile_loaded_paths()
        added: list[ReplayShardInfo] = []
        for path in self._iter_candidate_paths():
            resolved = path.resolve()
            if resolved in self._loaded_paths or resolved in self._bad_paths:
                continue
            try:
                shard = self._load_shard(resolved)
            except (OSError, ReplayShardError, KeyError, ValueError) as exc:
                self._bad_paths[resolved] = str(exc)
                logging.warning("Skipping invalid RLT replay shard %s: %s", resolved, exc)
                continue
            self._append_shard(shard)
            added.append(shard.info)
        if added:
            self._refresh_z_rl_normalization()
            logging.info(
                "Loaded %d new RLT replay shard(s), replay_size=%d",
                len(added),
                self._replay_size,
            )
        return added

    def ready(
        self,
        *,
        min_replay_samples: int,
        min_success_episodes: int = 0,
        min_failure_episodes: int = 0,
        min_replay_shards: int = 0,
    ) -> bool:
        stats = self.stats
        return (
            stats.replay_size >= min_replay_samples
            and stats.success_episodes >= min_success_episodes
            and stats.failure_episodes >= min_failure_episodes
            and stats.num_shards >= min_replay_shards
        )

    def sample_batch(self, rng: np.random.Generator, batch_size: int) -> rlt_training.RLTReplayBatch:
        if self._replay_size <= 0:
            raise ValueError("Cannot sample from an empty RLT replay store.")
        indices = rng.integers(0, self._replay_size, size=batch_size)
        arrays = self._slice_sample_horizon(self._gather(indices))
        arrays = self._standardize_z_arrays(arrays)
        return rlt_training.make_replay_batch(
            z_rl=jnp.asarray(arrays["z_rl"]),
            proprio=jnp.asarray(arrays["proprio"]),
            action=jnp.asarray(arrays["action"]),
            reference_action=jnp.asarray(arrays["reference_action"]),
            reward_seq=jnp.asarray(arrays["reward_seq"]),
            next_z_rl=jnp.asarray(arrays["next_z_rl"]),
            next_proprio=jnp.asarray(arrays["next_proprio"]),
            next_reference_action=jnp.asarray(arrays["next_reference_action"]),
            done=jnp.asarray(arrays["done"].astype(np.bool_)),
            episode_success=jnp.asarray(arrays["episode_success"].astype(np.bool_)),
        )


    def _slice_sample_horizon(self, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._sample_action_horizon is None:
            return arrays
        horizon = self._sample_action_horizon
        sliced = dict(arrays)
        sliced["action"] = arrays["action"][:, :horizon, :]
        sliced["reference_action"] = arrays["reference_action"][:, :horizon, :]
        sliced["next_reference_action"] = arrays["next_reference_action"][:, :horizon, :]
        sliced["reward_seq"] = arrays["reward_seq"][:, :horizon]
        return sliced

    def bad_shards(self) -> dict[pathlib.Path, str]:
        return dict(self._bad_paths)

    def _iter_candidate_paths(self) -> list[pathlib.Path]:
        if self._manifest_path is not None:
            return self._accepted_shard_paths()
        if self._segment_db_path is not None:
            return self._accepted_shard_paths()
        if not self._replay_dir.exists():
            return []
        if self._recursive:
            paths = self._replay_dir.glob("**/shards/*.npz")
        else:
            shards_dir = self._replay_dir / "shards"
            paths = shards_dir.glob("*.npz") if shards_dir.exists() else self._replay_dir.glob("*.npz")
        return sorted(path for path in paths if path.is_file() and path.suffix == ".npz")

    def _accepted_shard_paths(self) -> list[pathlib.Path]:
        if self._manifest_path is not None:
            if not self._manifest_path.exists():
                return []
            paths = rlt_trainable_manifest.read_manifest_paths(self._manifest_path)
            return sorted(path.resolve() for path in paths if path.exists() and path.suffix == ".npz")
        if self._segment_db_path is None or not self._segment_db_path.exists():
            return []
        with sqlite3.connect(self._segment_db_path) as conn:
            rows = conn.execute(
                "SELECT shard_path FROM segments WHERE status = 'committed' AND shard_path IS NOT NULL"
            ).fetchall()
        paths = [pathlib.Path(row[0]) for row in rows if row[0]]
        return sorted(path.resolve() for path in paths if path.exists() and path.suffix == ".npz")

    def _reconcile_loaded_paths(self) -> None:
        if self._segment_db_path is None and self._manifest_path is None:
            return
        accepted = set(self._accepted_shard_paths())
        if not accepted:
            if self._shards:
                self._shards = []
                self._loaded_paths = set()
                self._replay_size = 0
                self._shape = None
                self._refresh_z_rl_normalization()
            return
        if self._loaded_paths.issubset(accepted):
            return
        self._shards = [shard for shard in self._shards if shard.info.path in accepted]
        self._loaded_paths = {shard.info.path for shard in self._shards}
        self._replay_size = sum(shard.info.num_transitions for shard in self._shards)
        self._shape = self._shards[0].arrays and _shape_from_arrays(self._shards[0].arrays) if self._shards else None
        self._refresh_z_rl_normalization()

    def _load_shard(self, path: pathlib.Path) -> _LoadedShard:
        with np.load(path) as data:
            missing = [key for key in REQUIRED_REPLAY_KEYS if key not in data]
            if missing:
                raise ReplayShardError(f"missing required arrays: {missing}")
            arrays = {key: np.asarray(data[key]) for key in REQUIRED_REPLAY_KEYS}
            manifest = _load_manifest(data)

        _validate_arrays(path, arrays)
        shape = _shape_from_arrays(arrays)
        _validate_manifest(
            manifest,
            shape,
            require_formal_replay=self._require_formal_replay,
            expected_formal_z_dim=self._expected_formal_z_dim,
        )
        if self._sample_action_horizon is not None and self._sample_action_horizon > shape.action_horizon:
            raise ReplayShardError(
                f"sample_action_horizon={self._sample_action_horizon} exceeds replay action_horizon={shape.action_horizon}"
            )
        if self._shape is None:
            self._shape = shape
        elif shape != self._shape:
            raise ReplayShardError(f"shape {shape} does not match replay store shape {self._shape}")

        done = arrays["done"].astype(np.bool_)
        terminal_rewards = np.sum(arrays["reward_seq"][done], axis=-1) if np.any(done) else np.asarray([], dtype=np.float32)
        episode_success = bool(np.any(terminal_rewards > 0.0))
        arrays["episode_success"] = np.full((len(arrays["z_rl"]),), episode_success, dtype=np.bool_)
        info = ReplayShardInfo(
            path=path,
            num_transitions=len(arrays["z_rl"]),
            num_terminal=int(np.sum(done)),
            num_success=int(np.sum(terminal_rewards > 0.0)),
            num_failure=int(np.sum(done) - np.sum(terminal_rewards > 0.0)),
        )
        return _LoadedShard(info=info, arrays=arrays)

    def _append_shard(self, shard: _LoadedShard) -> None:
        self._shards.append(shard)
        self._loaded_paths.add(shard.info.path)
        self._replay_size += shard.info.num_transitions
        self._trim_oldest_if_needed()

    def _trim_oldest_if_needed(self) -> None:
        if self._max_replay_samples is None:
            return
        removed_any = False
        while self._shards and self._replay_size - self._shards[0].info.num_transitions >= self._max_replay_samples:
            removed = self._shards.pop(0)
            self._replay_size -= removed.info.num_transitions
            removed_any = True
        if removed_any:
            self._loaded_paths = {shard.info.path for shard in self._shards}
            self._refresh_z_rl_normalization()

    def _gather(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        cumulative = np.cumsum([shard.info.num_transitions for shard in self._shards])
        shard_indices = np.searchsorted(cumulative, indices, side="right")
        previous = np.concatenate([np.asarray([0]), cumulative[:-1]])

        pieces: dict[str, list[np.ndarray]] = {key: [] for key in (*REQUIRED_REPLAY_KEYS, "episode_success")}
        order: list[np.ndarray] = []
        for shard_index, shard in enumerate(self._shards):
            positions = np.flatnonzero(shard_indices == shard_index)
            if len(positions) == 0:
                continue
            local_indices = indices[positions] - previous[shard_index]
            order.append(positions)
            for key in pieces:
                pieces[key].append(shard.arrays[key][local_indices])

        if not order:
            raise ValueError("No replay samples were gathered.")

        concatenated = {key: np.concatenate(value, axis=0) for key, value in pieces.items()}
        restore_order = np.argsort(np.concatenate(order))
        return {key: value[restore_order] for key, value in concatenated.items()}

    def _refresh_z_rl_normalization(self) -> None:
        if not self._normalize_z_rl or not self._shards:
            self._z_rl_normalization = None
            return
        z_rl = np.concatenate([shard.arrays["z_rl"] for shard in self._shards], axis=0).astype(np.float32)
        mean = np.mean(z_rl, axis=0, dtype=np.float64).astype(np.float32)
        std = np.std(z_rl, axis=0, dtype=np.float64).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        self._z_rl_normalization = ZRLNormalization(mean=mean, std=std)

    def _standardize_z_arrays(self, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._z_rl_normalization is None:
            return arrays
        normalized = dict(arrays)
        mean = self._z_rl_normalization.mean
        std = self._z_rl_normalization.std
        normalized["z_rl"] = ((arrays["z_rl"].astype(np.float32) - mean) / std).astype(np.float32)
        normalized["next_z_rl"] = ((arrays["next_z_rl"].astype(np.float32) - mean) / std).astype(np.float32)
        return normalized


def _shape_from_arrays(arrays: dict[str, np.ndarray]) -> ReplayShape:
    return ReplayShape(
        z_dim=int(arrays["z_rl"].shape[-1]),
        proprio_dim=int(arrays["proprio"].shape[-1]),
        action_horizon=int(arrays["action"].shape[-2]),
        action_dim=int(arrays["action"].shape[-1]),
    )


def _validate_arrays(path: pathlib.Path, arrays: dict[str, np.ndarray]) -> None:
    replay_size = len(arrays["z_rl"])
    if replay_size == 0:
        raise ReplayShardError("shard has zero transitions")
    for key, array in arrays.items():
        if len(array) != replay_size:
            raise ReplayShardError(f"{key} has length {len(array)} but expected {replay_size}")
    if arrays["z_rl"].ndim != 2:
        raise ReplayShardError("z_rl must have shape [N, z_dim]")
    if arrays["proprio"].ndim != 2:
        raise ReplayShardError("proprio must have shape [N, proprio_dim]")
    if arrays["action"].ndim != 3:
        raise ReplayShardError("action must have shape [N, horizon, action_dim]")
    if arrays["reference_action"].shape != arrays["action"].shape:
        raise ReplayShardError("reference_action shape must match action")
    if arrays["next_reference_action"].shape != arrays["action"].shape:
        raise ReplayShardError("next_reference_action shape must match action")
    if arrays["reward_seq"].shape != arrays["action"].shape[:2]:
        raise ReplayShardError("reward_seq must have shape [N, horizon]")
    if arrays["next_z_rl"].shape != arrays["z_rl"].shape:
        raise ReplayShardError("next_z_rl shape must match z_rl")
    if arrays["next_proprio"].shape != arrays["proprio"].shape:
        raise ReplayShardError("next_proprio shape must match proprio")
    if arrays["done"].shape != (replay_size,):
        raise ReplayShardError("done must have shape [N]")
    if not all(np.all(np.isfinite(arrays[key])) for key in REQUIRED_REPLAY_KEYS if key != "done"):
        raise ReplayShardError(f"{path} contains non-finite replay values")


def _load_manifest(data) -> dict:
    return rlt_replay_schema.load_manifest_from_npz(data)


def _validate_manifest(
    manifest: dict,
    shape: ReplayShape,
    *,
    require_formal_replay: bool = False,
    expected_formal_z_dim: int | None = None,
) -> None:
    if not manifest:
        if require_formal_replay:
            raise ReplayShardError("not formal replay trainable: missing manifest")
        return
    if manifest.get("voided") is True:
        raise ReplayShardError("manifest marks shard as voided")
    if manifest.get("train_eligible") is False:
        raise ReplayShardError("manifest train_eligible is false")
    schema_version = manifest.get("schema_version", 1)
    if schema_version != 1:
        raise ReplayShardError(f"unsupported schema_version={schema_version}")
    action_space = manifest.get("action_space", "aloha_exec")
    if action_space != "aloha_exec":
        raise ReplayShardError(f"action_space must be aloha_exec, got {action_space}")
    reward_placement = manifest.get("reward_placement", "terminal_last_train_step")
    if reward_placement != "terminal_last_train_step":
        raise ReplayShardError(f"unsupported reward_placement={reward_placement}")
    action_dim = manifest.get("action_dim")
    if action_dim is not None and int(action_dim) != shape.action_dim:
        raise ReplayShardError(f"manifest action_dim={action_dim} does not match replay action_dim={shape.action_dim}")
    policy_horizon = manifest.get("policy_horizon")
    if policy_horizon is not None and int(policy_horizon) != shape.action_horizon:
        raise ReplayShardError(
            f"manifest policy_horizon={policy_horizon} does not match replay action_horizon={shape.action_horizon}"
        )
    if require_formal_replay:
        try:
            rlt_replay_schema.require_formal_trainable_manifest(
                manifest,
                z_dim=shape.z_dim,
                expected_z_dim=expected_formal_z_dim,
            )
        except ValueError as exc:
            raise ReplayShardError(str(exc)) from exc
