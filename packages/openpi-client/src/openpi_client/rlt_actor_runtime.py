
from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import time
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class RLTActorApplyResult:
    actions: np.ndarray
    applied: bool
    reason: str | None
    actor_dir: str | None
    actor_step: int | None
    delta_norm: float | None
    max_abs_delta: float | None


class RLTActorRuntime:
    def __init__(self, actor_path: str | None, poll_interval_seconds: float = 1.0) -> None:
        self._actor_path = pathlib.Path(actor_path) if actor_path else None
        self._poll_interval_seconds = poll_interval_seconds
        self._last_poll = 0.0
        self._actor = None
        self._config = None
        self._actor_dir: pathlib.Path | None = None
        self._actor_step: int | None = None
        self._reason: str | None = "actor_path_not_configured" if self._actor_path is None else None

    def status(self) -> dict[str, Any]:
        return {
            "actor_ready": self._actor is not None,
            "actor_dir": None if self._actor_dir is None else str(self._actor_dir),
            "actor_step": self._actor_step,
            "actor_load_error": self._reason,
        }

    def maybe_reload(self, *, force: bool = False) -> None:
        if self._actor_path is None:
            self._reason = "actor_path_not_configured"
            return
        now = time.monotonic()
        if not force and now - self._last_poll < self._poll_interval_seconds:
            return
        self._last_poll = now
        try:
            actor_dir = self._resolve_actor_dir()
            if actor_dir == self._actor_dir and self._actor is not None:
                return
            self._load_actor(actor_dir)
        except Exception as exc:
            self._actor = None
            self._config = None
            self._actor_dir = None
            self._actor_step = None
            self._reason = str(exc)
            logging.warning("RLT actor runtime load failed: %s", exc)

    def apply(
        self,
        *,
        reference_actions: np.ndarray,
        z_rl: np.ndarray,
        proprio: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> RLTActorApplyResult:
        reference = np.array(reference_actions, dtype=np.float32, copy=True)
        context = context or {}
        if not bool(context.get("actor_requested", False)):
            return self._fail(reference, "actor_not_requested")
        self.maybe_reload()
        if self._actor is None or self._config is None:
            return self._fail(reference, self._reason or "actor_not_loaded")
        reason = self._validate_shapes(reference, z_rl, proprio)
        if reason is not None:
            return self._fail(reference, reason)
        try:
            from openpi.models import rlt
            import jax
            import jax.numpy as jnp

            horizon = int(self._config.action_horizon)
            prefix = reference[:horizon]
            x = rlt.make_state(
                jnp.asarray(np.asarray(z_rl, dtype=np.float32)[None, :]),
                jnp.asarray(np.asarray(proprio, dtype=np.float32)[None, :]),
            )
            action = self._actor(
                x,
                jnp.asarray(prefix[None, :, :]),
                sample=False,
                intervention_scale=float(context.get("intervention_scale", 1.0)),
            )
            adjusted_prefix = np.asarray(jax.device_get(action[0]), dtype=np.float32)
            if not np.all(np.isfinite(adjusted_prefix)):
                return self._fail(reference, "actor_output_non_finite")
            adjusted = np.array(reference, copy=True)
            adjusted[:horizon] = adjusted_prefix
            delta = adjusted[:horizon] - reference[:horizon]
            return RLTActorApplyResult(
                actions=adjusted,
                applied=True,
                reason=None,
                actor_dir=None if self._actor_dir is None else str(self._actor_dir),
                actor_step=self._actor_step,
                delta_norm=float(np.linalg.norm(delta.reshape(-1))),
                max_abs_delta=float(np.max(np.abs(delta))) if delta.size else 0.0,
            )
        except Exception as exc:
            logging.warning("RLT actor runtime apply failed: %s", exc)
            return self._fail(reference, str(exc))

    def _fail(self, reference: np.ndarray, reason: str | None) -> RLTActorApplyResult:
        return RLTActorApplyResult(
            actions=np.array(reference, copy=True),
            applied=False,
            reason=reason,
            actor_dir=None if self._actor_dir is None else str(self._actor_dir),
            actor_step=self._actor_step,
            delta_norm=None,
            max_abs_delta=None,
        )

    def _resolve_actor_dir(self) -> pathlib.Path:
        assert self._actor_path is not None
        path = self._actor_path
        if path.name == "LATEST":
            if not path.exists():
                raise FileNotFoundError(f"actor LATEST not found: {path}")
            return pathlib.Path(path.read_text().strip())
        if path.is_dir() and (path / "LATEST").exists():
            return pathlib.Path((path / "LATEST").read_text().strip())
        if path.is_dir():
            return path
        raise FileNotFoundError(f"actor path not found: {path}")

    def _load_actor(self, actor_dir: pathlib.Path) -> None:
        metadata_path = actor_dir / "metadata.json"
        actor_path = actor_dir / "actor.msgpack"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {actor_dir}")
        if not actor_path.exists():
            raise FileNotFoundError(f"actor.msgpack not found in {actor_dir}")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("type") != "rlt_inference_actor":
            raise ValueError("metadata type is not rlt_inference_actor")
        if "rlt_config" not in metadata:
            raise ValueError("metadata missing rlt_config")

        from flax import nnx
        from flax import serialization
        from openpi.models import rlt

        config = rlt.RLTConfig(**metadata["rlt_config"])
        actor = rlt.RLTActor(config, rngs=nnx.Rngs(0))
        state = nnx.state(actor)
        pure_state = serialization.from_bytes(state.to_pure_dict(), actor_path.read_bytes())
        state.replace_by_pure_dict(pure_state)
        nnx.update(actor, state)
        self._actor = actor
        self._config = config
        self._actor_dir = actor_dir
        self._actor_step = None if metadata.get("step") is None else int(metadata["step"])
        self._reason = None

    def _validate_shapes(self, reference: np.ndarray, z_rl: np.ndarray, proprio: np.ndarray) -> str | None:
        if reference.ndim != 2:
            return "reference_actions must have shape [horizon, action_dim]"
        if reference.shape[0] < int(self._config.action_horizon):
            return f"reference horizon {reference.shape[0]} < actor horizon {self._config.action_horizon}"
        if reference.shape[1] != int(self._config.action_dim):
            return f"action_dim mismatch: reference={reference.shape[1]} actor={self._config.action_dim}"
        if np.asarray(z_rl).shape != (int(self._config.z_dim),):
            return f"z_rl shape mismatch: got {np.asarray(z_rl).shape}, expected {(int(self._config.z_dim),)}"
        if np.asarray(proprio).shape != (int(self._config.proprio_dim),):
            return f"proprio shape mismatch: got {np.asarray(proprio).shape}, expected {(int(self._config.proprio_dim),)}"
        return None
