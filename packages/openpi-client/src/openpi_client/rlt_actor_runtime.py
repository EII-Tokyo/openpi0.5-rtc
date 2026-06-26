
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
    reference_q_value: float | None = None
    actor_q_value: float | None = None
    q_advantage: float | None = None
    key_region_probability: float | None = None
    gate_reason: str | None = None
    critic_ready: bool = False
    critic_gate_enabled: bool = False
    action_start_index: int | None = None
    action_horizon: int | None = None
    action_end_index: int | None = None


class RLTActorRuntime:
    def __init__(self, actor_path: str | None, poll_interval_seconds: float = 1.0) -> None:
        self._actor_path = pathlib.Path(actor_path) if actor_path else None
        self._poll_interval_seconds = poll_interval_seconds
        self._last_poll = 0.0
        self._actor = None
        self._critic = None
        self._config = None
        self._actor_dir: pathlib.Path | None = None
        self._actor_step: int | None = None
        self._rng_key = None
        self._reason: str | None = "actor_path_not_configured" if self._actor_path is None else None

    def status(self) -> dict[str, Any]:
        return {
            "actor_ready": self._actor is not None,
            "critic_ready": self._critic is not None,
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
            self._critic = None
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
        action_start_index: int = 0,
    ) -> RLTActorApplyResult:
        reference = np.array(reference_actions, dtype=np.float32, copy=True)
        context = context or {}
        if not bool(context.get("actor_requested", False)):
            return self._fail(reference, "actor_not_requested")
        self.maybe_reload(force=True)
        if self._actor is None or self._config is None:
            return self._fail(reference, self._reason or "actor_not_loaded")
        action_start_index = int(action_start_index)
        reason = self._validate_shapes(reference, z_rl, proprio, action_start_index)
        if reason is not None:
            return self._fail(reference, reason)
        try:
            from openpi.models import rlt
            import jax
            import jax.numpy as jnp

            if self._rng_key is None:
                self._rng_key = jax.random.key(time.time_ns() % (2**31 - 1))
            self._rng_key, actor_rng = jax.random.split(self._rng_key)
            horizon = int(self._config.action_horizon)
            action_end_index = action_start_index + horizon
            prefix = reference[action_start_index:action_end_index]
            x = rlt.make_state(
                jnp.asarray(np.asarray(z_rl, dtype=np.float32)[None, :]),
                jnp.asarray(np.asarray(proprio, dtype=np.float32)[None, :]),
            )
            prefix_jax = jnp.asarray(prefix[None, :, :])
            action = self._actor(
                x,
                prefix_jax,
                rng=actor_rng,
                sample=False,
                intervention_scale=float(context.get("intervention_scale", 1.0)),
            )
            adjusted_prefix = np.asarray(jax.device_get(action[0]), dtype=np.float32)
            if not np.all(np.isfinite(adjusted_prefix)):
                return self._fail(reference, "actor_output_non_finite")
            adjusted_prefix = _ramp_adjusted_action(
                reference_prefix=prefix,
                adjusted_prefix=adjusted_prefix,
                ramp_steps=int(context.get("intervention_ramp_steps", 0) or 0),
            )
            adjusted_prefix = _clip_adjusted_action_delta(
                reference_prefix=prefix,
                adjusted_prefix=adjusted_prefix,
                max_delta=float(context.get("max_delta", 0.0) or 0.0),
            )
            adjusted = np.array(reference, copy=True)
            adjusted[action_start_index:action_end_index] = adjusted_prefix
            delta = adjusted[action_start_index:action_end_index] - reference[action_start_index:action_end_index]
            critic_gate_enabled = bool(context.get("critic_gate_enabled", False))
            gate_margin = float(context.get("critic_gate_margin", 0.0) or 0.0)
            gate_temperature = max(1e-6, float(context.get("critic_gate_temperature", 0.05) or 0.05))
            q_metrics = self._critic_metrics(x, prefix_jax, jnp.asarray(adjusted_prefix[None, :, :]), gate_margin, gate_temperature)
            if critic_gate_enabled and self._critic is None:
                return self._fail(
                    reference,
                    "critic_gate_critic_not_loaded",
                    gate_reason="critic_gate_critic_not_loaded",
                    critic_gate_enabled=True,
                )
            if critic_gate_enabled and q_metrics.get("q_advantage") is not None and q_metrics["q_advantage"] < gate_margin:
                return self._fail(
                    reference,
                    "critic_gate_q_advantage_low",
                    gate_reason="critic_gate_q_advantage_low",
                    critic_gate_enabled=True,
                    **q_metrics,
                )
            gate_reason = "critic_gate_actor_active" if critic_gate_enabled else None
            return RLTActorApplyResult(
                actions=adjusted,
                applied=True,
                reason=None,
                actor_dir=None if self._actor_dir is None else str(self._actor_dir),
                actor_step=self._actor_step,
                delta_norm=float(np.linalg.norm(delta.reshape(-1))),
                max_abs_delta=float(np.max(np.abs(delta))) if delta.size else 0.0,
                gate_reason=gate_reason,
                critic_gate_enabled=critic_gate_enabled,
                action_start_index=action_start_index,
                action_horizon=horizon,
                action_end_index=action_end_index,
                **q_metrics,
            )
        except Exception as exc:
            logging.warning("RLT actor runtime apply failed: %s", exc)
            return self._fail(reference, str(exc))

    def _fail(
        self,
        reference: np.ndarray,
        reason: str | None,
        *,
        reference_q_value: float | None = None,
        actor_q_value: float | None = None,
        q_advantage: float | None = None,
        key_region_probability: float | None = None,
        gate_reason: str | None = None,
        critic_ready: bool | None = None,
        critic_gate_enabled: bool = False,
    ) -> RLTActorApplyResult:
        return RLTActorApplyResult(
            actions=np.array(reference, copy=True),
            applied=False,
            reason=reason,
            actor_dir=None if self._actor_dir is None else str(self._actor_dir),
            actor_step=self._actor_step,
            delta_norm=None,
            max_abs_delta=None,
            reference_q_value=reference_q_value,
            actor_q_value=actor_q_value,
            q_advantage=q_advantage,
            key_region_probability=key_region_probability,
            gate_reason=gate_reason,
            critic_ready=self._critic is not None if critic_ready is None else bool(critic_ready),
            critic_gate_enabled=critic_gate_enabled,
        )

    def _critic_metrics(self, x, reference_action, actor_action, margin: float, temperature: float) -> dict[str, Any]:
        if self._critic is None:
            return {
                "reference_q_value": None,
                "actor_q_value": None,
                "q_advantage": None,
                "key_region_probability": None,
                "critic_ready": False,
            }
        import jax

        reference_q = float(jax.device_get(self._critic.min_q(x, reference_action)[0]))
        actor_q = float(jax.device_get(self._critic.min_q(x, actor_action)[0]))
        advantage = actor_q - reference_q
        logit = float(np.clip((advantage - margin) / temperature, -60.0, 60.0))
        probability = float(1.0 / (1.0 + np.exp(-logit)))
        return {
            "reference_q_value": reference_q,
            "actor_q_value": actor_q,
            "q_advantage": advantage,
            "key_region_probability": probability,
            "critic_ready": True,
        }

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
        critic_path = actor_dir / "critic.msgpack"
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
        critic = None
        if critic_path.exists():
            critic = rlt.RLTTwinCritic(config, rngs=nnx.Rngs(0))
            critic_state = nnx.state(critic)
            pure_critic_state = serialization.from_bytes(critic_state.to_pure_dict(), critic_path.read_bytes())
            critic_state.replace_by_pure_dict(pure_critic_state)
            nnx.update(critic, critic_state)
        self._actor = actor
        self._critic = critic
        self._config = config
        self._actor_dir = actor_dir
        self._actor_step = None if metadata.get("step") is None else int(metadata["step"])
        self._reason = None
        logging.info(
            "Loaded RLT actor runtime: step=%s path=%s critic_ready=%s",
            self._actor_step,
            actor_dir,
            critic is not None,
        )

    def _validate_shapes(
        self,
        reference: np.ndarray,
        z_rl: np.ndarray,
        proprio: np.ndarray,
        action_start_index: int,
    ) -> str | None:
        if reference.ndim != 2:
            return "reference_actions must have shape [horizon, action_dim]"
        if action_start_index < 0:
            return f"action_start_index must be non-negative, got {action_start_index}"
        required_horizon = action_start_index + int(self._config.action_horizon)
        if reference.shape[0] < required_horizon:
            return (
                f"reference horizon {reference.shape[0]} < action_start_index "
                f"{action_start_index} + actor horizon {self._config.action_horizon}"
            )
        if reference.shape[1] != int(self._config.action_dim):
            return f"action_dim mismatch: reference={reference.shape[1]} actor={self._config.action_dim}"
        if np.asarray(z_rl).shape != (int(self._config.z_dim),):
            return f"z_rl shape mismatch: got {np.asarray(z_rl).shape}, expected {(int(self._config.z_dim),)}"
        if np.asarray(proprio).shape != (int(self._config.proprio_dim),):
            return f"proprio shape mismatch: got {np.asarray(proprio).shape}, expected {(int(self._config.proprio_dim),)}"
        return None


def _ramp_adjusted_action(
    *,
    reference_prefix: np.ndarray,
    adjusted_prefix: np.ndarray,
    ramp_steps: int,
) -> np.ndarray:
    """Blend actor delta in over the first ramp_steps frames.

    This preserves the actor's target action after the ramp while preventing a
    hard discontinuity at the RTC chunk handoff boundary.
    """
    if ramp_steps <= 1:
        return adjusted_prefix
    horizon = adjusted_prefix.shape[0]
    steps = min(int(ramp_steps), horizon)
    weights = np.ones((horizon,), dtype=np.float32)
    weights[:steps] = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    delta = adjusted_prefix - reference_prefix
    return reference_prefix + weights[:, None] * delta


def _clip_adjusted_action_delta(
    *,
    reference_prefix: np.ndarray,
    adjusted_prefix: np.ndarray,
    max_delta: float,
) -> np.ndarray:
    if max_delta <= 0:
        return adjusted_prefix
    delta = adjusted_prefix - reference_prefix
    clipped_delta = np.clip(delta, -float(max_delta), float(max_delta))
    return reference_prefix + clipped_delta
