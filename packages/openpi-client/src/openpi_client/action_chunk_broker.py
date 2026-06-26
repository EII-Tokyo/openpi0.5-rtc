from typing import Dict
import json
import logging
import pathlib
import time
import threading
import numpy as np
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from openpi_client.rlt_actor_runtime import RLTActorRuntime


_LEFT_ARM_ACTION_INDICES = (0, 1, 2, 3, 4, 5, 6)
_RIGHT_ARM_ACTION_INDICES = (7, 8, 9, 10, 11, 12, 13)


def _propagate_actor_residual_for_guidance(
    *,
    reference_actions: np.ndarray,
    adjusted_actions: np.ndarray,
    action_start_index: int,
    action_end_index: int,
    guidance_start_index: int,
    trend_window: int = 5,
    start_weight: float = 0.7,
    same_direction_scale: float = 0.35,
    opposing_direction_scale: float = 1.0,
    affected_indices: tuple[int, ...] = _LEFT_ARM_ACTION_INDICES,
) -> np.ndarray:
    reference = np.asarray(reference_actions, dtype=np.float32)
    adjusted = np.asarray(adjusted_actions, dtype=np.float32)
    guidance = np.array(adjusted, dtype=np.float32, copy=True)
    if reference.shape != adjusted.shape or reference.ndim != 2:
        return guidance
    affected_indices = tuple(index for index in affected_indices if 0 <= index < reference.shape[1])
    guidance = np.array(reference, dtype=np.float32, copy=True)
    if not affected_indices:
        return guidance
    guidance[:, affected_indices] = adjusted[:, affected_indices]

    action_start_index = max(0, int(action_start_index))
    action_end_index = min(int(action_end_index), reference.shape[0])
    if action_end_index <= action_start_index:
        return guidance

    residual = adjusted[action_start_index:action_end_index] - reference[action_start_index:action_end_index]
    window = max(1, min(int(trend_window), residual.shape[0]))
    trend = np.mean(residual[-window:], axis=0)

    tail_start = max(int(guidance_start_index), action_end_index)
    if tail_start >= reference.shape[0]:
        return guidance

    anchor = reference[action_end_index - 1]
    tail_trend = reference[tail_start] - anchor
    same_direction = np.sign(trend) == np.sign(tail_trend)
    active_tail = (np.abs(trend) > 1e-6) & (np.abs(tail_trend) > 1e-6)
    trend_scale = np.where(
        active_tail & same_direction,
        float(same_direction_scale),
        float(opposing_direction_scale),
    ).astype(np.float32)
    trend = trend * trend_scale

    weights = np.linspace(float(start_weight), 0.0, reference.shape[0] - tail_start, dtype=np.float32)[:, None]
    guidance[tail_start:, affected_indices] = (
        reference[tail_start:, affected_indices] + weights * trend[None, affected_indices]
    )
    return guidance


def _freeze_right_arm_actions(
    actions: np.ndarray,
    state: np.ndarray,
    *,
    right_arm_indices: tuple[int, ...] = _RIGHT_ARM_ACTION_INDICES,
) -> np.ndarray:
    frozen = np.array(actions, dtype=np.float32, copy=True)
    state = np.asarray(state, dtype=np.float32)
    if not right_arm_indices:
        return frozen
    if frozen.ndim != 2 or state.ndim != 1:
        return frozen
    if frozen.shape[1] <= max(right_arm_indices) or state.shape[0] <= max(right_arm_indices):
        return frozen
    frozen[:, right_arm_indices] = state[list(right_arm_indices)]
    return frozen


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        model_dir: str | None = None,
        adapt_to_pi: bool = True,
        use_rtc: bool = True,
        rlt_actor_path: str | None = None,
        rlt_actor_poll_interval: float = 1.0,
        rlt_actor_runtime=None,
    ):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None
        self._last_guidance_actions: np.ndarray | None = None
        self._last_origin_actions: np.ndarray | None = None
        self._last_reference_actions: np.ndarray | None = None
        self._last_rlt_context_signature: tuple | None = None
        self._background_results: Dict[str, np.ndarray] | None = None
        self._background_guidance_actions: np.ndarray | None = None
        self._background_rlt_context_signature: tuple | None = None
        self._right_arm_hold_state: np.ndarray | None = None
        self._background_running: bool = False
        self._cache_generation: int = 0
        self._cache_lock = threading.RLock()
        self._explicit_rlt_gate_enabled: bool | None = None
        self._explicit_rlt_gate_epoch: int | None = None
        self._rlt_actor = rlt_actor_runtime
        if self._rlt_actor is None and rlt_actor_path is not None:
            self._rlt_actor = RLTActorRuntime(rlt_actor_path, poll_interval_seconds=rlt_actor_poll_interval)

        self._obs: Dict[str, np.ndarray] | None = None
        self._s = 25
        self._d = 10
        self._use_rtc = use_rtc
        self._norm_stats = None
        self._joint_signs = np.ones(14)

        if self._use_rtc:
            if model_dir is None:
                raise ValueError("model_dir is required when use_rtc=True.")
            self._norm_stats, self._joint_signs = self._load_runtime_assets(model_dir, adapt_to_pi)

            self._infer_thread = threading.Thread(target=self._background_infer)
            self._infer_thread.start()

    @staticmethod
    def _joint_flip_mask() -> np.ndarray:
        return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])

    @staticmethod
    def _resolve_asset_id(model_dir: str) -> str:
        assets_dir = pathlib.Path(model_dir) / "assets"
        asset_dirs = sorted(
            p.name for p in assets_dir.iterdir() if p.is_dir() and (p / "norm_stats.json").exists()
        ) if assets_dir.exists() else []

        if len(asset_dirs) == 1:
            return asset_dirs[0]
        if "trossen" in asset_dirs:
            return "trossen"
        raise ValueError(
            f"Could not determine asset_id for checkpoint '{model_dir}'. "
            f"Found assets={asset_dirs}"
        )

    def _load_runtime_assets(self, model_dir: str, adapt_to_pi: bool):
        asset_id = self._resolve_asset_id(model_dir)
        norm_stats_path = pathlib.Path(model_dir) / "assets" / asset_id / "norm_stats.json"
        norm_stats = json.loads(norm_stats_path.read_text())["norm_stats"]
        joint_signs = self._joint_flip_mask() if adapt_to_pi else np.ones(14)
        return norm_stats, joint_signs

    def _background_infer(self):
        while True:
            if self._cur_step == self._s:
                total_start = time.monotonic()
                with self._cache_lock:
                    self._background_running = True
                    generation = self._cache_generation
                    obs = dict(self._obs or {})
                    context_signature = self._rlt_context_signature_from_obs(obs)
                    last_results = self._last_results
                    last_guidance_actions = self._last_guidance_actions
                    last_origin_actions = self._last_origin_actions
                    if last_results is None or last_origin_actions is None:
                        self._background_running = False
                        continue

                # flip, normalize joint actions
                # norm_action = (np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1]) * self._last_results["actions"] - np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1]) * self._obs["state"][:14] - np.array(self._norm_stats["actions"]["mean"])[:14]) / (np.array(self._norm_stats["actions"]["std"])[:14] + 1e-6)
                prep_start = time.monotonic()
                q01 = np.array(self._norm_stats["actions"]["q01"])[:14]
                q99 = np.array(self._norm_stats["actions"]["q99"])[:14]
                prev_actions = last_guidance_actions if last_guidance_actions is not None else last_results["actions"]
                scaled = self._joint_signs * (prev_actions - obs["state"][:14])
                norm_action = (scaled - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
                
                # get normalized gripper action
                norm_action[:, 6] = last_origin_actions[:, 6]
                norm_action[:, 13] = last_origin_actions[:, 13]
                
                zeros_padding = np.zeros((norm_action.shape[0], 18))
                norm_action = np.concatenate([norm_action, zeros_padding], axis=1)
                prep_ms = (time.monotonic() - prep_start) * 1000.0
                # np.savetxt("norm_action.txt", norm_action, fmt='%.6f')
                # np.savetxt("last_origin_actions.txt", self._last_origin_actions, fmt='%.6f')
                rpc_start = time.monotonic()
                policy_obs = self._policy_obs(obs)
                self._background_results = self._policy.infer(policy_obs, norm_action, self._use_rtc)
                background_results = self._apply_rlt_actor_to_policy_results(
                    self._background_results,
                    obs,
                    action_start_index=self._d,
                )
                with self._cache_lock:
                    if generation == self._cache_generation:
                        self._background_results = background_results
                        self._background_guidance_actions = background_results.get(
                            "rtc_guidance_actions",
                            background_results["actions"],
                        )
                        self._background_rlt_context_signature = context_signature
                rpc_ms = (time.monotonic() - rpc_start) * 1000.0
                # break
                # 将后面18列都设为0
                # modified_actions = None
                # if self._last_origin_actions is not None:
                #     modified_actions = self._last_origin_actions.copy()
                #     modified_actions[:, 14:] = 0
                
                # self._background_results = self._policy.infer(self._obs, modified_actions, self._use_rtc)

                with self._cache_lock:
                    if generation == self._cache_generation:
                        self._background_running = False
                total_ms = (time.monotonic() - total_start) * 1000.0
                server_timing = background_results.get("server_timing", {})
                policy_timing = background_results.get("policy_timing", {})
                server_infer_ms = server_timing.get("infer_ms")
                prev_total_ms = server_timing.get("prev_total_ms")
                policy_infer_ms = policy_timing.get("infer_ms")
                overhead_ms = None
                if server_infer_ms is not None:
                    overhead_ms = rpc_ms - float(server_infer_ms)
                log_parts = [
                    "RTC infer timing:",
                    f"prep_ms={prep_ms:.1f}",
                    f"rpc_ms={rpc_ms:.1f}",
                    f"total_ms={total_ms:.1f}",
                ]
                if server_infer_ms is not None:
                    log_parts.append(f"server_infer_ms={float(server_infer_ms):.1f}")
                if policy_infer_ms is not None:
                    log_parts.append(f"policy_infer_ms={float(policy_infer_ms):.1f}")
                if overhead_ms is not None:
                    log_parts.append(f"rpc_overhead_ms={float(overhead_ms):.1f}")
                if prev_total_ms is not None:
                    log_parts.append(f"prev_total_ms={float(prev_total_ms):.1f}")
                logging.info(" ".join(log_parts))
            else:
                time.sleep(0.01)

    def rlt_actor_status(self) -> dict:
        if self._rlt_actor is None:
            return {
                "actor_ready": False,
                "critic_ready": False,
                "actor_dir": None,
                "actor_step": None,
                "actor_load_error": "actor_runtime_not_configured",
            }
        maybe_reload = getattr(self._rlt_actor, "maybe_reload", None)
        if maybe_reload is not None:
            maybe_reload(force=True)
        status = getattr(self._rlt_actor, "status", None)
        if status is None:
            return {
                "actor_ready": False,
                "critic_ready": False,
                "actor_dir": None,
                "actor_step": None,
                "actor_load_error": "actor_runtime_status_unavailable",
            }
        return dict(status())

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        context_signature = self._rlt_context_signature_from_obs(obs)
        if self._use_rtc:
            if self._last_results is not None and context_signature != self._last_rlt_context_signature:
                self._clear_cached_results()
            # init     
            if self._last_results is None:
                if not self._refresh_current_results(obs, context_signature):
                    context_signature = self._rlt_context_signature_from_obs(obs)
                    self._refresh_current_results(obs, context_signature)

            results = self._slice_result_cache(self._last_results)
            self._obs = obs
            self._cur_step += 1

            # if current step equals s+d, wait for background inference to complete
            if self._cur_step == self._s + self._d:
                while self._background_running:
                    time.sleep(0.01)
                current_signature = self._rlt_context_signature_from_obs(obs)
                if (
                    self._background_results is not None
                    and self._background_rlt_context_signature == current_signature
                ):
                    self._last_origin_actions = self._background_results["origin_actions"]
                    self._last_reference_actions = self._background_results.get(
                        "reference_actions",
                        self._background_results["actions"],
                    )
                    self._last_state = self._background_results["state"]
                    self._last_guidance_actions = (
                        self._background_guidance_actions
                        if self._background_guidance_actions is not None
                        else self._background_results["actions"]
                    )
                    self._last_results = self._build_step_result_cache(self._background_results)
                    self._last_rlt_context_signature = self._background_rlt_context_signature
                    self._cur_step -= self._s
                else:
                    self._refresh_current_results(obs, current_signature)
            # print(results)
            return results
        else:
            if self._last_results is not None and context_signature != self._last_rlt_context_signature:
                self._clear_cached_results()
            if self._last_results is None:
                if not self._refresh_current_results(obs, context_signature):
                    context_signature = self._rlt_context_signature_from_obs(obs)
                    self._refresh_current_results(obs, context_signature)

            results = self._slice_result_cache(self._last_results)
            self._cur_step += 1

            if self._cur_step >= self._action_horizon:
                self._last_results = None

            return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._clear_cached_results()

    def _clear_cached_results(self) -> None:
        with self._cache_lock:
            self._cache_generation += 1
            self._last_results = None
            self._last_guidance_actions = None
            self._last_origin_actions = None
            self._last_reference_actions = None
            self._last_rlt_context_signature = None
            self._background_results = None
            self._background_guidance_actions = None
            self._background_rlt_context_signature = None
            self._right_arm_hold_state = None
            self._background_running = False
            self._cur_step = 0

    def flush_action_cache(self, reason: str | None = None) -> None:
        del reason
        self._clear_cached_results()

    def set_rlt_gate(self, *, enabled: bool, epoch: int, reason: str | None = None) -> None:
        logging.info("ActionChunkBroker RLT gate set: enabled=%s epoch=%s reason=%s", enabled, epoch, reason)
        with self._cache_lock:
            self._explicit_rlt_gate_enabled = bool(enabled)
            self._explicit_rlt_gate_epoch = int(epoch)
            self._clear_cached_results()

    def _policy_obs(self, obs: Dict) -> Dict:
        return {key: value for key, value in obs.items() if key != "rlt_context"}

    def _rlt_context_from_obs(self, obs: Dict) -> dict:
        context = obs.get("rlt_context") or {}
        context = dict(context) if isinstance(context, dict) else {}
        with self._cache_lock:
            if self._explicit_rlt_gate_enabled is not None:
                context["actor_requested"] = bool(self._explicit_rlt_gate_enabled)
                context["actor_effective"] = bool(self._explicit_rlt_gate_enabled)
                context["manual_actor_requested"] = bool(self._explicit_rlt_gate_enabled)
                context["rlt_context_epoch"] = self._explicit_rlt_gate_epoch
                if not self._explicit_rlt_gate_enabled:
                    context["actor_locked_reason"] = "actor_not_requested"
        return context

    def _rlt_context_signature_from_obs(self, obs: Dict) -> tuple:
        context = self._rlt_context_from_obs(obs)
        return tuple(
            (key, context.get(key))
            for key in (
                "rlt_context_epoch",
                "phase",
                "active_key_region_id",
                "actor_requested",
                "manual_actor_requested",
                "intervention_scale",
                "max_delta",
                "critic_gate_enabled",
                "critic_gate_margin",
                "critic_gate_temperature",
                "actor_ready",
                "loaded_actor_step",
            )
        )

    def _refresh_current_results(self, obs: Dict, context_signature: tuple | None = None) -> bool:
        if context_signature is None:
            context_signature = self._rlt_context_signature_from_obs(obs)
        with self._cache_lock:
            generation = self._cache_generation
        if self._use_rtc:
            policy_results = self._policy.infer(self._policy_obs(obs), None, self._use_rtc)
            policy_results = self._apply_rlt_actor_to_policy_results(policy_results, obs, action_start_index=0)
        else:
            policy_results = self._policy.infer(self._policy_obs(obs))
            policy_results = self._apply_rlt_actor_to_policy_results(policy_results, obs, action_start_index=0)
        with self._cache_lock:
            if generation != self._cache_generation:
                return False
            if self._use_rtc:
                self._last_origin_actions = policy_results["origin_actions"]
                self._last_reference_actions = policy_results.get("reference_actions", policy_results["actions"])
                self._last_state = policy_results["state"]
                self._last_guidance_actions = policy_results.get("rtc_guidance_actions", policy_results["actions"])
            self._last_results = self._build_step_result_cache(policy_results)
            self._last_rlt_context_signature = context_signature
            self._cur_step = 0
            return True

    def _apply_rlt_actor_to_policy_results(
        self,
        policy_results: Dict[str, np.ndarray],
        obs: Dict,
        *,
        action_start_index: int = 0,
    ) -> Dict[str, np.ndarray]:
        policy_results = dict(policy_results)
        context = self._rlt_context_from_obs(obs)
        policy_results["rlt_context_epoch"] = context.get("rlt_context_epoch")
        reference_actions = np.array(policy_results["actions"], dtype=np.float32, copy=True)
        policy_results["reference_actions"] = reference_actions
        if self._rlt_actor is None:
            policy_results.update(
                {
                    "rlt_actor_applied": False,
                    "rlt_actor_reason": "actor_runtime_not_configured",
                    "rlt_actor_step": None,
                    "rlt_actor_dir": None,
                    "rlt_actor_delta_norm": None,
                    "rlt_actor_max_abs_delta": None,
                    "rlt_reference_q": None,
                    "rlt_actor_q": None,
                    "rlt_q_advantage": None,
                    "rlt_key_region_probability": None,
                    "rlt_gate_reason": None,
                    "rlt_critic_ready": False,
                    "rlt_critic_gate_enabled": False,
                }
            )
            return policy_results

        z_rl = policy_results.get("z_rl", policy_results.get("rl_token"))
        proprio = policy_results.get("proprio", policy_results.get("state"))
        if z_rl is None or proprio is None:
            policy_results.update(
                {
                    "rlt_actor_applied": False,
                    "rlt_actor_reason": "missing_z_rl_or_proprio",
                    "rlt_actor_step": None,
                    "rlt_actor_dir": None,
                    "rlt_actor_delta_norm": None,
                    "rlt_actor_max_abs_delta": None,
                    "rlt_reference_q": None,
                    "rlt_actor_q": None,
                    "rlt_q_advantage": None,
                    "rlt_key_region_probability": None,
                    "rlt_gate_reason": None,
                    "rlt_critic_ready": False,
                    "rlt_critic_gate_enabled": False,
                }
            )
            return policy_results

        try:
            if context.get("phase") == "key_region" or self._explicit_rlt_gate_enabled is not None:
                logging.info(
                    "RLT actor apply context: requested=%s effective=%s epoch=%s phase=%s explicit_gate=%s start_index=%s",
                    context.get("actor_requested"),
                    context.get("actor_effective"),
                    context.get("rlt_context_epoch"),
                    context.get("phase"),
                    self._explicit_rlt_gate_enabled,
                    action_start_index,
                )
            result = self._rlt_actor.apply(
                reference_actions=reference_actions,
                z_rl=np.asarray(z_rl, dtype=np.float32),
                proprio=np.asarray(proprio, dtype=np.float32),
                context=context,
                action_start_index=action_start_index,
            )
        except Exception as exc:
            logging.exception("RLT actor failed; using reference VLA actions")
            policy_results.update(
                {
                    "rlt_actor_applied": False,
                    "rlt_actor_reason": str(exc),
                    "rlt_actor_step": None,
                    "rlt_actor_dir": None,
                    "rlt_actor_delta_norm": None,
                    "rlt_actor_max_abs_delta": None,
                    "rlt_reference_q": None,
                    "rlt_actor_q": None,
                    "rlt_q_advantage": None,
                    "rlt_key_region_probability": None,
                    "rlt_gate_reason": None,
                    "rlt_critic_ready": False,
                    "rlt_critic_gate_enabled": False,
                }
            )
            return policy_results

        if result.applied:
            policy_results["actions"] = np.asarray(result.actions, dtype=np.float32)
        action_start = result.action_start_index if result.action_start_index is not None else action_start_index
        action_end = result.action_end_index
        right_arm_frozen = False
        if context.get("phase") == "key_region" or bool(context.get("actor_requested", False)):
            actor_requested = bool(context.get("actor_requested", False))
            if actor_requested:
                before_freeze = np.asarray(policy_results["actions"], dtype=np.float32)
                robot_state = np.asarray(obs.get("state", proprio), dtype=np.float32)[:14]
                if self._right_arm_hold_state is None:
                    self._right_arm_hold_state = np.array(robot_state, dtype=np.float32, copy=True)
                frozen_actions = _freeze_right_arm_actions(before_freeze, self._right_arm_hold_state)
                if before_freeze.ndim == 2 and before_freeze.shape[1] > max(_RIGHT_ARM_ACTION_INDICES):
                    right_arm_frozen = not np.allclose(
                        before_freeze[:, _RIGHT_ARM_ACTION_INDICES],
                        frozen_actions[:, _RIGHT_ARM_ACTION_INDICES],
                    )
                policy_results["actions"] = frozen_actions
            else:
                self._right_arm_hold_state = None
        else:
            self._right_arm_hold_state = None
        if result.applied and action_end is not None:
            policy_results["rtc_guidance_actions"] = _propagate_actor_residual_for_guidance(
                reference_actions=reference_actions,
                adjusted_actions=policy_results["actions"],
                action_start_index=action_start,
                action_end_index=action_end,
                guidance_start_index=self._s,
            )
        elif right_arm_frozen:
            policy_results["rtc_guidance_actions"] = policy_results["actions"]
        policy_results.update(
            {
                "rlt_actor_applied": bool(result.applied),
                "rlt_actor_reason": result.reason,
                "rlt_actor_step": result.actor_step,
                "rlt_actor_dir": result.actor_dir,
                "rlt_actor_delta_norm": result.delta_norm,
                "rlt_actor_max_abs_delta": result.max_abs_delta,
                "rlt_reference_q": result.reference_q_value,
                "rlt_actor_q": result.actor_q_value,
                "rlt_q_advantage": result.q_advantage,
                "rlt_key_region_probability": result.key_region_probability,
                "rlt_gate_reason": result.gate_reason,
                "rlt_critic_ready": result.critic_ready,
                "rlt_critic_gate_enabled": result.critic_gate_enabled,
                "rlt_action_limited": False,
                "rlt_right_arm_frozen": right_arm_frozen,
            }
        )
        return policy_results

    def _build_step_result_cache(self, policy_results: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        reference_actions = policy_results.get("reference_actions", policy_results["actions"])
        cached: Dict[str, np.ndarray] = {
            "actions": policy_results["actions"],
            "reference_action": reference_actions,
            "action_full": policy_results.get("action_full", policy_results["actions"]),
            "reference_action_full": reference_actions,
        }
        if "origin_actions" in policy_results:
            cached["origin_actions"] = policy_results["origin_actions"]
        for source_key, target_key in (
            ("z_rl", "z_rl"),
            ("rl_token", "z_rl"),
            ("proprio", "proprio"),
            ("state", "proprio"),
            ("rlt_actor_applied", "rlt_actor_applied"),
            ("rlt_actor_reason", "rlt_actor_reason"),
            ("rlt_actor_step", "rlt_actor_step"),
            ("rlt_actor_dir", "rlt_actor_dir"),
            ("rlt_actor_delta_norm", "rlt_actor_delta_norm"),
            ("rlt_actor_max_abs_delta", "rlt_actor_max_abs_delta"),
            ("rlt_reference_q", "rlt_reference_q"),
            ("rlt_actor_q", "rlt_actor_q"),
            ("rlt_q_advantage", "rlt_q_advantage"),
            ("rlt_key_region_probability", "rlt_key_region_probability"),
            ("rlt_gate_reason", "rlt_gate_reason"),
            ("rlt_critic_ready", "rlt_critic_ready"),
            ("rlt_critic_gate_enabled", "rlt_critic_gate_enabled"),
            ("rlt_context_epoch", "rlt_context_epoch"),
        ):
            if source_key in policy_results and target_key not in cached:
                cached[target_key] = policy_results[source_key]
        return cached

    def _slice_result_cache(self, results: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        sliced = {}
        action_horizon = results["actions"].shape[0]
        for key, value in results.items():
            if key.endswith("_full"):
                sliced[key] = value
            elif isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == action_horizon:
                sliced[key] = value[self._cur_step, ...]
            else:
                sliced[key] = value
        return sliced
