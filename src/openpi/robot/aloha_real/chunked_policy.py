from typing import Dict, Literal
import json
import logging
import pathlib
import time
import threading
import numpy as np
import tree
from typing_extensions import override

from openpi.serving import base_policy as _base_policy


class ChunkedPolicy(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    _RLT_REPLAY_KEYS = {
        "rlt_token",
        "rlt_embeddings",
        "rlt_mask",
        "rlt_state",
        "rlt_state_is_normalized",
        "rlt_state_normalization",
        "rlt_reference_action_chunk",
        "rlt_policy_action_chunk",
        "rlt_actor_enabled",
        "rlt_chunk_q1",
        "rlt_chunk_q2",
        "rlt_chunk_q_min",
        "rlt_vla_chunk_q1",
        "rlt_vla_chunk_q2",
        "rlt_vla_chunk_q_min",
        "rlt_actor_chunk_q1",
        "rlt_actor_chunk_q2",
        "rlt_actor_chunk_q_min",
    }

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        model_dir: str | None = None,
        adapt_to_pi: bool = True,
        chunking_mode: Literal["inference_time"] = "inference_time",
        rtc_replan_start_step: int = 25,
        rtc_handoff_delay_steps: int = 10,
    ):
        self._policy = policy
        self._rtc_replan_start_step = rtc_replan_start_step
        self._rtc_handoff_delay_steps = rtc_handoff_delay_steps
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None
        self._last_model_actions: np.ndarray | None = None
        self._last_rlt_replay_chunk: dict | None = None
        self._chunk_index: int = 0
        self._background_results: Dict[str, np.ndarray] | None = None
        self._background_running: bool = False
        self._rlt_actor_enabled = False

        self._obs: Dict[str, np.ndarray] | None = None
        if chunking_mode != "inference_time":
            raise ValueError("Robot runtime only supports chunking_mode='inference_time'.")
        self._chunking_mode = chunking_mode
        self._norm_stats = None
        self._joint_signs = np.ones(14)

        if model_dir is None:
            raise ValueError("model_dir is required for inference-time RTC.")
        self._norm_stats, self._joint_signs = self._load_runtime_assets(model_dir, adapt_to_pi)

        self._infer_thread = threading.Thread(target=self._background_infer)
        self._infer_thread.daemon = True
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
            if self._cur_step == self._rtc_replan_start_step:
                total_start = time.monotonic()
                self._background_running = True

                # flip, normalize joint actions
                # norm_action = (np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1]) * self._last_results["actions"] - np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1]) * self._obs["state"][:14] - np.array(self._norm_stats["actions"]["mean"])[:14]) / (np.array(self._norm_stats["actions"]["std"])[:14] + 1e-6)
                prep_start = time.monotonic()
                q01 = np.array(self._norm_stats["actions"]["q01"])[:14]
                q99 = np.array(self._norm_stats["actions"]["q99"])[:14]
                scaled = self._joint_signs * (self._last_results["actions"] - self._obs["state"][:14])
                norm_action = (scaled - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
                
                # get normalized gripper action
                norm_action[:, 6] = self._last_model_actions[:, 6]
                norm_action[:, 13] = self._last_model_actions[:, 13]
                
                zeros_padding = np.zeros((norm_action.shape[0], 18))
                norm_action = np.concatenate([norm_action, zeros_padding], axis=1)
                prep_ms = (time.monotonic() - prep_start) * 1000.0
                # np.savetxt("norm_action.txt", norm_action, fmt='%.6f')
                # np.savetxt("last_model_actions.txt", self._last_model_actions, fmt='%.6f')
                rpc_start = time.monotonic()
                self._background_results = self._policy.infer(
                    self._obs,
                    prev_action=norm_action,
                    chunking_mode="inference_time",
                    rlt_actor_enabled=self._rlt_actor_enabled,
                )
                rpc_ms = (time.monotonic() - rpc_start) * 1000.0
                # break
                # 将后面18列都设为0
                # modified_actions = None
                # if self._last_model_actions is not None:
                #     modified_actions = self._last_model_actions.copy()
                #     modified_actions[:, 14:] = 0
                
                self._background_running = False
                total_ms = (time.monotonic() - total_start) * 1000.0
                server_timing = self._background_results.get("server_timing", {})
                policy_timing = self._background_results.get("policy_timing", {})
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
                logging.debug(" ".join(log_parts))
            else:
                time.sleep(0.01)

    def _pop_rlt_replay_chunk(self, results: Dict) -> dict | None:  # noqa: UP006
        replay = {}
        for key in self._RLT_REPLAY_KEYS:
            if key in results:
                replay[key] = results.pop(key)
        if not replay:
            return None
        replay["chunk_index"] = self._chunk_index
        self._chunk_index += 1
        return replay

    @staticmethod
    def _attach_rlt_replay_step(results: Dict, replay_chunk: dict | None, chunk_step_index: int) -> None:  # noqa: UP006
        if replay_chunk is None:
            return
        results["rlt_replay"] = {
            **replay_chunk,
            "chunk_step_index": chunk_step_index,
            "is_chunk_start": chunk_step_index == 0,
        }

    def set_rlt_actor_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._rlt_actor_enabled:
            return
        self._rlt_actor_enabled = enabled
        self._last_results = None
        self._last_model_actions = None
        self._last_rlt_replay_chunk = None
        self._background_results = None
        self._background_running = False
        self._cur_step = 0

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._last_results is None:
            full_results = self._policy.infer(
                obs,
                chunking_mode="inference_time",
                rlt_actor_enabled=self._rlt_actor_enabled,
            )
            self._last_rlt_replay_chunk = self._pop_rlt_replay_chunk(full_results)
            self._last_model_actions = full_results["model_actions"]
            self._last_state = full_results["state"]
            self._last_results = {"actions": full_results["actions"]}
            self._cur_step = 0

        chunk_step_index = self._cur_step
        replay_chunk = self._last_rlt_replay_chunk
        results = tree.map_structure(lambda x: x[self._cur_step, ...], self._last_results)
        self._obs = obs
        self._cur_step += 1

        # Wait until the handoff point, then swap in the background RTC chunk.
        if self._cur_step == self._rtc_replan_start_step + self._rtc_handoff_delay_steps:
            while self._background_running:
                time.sleep(0.01)
            self._last_rlt_replay_chunk = self._pop_rlt_replay_chunk(self._background_results)
            self._last_model_actions = self._background_results["model_actions"]
            self._last_state = self._background_results["state"]
            self._last_results = {"actions": self._background_results["actions"]}
            self._cur_step -= self._rtc_replan_start_step
        self._attach_rlt_replay_step(results, replay_chunk, chunk_step_index)
        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._last_rlt_replay_chunk = None
        self._cur_step = 0
        self._chunk_index = 0
