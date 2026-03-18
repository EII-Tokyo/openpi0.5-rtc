from typing import Dict
import json
import pathlib
import time
import threading
import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy


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
    ):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None
        self._last_origin_actions: np.ndarray | None = None
        self._background_results: Dict[str, np.ndarray] | None = None
        self._background_running: bool = False

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
                # start_time = time.time()
                self._background_running = True

                # flip, normalize joint actions
                # norm_action = (np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1]) * self._last_results["actions"] - np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1]) * self._obs["state"][:14] - np.array(self._norm_stats["actions"]["mean"])[:14]) / (np.array(self._norm_stats["actions"]["std"])[:14] + 1e-6)

                q01 = np.array(self._norm_stats["actions"]["q01"])[:14]
                q99 = np.array(self._norm_stats["actions"]["q99"])[:14]
                scaled = self._joint_signs * (self._last_results["actions"] - self._obs["state"][:14])
                norm_action = (scaled - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
                
                # get normalized gripper action
                norm_action[:, 6] = self._last_origin_actions[:, 6]
                norm_action[:, 13] = self._last_origin_actions[:, 13]
                
                zeros_padding = np.zeros((norm_action.shape[0], 18))
                norm_action = np.concatenate([norm_action, zeros_padding], axis=1)
                # np.savetxt("norm_action.txt", norm_action, fmt='%.6f')
                # np.savetxt("last_origin_actions.txt", self._last_origin_actions, fmt='%.6f')
                self._background_results = self._policy.infer(self._obs, norm_action, self._use_rtc)
                # break
                # 将后面18列都设为0
                # modified_actions = None
                # if self._last_origin_actions is not None:
                #     modified_actions = self._last_origin_actions.copy()
                #     modified_actions[:, 14:] = 0
                
                # self._background_results = self._policy.infer(self._obs, modified_actions, self._use_rtc)

                self._background_running = False
                # end_time = time.time()
                # print(f"Time taken to infer: {end_time - start_time}")
            else:
                time.sleep(0.01)

    def _slice_result(self, value):
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == self._action_horizon:
            return value[self._cur_step, ...]
        return value

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._use_rtc:
            # init     
            if self._last_results is None:
                self._last_results = self._policy.infer(obs, None, self._use_rtc)
                self._last_origin_actions = self._last_results["origin_actions"]
                self._cur_step = 0

            results = tree.map_structure(self._slice_result, self._last_results)
            self._obs = obs
            self._cur_step += 1

            # if current step equals s+d, wait for background inference to complete
            if self._cur_step == self._s + self._d:
                while self._background_running:
                    time.sleep(0.01)
                self._last_origin_actions = self._background_results["origin_actions"]
                self._last_results = self._background_results
                self._cur_step -= self._s
            return results
        else:
            if self._last_results is None:
                self._last_results = self._policy.infer(obs)
                self._cur_step = 0

            results = tree.map_structure(self._slice_result, self._last_results)
            self._cur_step += 1

            if self._cur_step >= self._action_horizon:
                self._last_results = None

            return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
