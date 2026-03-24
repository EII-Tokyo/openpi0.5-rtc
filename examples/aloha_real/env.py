import copy
from typing import List, Optional  # noqa: UP035
import dm_env
import einops
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override
from examples.aloha_real import real_env as _real_env

_DEFAULT_RESET_POSITION = [
    [0.0, -0.96, 1.16, 0.0, -0.0, 0.0],
    [0.0, -0.96, 1.16, 1.57, -0.0, -1.57],
]


class AlohaRealEnvironment(_environment.Environment):
    """An environment for an Aloha robot on real hardware."""

    def __init__(
        self,
        reset_position: Optional[List[List[float]]] = None,  # noqa: UP006,UP007
        gripper_current_limits: Optional[List[int]] = None,
        render_height: int = 224,
        render_width: int = 224,
    ) -> None:
        if reset_position is None:
            reset_position = [list(joints) for joints in _DEFAULT_RESET_POSITION]
        if gripper_current_limits is None:
            gripper_current_limits = [300, 500]
        self._env = _real_env.make_real_env(init_node=True, reset_position=reset_position, gripper_current_limits=gripper_current_limits)
        self._render_height = render_height
        self._render_width = render_width

        self._ts = None

    @override
    def reset(self) -> None:
        self._ts = self._env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")
        # Always pull a fresh sensor snapshot instead of reusing the last timestep's
        # cached observation. Otherwise task switches can feed stale camera frames and
        # robot state from the previous stop/reset event into high-level inference.
        fresh_observation = self._env.get_observation()
        self._ts = dm_env.TimeStep(
            step_type=self._ts.step_type,
            reward=self._ts.reward,
            discount=self._ts.discount,
            observation=fresh_observation,
        )
        origin_observation = copy.deepcopy(self._ts.observation)
        obs = self._ts.observation
        for k in list(obs["images"].keys()):
            if "_depth" in k:
                del obs["images"][k]

        # for cam_name in obs["images"]:
        #     img = image_tools.convert_to_uint8(
        #         image_tools.resize_with_pad(obs["images"][cam_name], self._render_height, self._render_width)
        #     )
        #     obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")
        return {
            "state": obs["qpos"],
            "qpos": obs["qpos"],
            "qvel": obs["qvel"],
            "effort": obs["effort"],
            "images": obs["images"],
            "origin_observation": origin_observation,
        }

    @override
    def apply_action(self, action: dict) -> None:
        self._ts = self._env.step(action["actions"])

    @override
    def stop(self) -> None:
        """Stop the environment."""
        self._ts = self._env.stop()

    @override
    def sleep_arms(self) -> None:
        """Sleep the arms."""
        self._ts = self._env.sleep_arms()
