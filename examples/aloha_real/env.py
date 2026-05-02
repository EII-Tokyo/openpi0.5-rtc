import copy
from typing import Any

from openpi_client.runtime import environment as _environment
from typing_extensions import override


class AlohaRealEnvironment(_environment.Environment):
    """An environment for an Aloha robot on real hardware."""

    def __init__(
        self,
        reset_position: list[list[float]] | None = None,
        gripper_current_limits: list[int] | None = None,
        action_label: str = "normal",
        render_height: int = 224,
        render_width: int = 224,
        video_memory_num_frames: int = 1,
        video_memory_stride_seconds: float = 1.0,
        continuous_roll_joints: bool = True,
    ) -> None:
        from examples.aloha_real import real_env as _real_env

        self._env = _real_env.make_real_env(
            init_node=True,
            reset_position=reset_position,
            gripper_current_limits=gripper_current_limits,
            video_memory_num_frames=video_memory_num_frames,
            video_memory_stride_seconds=video_memory_stride_seconds,
            continuous_roll_joints=continuous_roll_joints,
        )
        self._render_height = render_height
        self._render_width = render_width
        self._subtask: dict[str, Any] = {"good_bad_action": action_label}

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
            "subtask": self._subtask,
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
