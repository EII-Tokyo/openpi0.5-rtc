import dataclasses
import logging
from typing import List

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.aloha_real import env as _env
from examples.aloha_real import h5df_saver


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    action_horizon: int = 25

    num_episodes: int = 1
    max_episode_steps: int = 10000

    use_rtc: bool = True
    
    # reset_position: List[List[float]] = dataclasses.field(default_factory=lambda: [
    #         [0.0, -0.96, 1.16, 0.0, -0.0, 0.0],
    #         #[0.0, -0.96, 1.16, 0.0, -0.0, 0.0],
    #         [0.0, -0.96, 1.16, 0.0, -0.0, 0.0]         
    #     ])
    reset_position: List[List[float]] = dataclasses.field(default_factory=lambda: [
            [0.0, -0.96, 1.16, 0.0, -0.0, 0.0],
            # [0.0, -0.96, 1.16, 0.0, -0.0, 0.0],
            [0.0, -0.96, 1.16, 1.57, -0.0, -1.57]         
        ])
    
    # H5dfSaver 配置
    dataset_dir: str = "~/aloha_data"
    compress_images: bool = True
    is_mobile: bool = False


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    metadata = ws_client_policy.get_server_metadata()
    
    # 创建 H5dfSaver subscriber
    h5df_saver_instance = h5df_saver.H5dfSaver(
        dataset_dir=args.dataset_dir,
        compress_images=args.compress_images,
        is_mobile=args.is_mobile,
    )
    
    runtime = _runtime.Runtime(
        # environment=_env.AlohaRealEnvironment(reset_position=metadata.get("reset_pose")),
        environment=_env.AlohaRealEnvironment(reset_position=args.reset_position),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
                use_rtc=args.use_rtc,
            )
        ),
        subscribers=[h5df_saver_instance],
        max_hz=50,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
