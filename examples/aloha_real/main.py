import dataclasses
import logging
import signal
import sys
import threading
import time
from typing import Literal

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.aloha_real import env as _env
from examples.aloha_real import h5df_saver


@dataclasses.dataclass
class Args:
    model_dir: str
    adapt_to_pi: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    action_prompt: Literal["normal", "good", "bad"] = "normal"

    action_horizon: int = 25

    num_episodes: int = 1
    max_episode_steps: int = 10000

    use_rtc: bool = True
    policy_hz: float = 50.0
    manual_hz: float = 50.0

    # reset_position: List[List[float]] = dataclasses.field(default_factory=lambda: [
    #         [0.0, -0.96, 1.16, 0.0, -0.0, 0.0],
    #         #[0.0, -0.96, 1.16, 0.0, -0.0, 0.0],
    #         [0.0, -0.96, 1.16, 0.0, -0.0, 0.0]
    #     ])
    reset_position: list[list[float]] = dataclasses.field(
        default_factory=lambda: [
            [0.0, -0.96, 1.16, 0.0, -0.0, 0.0],
            # [0.0, -0.96, 1.16, 0.0, -0.0, 0.0],
            [0.0, -0.96, 1.16, 1.57, -0.0, -1.57],
        ]
    )
    gripper_current_limits: list[int] = dataclasses.field(default_factory=lambda: [400, 800])
    # H5dfSaver 配置
    dataset_dir: str = "/app/examples/aloha_real/error_hdf5/2026-03-11_inference_lora_error"
    manual_dataset_dir: str = "/app/examples/aloha_real/manual_override_hdf5/2026-03-11_inference_lora"
    compress_images: bool = True
    is_mobile: bool = False
    if_save_hdf5: bool = True
    # Set <= 0 to save the full episode instead of a rolling tail buffer.
    hdf5_max_buffer_seconds: float = 60.0


def main(args: Args) -> None:
    action_label_by_prompt = {
        "normal": "normal",
        "good": "good action",
        "bad": "bad action",
    }
    action_label = action_label_by_prompt[args.action_prompt]
    logging.info(
        "Using action prompt mode: %s (subtask.good_bad_action=%s)",
        args.action_prompt,
        action_label,
    )

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    # 创建 H5dfSaver subscriber
    h5df_saver_instance = h5df_saver.H5dfSaver(
        dataset_dir=args.dataset_dir,
        compress_images=args.compress_images,
        is_mobile=args.is_mobile,
        fps=args.policy_hz,
        max_buffer_seconds=args.hdf5_max_buffer_seconds,
    )

    runtime = _runtime.Runtime(
        # environment=_env.AlohaRealEnvironment(reset_position=metadata.get("reset_pose")),
        environment=_env.AlohaRealEnvironment(
            reset_position=args.reset_position,
            gripper_current_limits=args.gripper_current_limits,
            action_label=action_label,
        ),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
                model_dir=args.model_dir,
                adapt_to_pi=args.adapt_to_pi,
                use_rtc=args.use_rtc,
            )
        ),
        subscribers=[h5df_saver_instance] if args.if_save_hdf5 else [],
        max_hz=args.policy_hz,
        manual_hz=args.manual_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        manual_dataset_dir=args.manual_dataset_dir,
    )

    def _handle_exit_signal(signum, frame):
        logging.info(f"收到退出信号 {signum}, 准备退出")
        runtime.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handle_exit_signal)
    signal.signal(signal.SIGTERM, _handle_exit_signal)

    runtime.run()


def _start_logging_stdout_guard(interval: float = 1.0) -> None:
    """Keep stdout logging active even if ROS replaces root handlers."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    def ensure_stdout_handler() -> None:
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        if not any(
            isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout for h in root.handlers
        ):
            root.addHandler(handler)

    ensure_stdout_handler()

    def guard_loop() -> None:
        while True:
            ensure_stdout_handler()
            time.sleep(interval)

    threading.Thread(target=guard_loop, daemon=True).start()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None
    sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, "reconfigure") else None
    _start_logging_stdout_guard()
    args = tyro.cli(Args)
    main(args)
