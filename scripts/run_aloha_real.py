import dataclasses
import logging
import signal
import sys
import threading
import time
from typing import Literal

from openpi.robot.aloha_real import chunked_policy
from openpi.robot.client import websocket_client_policy as _websocket_client_policy
from openpi.robot.aloha_real import runtime as _runtime
import tyro



@dataclasses.dataclass
class Args:
    model_dir: str
    host: str = "0.0.0.0"
    port: int = 8000
    # Deprecated: ignored. Configure operator/rating in the web interface.
    action_quality: Literal["normal", "good", "bad"] = "normal"

    sync_replan_interval: int = 25
    rtc_replan_start_step: int = 25
    rtc_handoff_delay_steps: int = 10

    chunking_mode: Literal["sync", "inference_time", "training_time"] = "inference_time"
    latency_diagnostics: bool = False
    policy_hz: float = 50.0
    manual_hz: float = 50.0
    video_memory_num_frames: int = 1
    video_memory_stride_seconds: float = 1.0

    # reset_position: List[List[float]] = dataclasses.field(default_factory=lambda: [
    #         [0.0, -0.96, 1.16, 0.0, -0.0, 0.0],
    #         #[0.0, -0.96, 1.16, 0.0, -0.0, 0.0],
    #         [0.0, -0.96, 1.16, 0.0, -0.0, 0.0]
    #     ])
    reset_position: list[list[float]] = dataclasses.field(
        default_factory=lambda: [
            #[0.0, -0.96, 1.16, 0.0, -0.0, 0.0],
            # [0.0, -0.96, 1.16, 0.0, -0.0, 0.0],
            #[0.0, -0.96, 1.16, -1.57, -0.0, 1.57],
            [0.0, -0.96, 1.16, 1.57, -0.0, -1.57],
            [0.0, -0.96, 1.16, 0.0, -0.0, 0.0],
            #[0.0, -0.96, 1.16, -1.57, -0.0, 1.57],
        ]
    )
    gripper_current_limits: list[int] = dataclasses.field(default_factory=lambda: [300, 800])
    # Episodes are saved only through the explicit recording controls.
    manual_dataset_dir: str = "/app/data/aloha_real/manual_intervention_episodes"


def main(args: Args) -> None:
    from openpi.robot.aloha_real import real_env as _real_env

    if args.action_quality != "normal":
        logging.warning("Ignoring deprecated --action-quality; configure rating in the web interface.")

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    server_metadata = ws_client_policy.get_server_metadata()
    logging.info(f"Server metadata: {server_metadata}")
    adapt_to_pi = bool(server_metadata.get("runtime", {}).get("adapt_to_pi", True))
    logging.info("Using runtime.adapt_to_pi from policy metadata: %s", adapt_to_pi)

    # The server publishes the input pipeline of the configuration used to load
    # these weights. Never infer temporal settings from a model-name substring.
    from openpi.robot.aloha_real.video_input_config import configure_video_input
    video_input = configure_video_input(server_metadata, args.model_dir)
    args.video_memory_num_frames = video_input["num_frames"]
    args.video_memory_stride_seconds = video_input["stride_seconds"]
    if video_input["mosaic"] is not None:
        from openpi.robot.aloha_real import mosaic_runtime
        mosaic_runtime.install(video_input)
    logging.info("Using checkpoint video input (overrides CLI history): %s", video_input)

    if args.latency_diagnostics:
        from openpi.robot.aloha_real import latency_diagnostics
        latency_diagnostics.install()

    runtime = _runtime.Runtime(
        # environment=_real_env.AlohaRealEnvironment(reset_position=metadata.get("reset_pose")),
        environment=_real_env.AlohaRealEnvironment(
            reset_position=args.reset_position,
            gripper_current_limits=args.gripper_current_limits,
            video_memory_num_frames=args.video_memory_num_frames,
            video_memory_stride_seconds=args.video_memory_stride_seconds,
        ),
        policy=chunked_policy.ChunkedPolicy(
            policy=ws_client_policy,
            sync_replan_interval=args.sync_replan_interval,
            model_dir=args.model_dir,
            adapt_to_pi=adapt_to_pi,
            chunking_mode=args.chunking_mode,
            rtc_replan_start_step=args.rtc_replan_start_step,
            rtc_handoff_delay_steps=args.rtc_handoff_delay_steps,
        ),
        subscribers=[],
        max_hz=args.policy_hz,
        manual_hz=args.manual_hz,
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
