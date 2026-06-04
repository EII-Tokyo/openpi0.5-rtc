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
    action_quality: Literal["normal", "good", "bad"] = "normal"

    sync_replan_interval: int = 25
    rtc_replan_start_step: int = 25
    rtc_handoff_delay_steps: int = 10

    chunking_mode: Literal["sync", "inference_time", "training_time"] = "inference_time"
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
    # H5dfSaver 配置
    dataset_dir: str = "/app/data/aloha_real/policy_episodes"
    manual_dataset_dir: str = "/app/data/aloha_real/manual_intervention_episodes"
    compress_images: bool = True
    is_mobile: bool = False
    if_save_hdf5: bool = True
    # Set <= 0 to save the full episode instead of a rolling tail buffer.
    hdf5_max_buffer_seconds: float = 60.0


def main(args: Args) -> None:
    from openpi.robot.aloha_real import h5df_saver
    from openpi.robot.aloha_real import real_env as _real_env

    good_bad_action_by_quality = {
        "normal": "normal",
        "good": "good action",
        "bad": "bad action",
    }
    good_bad_action = good_bad_action_by_quality[args.action_quality]
    logging.info(
        "Using action quality mode: %s (subtask.good_bad_action=%s)",
        args.action_quality,
        good_bad_action,
    )

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    server_metadata = ws_client_policy.get_server_metadata()
    logging.info(f"Server metadata: {server_metadata}")
    adapt_to_pi = bool(server_metadata.get("runtime", {}).get("adapt_to_pi", True))
    logging.info("Using runtime.adapt_to_pi from policy metadata: %s", adapt_to_pi)

    # 创建 H5dfSaver subscriber
    h5df_saver_instance = h5df_saver.H5dfSaver(
        dataset_dir=args.dataset_dir,
        compress_images=args.compress_images,
        is_mobile=args.is_mobile,
        fps=args.policy_hz,
        max_buffer_seconds=args.hdf5_max_buffer_seconds,
    )

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
        subscribers=[h5df_saver_instance] if args.if_save_hdf5 else [],
        max_hz=args.policy_hz,
        manual_hz=args.manual_hz,
        manual_dataset_dir=args.manual_dataset_dir,
        good_bad_action=good_bad_action,
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
