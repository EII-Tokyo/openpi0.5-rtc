import dataclasses
import logging
import os
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
from examples.aloha_real import rlt_key_region_recorder
from examples.aloha_real import video_hdf5_saver


@dataclasses.dataclass
class Args:
    model_dir: str
    adapt_to_pi: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    action_quality: Literal["normal", "good", "bad"] = "normal"

    action_horizon: int = 25

    num_episodes: int = 1
    max_episode_steps: int = 10000

    use_rtc: bool = True
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
    dataset_dir: str = "/app/examples/aloha_real/error_hdf5/2026-03-11_inference_lora_error"
    manual_dataset_dir: str = "/app/examples/aloha_real/manual_override_hdf5/2026-03-11_inference_lora"
    compress_images: bool = True
    is_mobile: bool = False
    if_save_hdf5: bool = True
    save_format: Literal["hdf5", "video_hdf5", "rlt_key_region", "video_hdf5_and_rlt_key_region"] = "hdf5"
    video_codec: Literal["h264", "mp4v", "avc1"] = "h264"
    rlt_rollouts_root: str = "/data/openpi0.5-rtc-reward-learning/rollouts"
    rlt_replay_root: str = "/data/openpi0.5-rtc-reward-learning/replay"
    rlt_pre_roll_seconds: float = 2.0
    rlt_post_roll_seconds: float = 0.3
    rlt_max_key_region_seconds: float = 20.0
    rlt_train_horizon: int = 10
    rlt_full_horizon: int = 50
    rlt_chunk_horizon: int | None = None
    rlt_chunk_stride: int = 2
    rlt_prefer_gpu_video: bool = True
    rlt_actor_path: str | None = None
    rlt_actor_poll_interval: float = 1.0
    # Set <= 0 to save the full episode instead of a rolling tail buffer.
    hdf5_max_buffer_seconds: float = 60.0
    # Save one HDF5 rollout each time the robot leaves reset pose and returns.
    split_hdf5_on_reset: bool = False
    split_home_threshold: float = 0.15
    split_leave_threshold: float = 0.30
    split_stable_home_steps: int = 25
    split_min_episode_steps: int = 50


def main(args: Args) -> None:
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
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    subscribers = []
    if args.save_format in {"video_hdf5", "video_hdf5_and_rlt_key_region"}:
        saver_instance = video_hdf5_saver.VideoHdf5Saver(
            dataset_dir=args.dataset_dir,
            fps=args.policy_hz,
            is_mobile=args.is_mobile,
            split_on_reset=args.split_hdf5_on_reset,
            reset_position=args.reset_position,
            home_threshold=args.split_home_threshold,
            leave_threshold=args.split_leave_threshold,
            stable_home_steps=args.split_stable_home_steps,
            min_episode_steps=args.split_min_episode_steps,
            video_codec=args.video_codec,
        )
        subscribers.append(saver_instance)
    elif args.save_format == "hdf5":
        saver_instance = h5df_saver.H5dfSaver(
            dataset_dir=args.dataset_dir,
            compress_images=args.compress_images,
            is_mobile=args.is_mobile,
            fps=args.policy_hz,
            max_buffer_seconds=args.hdf5_max_buffer_seconds,
            split_on_reset=args.split_hdf5_on_reset,
            reset_position=args.reset_position,
            home_threshold=args.split_home_threshold,
            leave_threshold=args.split_leave_threshold,
            stable_home_steps=args.split_stable_home_steps,
            min_episode_steps=args.split_min_episode_steps,
        )
        subscribers.append(saver_instance)
    elif args.save_format != "rlt_key_region":
        raise ValueError(f"Unsupported save_format: {args.save_format}")

    if args.save_format in {"rlt_key_region", "video_hdf5_and_rlt_key_region"}:
        subscribers.append(
            rlt_key_region_recorder.KeyRegionReplayRecorder(
                rollouts_root=args.rlt_rollouts_root,
                replay_root=args.rlt_replay_root,
                fps=args.policy_hz,
                pre_roll_seconds=args.rlt_pre_roll_seconds,
                post_roll_seconds=args.rlt_post_roll_seconds,
                max_key_region_seconds=args.rlt_max_key_region_seconds,
                train_horizon=args.rlt_chunk_horizon or args.rlt_train_horizon,
                full_horizon=args.rlt_full_horizon,
                chunk_stride=args.rlt_chunk_stride,
                prefer_gpu_video=args.rlt_prefer_gpu_video,
            )
        )

    runtime = _runtime.Runtime(
        # environment=_env.AlohaRealEnvironment(reset_position=metadata.get("reset_pose")),
        environment=_env.AlohaRealEnvironment(
            reset_position=args.reset_position,
            gripper_current_limits=args.gripper_current_limits,
            video_memory_num_frames=args.video_memory_num_frames,
            video_memory_stride_seconds=args.video_memory_stride_seconds,
        ),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
                model_dir=args.model_dir,
                adapt_to_pi=args.adapt_to_pi,
                use_rtc=args.use_rtc,
                rlt_actor_path=args.rlt_actor_path or os.getenv("RLT_ACTOR_CHECKPOINT_PATH"),
                rlt_actor_poll_interval=args.rlt_actor_poll_interval,
            )
        ),
        subscribers=subscribers if args.if_save_hdf5 else [],
        max_hz=args.policy_hz,
        manual_hz=args.manual_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
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
