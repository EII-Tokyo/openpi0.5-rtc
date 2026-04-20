import dataclasses
import json
import logging
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, List
import urllib.error
import urllib.request

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
from openpi_client.runtime import gpt_high_level_policy as _gpt_high_level_policy
import tyro

from examples.aloha_real import env as _env
from examples.aloha_real import h5df_saver

_DEFAULT_RESET_POSITION = [
    0.0,
    -0.96,
    1.16,
    0.0,
    -0.0,
    0.0,
    0.0,
    0.0,
    -0.96,
    1.16,
    1.57,
    -0.0,
    -1.57,
    0.0,
]


@dataclasses.dataclass
class Args:
    model_dir: str
    prompt: str
    adapt_to_pi: bool = True
    # 与 docker compose（network_mode: host）同机部署时，策略服务在本机 8000/8001。
    # 远端策略机请用 CLI 或环境变量 OPENPI_LOW_LEVEL_HOST / OPENPI_HIGH_LEVEL_HOST 覆盖。
    low_level_host: str = "127.0.0.1"
    low_level_port: int = 8000
    high_level_host: str = "127.0.0.1"
    high_level_port: int = 8001
    high_level_hz: float = 0.0
    # Subtask History 下发到前端的最大条数（1–500），减轻 WebSocket 负载
    high_level_history_max_len: int = 50

    action_horizon: int = 25
    max_episode_steps: int = 10000

    use_rtc: bool = True
    policy_hz: float = 50.0
    manual_hz: float = 50.0
    
    # reset_position: List[List[float]] = dataclasses.field(default_factory=lambda: [
    #         [0.0, -0.96, 1.16, 0.0, -0.0, 0.0],
    #         #[0.0, -0.96, 1.16, 0.0, -0.0, 0.0],
    #         [0.0, -0.96, 1.16, 0.0, -0.0, 0.0]         
    #     ])
    reset_position: str = json.dumps(_DEFAULT_RESET_POSITION)
    gripper_current_limits: List[int] = dataclasses.field(default_factory=lambda: [300, 500])
    # H5dfSaver 配置
    dataset_dir: str | None = None
    manual_dataset_dir: str | None = None
    compress_images: bool = True
    is_mobile: bool = False
    if_save_hdf5: bool = False
    runtime_config_url: str = "http://127.0.0.1:8011/api/runtime/config"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_repo_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            value = value.strip().strip("'").strip('"')
            os.environ[key] = value
    except Exception:
        logging.exception("Failed to load .env from %s", env_path)


def _fetch_runtime_config(runtime_config_url: str) -> dict[str, Any] | None:
    url = str(runtime_config_url or "").strip()
    if not url:
        return None
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        logging.warning("Could not fetch runtime config from %s; using local defaults.", url)
    except Exception:
        logging.exception("Failed to fetch runtime config from %s", url)
    return None


def _parse_reset_position(raw_value: str) -> list[float]:
    try:
        parsed: Any = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --reset-position JSON: {exc}") from exc
    if not isinstance(parsed, list) or len(parsed) != 14:
        raise ValueError(
            "--reset-position must be a JSON array with exactly 14 values: "
            "[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]."
        )
    return [float(value) for value in parsed]


def main(args: Args) -> None:
    _load_repo_env(_repo_root() / ".env")
    reset_position = _parse_reset_position(args.reset_position)
    initial_runtime_config = _fetch_runtime_config(args.runtime_config_url)
    logging.info("Runtime startup config:\n%s", dataclasses.asdict(args))
    low_level_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.low_level_host,
        port=args.low_level_port,
    )
    low_level_metadata = low_level_policy.get_server_metadata()
    logging.info("Low-level server metadata: %s", low_level_metadata)

    runtime_policy = low_level_policy
    high_level_policy = _gpt_high_level_policy.RoutedHighLevelPolicy(
        service_host=args.high_level_host,
        service_port=args.high_level_port,
    )
    if initial_runtime_config and hasattr(high_level_policy, "set_config"):
        high_level_policy.set_config(
            source=initial_runtime_config.get("high_level_source"),
            gpt_model=initial_runtime_config.get("gpt_model"),
            gpt_image_mode=initial_runtime_config.get("gpt_image_mode"),
        )
    logging.info("High-level policy metadata: %s", high_level_policy.get_server_metadata())
    
    # 创建 H5dfSaver subscriber
    h5df_saver_instance = h5df_saver.H5dfSaver(
        dataset_dir=args.dataset_dir,
        compress_images=args.compress_images,
        is_mobile=args.is_mobile,
        fps=args.policy_hz,
    )
    
    runtime = _runtime.Runtime(
        # environment=_env.AlohaRealEnvironment(reset_position=metadata.get("reset_pose")),
        environment=_env.AlohaRealEnvironment(
            reset_position=reset_position,
            gripper_current_limits=args.gripper_current_limits,
        ),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=runtime_policy,
                action_horizon=args.action_horizon,
                model_dir=args.model_dir,
                adapt_to_pi=args.adapt_to_pi,
                use_rtc=args.use_rtc,
            )
        ),
        subscribers=[h5df_saver_instance] if args.if_save_hdf5 else [],
        max_hz=args.policy_hz,
        manual_hz=args.manual_hz,
        max_episode_steps=args.max_episode_steps,
        manual_dataset_dir=args.manual_dataset_dir,
        high_level_policy=high_level_policy,
        high_level_hz=args.high_level_hz,
        high_level_history_max_len=args.high_level_history_max_len,
        prompt=args.prompt,
    )
    if initial_runtime_config:
        runtime.apply_runtime_config(initial_runtime_config)
        logging.info(
            "Applied initial runtime config: high_level_source=%s gpt_model=%s gpt_image_mode=%s",
            initial_runtime_config.get("high_level_source"),
            initial_runtime_config.get("gpt_model"),
            initial_runtime_config.get("gpt_image_mode"),
        )

    shutdown_started = False

    def _shutdown_runtime_to_sleep() -> None:
        nonlocal shutdown_started
        if shutdown_started:
            return
        shutdown_started = True
        try:
            logging.info("准备关闭 runtime，先让机械臂回到 sleep 位置")
            runtime._environment.sleep_arms()
        except Exception:
            logging.exception("runtime 关闭前回 sleep 失败")
        finally:
            runtime.stop()

    def _handle_exit_signal(signum, frame):
        logging.info(f"收到退出信号 {signum}，准备退出")
        _shutdown_runtime_to_sleep()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handle_exit_signal)
    signal.signal(signal.SIGTERM, _handle_exit_signal)

    try:
        runtime.run()
    finally:
        _shutdown_runtime_to_sleep()


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
            isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout
            for h in root.handlers
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
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None
    _start_logging_stdout_guard()
    args = tyro.cli(Args)
    low_env = os.getenv("OPENPI_LOW_LEVEL_HOST", "").strip()
    if low_env:
        args.low_level_host = low_env
    high_env = os.getenv("OPENPI_HIGH_LEVEL_HOST", "").strip()
    if high_env:
        args.high_level_host = high_env
    main(args)
