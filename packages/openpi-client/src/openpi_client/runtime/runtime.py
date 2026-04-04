import logging
import threading
import time
import json
import hashlib
import io
import base64
import redis
import os
import re
import select
import subprocess
import sys
import termios
import tty
from collections import deque
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from openpi_client import subtask_parsing as _subtask_parsing
from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber
from examples.aloha_real import hdf5_utils as _hdf5_utils

# 确保 logging 有 handler（如果主程序没有配置）
_logger = logging.getLogger(__name__)
if not _logger.handlers and not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


class Runtime:
    """The core module orchestrating interactions between key components of the system."""

    _VALID_STATE_SUBTASK_PAIRS = {
        ("Bottle on table, opening faces left", "Rotate so opening faces right"),
        ("Bottle on table, opening faces right", "Pick up with left hand"),
        ("Bottle in left hand and capped", "Unscrew cap"),
        ("Bottle in left hand, cap removed, and cap in right hand", "Bottle to left trash bin, cap to right trash bin"),
        ("Bottle in left hand, cap removed, and cap not in right hand", "Bottle to left trash bin"),
        ("Bottle stuck in left hand", "Use right hand to remove and place into left trash bin"),
        ("Cap on table", "Pick up cap and place into right trash bin"),
        ("No bottle on table", "Return to initial pose"),
    }
    _VALID_SUBTASKS = {subtask for _, subtask in _VALID_STATE_SUBTASK_PAIRS}
    _SUBTASK_TO_BOTTLE_STATE = {subtask: bottle_state for bottle_state, subtask in _VALID_STATE_SUBTASK_PAIRS}
    _BOTTLE_START_SUBTASKS = {
        "Rotate so opening faces right",
        "Pick up with left hand",
    }
    _LOW_LEVEL_SUBTASK_OPTIONS = (
        "Rotate so opening faces right",
        "Pick up with left hand",
        "Unscrew cap",
        "Bottle to left trash bin, cap to right trash bin",
        "Bottle to left trash bin",
        "Use right hand to remove and place into left trash bin",
        "Pick up cap and place into right trash bin",
        "Return to initial pose",
    )

    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscribers: list[_subscriber.Subscriber],
        max_hz: float = 0,
        manual_hz: float = 0,
        max_episode_steps: int = 0,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        manual_dataset_dir: str | None = None,
        high_level_policy=None,
        high_level_hz: float = 0.0,
        good_bad_action: str | None = "normal",
    ) -> None:
        self._environment = environment
        self._agent = agent
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._manual_hz = manual_hz
        self._max_episode_steps = max_episode_steps

        self._step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        self._manual_step_time = 1 / self._manual_hz if self._manual_hz > 0 else 0
        self._manual_dataset_dir = manual_dataset_dir
        self._dataset_dir = None
        self._high_level_policy = high_level_policy
        self._high_level_step_time = 1 / high_level_hz if high_level_hz > 0 else 0.0
        self._good_bad_action = _subtask_parsing.normalize_good_bad_action(good_bad_action)

        self._in_episode = False
        self._episode_steps = 0
        
        # Redis配置
        self._redis_host = os.getenv('REDIS_HOST', redis_host)
        self._redis_port = int(os.getenv('REDIS_PORT', redis_port))
        self._redis_db = int(os.getenv('REDIS_DB', redis_db))
        
        # Redis相关
        self._redis_client = None
        self._redis_thread = None
        self._redis_running = False
        self._latest_task = None
        self._task_lock = threading.Lock()
        
        # 任务状态管理
        self._current_task = None
        self._is_waiting_for_task = False
        
        # 存储最后的action（用于task_num==3时移动master）
        self._last_action = None
        history_size = max(1, int((self._max_hz if self._max_hz > 0 else 1) * 10))
        self._recent_puppet_actions = deque(maxlen=history_size)

        self._high_level_running = False
        self._high_level_thread = None
        self._high_level_obs = None
        self._high_level_obs_version = 0
        self._high_level_obs_lock = threading.Lock()
        self._high_level_state_lock = threading.Lock()
        self._latest_structured_subtask = None
        self._latest_hierarchical_state = {}
        self._last_policy_action_ts = None
        self._locked_bottle_description = None
        self._committed_subtask = None
        self._pending_subtask = None
        self._pending_subtask_count = 0
        self._include_bottle_description = True
        self._include_bottle_position = False
        self._include_bottle_state = True
        self._include_subtask = True
        self._forced_low_level_subtask = None
        self._high_level_history: deque[dict] = deque(maxlen=100)
        self._video_memory_num_frames = 1
        self._video_memory_stride_seconds = 1.0
        self._temporal_build_interval_seconds = 10.0
        self._image_history_horizon_seconds = max(5.0, self._video_memory_num_frames * self._video_memory_stride_seconds + 2.0)
        self._image_history: dict[str, deque[tuple[float, np.ndarray]]] = {}
        self._image_history_lock = threading.Lock()
        self._cached_temporal_images: dict[str, np.ndarray] | None = None
        self._cached_temporal_images_ts: float | None = None
        self._temporal_cache_lock = threading.Lock()
        self._temporal_builder_running = False
        self._temporal_builder_thread = None
        self._temporal_video_debug_dir = Path.cwd() / "temporal_debug_videos"
        self._temporal_video_debug_dir.mkdir(parents=True, exist_ok=True)
        self._applied_action_trace_path = Path("/tmp/runtime_applied_actions.jsonl")
        self._applied_action_trace_path.write_text("")
        logging.info("Applied action trace will be written to %s", self._applied_action_trace_path)
        self._high_level_trace_path = Path("/tmp/runtime_high_level_requests.jsonl")
        self._high_level_trace_path.write_text("")
        logging.info("High-level request trace will be written to %s", self._high_level_trace_path)

        # 退出标志
        self._stop = False
        self._model_task_nums = {"1"}
        self._stop_task_nums = {"3", "4"}
        self._local_task_names = {
            "1": "process all bottles",
            "2": "Stop and human hand control",
            "3": "Return to home position and save hdf5",
            "4": "Return to sleep position, save hdf5 and quit robot runtime",
        }
        self._local_key_ttl_s = 1.0
        self._local_key_queue = deque()
        self._stdin_termios_backup = None

    def _apply_video_memory_config(self, num_frames: int) -> None:
        next_num_frames = int(num_frames)
        if next_num_frames not in (1, 4):
            return
        if next_num_frames == self._video_memory_num_frames:
            return
        self._video_memory_num_frames = next_num_frames
        self._image_history_horizon_seconds = max(
            5.0,
            self._video_memory_num_frames * self._video_memory_stride_seconds + 2.0,
        )
        self._reset_temporal_image_history()
        self._stop_temporal_builder()
        self._start_temporal_builder()

    def _stdin_is_tty(self) -> bool:
        try:
            return sys.stdin.isatty()
        except Exception:
            return False

    def _enable_local_keyboard_mode(self) -> None:
        if not self._stdin_is_tty() or self._stdin_termios_backup is not None:
            return
        self._stdin_termios_backup = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())

    def _restore_local_keyboard_mode(self) -> None:
        if self._stdin_termios_backup is None:
            return
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._stdin_termios_backup)
        self._stdin_termios_backup = None

    def _prompt_subfolder_name(self) -> str | None:
        if not self._stdin_is_tty():
            logging.error("stdin 不是 TTY，无法输入人工接管 subfolder。")
            return None
        self._restore_local_keyboard_mode()
        try:
            while True:
                raw_value = input("请输入人工接管保存 subfolder: ").strip()
                if not raw_value:
                    print("subfolder 不能为空。")
                    continue
                sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", raw_value)
                if not sanitized:
                    print("subfolder 无效，请重新输入。")
                    continue
                if sanitized != raw_value:
                    print(f"subfolder 已规范化为: {sanitized}")
                return sanitized
        finally:
            self._enable_local_keyboard_mode()

    def _read_local_key_raw(self, timeout: float = 0.0) -> str | None:
        if not self._stdin_is_tty():
            return None
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if not ready:
            return None
        first = sys.stdin.read(1)
        if first == "\x1b":
            ready, _, _ = select.select([sys.stdin], [], [], 0.02)
            if not ready:
                return "esc"
            second = sys.stdin.read(1)
            if second != "[":
                return None
            ready, _, _ = select.select([sys.stdin], [], [], 0.02)
            if not ready:
                return None
            third = sys.stdin.read(1)
            arrow_map = {
                "A": "up",
                "B": "down",
                "C": "right",
                "D": "left",
            }
            return arrow_map.get(third)
        return first

    def _prune_local_key_queue(self) -> None:
        if not self._local_key_queue:
            return
        now = time.time()
        while self._local_key_queue and now - self._local_key_queue[0][1] > self._local_key_ttl_s:
            self._local_key_queue.popleft()

    def _pump_local_keys(self, timeout: float = 0.0) -> None:
        self._prune_local_key_queue()
        key = self._read_local_key_raw(timeout)
        if key is not None:
            self._local_key_queue.append((key, time.time()))

    def _take_local_key(self, timeout: float = 0.0) -> str | None:
        self._prune_local_key_queue()
        if self._local_key_queue:
            key, _ = self._local_key_queue.popleft()
            return key
        self._pump_local_keys(timeout)
        if self._local_key_queue:
            key, _ = self._local_key_queue.popleft()
            return key
        return None

    def _take_local_task(self, allowed_task_nums: set[str] | None = None) -> dict | None:
        self._prune_local_key_queue()
        if not self._local_key_queue:
            return None
        keys = list(self._local_key_queue)
        for index, (key, key_ts) in enumerate(keys):
            if key not in self._local_task_names:
                continue
            if allowed_task_nums is not None and key not in allowed_task_nums:
                continue
            del keys[index]
            self._local_key_queue = deque(keys)
            return self._make_local_task(key)
        return None

    def _make_local_task(self, task_num: str) -> dict | None:
        task_num = str(task_num)
        task_name = self._local_task_names.get(task_num)
        if task_name is None:
            return None
        return {
            "task_num": task_num,
            "task_name": task_name,
            "timestamp": time.time(),
            "dataset_dir": self._dataset_dir,
            "manual_dataset_dir": self._manual_dataset_dir,
            "include_bottle_position": self._include_bottle_position,
            "include_bottle_description": self._include_bottle_description,
            "include_bottle_state": self._include_bottle_state,
            "include_subtask": self._include_subtask,
            "forced_low_level_subtask": self._forced_low_level_subtask,
        }

    def _rewind_action_history(self, real_env, action_history: deque[list[float]]) -> list[float] | None:
        if not action_history:
            logging.warning("没有可回退的动作历史。")
            return None
        rewind_steps = max(1, int(round((self._manual_hz if self._manual_hz > 0 else 50.0) * 0.25)))
        rewind_index = max(0, len(action_history) - 1 - rewind_steps)
        rewind_action = list(action_history[rewind_index])
        self._move_puppet_to_action(rewind_action, move_time=0.35)
        self._move_master_to_action(real_env, rewind_action, move_time=0.35)
        self._last_action = rewind_action
        logging.info("已回退到约 %.2f 秒前的动作状态。", rewind_steps / (self._manual_hz if self._manual_hz > 0 else 50.0))
        return rewind_action

    def _apply_task_paths(self, task_data: dict) -> None:
        dataset_dir = task_data.get("dataset_dir")
        manual_dataset_dir = task_data.get("manual_dataset_dir")
        include_bottle_description = task_data.get("include_bottle_description")
        include_bottle_position = task_data.get("include_bottle_position")
        include_bottle_state = task_data.get("include_bottle_state")
        include_subtask = task_data.get("include_subtask")
        forced_low_level_subtask = task_data.get("forced_low_level_subtask")
        video_memory_num_frames = task_data.get("video_memory_num_frames")
        if isinstance(dataset_dir, str) and dataset_dir.strip():
            self._dataset_dir = dataset_dir.strip()
        if isinstance(manual_dataset_dir, str) and manual_dataset_dir.strip():
            self._manual_dataset_dir = manual_dataset_dir.strip()
        if isinstance(include_bottle_description, bool):
            self._include_bottle_description = include_bottle_description
        if isinstance(include_bottle_position, bool):
            self._include_bottle_position = include_bottle_position
        if isinstance(include_bottle_state, bool):
            self._include_bottle_state = include_bottle_state
        if isinstance(include_subtask, bool):
            self._include_subtask = include_subtask
        if forced_low_level_subtask in self._LOW_LEVEL_SUBTASK_OPTIONS:
            self._forced_low_level_subtask = forced_low_level_subtask
        elif forced_low_level_subtask in ("", None):
            self._forced_low_level_subtask = None
        if isinstance(video_memory_num_frames, int):
            self._apply_video_memory_config(video_memory_num_frames)
        for subscriber in self._subscribers:
            if hasattr(subscriber, "set_dataset_dir"):
                subscriber.set_dataset_dir(self._dataset_dir)

    def _setup_redis(self) -> None:
        """设置Redis连接"""
        try:
            self._redis_client = redis.Redis(
                host=self._redis_host,
                port=self._redis_port,
                db=self._redis_db,
                decode_responses=True
            )
            self._redis_client.ping()
            logging.info(f"Redis连接成功: {self._redis_host}:{self._redis_port}")
        except Exception as e:
            logging.error(f"Redis连接失败: {e}")
            raise

    def _redis_listener(self) -> None:
        """Redis pub/sub监听线程"""
        pubsub = self._redis_client.pubsub()
        pubsub.subscribe("aloha_voice_commands")
        
        logging.info("开始监听Redis pub/sub频道: aloha_voice_commands")
        
        try:
            while self._redis_running:
                message = pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        if not data.get('task'):
                            self._apply_task_paths(data)
                            self._publish_runtime_state(mode="waiting" if self._is_waiting_for_task else None)
                            logging.info(
                                "收到Redis运行配置更新: include_bottle_description=%s include_bottle_position=%s include_bottle_state=%s include_subtask=%s forced_low_level_subtask=%s video_memory_num_frames=%s",
                                self._include_bottle_description,
                                self._include_bottle_position,
                                self._include_bottle_state,
                                self._include_subtask,
                                self._forced_low_level_subtask,
                                self._video_memory_num_frames,
                            )
                            continue
                        task_num = data.get('task')
                        task_name = data.get('task_name', '未知任务')
                        timestamp = data.get('timestamp', time.time())
                        
                        logging.info(f"收到Redis任务: {task_num} - {task_name}")
                        
                        with self._task_lock:
                            self._latest_task = {
                                'task_num': task_num,
                                'task_name': task_name,
                                'timestamp': timestamp,
                                'dataset_dir': data.get('dataset_dir'),
                                'manual_dataset_dir': data.get('manual_dataset_dir'),
                                'include_bottle_description': data.get('include_bottle_description'),
                                'include_bottle_position': data.get('include_bottle_position'),
                                'include_bottle_state': data.get('include_bottle_state'),
                                'include_subtask': data.get('include_subtask'),
                                'forced_low_level_subtask': data.get('forced_low_level_subtask'),
                                'video_memory_num_frames': data.get('video_memory_num_frames'),
                            }
                            
                    except json.JSONDecodeError as e:
                        logging.error(f"Redis消息JSON解析失败: {e}")
                    except Exception as e:
                        logging.error(f"处理Redis消息失败: {e}")
                        
        except Exception as e:
            logging.error(f"Redis监听线程异常: {e}")
        finally:
            pubsub.close()
            logging.info("Redis监听线程结束")

    def _start_redis_listener(self) -> None:
        """启动Redis监听线程"""
        if self._redis_client is None:
            self._setup_redis()
        
        self._redis_running = True
        self._redis_thread = threading.Thread(target=self._redis_listener, daemon=True)
        self._redis_thread.start()
        logging.info("Redis监听线程已启动")

    def _stop_redis_listener(self) -> None:
        """停止Redis监听线程"""
        self._redis_running = False
        if self._redis_thread and self._redis_thread.is_alive():
            self._redis_thread.join(timeout=2.0)
        logging.info("Redis监听线程已停止")

    def _reset_high_level_state(self) -> None:
        with self._high_level_obs_lock:
            self._high_level_obs = None
            self._high_level_obs_version += 1
        with self._high_level_state_lock:
            self._latest_structured_subtask = None
            self._latest_hierarchical_state = {}
        self._locked_bottle_description = None
        self._committed_subtask = None
        self._pending_subtask = None
        self._pending_subtask_count = 0
        self._awaiting_initial_subtask = False
        self._last_initial_subtask_wait_log_ts = 0.0
        if self._high_level_policy is not None:
            self._high_level_policy.reset()

    def _stabilize_high_level_payload(self, parsed: dict) -> dict | None:
        bottle_description = parsed.get("bottle_description")
        bottle_state = parsed.get("bottle_state")

        raw_subtask = parsed.get("subtask")
        if (bottle_state, raw_subtask) not in self._VALID_STATE_SUBTASK_PAIRS:
            raw_subtask = None
            bottle_state = None

        if raw_subtask:
            should_refresh_description = False
            if self._locked_bottle_description is None:
                should_refresh_description = True
            elif (
                raw_subtask in self._BOTTLE_START_SUBTASKS
                and self._committed_subtask not in self._BOTTLE_START_SUBTASKS
            ):
                should_refresh_description = True

            if should_refresh_description and bottle_description:
                self._locked_bottle_description = bottle_description

            self._committed_subtask = raw_subtask
            self._pending_subtask = None
            self._pending_subtask_count = 0

        effective = {
            "bottle_description": self._locked_bottle_description,
            "bottle_position": parsed.get("bottle_position"),
            "bottle_state": bottle_state,
            "subtask": self._committed_subtask,
        }
        if all(value is None for value in effective.values()):
            return None
        if self._good_bad_action is not None:
            effective["good_bad_action"] = self._good_bad_action
        return effective

    def _start_high_level_worker(self) -> None:
        if self._high_level_policy is None or self._high_level_thread is not None:
            return
        self._high_level_running = True
        self._high_level_thread = threading.Thread(target=self._high_level_worker, daemon=True)
        self._high_level_thread.start()

    def _stop_high_level_worker(self) -> None:
        self._high_level_running = False
        if self._high_level_thread and self._high_level_thread.is_alive():
            self._high_level_thread.join(timeout=2.0)
        self._high_level_thread = None

    def _update_high_level_observation(self, observation_with_task: dict) -> None:
        if self._high_level_policy is None:
            return
        obs_for_high_level = {
            k: v for k, v in observation_with_task.items() if k != "origin_observation" and k != "subtask"
        }
        with self._high_level_obs_lock:
            self._high_level_obs = obs_for_high_level
            self._high_level_obs_version += 1

    def _get_environment_observation(self, *, fresh: bool = False) -> dict:
        if fresh and hasattr(self._environment, "get_fresh_observation"):
            return self._environment.get_fresh_observation()
        return self._environment.get_observation()

    def _decorate_observation(self, observation: dict) -> dict:
        temporal_timestamp = time.time()
        return {
            **observation,
            "images": self._get_temporal_images(observation.get("images", {}), temporal_timestamp),
        }

    def _refresh_high_level_observation(self) -> None:
        if self._high_level_policy is None or self._current_task is None:
            return
        observation = self._decorate_observation(self._get_environment_observation(fresh=True))
        observation_with_task = {
            **observation,
            "prompt": self._current_task.get("task_name"),
        }
        self._update_high_level_observation(observation_with_task)

    def _reset_temporal_image_history(self) -> None:
        with self._image_history_lock:
            self._image_history.clear()
        with self._temporal_cache_lock:
            self._cached_temporal_images = None
            self._cached_temporal_images_ts = None

    def _record_temporal_images(self, images: dict, timestamp: float) -> None:
        if self._video_memory_num_frames <= 1 or not isinstance(images, dict):
            return
        with self._image_history_lock:
            for key, value in images.items():
                arr = np.asarray(value)
                history = self._image_history.setdefault(key, deque())
                history.append((timestamp, np.array(arr, copy=True)))
                while history and timestamp - history[0][0] > self._image_history_horizon_seconds:
                    history.popleft()

    def _build_temporal_images(self, images: dict, timestamp: float) -> dict:
        if self._video_memory_num_frames <= 1 or not isinstance(images, dict):
            return images
        history_copy_start = time.perf_counter()
        with self._image_history_lock:
            image_history = {
                key: list(history)
                for key, history in self._image_history.items()
            }
        history_copy_ms = (time.perf_counter() - history_copy_start) * 1000.0
        targets = [
            timestamp - (self._video_memory_num_frames - 1 - i) * self._video_memory_stride_seconds
            for i in range(self._video_memory_num_frames)
        ]
        stacked_images = {}
        lookup_ms = 0.0
        stack_ms = 0.0
        for key, current_value in images.items():
            history = image_history.get(key, ())
            if not history:
                stack_start = time.perf_counter()
                base = np.asarray(current_value)
                stacked_images[key] = np.stack([base] * self._video_memory_num_frames, axis=0)
                stack_ms += (time.perf_counter() - stack_start) * 1000.0
                continue

            lookup_start = time.perf_counter()
            selected_frames: list[np.ndarray] = []
            earliest_frame = history[0][1]
            for target_ts in targets:
                chosen = None
                for hist_ts, hist_frame in history:
                    if hist_ts <= target_ts:
                        chosen = hist_frame
                    else:
                        break
                if chosen is None:
                    chosen = earliest_frame
                selected_frames.append(np.asarray(chosen))
            lookup_ms += (time.perf_counter() - lookup_start) * 1000.0
            stack_start = time.perf_counter()
            stacked_images[key] = np.stack(selected_frames, axis=0)
            stack_ms += (time.perf_counter() - stack_start) * 1000.0
        total_ms = history_copy_ms + lookup_ms + stack_ms
        logging.info(
            "_build_temporal_images timing: total=%.2f ms copy_history=%.2f ms lookup=%.2f ms stack=%.2f ms",
            total_ms,
            history_copy_ms,
            lookup_ms,
            stack_ms,
        )
        return stacked_images

    def _get_temporal_images(self, images: dict, timestamp: float) -> dict:
        if self._video_memory_num_frames <= 1 or not isinstance(images, dict):
            return images
        self._record_temporal_images(images, timestamp)
        with self._temporal_cache_lock:
            cached = self._cached_temporal_images
        if cached is not None:
            return cached
        start_time = time.perf_counter()
        stacked_images = self._build_temporal_images(images, timestamp)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logging.info("Temporal image build took %.1f ms (foreground bootstrap)", elapsed_ms)
        with self._temporal_cache_lock:
            self._cached_temporal_images = stacked_images
            self._cached_temporal_images_ts = timestamp
        return stacked_images

    def _temporal_builder_loop(self) -> None:
        while self._temporal_builder_running:
            try:
                with self._image_history_lock:
                    latest_images = {
                        key: np.array(history[-1][1], copy=True)
                        for key, history in self._image_history.items()
                        if history
                    }
                    latest_ts = max((history[-1][0] for history in self._image_history.values() if history), default=None)
                if latest_images and latest_ts is not None:
                    start_time = time.perf_counter()
                    stacked_images = self._build_temporal_images(latest_images, latest_ts)
                    self._export_temporal_images_video(stacked_images)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    with self._temporal_cache_lock:
                        self._cached_temporal_images = stacked_images
                        self._cached_temporal_images_ts = latest_ts
                    logging.info("Temporal image build took %.1f ms", elapsed_ms)
            except Exception:
                logging.exception("后台 temporal image build 失败")
            for _ in range(int(self._temporal_build_interval_seconds * 10)):
                if not self._temporal_builder_running:
                    break
                time.sleep(0.1)

    def _start_temporal_builder(self) -> None:
        if self._video_memory_num_frames <= 1 or self._temporal_builder_thread is not None:
            return
        self._temporal_builder_running = True
        self._temporal_builder_thread = threading.Thread(target=self._temporal_builder_loop, daemon=True)
        self._temporal_builder_thread.start()

    def _stop_temporal_builder(self) -> None:
        self._temporal_builder_running = False
        if self._temporal_builder_thread and self._temporal_builder_thread.is_alive():
            self._temporal_builder_thread.join(timeout=2.0)
        self._temporal_builder_thread = None

    def _export_temporal_images_video(self, stacked_images: dict) -> None:
        return
        if not isinstance(stacked_images, dict) or not stacked_images:
            return

        def _to_hwc_uint8(frame: np.ndarray) -> np.ndarray:
            arr = np.asarray(frame)
            if arr.ndim != 3:
                raise ValueError(f"expected 3D frame, got {arr.shape}")
            if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            return arr

        camera_order = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
        first_stack = next((np.asarray(v) for v in stacked_images.values() if np.asarray(v).ndim == 4), None)
        if first_stack is None:
            return

        num_frames = int(first_stack.shape[0])
        sample_frame = _to_hwc_uint8(first_stack[0])
        h, w = sample_frame.shape[:2]
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        def _build_grid(frame_idx: int) -> np.ndarray:
            tiles = []
            for cam_name in camera_order:
                stack = stacked_images.get(cam_name)
                if stack is None:
                    tiles.append(blank)
                    continue
                stack_arr = np.asarray(stack)
                if stack_arr.ndim != 4 or frame_idx >= stack_arr.shape[0]:
                    tiles.append(blank)
                    continue
                tiles.append(_to_hwc_uint8(stack_arr[frame_idx]))
            top = np.concatenate([tiles[0], tiles[1]], axis=1)
            bottom = np.concatenate([tiles[2], tiles[3]], axis=1)
            return np.concatenate([top, bottom], axis=0)

        if num_frames <= 1:
            out_path = self._temporal_video_debug_dir / "stacked_images_grid.png"
            try:
                grid = _build_grid(0)
                Image.fromarray(grid).save(out_path)
            except Exception as exc:
                logging.warning("导出 temporal stacked image 失败: %s", exc)
            return

        out_path = self._temporal_video_debug_dir / "stacked_images_grid.mp4"

        try:
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{w * 2}x{h * 2}",
                "-r",
                "1",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(out_path),
            ]
            proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                for idx in range(num_frames):
                    grid = _build_grid(idx)
                    assert proc.stdin is not None
                    proc.stdin.write(grid.tobytes())
            finally:
                if proc.stdin is not None:
                    proc.stdin.close()
                proc.wait(timeout=10)
        except Exception as exc:
            logging.warning("导出 temporal stacked video 失败: %s", exc)

    def _get_high_level_state(self) -> tuple[dict | None, dict]:
        with self._high_level_state_lock:
            structured_subtask = self._latest_structured_subtask
            hierarchical_state = dict(self._latest_hierarchical_state)
        return structured_subtask, hierarchical_state

    def _summarize_high_level_obs(self, obs: dict) -> dict:
        images = obs.get("images") or {}
        image_summary = {}
        for key, image in images.items():
            np_image = np.asarray(image)
            digest = hashlib.sha1(np_image.tobytes()).hexdigest()[:16]
            image_summary[key] = {
                "shape": list(np_image.shape),
                "dtype": str(np_image.dtype),
                "sha1_16": digest,
            }

        state = obs.get("state")
        state_summary = None
        if state is not None:
            np_state = np.asarray(state, dtype=np.float32)
            state_summary = {
                "shape": list(np_state.shape),
                "values": np_state.tolist(),
                "min": float(np.min(np_state)),
                "max": float(np.max(np_state)),
            }

        return {
            "timestamp": time.time(),
            "prompt": str(obs.get("prompt") or ""),
            "image_summary": image_summary,
            "state_summary": state_summary,
        }

    def _append_high_level_trace(self, payload: dict) -> None:
        try:
            with self._high_level_trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            logging.debug("写 high-level trace 失败: %s", exc)

    def _encode_history_images(self, obs: dict) -> dict[str, str]:
        encoded: dict[str, str] = {}
        images = obs.get("images", {})
        if not isinstance(images, dict):
            return encoded
        for key, value in images.items():
            try:
                arr = np.asarray(value)
                if arr.ndim == 4:
                    arr = arr[-1]
                if arr.ndim != 3:
                    continue
                if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                image = Image.fromarray(arr)
                image.thumbnail((200, 150))
                buf = io.BytesIO()
                image.save(buf, format="JPEG", quality=65)
                encoded[key] = base64.b64encode(buf.getvalue()).decode("ascii")
            except Exception as exc:
                logging.debug("编码 high-level 历史图片失败 %s: %s", key, exc)
        return encoded

    def _high_level_worker(self) -> None:
        last_processed_version = -1
        while self._high_level_running:
            if self._is_waiting_for_task or self._current_task is None:
                time.sleep(0.01)
                continue

            self._refresh_high_level_observation()

            with self._high_level_obs_lock:
                obs = self._high_level_obs
                version = self._high_level_obs_version

            if obs is None or version == last_processed_version:
                time.sleep(0.005)
                continue

            try:
                trace_payload = {
                    "kind": "request",
                    "obs_version": version,
                    **self._summarize_high_level_obs(obs),
                }
                self._append_high_level_trace(trace_payload)
                high_level_result = self._high_level_policy.infer_subtask(obs)
                high_level_text = str(high_level_result.get("subtask_text") or "").strip()
                parsed = _subtask_parsing.parse_structured_fields(high_level_text)
                structured_subtask = self._stabilize_high_level_payload(parsed)
                history_entry = {
                    "id": hashlib.md5(f"{version}:{high_level_text}".encode("utf-8")).hexdigest()[:12],
                    "timestamp": time.time(),
                    "obs_version": version,
                    "task_prompt": str(obs.get("prompt") or "").strip(),
                    "high_level_text": high_level_text,
                    "bottle_description": parsed.get("bottle_description"),
                    "bottle_state": parsed.get("bottle_state"),
                    "bottle_position": parsed.get("bottle_position"),
                    "subtask": parsed.get("subtask"),
                    "images": self._encode_history_images(obs),
                }
                self._high_level_history.append(history_entry)
                hierarchical_state = {
                    "task_prompt": str(obs.get("prompt") or "").strip(),
                    "low_level_prompt": json.dumps(structured_subtask, ensure_ascii=False)
                    if structured_subtask is not None
                    else (high_level_text or str(obs.get("prompt") or "").strip()),
                    "high_level_server_timing": high_level_result.get("server_timing", {}),
                    "low_level_server_timing": {},
                    "good_bad_action": self._good_bad_action,
                    "locked_bottle_description": self._locked_bottle_description,
                    "raw_subtask": parsed.get("subtask"),
                    "effective_subtask": self._committed_subtask,
                    "forced_low_level_subtask": self._forced_low_level_subtask,
                    "pending_subtask": self._pending_subtask,
                    "pending_subtask_count": self._pending_subtask_count,
                    "history": list(self._high_level_history),
                    **parsed,
                }
                self._append_high_level_trace(
                    {
                        "kind": "response",
                        "obs_version": version,
                        "timestamp": time.time(),
                        "subtask_text": high_level_text,
                        "structured_subtask": structured_subtask,
                        "server_timing": high_level_result.get("server_timing", {}),
                    }
                )
                logging.info("High-level raw output: %s", high_level_text)
                logging.info("Structured subtask payload: %s", structured_subtask)
                with self._high_level_state_lock:
                    self._latest_structured_subtask = structured_subtask
                    self._latest_hierarchical_state = hierarchical_state
                last_processed_version = version
            except Exception as exc:
                logging.warning("High-level infer failed: %s", exc)
                time.sleep(0.05)
                continue

            if self._high_level_step_time > 0:
                time.sleep(self._high_level_step_time)

    def _publish_runtime_state(self, *, qpos=None, latest_action=None, mode: str | None = None) -> None:
        """发布轻量运行时状态给可视化前端。"""
        if self._redis_client is None:
            return

        current_task = self._current_task.get("task_name") if self._current_task else None
        if qpos is None and hasattr(self._environment, "_ts") and getattr(self._environment, "_ts") is not None:
            qpos = self._environment._ts.observation.get("qpos")
        _, hierarchical = self._get_high_level_state()
        if not isinstance(hierarchical, dict):
            hierarchical = {}

        payload = {
            "timestamp": time.time(),
            "mode": mode or ("waiting" if self._is_waiting_for_task else "policy"),
            "current_task": current_task,
            "qpos": list(qpos) if qpos is not None else [],
            "latest_action": list(latest_action) if latest_action is not None else [],
            "hierarchical": hierarchical,
        }
        try:
            self._redis_client.publish("aloha_runtime_state", json.dumps(payload))
        except Exception as exc:
            logging.debug("发布运行时状态失败: %s", exc)

    def _take_latest_task(self, allowed_task_nums: set[str] | None = None):
        """获取并消费最新的 Redis 任务。"""
        self._pump_local_keys(timeout=0.0)
        local_task = self._take_local_task(allowed_task_nums)
        if local_task is not None:
            return local_task
        with self._task_lock:
            if self._latest_task is None:
                return None
            task_num = str(self._latest_task.get("task_num"))
            if allowed_task_nums is not None and task_num not in allowed_task_nums:
                return None
            latest_task = self._latest_task
            self._latest_task = None
            return latest_task
    
    def is_waiting_for_task(self) -> bool:
        """检查是否正在等待任务"""
        return self._is_waiting_for_task
    
    def get_current_task(self):
        """获取当前任务"""
        return self._current_task
    
    def set_waiting_state(self, waiting: bool):
        """设置等待状态"""
        self._is_waiting_for_task = waiting
        if waiting:
            self._current_task = None

    def run(self) -> None:
        """Runs the runtime loop continuously until stop() is called or the environment is done."""
        self._enable_local_keyboard_mode()
        # 启动Redis监听
        self._start_redis_listener()
        self._start_temporal_builder()
        
        try:
            self._run()
        finally:
            # 停止Redis监听
            self._stop_temporal_builder()
            self._stop_high_level_worker()
            self._stop_redis_listener()
            self._restore_local_keyboard_mode()

    def mark_episode_complete(self) -> None:
        """Marks the end of an episode."""
        self._in_episode = False

    def stop(self) -> None:
        """Request the runtime loop to stop."""
        self._stop = True

    def _run(self) -> None:
        """Runs a single episode."""
        logging.info("Starting episode...")
        self._environment.reset()
        self._agent.reset()
        self._reset_high_level_state()
        self._reset_temporal_image_history()
        self._start_high_level_worker()
        self._last_policy_action_ts = None

        self._in_episode = True
        self._episode_steps = 0
        
        last_step_time = time.time()
        last_waiting_state_publish = 0.0
        
        self._is_waiting_for_task = True
        self._current_task = None
        self._publish_runtime_state(mode="waiting")
        logging.info("Runtime 默认仅接受 Redis / voice web 任务；仅在人工接管 teleop 阶段监听本地按键")

        while not self._stop:
            task_data = self._take_latest_task()
            if task_data:
                self._handle_task(task_data)
            
            if self._is_waiting_for_task:
                now = time.time()
                if now - last_waiting_state_publish >= 0.2:
                    observation = self._get_environment_observation()
                    self._publish_runtime_state(qpos=observation.get("qpos"), mode="waiting")
                    last_waiting_state_publish = now
                time.sleep(0.05)
            else:
                action_applied = self._step()
                if action_applied:
                    self._episode_steps += 1
                now = time.time()
                dt = now - last_step_time
                if dt < self._step_time:
                    time.sleep(self._step_time - dt)
                    last_step_time = time.time()
                else:
                    last_step_time = now
        

    def _handle_task(self, task_data) -> None:
        """处理来自 Redis / voice web 的任务。"""
        task_num = task_data.get('task_num')
        task_name = task_data.get('task_name', '未知任务')
        
        logging.info(f"处理语音任务: {task_num} - {task_name}")
        self._apply_task_paths(task_data)
        
        if task_num in self._model_task_nums:
            logging.info(f"开始执行任务: {task_name}")
            for subscriber in self._subscribers:
                subscriber.on_episode_start()
            # 设置当前任务
            self._current_task = task_data
            self._is_waiting_for_task = False 
            self._high_level_history.clear()
            self._reset_high_level_state()
            self._reset_temporal_image_history()
            self._refresh_high_level_observation()
            self._awaiting_initial_subtask = self._high_level_policy is not None
            self._last_initial_subtask_wait_log_ts = 0.0
            self._publish_runtime_state(mode="policy")
        elif task_num == "2":
            logging.info("收到停止指令，进入人机协作模式")
            self._current_task = task_data
            self._agent.reset() 
            self._reset_high_level_state()
            self._reset_temporal_image_history()
            for subscriber in self._subscribers:
                subscriber.on_episode_end()
            self._publish_runtime_state(mode="teleop_prepare")
            self._handle_human_teleop_mode()  
        elif task_num == "3":
            logging.info("收到停止指令，回到初始位置并停止agent")
            # 设置等待状态
            self._is_waiting_for_task = True
            self._current_task = None
            self._reset_high_level_state()
            self._reset_temporal_image_history()
            # 回到初始位置
            self._environment.stop()
            if hasattr(self._environment, "close_grippers"):
                self._environment.close_grippers()
            # 停止agent
            self._agent.reset()   
            # 通知subscriber episode结束
            for subscriber in self._subscribers:
                subscriber.on_episode_end()   
            self._publish_runtime_state(mode="waiting")
        elif task_num == "4":
            logging.info("收到回到sleep位置并退出指令，退出程序")
            self._environment.sleep_arms()
            self._agent.reset()
            self._reset_high_level_state()
            self._reset_temporal_image_history()
            for subscriber in self._subscribers:
                subscriber.on_episode_end()
            self._publish_runtime_state(mode="sleep")
            self._stop = True
        else:
            logging.warning(f"未知任务编号: {task_num}")

    def _handle_voice_task(self, task_data) -> None:
        """兼容旧调用点。"""
        self._handle_task(task_data)

    def _step(self) -> None:
        """A single step of the runtime loop."""
        step_started_at = time.monotonic()
        observation = self._get_environment_observation()
        observation_ready_at = time.monotonic()
        observation = self._decorate_observation(observation)
        assert self._current_task is not None, "_current_task must be set before calling _step()"
        observation_with_task = {
            **{k: v for k, v in observation.items() if k != "origin_observation"},
            'prompt': self._current_task.get('task_name')
        }

        self._update_high_level_observation(observation_with_task)
        structured_subtask, hierarchical = self._get_high_level_state()
        if self._awaiting_initial_subtask:
            if structured_subtask is None:
                now = time.monotonic()
                if now - self._last_initial_subtask_wait_log_ts >= 0.5:
                    logging.info(
                        "Waiting for initial high-level subtask before starting low-level control "
                        "prompt=%s state_shape=%s",
                        observation_with_task.get("prompt"),
                        list(observation_with_task.get("state").shape)
                        if hasattr(observation_with_task.get("state"), "shape")
                        else None,
                    )
                    self._last_initial_subtask_wait_log_ts = now
                self._publish_runtime_state(
                    qpos=observation.get("qpos"),
                    latest_action=self._last_action,
                    mode="policy_waiting_subtask",
                )
                return False
            self._awaiting_initial_subtask = False
            self._last_initial_subtask_wait_log_ts = 0.0
            logging.info("Initial high-level subtask ready: %s", structured_subtask)
        if structured_subtask is not None:
            if not self._include_bottle_position and isinstance(structured_subtask, dict):
                structured_subtask = {
                    **structured_subtask,
                    "bottle_position": None,
                }
            if not self._include_bottle_description and isinstance(structured_subtask, dict):
                structured_subtask = {
                    **structured_subtask,
                    "bottle_description": None,
                }
            if not self._include_bottle_state and isinstance(structured_subtask, dict):
                structured_subtask = {
                    **structured_subtask,
                    "bottle_state": None,
                }
            if not self._include_subtask and isinstance(structured_subtask, dict):
                structured_subtask = {
                    **structured_subtask,
                    "subtask": None,
                }
            if self._forced_low_level_subtask is not None and isinstance(structured_subtask, dict):
                structured_subtask = {
                    **structured_subtask,
                    "bottle_state": self._SUBTASK_TO_BOTTLE_STATE.get(
                        self._forced_low_level_subtask,
                        structured_subtask.get("bottle_state"),
                    ),
                    "subtask": self._forced_low_level_subtask,
                }
            observation_with_task["subtask"] = structured_subtask
            if not isinstance(hierarchical, dict):
                hierarchical = {}
            hierarchical["low_level_prompt"] = (
                json.dumps(structured_subtask, ensure_ascii=False)
                if isinstance(structured_subtask, dict)
                else str(structured_subtask)
            )
            if isinstance(structured_subtask, dict):
                hierarchical["effective_subtask"] = structured_subtask.get("subtask")
                hierarchical["bottle_state"] = structured_subtask.get("bottle_state")
                hierarchical["bottle_position"] = structured_subtask.get("bottle_position")
        high_level_state_ready_at = time.monotonic()
        if self._episode_steps < 5 or self._episode_steps % 25 == 0:
            state = observation_with_task.get("state")
            images = observation_with_task.get("images", {})
            image_shapes = {
                key: list(value.shape) if hasattr(value, "shape") else None
                for key, value in images.items()
            } if isinstance(images, dict) else {}
            logging.info(
                "Low-level input step=%d prompt=%s subtask=%s state_shape=%s image_shapes=%s",
                self._episode_steps,
                observation_with_task.get("prompt"),
                observation_with_task.get("subtask"),
                list(state.shape) if hasattr(state, "shape") else None,
                image_shapes,
            )

        action = self._agent.get_action(observation_with_task)
        action_ready_at = time.monotonic()
        rtc_debug = action.get("rtc_debug", {}) if isinstance(action, dict) else {}
        if not isinstance(rtc_debug, dict):
            rtc_debug = {}
        qpos_before = observation.get("qpos")
        applied_action = action.get("actions") if isinstance(action, dict) and "actions" in action else None
        trace_record = {
            "timestamp": time.time(),
            "episode_step": self._episode_steps,
            "task": self._current_task.get("task_name") if self._current_task else None,
            "qpos_before": list(qpos_before) if qpos_before is not None else None,
            "applied_action": list(applied_action) if applied_action is not None else None,
            "rtc_debug": rtc_debug,
            "subtask": hierarchical.get("subtask") if isinstance(hierarchical, dict) else None,
            "bottle_state": hierarchical.get("bottle_state") if isinstance(hierarchical, dict) else None,
        }
        with self._applied_action_trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace_record, ensure_ascii=True) + "\n")
        if rtc_debug.get("switched"):
            logging.info(
                "RTC chunk switched at episode_step=%d chunk_id=%s local_step=%s trace=%s",
                self._episode_steps,
                rtc_debug.get("chunk_id"),
                rtc_debug.get("local_step"),
                self._applied_action_trace_path,
            )
        self._environment.apply_action(action)
        action_applied_at = time.monotonic()
        self._last_action = action.get("actions") if isinstance(action, dict) and "actions" in action else None
        if self._last_action is not None:
            self._recent_puppet_actions.append(list(self._last_action))
        if not isinstance(hierarchical, dict):
            hierarchical = {}
        low_level_timing = action.get("server_timing", {}) if isinstance(action, dict) else {}
        if not isinstance(low_level_timing, dict):
            low_level_timing = {}
        hierarchical["low_level_server_timing"] = low_level_timing
        if isinstance(action, dict):
            action["hierarchical"] = hierarchical
        action_interval_ms = None
        if self._last_policy_action_ts is not None:
            action_interval_ms = (action_applied_at - self._last_policy_action_ts) * 1000.0
        self._last_policy_action_ts = action_applied_at
        low_level_infer_ms = low_level_timing.get("infer_ms")
        if low_level_infer_ms is not None:
            low_level_infer_ms = round(float(low_level_infer_ms), 1)
        low_level_recv_ms = low_level_timing.get("recv_ms")
        if low_level_recv_ms is not None:
            low_level_recv_ms = round(float(low_level_recv_ms), 1)
        low_level_unpack_ms = low_level_timing.get("unpack_ms")
        if low_level_unpack_ms is not None:
            low_level_unpack_ms = round(float(low_level_unpack_ms), 1)
        low_level_pack_ms = low_level_timing.get("pack_ms")
        if low_level_pack_ms is not None:
            low_level_pack_ms = round(float(low_level_pack_ms), 1)
        low_level_prev_send_ms = low_level_timing.get("prev_send_ms")
        if low_level_prev_send_ms is not None:
            low_level_prev_send_ms = round(float(low_level_prev_send_ms), 1)
        low_level_prev_total_ms = low_level_timing.get("prev_total_ms")
        if low_level_prev_total_ms is not None:
            low_level_prev_total_ms = round(float(low_level_prev_total_ms), 1)
        high_level_timing = hierarchical.get("high_level_server_timing", {}) if isinstance(hierarchical, dict) else {}
        if not isinstance(high_level_timing, dict):
            high_level_timing = {}
        high_level_infer_ms = high_level_timing.get("infer_ms")
        if high_level_infer_ms is not None:
            high_level_infer_ms = round(float(high_level_infer_ms), 1)
        high_level_prev_total_ms = high_level_timing.get("prev_total_ms")
        if high_level_prev_total_ms is not None:
            high_level_prev_total_ms = round(float(high_level_prev_total_ms), 1)
        effective_subtask = observation_with_task.get("subtask")
        if isinstance(effective_subtask, dict):
            effective_subtask = effective_subtask.get("subtask")
        logging.info(
            "Policy action step=%d interval_ms=%s target_ms=%.1f step_compute_ms=%.1f "
            "high_level_infer_ms=%s high_level_prev_total_ms=%s "
            "low_level_recv_ms=%s low_level_unpack_ms=%s low_level_infer_ms=%s low_level_pack_ms=%s "
            "low_level_prev_send_ms=%s low_level_prev_total_ms=%s subtask=%s",
            self._episode_steps,
            f"{action_interval_ms:.1f}" if action_interval_ms is not None else "n/a",
            self._step_time * 1000.0,
            (action_applied_at - step_started_at) * 1000.0,
            high_level_infer_ms if high_level_infer_ms is not None else "n/a",
            high_level_prev_total_ms if high_level_prev_total_ms is not None else "n/a",
            low_level_recv_ms if low_level_recv_ms is not None else "n/a",
            low_level_unpack_ms if low_level_unpack_ms is not None else "n/a",
            low_level_infer_ms if low_level_infer_ms is not None else "n/a",
            low_level_pack_ms if low_level_pack_ms is not None else "n/a",
            low_level_prev_send_ms if low_level_prev_send_ms is not None else "n/a",
            low_level_prev_total_ms if low_level_prev_total_ms is not None else "n/a",
            effective_subtask or "n/a",
        )

        qpos = observation.get("qpos")
        payload = {
            "timestamp": time.time(),
            "mode": "policy",
            "current_task": self._current_task.get("task_name"),
            "qpos": list(qpos) if qpos is not None else [],
            "latest_action": list(self._last_action) if self._last_action is not None else [],
            "hierarchical": hierarchical,
        }
        try:
            self._redis_client.publish("aloha_runtime_state", json.dumps(payload))
        except Exception as exc:
            logging.debug("发布运行时状态失败: %s", exc)
        publish_done_at = time.monotonic()

        if self._episode_steps < 5 or self._episode_steps % 25 == 0:
            logging.info(
                "Policy step breakdown step=%d get_obs_ms=%.1f high_level_state_ms=%.1f "
                "get_action_ms=%.1f apply_action_ms=%.1f publish_state_ms=%.1f total_step_ms=%.1f",
                self._episode_steps,
                (observation_ready_at - step_started_at) * 1000.0,
                (high_level_state_ready_at - observation_ready_at) * 1000.0,
                (action_ready_at - high_level_state_ready_at) * 1000.0,
                (action_applied_at - action_ready_at) * 1000.0,
                (publish_done_at - action_applied_at) * 1000.0,
                (publish_done_at - step_started_at) * 1000.0,
            )

        for subscriber in self._subscribers:
            subscriber.on_step(observation["origin_observation"], action)
        return True

    def _move_master_to_action(self, real_env, action, move_time: float = 0.5) -> None:
        """仅将master移动到指定action。"""
        from examples.aloha_real import robot_utils
        from examples.aloha_real import constants

        master_bot_left = real_env.master_bot_left
        master_bot_right = real_env.master_bot_right

        left_arm_pos = action[:6]
        left_gripper_normalized = action[6]
        right_arm_pos = action[7:13]
        right_gripper_normalized = action[13]

        master_left_gripper_joint = constants.MASTER_GRIPPER_JOINT_UNNORMALIZE_FN(left_gripper_normalized)
        master_right_gripper_joint = constants.MASTER_GRIPPER_JOINT_UNNORMALIZE_FN(right_gripper_normalized)

        robot_utils.torque_on(master_bot_left)
        robot_utils.torque_on(master_bot_right)
        robot_utils.move_arms(
            [master_bot_left, master_bot_right],
            [left_arm_pos, right_arm_pos],
            move_time=move_time,
        )
        robot_utils.move_grippers(
            [master_bot_left, master_bot_right],
            [master_left_gripper_joint, master_right_gripper_joint],
            move_time=min(move_time, 0.5),
        )

    def _move_puppet_to_action(self, target_action, move_time: float = 0.35) -> None:
        """平滑回退 puppet，避免单次 apply_action 造成机械臂突跳。"""
        if target_action is None:
            return

        current_qpos = None
        if hasattr(self._environment, "_ts") and getattr(self._environment, "_ts") is not None:
            current_qpos = self._environment._ts.observation.get("qpos")
        if current_qpos is None:
            current_qpos = target_action

        start = np.asarray(current_qpos, dtype=float)
        target = np.asarray(target_action, dtype=float)
        if start.shape != target.shape:
            logging.warning("回退动作维度不匹配，跳过平滑插值。")
            self._environment.apply_action({"actions": list(target)})
            return

        control_hz = self._manual_hz if self._manual_hz > 0 else (self._max_hz if self._max_hz > 0 else 50.0)
        num_steps = max(2, int(round(move_time * control_hz)))
        trajectory = np.linspace(start, target, num_steps)
        step_sleep = move_time / max(1, num_steps - 1)

        for step_action in trajectory[1:]:
            self._environment.apply_action({"actions": step_action.tolist()})
            if step_sleep > 0:
                time.sleep(step_sleep)

    def _handle_human_teleop_mode(self) -> None:
        """处理人机协作模式（task_num==3）"""
        try:
            from examples.aloha_real import robot_utils
            from examples.aloha_real.real_env import get_action
            
            if not hasattr(self._environment, '_env'):
                logging.error("无法访问real_env，跳过人机协作模式")
                return
            
            real_env = self._environment._env
            master_bot_left = real_env.master_bot_left
            master_bot_right = real_env.master_bot_right

            if self._last_action is None:
                logging.warning("没有上次的action，退出人机协作模式")
                self._is_waiting_for_task = True
                self._current_task = None
                self._publish_runtime_state(mode="waiting")
                return

            if not self._recent_puppet_actions:
                self._recent_puppet_actions.append(list(self._last_action))

            action_history = deque((list(action) for action in self._recent_puppet_actions), maxlen=self._recent_puppet_actions.maxlen)
            action = self._last_action
            self._move_master_to_action(real_env, action, move_time=0.5)
            logging.info("leader已移动到上次模型输出位置")

            timesteps = []
            actions = []
            timestamps = []
            actual_dt_history = []

            logging.info("进入人工接管准备阶段：按 b 开始采集，按左方向键回退约 0.25 秒，按 1/2/3/4 切换状态。")

            if not self._manual_dataset_dir:
                logging.warning("未从 voice web 收到人工接管保存路径，取消本次人工接管数据保存。")
                self._is_waiting_for_task = True
                self._current_task = None
                self._publish_runtime_state(mode="waiting")
                return
            episode_subfolder = self._prompt_subfolder_name()
            if not episode_subfolder:
                logging.warning("未输入人工接管 subfolder，取消本次人工接管数据保存。")
                self._is_waiting_for_task = True
                self._current_task = None
                self._publish_runtime_state(mode="waiting")
                return
            episode_dataset_dir = os.path.join(self._manual_dataset_dir, episode_subfolder)
            os.makedirs(episode_dataset_dir, exist_ok=True)

            step_count = 0
            latest_task = None

            if not self._stdin_is_tty():
                logging.warning("stdin 不是 TTY，人工接管只能通过 voice web 任务切换，无法使用本地按键。")

            try:
                while True:
                    latest_task = self._take_latest_task(
                        allowed_task_nums=self._model_task_nums | self._stop_task_nums
                    )
                    if latest_task:
                        logging.info(
                            "人工接管准备阶段收到任务 %s，结束当前人工接管并切换流程",
                            latest_task["task_num"],
                        )
                        break

                    local_key = self._take_local_key(timeout=0.05)
                    if local_key == "left":
                        rewound_action = self._rewind_action_history(real_env, action_history)
                        if rewound_action is not None:
                            self._publish_runtime_state(latest_action=rewound_action, mode="teleop_prepare")
                        continue
                    if local_key == "b":
                        robot_utils.torque_off(master_bot_left)
                        robot_utils.torque_off(master_bot_right)
                        logging.info("master torque已关闭，开始人工接管数据采集。再次按 b 可停止采集。")
                        break
                    self._publish_runtime_state(mode="teleop_prepare")

                if latest_task is None:
                    while True:
                        t0 = time.time()
                        action = get_action(master_bot_left, master_bot_right)
                        t1 = time.time()

                        self._environment.apply_action({"actions": action})
                        ts = self._environment._ts
                        self._publish_runtime_state(qpos=ts.observation.get("qpos"), latest_action=action, mode="human_teleop")
                        t2 = time.time()

                        action_list = list(action)
                        timesteps.append(ts)
                        actions.append(action_list)
                        action_history.append(action_list)
                        actual_dt_history.append([t0, t1, t2])
                        timestamps.append(t0)
                        step_count += 1

                        stop_key = self._take_local_key(timeout=max(0, self._manual_step_time - (time.time() - t0)))
                        if stop_key == "b":
                            logging.info("检测到按键 b，停止人工接管数据采集。")
                            break
            finally:
                robot_utils.torque_on(master_bot_left)
                robot_utils.torque_on(master_bot_right)
                logging.info("master torque已恢复")
            
            logging.info(f"停止数据收集，共收集 {step_count} 步数据")
            
            if not timesteps:
                logging.warning("没有数据可保存，跳过保存。")
            else:
                observations = [ts.observation for ts in timesteps]
                _hdf5_utils.save_hdf5_episode(
                    observations,
                    actions,
                    episode_dataset_dir,
                    compress_images=True,
                    is_mobile=False,
                    fps=self._manual_hz if self._manual_hz > 0 else None,
                    timestamps=timestamps,
                )
            
            self._is_waiting_for_task = True
            self._current_task = None
            self._publish_runtime_state(mode="waiting")
            if latest_task:
                self._handle_task(latest_task)
                
        except Exception as e:
            logging.error(f"人机协作模式出错: {e}", exc_info=True)
            try:
                from examples.aloha_real import robot_utils
                if hasattr(self._environment, '_env'):
                    real_env = self._environment._env
                    robot_utils.torque_on(real_env.master_bot_left)
                    robot_utils.torque_on(real_env.master_bot_right)
            except:
                pass
            self._is_waiting_for_task = True
            self._current_task = None
            self._publish_runtime_state(mode="waiting")
        finally:
            self._last_action = None
