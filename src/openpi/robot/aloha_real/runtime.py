import logging
import threading
import time
import json
import redis
import os
import re
import sys
import termios
import tty
import select
from collections import deque

from openpi.robot.aloha_real.leader_follower_demo import LeaderFollowerDemoController
from openpi.robot.aloha_real.manual_intervention import ManualInterventionController
from openpi.robot.aloha_real.policy_recording import PolicyRecording
from openpi.robot.aloha_real.video_memory_preview import VideoMemoryPreview
from openpi.serving import base_policy as _base_policy
from openpi.shared.prompt_metadata import INFERENCE_SETTINGS_KEY, INFERENCE_UPDATES_CHANNEL, METADATA_KEYS, normalize_prompt_metadata

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

    _TASK_PROMPT_BY_NUM = {
        "1": os.getenv("ALOHA_TASK_1_PROMPT", "Twist off the bottle cap"),
        "2": os.getenv("ALOHA_TASK_2_PROMPT", "Rinse bottle"),
    }

    def __init__(
        self,
        environment,
        policy: _base_policy.BasePolicy,
        subscribers: list,
        max_hz: float = 0,
        manual_hz: float = 0,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        manual_dataset_dir: str | None = None,
        good_bad_action: str | None = None,  # Deprecated, accepted for old callers only.
    ) -> None:
        self._environment = environment
        self._policy = policy
        self._subscribers = subscribers
        self._control_samples = deque(maxlen=500)
        self._last_control_tick = None
        self._next_control_report = 0.0
        self._max_hz = max_hz
        self._manual_hz = manual_hz

        self._step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        self._manual_step_time = 1 / self._manual_hz if self._manual_hz > 0 else 0
        self._manual_dataset_dir = manual_dataset_dir or "/app/data/aloha_real/manual_intervention_episodes"
        
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
        self._pending_inference_settings = None
        # Recording commands are consumed only by the actuator/control thread.
        self._recording_commands = deque()
        self._recording_hold = threading.Event()
        self._recording_paused = False
        self._recording_command_id = None
        self._recording_command_error = None
        self._recording = PolicyRecording(
            os.getenv("ALOHA_ROLLOUT_DIR", "/app/data/aloha_real/rollout_episodes"), fps=max_hz or 50.0
        )
        
        # 任务状态管理
        self._current_task = None
        self._is_waiting_for_task = False
        
        # 存储最后的action（用于task_num==3时移动master）
        self._last_action = None
        history_size = max(1, int((self._max_hz if self._max_hz > 0 else 1) * 10))
        self._recent_puppet_actions = deque(maxlen=history_size)

        # 退出标志
        self._stop = False
        self._keyboard_task_mapping = {
            "1": self._TASK_PROMPT_BY_NUM["1"],
            "2": self._TASK_PROMPT_BY_NUM["2"],
            "3": "Stop and human hand control",
            "4": "Return to home position and save hdf5",
            "5": "Return to sleep position, save hdf5 and quit robot runtime",
            "6": "Leader follower demo",
        }
        self._model_task_nums = {"1", "2"}
        self._stop_task_nums = {"4", "5"}
        self._manual_intervention = ManualInterventionController(
            environment=self._environment,
            manual_dataset_dir=self._manual_dataset_dir,
            manual_step_time=self._manual_step_time,
            manual_hz=self._manual_hz,
            policy_step_time=self._step_time,
            model_task_nums=self._model_task_nums,
            stop_task_nums=self._stop_task_nums,
            poll_key=self._poll_single_key,
            take_latest_task=self._take_latest_task,
            build_task_from_key=self._build_task_from_key,
            handle_task=self._handle_task,
            publish_runtime_state=self._publish_runtime_state,
            enter_waiting=self._enter_waiting,
        )
        self._leader_follower_demo = LeaderFollowerDemoController(
            environment=self._environment,
            manual_step_time=self._manual_step_time,
            model_task_nums=self._model_task_nums,
            stop_task_nums=self._stop_task_nums,
            poll_key=self._poll_single_key,
            take_latest_task=self._take_latest_task,
            build_task_from_key=self._build_task_from_key,
            handle_task=self._handle_task,
            publish_runtime_state=self._publish_runtime_state,
            enter_waiting=self._enter_waiting,
            should_stop=lambda: self._stop,
        )

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
        pubsub.subscribe("aloha_voice_commands", INFERENCE_UPDATES_CHANNEL)
        
        logging.info("开始监听Redis pub/sub频道: aloha_voice_commands")
        
        try:
            while self._redis_running:
                message = pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        if data.get("type") == "recording_control":
                            # Latch stop before an in-flight inference dispatches its next action.
                            if data.get("action") == "stop":
                                self._recording_hold.set()
                            with self._task_lock:
                                if len(self._recording_commands) < 16:
                                    self._recording_commands.append(data)
                            continue
                        if data.get('type') == 'inference_settings':
                            metadata = normalize_prompt_metadata(**{
                                key: data[key] for key in METADATA_KEYS if key in data
                            })
                            with self._task_lock:
                                self._pending_inference_settings = metadata
                            logging.info("Inference conditioning update queued: %s", metadata)
                            continue
                        task_num = data.get('task')
                        task_name = data.get('task_name', '未知任务')
                        timestamp = data.get('timestamp', time.time())
                        
                        logging.info(f"收到Redis任务: {task_num} - {task_name}")
                        
                        with self._task_lock:
                            self._latest_task = {
                                'task_num': task_num,
                                'task_name': task_name,
                                'timestamp': timestamp
                            }
                            self._latest_task.update({key: data[key] for key in METADATA_KEYS if key in data})
                            
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

    def _publish_runtime_state(self, *, qpos=None, latest_action=None, mode: str | None = None) -> None:
        """发布轻量运行时状态给可视化前端。"""
        if self._redis_client is None:
            return

        current_task = self._current_task.get("task_name") if self._current_task else None
        if qpos is None and hasattr(self._environment, "_ts") and getattr(self._environment, "_ts") is not None:
            qpos = self._environment._ts.observation.get("qpos")

        payload = {
            "timestamp": time.time(),
            "mode": mode or ("waiting" if self._is_waiting_for_task else "policy"),
            "current_task": current_task,
            "inference_prompt": getattr(self._policy, "inference_prompt", None) if current_task else None,
            "operator": self._current_task.get("operator") if self._current_task else None,
            "camera_low_position": self._current_task.get("camera_low_position") if self._current_task else None,
            "rating": self._current_task.get("rating") if self._current_task else None,
            "qpos": list(qpos) if qpos is not None else [],
            "latest_action": list(latest_action) if latest_action is not None else [],
            "recording": self._recording_status(),
        }
        if getattr(self, "_recording_paused", False):
            payload["mode"] = "paused"
        try:
            self._redis_client.set("aloha_recording_state", json.dumps({
                **payload["recording"], "timestamp": payload["timestamp"], "mode": payload["mode"]
            }))
            self._redis_client.publish("aloha_runtime_state", json.dumps(payload))
        except Exception as exc:
            logging.debug("发布运行时状态失败: %s", exc)

    def _take_latest_task(self, allowed_task_nums: set[str] | None = None):
        """获取并消费最新的 Redis 任务。"""
        with self._task_lock:
            if self._latest_task is None:
                return None
            task_num = str(self._latest_task.get("task_num"))
            if allowed_task_nums is not None and task_num not in allowed_task_nums:
                return None
            latest_task = self._latest_task
            self._latest_task = None
            return latest_task

    def _normalize_task_data(self, task_data):
        """Canonicalize task prompts before they are shown or sent to the policy."""
        if task_data is None:
            return None
        normalized = dict(task_data)
        task_num = str(normalized.get("task_num"))
        if task_num in self._TASK_PROMPT_BY_NUM:
            normalized["task_name"] = self._TASK_PROMPT_BY_NUM[task_num]
        return normalized
    
    def run(self) -> None:
        """Runs the runtime loop continuously until stop() is called or the environment is done."""
        # 启动Redis监听
        self._start_redis_listener()
        
        try:
            self._run()
        finally:
            # 停止Redis监听
            self._stop_redis_listener()
            self._recording.close()

    def stop(self) -> None:
        """Request the runtime loop to stop."""
        self._stop = True

    def _recording_status(self):
        """Expose file finalization and command acknowledgements independently."""
        recorder = getattr(self, "_recording", None)
        state = recorder.snapshot() if recorder is not None else {"status": "idle", "frames": 0}
        return {**state, "paused": getattr(self, "_recording_paused", False),
                "command_id": getattr(self, "_recording_command_id", None),
                "command_error": getattr(self, "_recording_command_error", None)}

    def _pause_recording(self):
        """Hold the last commanded pose, without home, sleep or torque-off commands."""
        self._recording_hold.set()
        self._recording_paused = True
        self._is_waiting_for_task = True
        self._recording.finish()
        self._publish_runtime_state(mode="paused")
        self._policy.reset()  # Wait for an old background RPC, then discard its chunks.

    def _process_recording_commands(self):
        """Serialize recording state changes with action execution on the control thread."""
        with self._task_lock:
            commands = list(self._recording_commands)
            self._recording_commands.clear()
        for command in commands:
            self._recording_command_id = command.get("command_id")
            self._recording_command_error = None
            try:
                action = command["action"]
                if action == "start":
                    if self._is_waiting_for_task or self._recording_paused or self._current_task is None:
                        raise ValueError("Start the robot task or resume first; recording never starts robot motion")
                    if self._recording.snapshot()["status"] in {"recording", "saving"}:
                        raise ValueError("A recording is already active or saving")
                    self._recording.start({"source": "policy_rollout", "task": dict(self._current_task),
                                           "annotations": command.get("annotations", {})}, batch_name=command.get("batch_name"))
                elif action == "stop":
                    self._recording.finish(command.get("annotations", {}))
                    self._pause_recording()
                elif action == "resume":
                    if not self._recording_paused or self._current_task is None:
                        raise ValueError("No paused policy task to resume")
                    if self._recording.snapshot()["status"] in {"recording", "saving", "error"}:
                        raise ValueError("Wait until recording is saved; resolve any recording error first")
                    self._policy.reset()
                    self._recording_paused = False
                    self._recording_hold.clear()
                    self._is_waiting_for_task = False
                else:
                    raise ValueError("Unknown recording command")
            except Exception as exc:
                self._recording_command_error = str(exc)
                logging.exception("Recording command failed")
            self._publish_runtime_state()
        if self._recording.snapshot()["status"] == "error" and not self._recording_paused:
            self._pause_recording()

    def _enter_waiting(self) -> None:
        self._is_waiting_for_task = True
        self._current_task = None
        self._publish_runtime_state(mode="waiting")

    def _run(self) -> None:
        """Runs a single episode."""
        logging.info("Starting episode...")
        self._environment.reset()
        self._policy.reset()
        
        last_step_time = time.time()
        
        # 初始状态为等待任务
        self._is_waiting_for_task = True
        self._current_task = None
        self._publish_runtime_state(mode="waiting")
        fd = None
        old_settings = None
        if sys.stdin.isatty():
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            logging.info(
                "键盘快捷键已启用：1 拧瓶盖任务，2 冲洗瓶子，3 人工接管，4 回 home 并保存，5 回 sleep 并退出，6 遥操作体验"
            )
        else:
            logging.warning("stdin 不是 TTY，主循环中无法监听键盘快捷键")

        try:
            while not self._stop:
                self._process_recording_commands()
                task_data = self._poll_task_from_inputs()
                if task_data:
                    self._handle_task(task_data)
                
                if self._is_waiting_for_task:
                    # 等待状态下，短sleep并持续监听键盘/Redis
                    self._publish_runtime_state()
                    time.sleep(0.05)
                else:
                    # 有任务时正常执行step
                    self._step()
                    # Sleep to maintain the desired frame rate
                    now = time.time()
                    dt = now - last_step_time
                    if dt < self._step_time:
                        time.sleep(self._step_time - dt)
                        last_step_time = time.time()
                    else:
                        last_step_time = now
        finally:
            if fd is not None and old_settings is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _poll_single_key(self, timeout: float = 0.0) -> str | None:
        """非阻塞读取单个按键，支持方向键。"""
        if not sys.stdin.isatty():
            return None

        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if not ready:
            return None

        key = sys.stdin.read(1)
        if key == "\x03":
            raise KeyboardInterrupt
        if key == "\x1b":
            next_1 = sys.stdin.read(1)
            if next_1 == "[":
                next_2 = sys.stdin.read(1)
                return f"\x1b[{next_2}"
            return key + next_1
        return key

    def _build_task_from_key(
        self,
        key: str | None,
        *,
        allowed_task_nums: set[str] | None = None,
        prompt_for_manual_dataset: bool = True,
        log_invalid: bool = True,
    ):
        """将单个键盘输入解析成统一 task_data。"""
        if key is None or key in ("\n", "\r"):
            return None

        task_name = self._keyboard_task_mapping.get(key)
        if task_name is None:
            if log_invalid:
                valid_keys = "/".join(sorted(allowed_task_nums or set(self._keyboard_task_mapping)))
                logging.info("忽略键盘输入 %r；可用快捷键: %s", key, valid_keys)
            return None
        if allowed_task_nums is not None and key not in allowed_task_nums:
            return None

        task_data = {
            "task_num": key,
            "task_name": task_name,
            "timestamp": time.time(),
        }
        if key == "3" and prompt_for_manual_dataset:
            while True:
                dataset_subdir = self._read_line_from_keyboard(
                    "请输入人工接管数据保存子目录名，然后回车: "
                ).strip()
                if not dataset_subdir:
                    logging.warning("未输入人工接管数据保存子目录名，已取消进入人工接管模式。")
                    return None
                if re.fullmatch(r"[A-Za-z0-9]+", dataset_subdir):
                    break
                logging.warning("目录名只允许字母和数字，请重新输入。")
            task_data["manual_dataset_subdir"] = dataset_subdir
            logging.info("人工接管数据将保存到子目录: %s", dataset_subdir)

        task_data = self._normalize_task_data(task_data)
        logging.info("收到键盘任务: %s - %s", key, task_data["task_name"])
        return task_data

    def _poll_task_from_inputs(
        self,
        *,
        allowed_task_nums: set[str] | None = None,
        keyboard_timeout: float = 0.0,
        prompt_for_manual_dataset: bool = True,
    ):
        """统一轮询键盘和 Redis 任务输入。"""
        key = self._poll_single_key(timeout=keyboard_timeout)
        task_data = self._build_task_from_key(
            key,
            allowed_task_nums=allowed_task_nums,
            prompt_for_manual_dataset=prompt_for_manual_dataset,
        )
        if task_data is not None:
            return task_data
        return self._normalize_task_data(self._take_latest_task(allowed_task_nums=allowed_task_nums))

    def _read_line_from_keyboard(self, prompt: str) -> str:
        """在cbreak模式下读取一行输入。"""
        sys.stdout.write(prompt)
        sys.stdout.flush()
        chars: list[str] = []
        while True:
            ch = sys.stdin.read(1)
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch in ("\n", "\r"):
                sys.stdout.write("\n")
                sys.stdout.flush()
                return "".join(chars)
            if ch in ("\x7f", "\b"):
                if chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            chars.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()
        

    def _handle_task(self, task_data) -> None:
        """处理来自键盘或 Redis 的任务。"""
        task_num = task_data.get('task_num')
        task_name = task_data.get('task_name', '未知任务')
        
        logging.info(f"处理语音任务: {task_num} - {task_name}")

        if getattr(self, "_recording_paused", False) and task_num in {"1", "2", "6"}:
            self._recording_command_error = "Recording paused the robot; use Resume execution first"
            self._publish_runtime_state(mode="paused")
            return
        recorder = getattr(self, "_recording", None)
        if recorder is not None and recorder.snapshot()["status"] == "recording":
            recorder.finish()  # Never mix different robot tasks in the same recording.
        if task_num in {"3", "4", "5"} and recorder is not None:
            self._recording_paused = False
            self._recording_hold.clear()
        
        if task_num in self._model_task_nums:
            task_data = dict(task_data)
            with self._task_lock:
                self._pending_inference_settings = None
                if self._redis_client is not None:
                    raw = self._redis_client.get(INFERENCE_SETTINGS_KEY)
                    if raw:
                        stored = json.loads(raw)
                        for key in METADATA_KEYS:
                            task_data.pop(key, None)
                        task_data.update({key: stored[key] for key in METADATA_KEYS if key in stored})
            metadata = normalize_prompt_metadata(**{
                key: task_data[key] for key in METADATA_KEYS if key in task_data
            })
            for key in METADATA_KEYS:
                task_data.pop(key, None)
            task_data.update(metadata)
            self._policy.reset()
            logging.info(f"开始执行任务: {task_name}")
            for subscriber in self._subscribers:
                subscriber.on_episode_start()
            # 设置当前任务
            self._current_task = task_data
            self._is_waiting_for_task = False 
            self._publish_runtime_state(mode="policy")
        elif task_num == "3":
            logging.info("收到停止指令，进入人机协作模式")
            self._current_task = task_data
            self._policy.reset()
            episode_subdir = task_data.get("manual_dataset_subdir")
            for subscriber in self._subscribers:
                subscriber.on_episode_end(episode_subdir=episode_subdir)
            self._publish_runtime_state(mode="teleop_prepare")
            self._manual_intervention.run(
                task_data,
                last_action=self._last_action,
                recent_puppet_actions=list(self._recent_puppet_actions),
            )
            self._last_action = None
        elif task_num == "6":
            logging.info("收到遥操作体验指令，进入leader-follower演示模式")
            self._current_task = task_data
            self._is_waiting_for_task = False
            self._policy.reset()
            if self._last_action is not None:
                for subscriber in self._subscribers:
                    subscriber.on_episode_end()
            self._publish_runtime_state(mode="leader_follower_prepare")
            self._leader_follower_demo.run()
        elif task_num == "4":
            logging.info("收到停止指令，回到初始位置并停止agent")
            self._enter_waiting()
            # 回到初始位置
            self._environment.stop()
            # 停止agent
            self._policy.reset()   
            # 通知subscriber episode结束
            for subscriber in self._subscribers:
                subscriber.on_episode_end()   
            self._publish_runtime_state(mode="waiting")
        elif task_num == "5":
            logging.info("收到回到sleep位置并退出指令，退出程序")
            self._environment.sleep_arms()
            self._policy.reset()
            for subscriber in self._subscribers:
                subscriber.on_episode_end()
            self._publish_runtime_state(mode="sleep")
            self._stop = True
        else:
            logging.warning(f"未知任务编号: {task_num}")

    def _apply_pending_inference_settings(self) -> None:
        with self._task_lock:
            metadata = self._pending_inference_settings
            self._pending_inference_settings = None
        if metadata is None or self._current_task is None:
            return
        current = {key: self._current_task[key] for key in METADATA_KEYS if key in self._current_task}
        if metadata == current:
            return
        # Called by the control thread, never by the Redis listener. Reset waits
        # for an in-flight RPC, then discards all actions from the old condition.
        self._policy.reset()
        self._current_task = {key: value for key, value in self._current_task.items() if key not in METADATA_KEYS}
        self._current_task.update(metadata)
        logging.info("Inference conditioning applied live: %s", metadata)
        self._publish_runtime_state(mode="policy")

    def _step(self) -> None:
        """A single step of the runtime loop."""
        hold = getattr(self, "_recording_hold", None)
        if hold is not None and hold.is_set():
            return
        self._apply_pending_inference_settings()
        step_started = time.monotonic()
        observation = self._environment.get_observation()
        observation_done = time.monotonic()
        # Inspect the exact pre-model image history without blocking robot control.
        if not hasattr(self, "_video_memory_preview"):
            self._video_memory_preview = VideoMemoryPreview()
        self._video_memory_preview.submit(
            observation["images"],
            stride_seconds=self._environment.image_recorder.history_stride_seconds
            if hasattr(self._environment, "image_recorder")
            else self._environment._env.image_recorder.history_stride_seconds,
        )
        assert self._current_task is not None, "_current_task must be set before calling _step()"
        observation_with_task = {
            **observation,
            'task': self._current_task.get('task_name'),
            **{key: self._current_task[key] for key in METADATA_KEYS if key in self._current_task},
        }

        infer_started = time.monotonic()
        action = self._policy.infer(observation_with_task)
        infer_done = time.monotonic()
        # A stop arriving during inference must not execute the newly returned chunk.
        if hold is not None and hold.is_set():
            return
        self._environment.apply_action(action)
        action_done = time.monotonic()
        self._report_control_timing(step_started, observation_done, infer_started, infer_done, action_done)
        recorder = getattr(self, "_recording", None)
        if recorder is not None:
            try:
                recorder.append(observation["origin_observation"], action, dict(self._current_task))
            except Exception:
                logging.exception("Recording failed; pausing robot")
                self._pause_recording()
                return
        # 存储最后的action（用于task_num==3时移动master）
        self._last_action = action.get("actions") if isinstance(action, dict) and "actions" in action else None
        if self._last_action is not None:
            self._recent_puppet_actions.append(list(self._last_action))
        self._publish_runtime_state(latest_action=self._last_action, mode="policy")

        for subscriber in self._subscribers:
            subscriber.on_step(observation["origin_observation"], action)

    def _report_control_timing(self, started, observation_done, infer_started, infer_done, action_done):
        previous = self._last_control_tick
        self._last_control_tick = started
        if previous is None or started - previous > 1.0:
            self._control_samples.clear()
            self._next_control_report = started + 5.0
            return
        self._control_samples.append((started - previous, observation_done - started,
                                      infer_done - infer_started, action_done - infer_done))
        if started < self._next_control_report or len(self._control_samples) < 50:
            return
        self._next_control_report = started + 5.0
        samples = list(self._control_samples)
        count = len(samples)
        hz = count / sum(row[0] for row in samples)
        p95 = sorted(row[0] for row in samples)[int((count - 1) * 0.95)] * 1000
        self._control_health = {"target_hz": self._max_hz, "actual_hz": round(hz, 2),
            "interval_p95_ms": round(p95, 2), "samples": count,
            "observation_ms": round(sum(r[1] for r in samples) * 1000 / count, 2),
            "policy_ms": round(sum(r[2] for r in samples) * 1000 / count, 2),
            "apply_and_images_ms": round(sum(r[3] for r in samples) * 1000 / count, 2),
            "timestamp": time.time()}
        log = logging.warning if hz < self._max_hz * 0.9 else logging.info
        log("CONTROL_RATE %s", json.dumps(self._control_health))
        if self._redis_client is not None:
            self._redis_client.set("aloha_control_health", json.dumps(self._control_health), ex=15)
