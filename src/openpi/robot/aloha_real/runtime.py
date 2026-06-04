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
from openpi.serving import base_policy as _base_policy

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
        "1": "Twist off the bottle cap",
        "2": "Rinse bottle",
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
        good_bad_action: str = "normal",
    ) -> None:
        self._environment = environment
        self._policy = policy
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._manual_hz = manual_hz
        self._good_bad_action = good_bad_action

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
        pubsub.subscribe("aloha_voice_commands")
        
        logging.info("开始监听Redis pub/sub频道: aloha_voice_commands")
        
        try:
            while self._redis_running:
                message = pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
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
            "qpos": list(qpos) if qpos is not None else [],
            "latest_action": list(latest_action) if latest_action is not None else [],
        }
        try:
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

    def stop(self) -> None:
        """Request the runtime loop to stop."""
        self._stop = True

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
                task_data = self._poll_task_from_inputs()
                if task_data:
                    self._handle_task(task_data)
                
                if self._is_waiting_for_task:
                    # 等待状态下，短sleep并持续监听键盘/Redis
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
        
        if task_num in self._model_task_nums:
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

    def _step(self) -> None:
        """A single step of the runtime loop."""
        observation = self._environment.get_observation()
        assert self._current_task is not None, "_current_task must be set before calling _step()"
        observation_with_task = {
            **observation,
            'task': self._current_task.get('task_name'),
            'subtask': json.dumps({'good_bad_action': self._good_bad_action}),
        }

        action = self._policy.infer(observation_with_task)
        self._environment.apply_action(action)
        # 存储最后的action（用于task_num==3时移动master）
        self._last_action = action.get("actions") if isinstance(action, dict) and "actions" in action else None
        if self._last_action is not None:
            self._recent_puppet_actions.append(list(self._last_action))
        self._publish_runtime_state(latest_action=self._last_action, mode="policy")

        for subscriber in self._subscribers:
            subscriber.on_step(observation["origin_observation"], action)
