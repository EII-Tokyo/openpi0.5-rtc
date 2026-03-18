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

    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscribers: list[_subscriber.Subscriber],
        max_hz: float = 0,
        manual_hz: float = 0,
        num_episodes: int = 1,
        max_episode_steps: int = 0,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        manual_dataset_dir: str | None = None,
    ) -> None:
        self._environment = environment
        self._agent = agent
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._manual_hz = manual_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = max_episode_steps

        self._step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        self._manual_step_time = 1 / self._manual_hz if self._manual_hz > 0 else 0
        self._manual_dataset_dir = manual_dataset_dir or "/app/examples/aloha_real/manual_override"

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

        # 退出标志
        self._stop = False
        self._keyboard_task_mapping = {
            "1": "Remove the label from the bottle with the knife in the right hand.",
            "2": "process all bottles",
            "3": "Stop and human hand control",
            "4": "Return to home position and save hdf5",
            "5": "Return to sleep position, save hdf5 and quit robot runtime",
        }
        self._model_task_nums = {"1", "2"}
        self._stop_task_nums = {"4", "5"}

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
            "hierarchical": {},
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
        # 启动Redis监听
        self._start_redis_listener()
        
        try:
            self._run()
        finally:
            # 停止Redis监听
            self._stop_redis_listener()

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

        self._in_episode = True
        self._episode_steps = 0
        
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
            logging.info("键盘快捷键已启用：2 执行任务，3 人工接管，4 回 home 并保存，5 回 sleep 并退出")
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
                    self._episode_steps += 1
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

        logging.info("收到键盘任务: %s - %s", key, task_name)
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
        return self._take_latest_task(allowed_task_nums=allowed_task_nums)

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
            # 停止agent
            self._agent.reset() 
            # 通知subscriber episode结束, 并录制数据
            episode_subdir = task_data.get("manual_dataset_subdir")
            for subscriber in self._subscribers:
                subscriber.on_episode_end(episode_subdir=episode_subdir) 
            self._publish_runtime_state(mode="teleop_prepare")
            self._handle_human_teleop_mode()  
        elif task_num == "4":
            logging.info("收到停止指令，回到初始位置并停止agent")
            # 设置等待状态
            self._is_waiting_for_task = True
            self._current_task = None
            # 回到初始位置
            self._environment.stop()
            # 停止agent
            self._agent.reset()   
            # 通知subscriber episode结束
            for subscriber in self._subscribers:
                subscriber.on_episode_end()   
            self._publish_runtime_state(mode="waiting")
        elif task_num == "5":
            logging.info("收到回到sleep位置并退出指令，退出程序")
            self._environment.sleep_arms()
            self._agent.reset()
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
        observation = self._environment.get_observation()
        assert self._current_task is not None, "_current_task must be set before calling _step()"
        observation_with_task = {
            **observation,
            'prompt': self._current_task.get('task_name')
        }
            
        action = self._agent.get_action(observation_with_task)
        self._environment.apply_action(action)
        # 存储最后的action（用于task_num==3时移动master）
        self._last_action = action.get("actions") if isinstance(action, dict) and "actions" in action else None
        if self._last_action is not None:
            self._recent_puppet_actions.append(list(self._last_action))
        hierarchical = action.get("hierarchical", {}) if isinstance(action, dict) else {}
        if not isinstance(hierarchical, dict):
            hierarchical = {}

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

        for subscriber in self._subscribers:
            subscriber.on_step(observation["origin_observation"], action)

    def _move_robots_to_action(self, real_env, action, step_sleep: float = 0.0) -> None:
        """将puppet和master同步到单个action。"""
        from examples.aloha_real import robot_utils
        from examples.aloha_real import constants
        from interbotix_xs_msgs.msg import JointSingleCommand

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

        # 通过环境 wrapper 驱动 puppet，这样 AlohaRealEnvironment._ts 会按既有逻辑更新。
        self._environment.apply_action({"actions": action})
        self._publish_runtime_state(latest_action=action, mode="teleop_preview")

        master_bot_left.arm.set_joint_positions(left_arm_pos, blocking=False)
        master_bot_right.arm.set_joint_positions(right_arm_pos, blocking=False)
        gripper_command = JointSingleCommand(name="gripper")
        gripper_command.cmd = master_left_gripper_joint
        master_bot_left.gripper.core.pub_single.publish(gripper_command)
        gripper_command.cmd = master_right_gripper_joint
        master_bot_right.gripper.core.pub_single.publish(gripper_command)

        if step_sleep > 0:
            time.sleep(step_sleep)

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

    def _replay_history_actions(self, real_env, history_actions, start_index: int, target_index: int) -> int:
        """按history里的action逐步回放到目标帧。"""
        if target_index >= start_index:
            return start_index

        step_sleep = self._step_time if self._step_time > 0 else 0.0
        for idx in range(start_index - 1, target_index - 1, -1):
            self._move_robots_to_action(real_env, history_actions[idx], step_sleep=step_sleep)
        return target_index


    def _handle_human_teleop_mode(self) -> None:
        """处理人机协作模式（task_num==3）"""
        try:
            # 导入必要的模块
            from examples.aloha_real import robot_utils
            from examples.aloha_real.real_env import get_action
            
            # 获取real_env实例
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
                return

            if not self._recent_puppet_actions:
                self._recent_puppet_actions.append(list(self._last_action))

            rewind_steps = max(1, int(round((self._max_hz if self._max_hz > 0 else 1) * 0.25)))
            history_actions = list(self._recent_puppet_actions)
            history_index = len(history_actions) - 1
            
            # 步骤1: 将master移动到上次模型输出的位置
            action = self._last_action
            self._move_master_to_action(real_env, action, move_time=0.5)
                
            logging.info("leader已移动到上次模型输出位置")
            
            # 步骤2: 等待按键
            logging.info("等待按下'b'键开始人机控制...")
            logging.info(f"（按左方向键每次回退0.25秒，约 {rewind_steps} 个policy step；按'b'键开始）")
            
            while True:
                latest_task = self._take_latest_task(
                    allowed_task_nums=self._model_task_nums | self._stop_task_nums
                )
                if latest_task:
                    logging.info(
                        "人工接管准备阶段收到Redis任务 %s，退出人工接管准备并执行对应任务",
                        latest_task["task_num"],
                    )
                    self._handle_task(latest_task)
                    return

                key = self._poll_single_key(timeout=0.05)
                if key is None:
                    continue
                if key.lower() == 'b':
                    break
                if key == "\x1b[D":
                    target_index = max(0, history_index - rewind_steps)
                    history_index = self._replay_history_actions(real_env, history_actions, history_index, target_index)
                    self._last_action = list(history_actions[history_index])
                    logging.info(
                        "已回退0.25秒，当前位于最近轨迹第 %d/%d 帧",
                        history_index + 1,
                        len(history_actions),
                    )
                    continue
                task_data = self._build_task_from_key(
                    key,
                    allowed_task_nums=self._model_task_nums | self._stop_task_nums,
                    prompt_for_manual_dataset=False,
                    log_invalid=False,
                )
                if task_data is not None:
                    logging.info("收到输入 %r，退出人工接管准备并执行对应任务", key)
                    self._handle_task(task_data)
                    return
                logging.info("收到输入 %r；按左方向键回退，按'1'/'2'继续模型，按'4'/'5'执行停止任务，按'b'键开始", key)
            
            logging.info("收到'b'键，开始人机控制模式")
            
            # 步骤3: master torque off，puppet跟随master
            robot_utils.torque_off(master_bot_left)
            robot_utils.torque_off(master_bot_right)
            logging.info("master torque已关闭")
            
            # 步骤4: 数据收集循环
            timesteps = []
            actions = []
            timestamps = []
            actual_dt_history = []
            
            logging.info("=" * 60)
            logging.info("开始数据收集...")
            logging.info("提示：在当前终端直接按 'b' 键退出并保存数据")
            logging.info("（后台线程正在等待您的输入）")
            logging.info("=" * 60)

            if not self._current_task or not self._current_task.get("manual_dataset_subdir"):
                logging.warning("未找到人工接管数据保存子目录名，取消本次人工接管数据保存。")
                self._is_waiting_for_task = True
                self._current_task = None
                return
            episode_dataset_dir = os.path.join(
                self._manual_dataset_dir,
                self._current_task["manual_dataset_subdir"],
            )
            os.makedirs(episode_dataset_dir, exist_ok=True)
            
            # 使用线程来监听键盘输入（非阻塞方式）
            key_pressed = threading.Event()
            
            def keyboard_listener():
                """在后台线程中监听键盘输入"""
                try:
                    while not key_pressed.is_set():
                        # 等待用户按键（用户可以直接在当前终端按键）
                        key = _read_single_key().lower()
                        if key == 'b':
                            key_pressed.set()
                            print("收到退出信号，准备保存数据...")
                            logging.info("收到退出信号，准备保存数据...")
                        else:
                            logging.info(f"收到输入 '{key}'，但需要输入 'b' 才能退出")
                except (EOFError, KeyboardInterrupt):
                    key_pressed.set()
                    logging.info("收到退出信号")
                except Exception as e:
                    logging.warning(f"键盘监听出错: {e}")
            
            listener_thread = threading.Thread(target=keyboard_listener, daemon=True)
            listener_thread.start()
            
            # 数据收集循环（只收集原始数据，不填充data_dict）
            step_count = 0
            while not key_pressed.is_set():
                t0 = time.time()
                action = get_action(master_bot_left, master_bot_right)
                t1 = time.time()
                
                # 应用action到puppet
                self._environment.apply_action({"actions": action})
                ts = self._environment._ts
                self._publish_runtime_state(qpos=ts.observation.get("qpos"), latest_action=action, mode="human_teleop")
                t2 = time.time()
                
                timesteps.append(ts)
                actions.append(action)
                actual_dt_history.append([t0, t1, t2])
                timestamps.append(t0)
                
                # Sleep to maintain the desired frame rate
                time.sleep(max(0, self._manual_step_time - (time.time() - t0)))
                
                step_count += 1
            
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
            
            # 设置等待状态
            self._is_waiting_for_task = True
            self._current_task = None
            self._publish_runtime_state(mode="waiting")
                
        except Exception as e:
            logging.error(f"人机协作模式出错: {e}", exc_info=True)
            # 确保恢复master torque
            try:
                from examples.aloha_real import robot_utils
                if hasattr(self._environment, '_env'):
                    real_env = self._environment._env
                    robot_utils.torque_on(real_env.master_bot_left)
                    robot_utils.torque_on(real_env.master_bot_right)
            except:
                pass
            # 设置等待状态
            self._is_waiting_for_task = True
            self._current_task = None
            self._publish_runtime_state(mode="waiting")
        finally:
            self._last_action = None
