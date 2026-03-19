import logging
import threading
import time
import json
import redis
import os
import sys
from collections import deque

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
        high_level_policy=None,
        high_level_hz: float = 0.0,
        good_bad_action: str | None = "good action",
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

        # 退出标志
        self._stop = False
        self._model_task_nums = {"1", "2"}
        self._stop_task_nums = {"4", "5"}

    def _apply_task_paths(self, task_data: dict) -> None:
        dataset_dir = task_data.get("dataset_dir")
        manual_dataset_dir = task_data.get("manual_dataset_dir")
        if isinstance(dataset_dir, str) and dataset_dir.strip():
            self._dataset_dir = dataset_dir.strip()
        if isinstance(manual_dataset_dir, str) and manual_dataset_dir.strip():
            self._manual_dataset_dir = manual_dataset_dir.strip()
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
        if self._high_level_policy is not None:
            self._high_level_policy.reset()

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

    def _get_high_level_state(self) -> tuple[dict | None, dict]:
        with self._high_level_state_lock:
            structured_subtask = self._latest_structured_subtask
            hierarchical_state = dict(self._latest_hierarchical_state)
        return structured_subtask, hierarchical_state

    def _high_level_worker(self) -> None:
        last_processed_version = -1
        while self._high_level_running:
            if self._is_waiting_for_task or self._current_task is None:
                time.sleep(0.01)
                continue

            with self._high_level_obs_lock:
                obs = self._high_level_obs
                version = self._high_level_obs_version

            if obs is None or version == last_processed_version:
                time.sleep(0.005)
                continue

            try:
                high_level_result = self._high_level_policy.infer_subtask(obs)
                high_level_text = str(high_level_result.get("subtask_text") or "").strip()
                structured_subtask = _subtask_parsing.build_low_level_subtask_payload(
                    high_level_text,
                    good_bad_action=self._good_bad_action,
                )
                parsed = _subtask_parsing.parse_structured_fields(high_level_text)
                hierarchical_state = {
                    "task_prompt": str(obs.get("prompt") or "").strip(),
                    "low_level_prompt": json.dumps(structured_subtask, ensure_ascii=False)
                    if structured_subtask is not None
                    else (high_level_text or str(obs.get("prompt") or "").strip()),
                    "high_level_server_timing": high_level_result.get("server_timing", {}),
                    "low_level_server_timing": {},
                    "good_bad_action": structured_subtask.get("good_bad_action") if structured_subtask is not None else None,
                    **parsed,
                }
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
            self._stop_high_level_worker()
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
        self._reset_high_level_state()
        self._start_high_level_worker()

        self._in_episode = True
        self._episode_steps = 0
        
        last_step_time = time.time()
        
        self._is_waiting_for_task = True
        self._current_task = None
        self._publish_runtime_state(mode="waiting")
        logging.info("Runtime 已切换为仅接受 Redis / voice web 任务，不再监听本地键盘")

        while not self._stop:
            task_data = self._take_latest_task()
            if task_data:
                self._handle_task(task_data)
            
            if self._is_waiting_for_task:
                time.sleep(0.05)
            else:
                self._step()
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
            self._reset_high_level_state()
            self._publish_runtime_state(mode="policy")
        elif task_num == "3":
            logging.info("收到停止指令，进入人机协作模式")
            self._current_task = task_data
            self._agent.reset() 
            self._reset_high_level_state()
            for subscriber in self._subscribers:
                subscriber.on_episode_end()
            self._publish_runtime_state(mode="teleop_prepare")
            self._handle_human_teleop_mode()  
        elif task_num == "4":
            logging.info("收到停止指令，回到初始位置并停止agent")
            # 设置等待状态
            self._is_waiting_for_task = True
            self._current_task = None
            self._reset_high_level_state()
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
            self._reset_high_level_state()
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

        self._update_high_level_observation(observation_with_task)
        structured_subtask, hierarchical = self._get_high_level_state()
        if structured_subtask is not None:
            observation_with_task["subtask"] = structured_subtask

        action = self._agent.get_action(observation_with_task)
        self._environment.apply_action(action)
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
            
            action = self._last_action
            self._move_master_to_action(real_env, action, move_time=0.5)
            logging.info("leader已移动到上次模型输出位置")

            robot_utils.torque_off(master_bot_left)
            robot_utils.torque_off(master_bot_right)
            logging.info("master torque已关闭")

            timesteps = []
            actions = []
            timestamps = []
            actual_dt_history = []
            
            logging.info("开始人机协作数据收集；后续通过 voice web 发送任务 1/2/4/5 退出或切换模式")

            if not self._manual_dataset_dir:
                logging.warning("未从 voice web 收到人工接管保存路径，取消本次人工接管数据保存。")
                self._is_waiting_for_task = True
                self._current_task = None
                self._publish_runtime_state(mode="waiting")
                return
            episode_dataset_dir = self._manual_dataset_dir
            os.makedirs(episode_dataset_dir, exist_ok=True)

            step_count = 0
            latest_task = None
            while True:
                latest_task = self._take_latest_task(
                    allowed_task_nums=self._model_task_nums | self._stop_task_nums
                )
                if latest_task:
                    logging.info(
                        "人机协作模式收到任务 %s，结束当前人工接管并切换流程",
                        latest_task["task_num"],
                    )
                    break

                t0 = time.time()
                action = get_action(master_bot_left, master_bot_right)
                t1 = time.time()
                
                self._environment.apply_action({"actions": action})
                ts = self._environment._ts
                self._publish_runtime_state(qpos=ts.observation.get("qpos"), latest_action=action, mode="human_teleop")
                t2 = time.time()
                
                timesteps.append(ts)
                actions.append(action)
                actual_dt_history.append([t0, t1, t2])
                timestamps.append(t0)
                
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
