import logging
import threading
import time
import json
import redis
import os

from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber


class Runtime:
    """The core module orchestrating interactions between key components of the system."""

    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscribers: list[_subscriber.Subscriber],
        max_hz: float = 0,
        num_episodes: int = 1,
        max_episode_steps: int = 0,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
    ) -> None:
        self._environment = environment
        self._agent = agent
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = max_episode_steps

        self._in_episode = False
        self._episode_steps = 0
        
        # Redis配置
        self._redis_host = os.getenv('REDIS_HOST', redis_host)
        self._redis_port = int(os.getenv('REDIS_PORT', redis_port))
        self._redis_db = int(os.getenv('REDIS_DB', redis_db))
        
        # 任务映射 - 与voice_assistant.py保持一致
        self._task_mapping = {
            "1": "Remove the label from the bottle with the knife in the right hand.",
            "2": "Twist off the bottle cap.", 
            "3": "Stop",
            "4": "Return to sleep position"
        }
        
        # Redis相关
        self._redis_client = None
        self._redis_thread = None
        self._redis_running = False
        self._latest_task = None
        self._task_lock = threading.Lock()
        
        # 任务状态管理
        self._current_task = None
        self._is_waiting_for_task = False

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

    def get_latest_task(self):
        """获取最新的任务"""
        with self._task_lock:
            return self._latest_task

    def clear_latest_task(self):
        """清除最新任务"""
        with self._task_lock:
            self._latest_task = None
    
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
            for _ in range(self._num_episodes):
                self._run_episode()

            # Final reset, this is important for real environments to move the robot to its home position.
            self._environment.reset()
        finally:
            # 停止Redis监听
            self._stop_redis_listener()

    def run_in_new_thread(self) -> threading.Thread:
        """Runs the runtime loop in a new thread."""
        thread = threading.Thread(target=self.run)
        thread.start()
        return thread

    def mark_episode_complete(self) -> None:
        """Marks the end of an episode."""
        self._in_episode = False

    def _run_episode(self) -> None:
        """Runs a single episode."""
        logging.info("Starting episode...")
        self._environment.reset()
        self._agent.reset()
        for subscriber in self._subscribers:
            subscriber.on_episode_start()

        self._in_episode = True
        self._episode_steps = 0
        step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        last_step_time = time.time()
        
        # 初始状态为等待任务
        self._is_waiting_for_task = True
        self._current_task = None

        while self._in_episode:
            # 检查是否有新的Redis任务
            latest_task = self.get_latest_task()
            if latest_task:
                self._handle_voice_task(latest_task)
                self.clear_latest_task()
            
            if self._is_waiting_for_task:
                # 等待状态下，1秒sleep一次
                logging.debug("等待新任务中...")
                time.sleep(1.0)
            else:
                # 有任务时正常执行step
                self._step()
                self._episode_steps += 1

                # Sleep to maintain the desired frame rate
                now = time.time()
                dt = now - last_step_time
                if dt < step_time:
                    time.sleep(step_time - dt)
                    last_step_time = time.time()
                else:
                    last_step_time = now

        logging.info("Episode completed.")
        for subscriber in self._subscribers:
            subscriber.on_episode_end()

    def _handle_voice_task(self, task_data) -> None:
        """处理语音任务"""
        task_num = task_data.get('task_num')
        task_name = task_data.get('task_name', '未知任务')
        
        logging.info(f"处理语音任务: {task_num} - {task_name}")
        
        if task_num == "3":  # stop任务
            logging.info("收到停止指令，回到初始位置并停止agent")
            # 回到初始位置
            self._environment.stop()
            # 停止agent
            self._agent.reset()
            # 设置等待状态
            self._is_waiting_for_task = True
            self._current_task = None
        elif task_num == "4":  # 回到sleep位置任务
            logging.info("收到回到sleep位置指令，回到初始位置并停止agent")
            # 回到初始位置
            self._environment.sleep_arms()
            # 停止agent
            self._agent.reset()
            # 设置等待状态
            self._is_waiting_for_task = True
            self._current_task = None
        elif task_num in ["1", "2"]:  # 其他任务
            logging.info(f"开始执行任务: {task_name}")
            # 设置当前任务
            self._current_task = task_data
            self._is_waiting_for_task = False
        else:
            logging.warning(f"未知任务编号: {task_num}")

    def _step(self) -> None:
        """A single step of the runtime loop."""
        observation = self._environment.get_observation()
        
        # 将当前任务信息添加到observation中传递给agent
        if self._current_task:
            observation_with_task = {
                **observation,
                'prompt': self._current_task.get('task_name')
            }
        else:
            observation_with_task = observation
            
        action = self._agent.get_action(observation_with_task)
        self._environment.apply_action(action)

        for subscriber in self._subscribers:
            subscriber.on_step(observation_with_task, action)

        # if self._environment.is_episode_complete() or (
        #     self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps
        # ):
        #     self.mark_episode_complete()
