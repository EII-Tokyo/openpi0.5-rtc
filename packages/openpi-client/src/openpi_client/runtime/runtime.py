import logging
import threading
import time
import json
import redis
import os
import sys
import termios
import tty

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

        # 退出标志
        self._stop = False

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

        while not self._stop:
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
                if dt < self._step_time:
                    time.sleep(self._step_time - dt)
                    last_step_time = time.time()
                else:
                    last_step_time = now
        

    def _handle_voice_task(self, task_data) -> None:
        """处理语音任务"""
        task_num = task_data.get('task_num')
        task_name = task_data.get('task_name', '未知任务')
        
        logging.info(f"处理语音任务: {task_num} - {task_name}")
        
        if task_num in ["1", "2"]:  # 其他任务
            logging.info(f"开始执行任务: {task_name}")
            for subscriber in self._subscribers:
                subscriber.on_episode_start()
            # 设置当前任务
            self._current_task = task_data
            self._is_waiting_for_task = False 
        elif task_num == "3":  # stop任务 - 人机协作模式
            logging.info("收到停止指令，进入人机协作模式")
            # 停止agent
            self._agent.reset() 
            # 通知subscriber episode结束, 并录制数据
            for subscriber in self._subscribers:
                subscriber.on_episode_end() 
            self._handle_human_teleop_mode()  
            self._environment.reset(reset_position=False)      
        elif task_num == "4":  # 回到初始位置任务
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
        elif task_num == "5":  # quit任务
            logging.info("收到回到sleep位置并退出指令，退出程序")
            self._environment.sleep_arms()
            self._agent.reset()
            for subscriber in self._subscribers:
                subscriber.on_episode_end()
            self._stop = True
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
        # 存储最后的action（用于task_num==3时移动master）
        self._last_action = action.get("actions") if isinstance(action, dict) and "actions" in action else None

        for subscriber in self._subscribers:
            subscriber.on_step(observation["origin_observation"], action)


    def _handle_human_teleop_mode(self) -> None:
        """处理人机协作模式（task_num==3）"""
        try:
            # 导入必要的模块
            from examples.aloha_real import robot_utils
            from examples.aloha_real import constants 
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
            
            # 步骤1: 将master移动到上次模型输出的位置
            robot_utils.torque_on(master_bot_left)
            robot_utils.torque_on(master_bot_right)
            action = self._last_action

            # action格式: [left_arm_qpos (6), left_gripper (1), right_arm_qpos (6), right_gripper (1)]
            left_arm_pos = action[:6]
            left_gripper_normalized = action[6]
            right_arm_pos = action[7:7+6]
            right_gripper_normalized = action[7+6]

            # 转换gripper从normalized到master joint位置
            left_gripper_joint = constants.MASTER_GRIPPER_JOINT_UNNORMALIZE_FN(left_gripper_normalized)
            right_gripper_joint = constants.MASTER_GRIPPER_JOINT_UNNORMALIZE_FN(right_gripper_normalized)

            # 移动master arms
            robot_utils.move_arms(
                [master_bot_left, master_bot_right],
                [left_arm_pos, right_arm_pos],
                move_time=1.0
            )

            # 移动master grippers
            robot_utils.move_grippers(
                [master_bot_left, master_bot_right],
                [left_gripper_joint, right_gripper_joint],
                move_time=0.5
            )
                
            logging.info("master已移动到上次模型输出位置")
            
            # 步骤2: 等待s键按下
            logging.info("等待按下'b'键开始人机控制...")
            logging.info("（按'b'键开始，按'b'键退出并保存）")
            
            def _read_single_key() -> str:
                """Read a single keypress without requiring Enter."""
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setcbreak(fd)
                    key = sys.stdin.read(1)
                    if key == "\x03":
                        raise KeyboardInterrupt
                    return key
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

            # 等待b键开始（阻塞方式）
            key = _read_single_key()
            if key.lower() != 'b':
                logging.info("未收到'b'键，取消人机协作模式")
                self._is_waiting_for_task = True
                self._current_task = None
                return
            
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
                ts = real_env.step(action)
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
                    self._manual_dataset_dir,
                    compress_images=True,
                    is_mobile=False,
                    fps=self._manual_hz if self._manual_hz > 0 else None,
                    timestamps=timestamps,
                )
            
            # 设置等待状态
            self._is_waiting_for_task = True
            self._current_task = None
                
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
        finally:
            self._last_action = None
