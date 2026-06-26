import logging
import threading
import time
import json
import redis
import os
import sys
import termios
import tty
import select
from collections import deque

from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber

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

    _KEY_LEFT_ARROW = "\x1b[D"
    _KEY_RIGHT_ARROW = "\x1b[C"

    _TASK_PROMPT_BY_NUM = {
        "1": "Twist off the bottle cap",
        "2": "Rinse bottle",
    }

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
        good_bad_action: str = "normal",
    ) -> None:
        self._environment = environment
        self._agent = agent
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._manual_hz = manual_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = max_episode_steps
        self._good_bad_action = good_bad_action

        self._step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        self._manual_step_time = 1 / self._manual_hz if self._manual_hz > 0 else 0
        self._manual_dataset_dir = manual_dataset_dir or "/app/examples/aloha_real/manual_override"

        self._in_episode = False
        self._episode_steps = 0
        
        # Redis配置
        self._redis_host = os.getenv('REDIS_HOST', redis_host)
        self._redis_port = int(os.getenv('REDIS_PORT', redis_port))
        self._redis_db = int(os.getenv('REDIS_DB', redis_db))
        self._rlt_control_channel = os.getenv("RLT_CONTROL_CHANNEL", "aloha_rlt_control")
        self._rlt_state_channel = os.getenv("RLT_STATE_CHANNEL", "aloha_rlt_state")
        
        # Redis相关
        self._redis_client = None
        self._redis_thread = None
        self._redis_running = False
        self._latest_task = None
        self._preempt_task = None
        self._rlt_state = {
            "phase": "idle",
            "training_phase": "warmup",
            "warmup_target": int(os.getenv("RLT_DEFAULT_WARMUP_TARGET", "100")),
            "warmup_count": 0,
            "auto_rollout_count": 0,
            "actor_enabled": False,
            "actor_effective": False,
            "beta": float(os.getenv("RLT_DEFAULT_BETA", "10.0")),
            "intervention_scale": float(os.getenv("RLT_DEFAULT_INTERVENTION_SCALE", "0.25")),
            "max_delta": float(os.getenv("RLT_DEFAULT_MAX_DELTA", "0.1")),
            "actor_handoff_steps": int(os.getenv("RLT_DEFAULT_ACTOR_HANDOFF_STEPS", "4")),
            "actor_delta_ema_alpha": float(os.getenv("RLT_DEFAULT_ACTOR_DELTA_EMA_ALPHA", "0.35")),
            "critic_gate_enabled": os.getenv("RLT_DEFAULT_CRITIC_GATE_ENABLED", "1") in {"1", "true", "True"},
            "critic_gate_margin": float(os.getenv("RLT_DEFAULT_CRITIC_GATE_MARGIN", "0.0")),
            "critic_gate_temperature": float(os.getenv("RLT_DEFAULT_CRITIC_GATE_TEMPERATURE", "0.05")),
            "critic_ready": False,
            "inference_actor_active": False,
            "inference_delta_norm": None,
            "inference_gate_reason": None,
            "key_region_probability": None,
            "loaded_actor_step": None,
            "inference_reference_q_value": None,
            "inference_actor_q_value": None,
            "inference_q_advantage": None,
            "rl_token_checkpoint_path": os.getenv(
                "RLT_RL_TOKEN_CHECKPOINT_PATH",
                "/app/checkpoints/eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo_rl_token_query/rl_token_2048_enc4_dec4_query_from_19000_20260528/12000",
            ),
            "active_key_region_id": None,
            "last_reward": None,
        }
        self._task_lock = threading.Lock()
        
        # 任务状态管理
        self._current_task = None
        self._is_waiting_for_task = False
        self._standby_mode = "waiting"
        
        # 存储最近的 puppet action，用于遥操作体验模式对齐 leader/follower。
        self._last_action = None
        history_size = max(1, int((self._max_hz if self._max_hz > 0 else 1) * 10))
        self._recent_puppet_actions = deque(maxlen=history_size)

        # 退出标志
        self._stop = False
        self._manual_actor_requested = False
        self._rlt_context_epoch = 0
        self._keyboard_task_mapping = {
            "1": self._TASK_PROMPT_BY_NUM["1"],
            "2": self._TASK_PROMPT_BY_NUM["2"],
            "4": "Return to home position and save hdf5",
            "5": "Return to sleep position and keep robot runtime standby",
            "6": "Leader follower demo",
            "9": "Shutdown robot runtime",
        }
        self._model_task_nums = {"1", "2"}
        self._stop_task_nums = {"4", "5", "9"}

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
        pubsub.subscribe(self._rlt_control_channel)

        logging.info("开始监听Redis pub/sub频道: %s", self._rlt_control_channel)
        
        try:
            while self._redis_running:
                message = pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        self._handle_rlt_control_event(data)
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
        self._publish_rlt_state()

    def _rlt_actor_runtime_status(self) -> dict | None:
        status = getattr(self._agent, "rlt_actor_status", None)
        if status is None:
            return None
        try:
            value = status()
        except Exception as exc:
            logging.debug("读取 RLT actor runtime 状态失败: %s", exc)
            return {
                "actor_ready": False,
                "critic_ready": False,
                "actor_dir": None,
                "actor_step": None,
                "actor_load_error": str(exc),
            }
        return dict(value) if isinstance(value, dict) else None

    def _sync_rlt_actor_runtime_status_locked(self, status: dict | None) -> None:
        if not status:
            return
        actor_ready = bool(status.get("actor_ready"))
        critic_ready = bool(status.get("critic_ready"))
        actor_step = status.get("actor_step")
        actor_dir = status.get("actor_dir")
        load_error = status.get("actor_load_error")
        self._rlt_state["actor_runtime_checkpoint_path"] = actor_dir
        self._rlt_state["actor_runtime_checkpoint_step"] = actor_step
        if actor_ready:
            self._rlt_state["actor_ready"] = True
            self._rlt_state["critic_ready"] = critic_ready
            self._rlt_state["loaded_actor_step"] = actor_step
            if self._rlt_state.get("inference_actor_active") is not True and self._rlt_state.get("inference_gate_reason") in {
                None,
                "actor_runtime_not_configured",
                "actor_path_not_configured",
                "actor_not_loaded",
            }:
                self._rlt_state["inference_gate_reason"] = "waiting_for_inference"
        elif load_error:
            self._rlt_state["critic_ready"] = False
            self._rlt_state["actor_ready"] = False
            self._rlt_state["loaded_actor_step"] = None
            self._rlt_state["inference_actor_active"] = False
            self._rlt_state["inference_gate_reason"] = str(load_error)

    def _sync_manual_actor_gate_locked(self) -> None:
        if self._rlt_state.get("phase") == "key_region":
            self._manual_actor_requested = True
        in_warmup = int(self._rlt_state.get("warmup_count", 0)) < int(self._rlt_state.get("warmup_target", 0))
        actor_ready = bool(self._rlt_state.get("actor_ready", False))
        actor_effective = bool(self._manual_actor_requested and actor_ready and not in_warmup)
        self._rlt_state["actor_enabled"] = self._manual_actor_requested
        self._rlt_state["actor_effective"] = actor_effective
        if in_warmup:
            self._rlt_state["actor_locked_reason"] = "warmup"
        elif not self._manual_actor_requested:
            self._rlt_state["actor_locked_reason"] = "manual_arrow_disabled"
        elif not actor_ready:
            self._rlt_state["actor_locked_reason"] = "actor_not_ready"
        elif self._rlt_state.get("actor_locked_reason") in {"warmup", "manual_arrow_disabled", "actor_not_ready"}:
            self._rlt_state["actor_locked_reason"] = None

    def _bump_rlt_context_epoch_locked(self) -> int:
        self._rlt_context_epoch += 1
        return self._rlt_context_epoch

    def _flush_agent_action_cache(self, reason: str) -> None:
        flush = getattr(self._agent, "flush_action_cache", None)
        if flush is not None:
            flush(reason)

    def _set_agent_rlt_gate(self, *, enabled: bool, reason: str) -> None:
        logging.info("RLT broker gate set: enabled=%s epoch=%s reason=%s", enabled, self._rlt_context_epoch, reason)
        setter = getattr(self._agent, "set_rlt_gate", None)
        if setter is not None:
            setter(enabled=enabled, epoch=self._rlt_context_epoch, reason=reason)
        else:
            self._flush_agent_action_cache(reason)

    def _publish_rlt_state(self) -> None:
        if self._redis_client is None:
            return
        runtime_status = self._rlt_actor_runtime_status()
        with self._task_lock:
            self._sync_rlt_actor_runtime_status_locked(runtime_status)
            self._sync_manual_actor_gate_locked()
            payload = dict(self._rlt_state)
        payload["timestamp"] = time.time()
        try:
            self._redis_client.publish(self._rlt_state_channel, json.dumps(payload))
        except Exception as exc:
            logging.debug("发布 RLT 状态失败: %s", exc)

    def _handle_rlt_control_event(self, data: dict) -> None:
        event_type = data.get("type")
        state = data.get("state") or {}
        logging.info("收到 RLT 控制事件: %s", event_type)
        if event_type == "robot_task":
            task_data = self._normalize_task_data(
                {
                    "task_num": str(data.get("task_num")),
                    "task_name": data.get("task_name"),
                    "timestamp": data.get("timestamp", time.time()),
                }
            )
            should_preempt = str(task_data["task_num"]) in self._stop_task_nums
            with self._task_lock:
                self._latest_task = task_data
                if should_preempt:
                    self._preempt_task = task_data
            if should_preempt:
                self._flush_agent_action_cache(f"preempt_task_{task_data['task_num']}")
            logging.info("收到前端机器人任务: %s - %s", task_data["task_num"], task_data["task_name"])
            return
        should_notify_key_region = event_type in {"key_region_start", "key_region_end", "score", "key_region_discard"}
        with self._task_lock:
            previous_active_key_region_id = self._rlt_state.get("active_key_region_id")
            for key in (
                "warmup_target",
                "warmup_count",
                "warmup_success",
                "warmup_failure",
                "warmup_attempts",
                "warmup_invalid",
                "auto_rollout_count",
                "auto_rollout_attempts",
                "auto_rollout_invalid",
                "training_phase",
                "actor_enabled",
                "actor_effective",
                "actor_ready",
                "actor_locked_reason",
                "actor_checkpoint_path",
                "actor_checkpoint_step",
                "beta",
                "intervention_scale",
                "max_delta",
                "actor_handoff_steps",
                "actor_delta_ema_alpha",
                "critic_gate_enabled",
                "critic_gate_margin",
                "critic_gate_temperature",
                "critic_ready",
                "inference_actor_active",
                "inference_delta_norm",
                "inference_gate_reason",
                "key_region_probability",
                "loaded_actor_step",
                "inference_reference_q_value",
                "inference_actor_q_value",
                "inference_q_advantage",
            ):
                if key in state:
                    self._rlt_state[key] = state[key]
                elif key in data:
                    self._rlt_state[key] = data[key]
            if event_type == "key_region_start":
                self._bump_rlt_context_epoch_locked()
                self._manual_actor_requested = True
                self._rlt_state["phase"] = "key_region"
                self._rlt_state["active_key_region_id"] = data.get("key_region_id") or state.get("active_key_region_id")
                self._set_agent_rlt_gate(enabled=True, reason="key_region_start")
            elif event_type == "key_region_end":
                self._bump_rlt_context_epoch_locked()
                self._manual_actor_requested = False
                self._rlt_state["phase"] = "await_score"
                self._rlt_state["inference_actor_active"] = False
                self._rlt_state["inference_delta_norm"] = None
                self._rlt_state["inference_gate_reason"] = "actor_not_requested"
                self._set_agent_rlt_gate(enabled=False, reason="key_region_end")
            elif event_type == "score":
                reward = data.get("reward")
                self._rlt_state["phase"] = "idle"
                self._rlt_state["active_key_region_id"] = None
                self._rlt_state["last_reward"] = reward
                if "warmup_attempts" in state:
                    self._rlt_state["warmup_attempts"] = state["warmup_attempts"]
                if "auto_rollout_attempts" in state:
                    self._rlt_state["auto_rollout_attempts"] = state["auto_rollout_attempts"]
            elif event_type == "key_region_discard":
                self._bump_rlt_context_epoch_locked()
                self._rlt_state["phase"] = "idle"
                self._rlt_state["active_key_region_id"] = None
                self._rlt_state["inference_actor_active"] = False
                self._rlt_state["inference_delta_norm"] = None
                self._rlt_state["inference_gate_reason"] = "actor_not_requested"
                self._set_agent_rlt_gate(enabled=False, reason="key_region_discard")
            if event_type == "config_update":
                gate_keys = {
                    "actor_enabled",
                    "intervention_scale",
                    "max_delta",
                    "actor_handoff_steps",
                    "actor_delta_ema_alpha",
                    "critic_gate_enabled",
                    "critic_gate_margin",
                    "critic_gate_temperature",
                }
                if any(key in state or key in data for key in gate_keys):
                    self._bump_rlt_context_epoch_locked()
                    gate_enabled = bool(state.get("actor_enabled", data.get("actor_enabled", self._manual_actor_requested)))
                    self._set_agent_rlt_gate(enabled=gate_enabled, reason="config_update")
                if "actor_enabled" in state:
                    self._manual_actor_requested = bool(state["actor_enabled"])
                elif "actor_enabled" in data:
                    self._manual_actor_requested = bool(data["actor_enabled"])
            in_warmup = self._rlt_state["warmup_count"] < self._rlt_state["warmup_target"]
            if "training_phase" not in state:
                self._rlt_state["training_phase"] = "warmup" if in_warmup else "rl"
            self._sync_manual_actor_gate_locked()
            rlt_state_snapshot = dict(self._rlt_state)
            current_task_snapshot = dict(self._current_task) if self._current_task is not None else None
        self._publish_rlt_state()
        if should_notify_key_region:
            event_payload = dict(data)
            event_payload["type"] = event_type
            event_payload.setdefault("timestamp", time.time())
            if not event_payload.get("key_region_id"):
                event_payload["key_region_id"] = (
                    state.get("active_key_region_id")
                    or rlt_state_snapshot.get("active_key_region_id")
                    or previous_active_key_region_id
                )
            event_payload["state"] = rlt_state_snapshot
            if current_task_snapshot is not None:
                event_payload["current_task"] = current_task_snapshot
            self._notify_key_region_subscribers(event_type, event_payload)

    def _build_rlt_context(self) -> dict:
        with self._task_lock:
            state = dict(self._rlt_state)
            current_task = dict(self._current_task) if self._current_task is not None else None
            manual_actor_requested = self._manual_actor_requested
        in_warmup = int(state.get("warmup_count", 0)) < int(state.get("warmup_target", 0))
        actor_ready = bool(state.get("actor_ready", False))
        key_region_actor_requested = state.get("phase") == "key_region"
        gate_requested = bool(manual_actor_requested or key_region_actor_requested)
        actor_requested = bool(
            not in_warmup and (key_region_actor_requested or (manual_actor_requested and actor_ready))
        )
        actor_locked_reason = state.get("actor_locked_reason")
        if in_warmup:
            actor_locked_reason = "warmup"
        elif not gate_requested:
            actor_locked_reason = "manual_arrow_disabled"
        elif not actor_ready and not key_region_actor_requested:
            actor_locked_reason = "actor_not_ready"
        elif key_region_actor_requested:
            actor_locked_reason = None
        return {
            **state,
            "actor_requested": actor_requested,
            "actor_effective": actor_requested,
            "actor_locked_reason": actor_locked_reason,
            "manual_actor_requested": manual_actor_requested,
            "rlt_context_epoch": self._rlt_context_epoch,
            "current_task": current_task,
            "episode_step": self._episode_steps,
        }

    def _set_manual_actor_requested(self, enabled: bool) -> None:
        with self._task_lock:
            self._bump_rlt_context_epoch_locked()
            self._manual_actor_requested = enabled
            self._sync_manual_actor_gate_locked()
            self._set_agent_rlt_gate(
                enabled=enabled,
                reason="manual_actor_enabled" if enabled else "manual_actor_disabled",
            )
            if enabled and self._rlt_state.get("actor_locked_reason") == "warmup":
                self._rlt_state["actor_locked_reason"] = "warmup"
                self._rlt_state["inference_gate_reason"] = "warmup"
            elif enabled:
                self._rlt_state["inference_gate_reason"] = "manual_arrow_enabled"
            else:
                self._rlt_state["inference_actor_active"] = False
                self._rlt_state["inference_delta_norm"] = None
                self._rlt_state["inference_gate_reason"] = "manual_arrow_disabled"
        logging.info("RLT actor %s（%s）", "已接入" if enabled else "已停止", "左箭头" if enabled else "右箭头")
        self._publish_rlt_state()

    def _handle_policy_control_key(self, key: str | None) -> bool:
        if key == self._KEY_LEFT_ARROW:
            self._set_manual_actor_requested(True)
            return True
        if key == self._KEY_RIGHT_ARROW:
            self._set_manual_actor_requested(False)
            return True
        return False

    def _update_rlt_actor_status_from_action(self, action: dict) -> None:
        if not isinstance(action, dict) or "rlt_actor_applied" not in action:
            return
        with self._task_lock:
            action_epoch = action.get("rlt_context_epoch")
            if action_epoch is not None and int(action_epoch) != int(self._rlt_context_epoch):
                self._rlt_state["inference_actor_active"] = False
                self._rlt_state["inference_delta_norm"] = None
                self._rlt_state["inference_gate_reason"] = "stale_actor_context"
                self._sync_manual_actor_gate_locked()
                return
            self._rlt_state["actor_runtime_applied"] = bool(action.get("rlt_actor_applied"))
            self._rlt_state["actor_runtime_reason"] = action.get("rlt_actor_reason")
            self._rlt_state["actor_runtime_checkpoint_step"] = action.get("rlt_actor_step")
            self._rlt_state["actor_runtime_checkpoint_path"] = action.get("rlt_actor_dir")
            self._rlt_state["actor_last_delta_norm"] = action.get("rlt_actor_delta_norm")
            self._rlt_state["inference_actor_active"] = bool(action.get("rlt_actor_applied"))
            self._rlt_state["inference_delta_norm"] = action.get("rlt_actor_delta_norm")
            self._rlt_state["inference_gate_reason"] = action.get("rlt_gate_reason") or action.get("rlt_actor_reason")
            self._rlt_state["key_region_probability"] = action.get("rlt_key_region_probability")
            self._rlt_state["loaded_actor_step"] = action.get("rlt_actor_step")
            self._rlt_state["inference_reference_q_value"] = action.get("rlt_reference_q")
            self._rlt_state["inference_actor_q_value"] = action.get("rlt_actor_q")
            self._rlt_state["inference_q_advantage"] = action.get("rlt_q_advantage")
            if "rlt_critic_ready" in action:
                self._rlt_state["critic_ready"] = bool(action.get("rlt_critic_ready"))
            if "rlt_critic_gate_enabled" in action:
                self._rlt_state["critic_gate_enabled"] = bool(action.get("rlt_critic_gate_enabled"))
            self._sync_manual_actor_gate_locked()

    def _notify_key_region_subscribers(self, event_type: str, event: dict) -> None:
        hook_name_by_type = {
            "key_region_start": "on_key_region_start",
            "key_region_end": "on_key_region_end",
            "score": "on_key_region_score",
            "key_region_discard": "on_key_region_discard",
        }
        hook_name = hook_name_by_type.get(event_type)
        if hook_name is None:
            return
        for subscriber in self._subscribers:
            hook = getattr(subscriber, hook_name, None)
            if hook is None:
                continue
            try:
                hook(event)
            except Exception as exc:
                logging.exception("Subscriber %s failed handling %s: %s", subscriber, event_type, exc)

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

    def _take_preempt_task(self):
        """Consume a pending stop/home/sleep/shutdown task.

        This lets the policy loop drop any just-computed action before it can
        reach the robot when a stop-class command arrives during inference.
        """
        with self._task_lock:
            if self._preempt_task is None:
                return None
            task = self._preempt_task
            self._preempt_task = None
            latest_num = str(self._latest_task.get("task_num")) if self._latest_task is not None else None
            if latest_num == str(task.get("task_num")):
                self._latest_task = None
            return task

    def _clear_preempt_task(self, task_num: str | None = None) -> None:
        with self._task_lock:
            if self._preempt_task is None:
                return
            if task_num is None or str(self._preempt_task.get("task_num")) == str(task_num):
                self._preempt_task = None

    def _normalize_task_data(self, task_data):
        """Canonicalize task prompts before they are shown or sent to the policy."""
        if task_data is None:
            return None
        normalized = dict(task_data)
        task_num = str(normalized.get("task_num"))
        if task_num in self._TASK_PROMPT_BY_NUM:
            normalized["task_name"] = self._TASK_PROMPT_BY_NUM[task_num]
        return normalized
    
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
            self._close_subscribers()

    def mark_episode_complete(self) -> None:
        """Marks the end of an episode."""
        self._in_episode = False

    def stop(self) -> None:
        """Request the runtime loop to stop."""
        self._stop = True

    def _close_subscribers(self) -> None:
        for subscriber in self._subscribers:
            close = getattr(subscriber, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception as exc:
                logging.exception("Subscriber %s failed during close: %s", subscriber, exc)

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
        self._standby_mode = "waiting"
        self._publish_runtime_state(mode="waiting")
        last_standby_state_publish = time.time()
        fd = None
        old_settings = None
        if sys.stdin.isatty():
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            logging.info(
                "键盘快捷键已启用：1 拧瓶盖任务，2 冲洗瓶子，4 回 home 并保存，5 回 sleep 待机，6 遥操作体验，9 关闭 runtime"
            )
        else:
            logging.warning("stdin 不是 TTY，主循环中无法监听键盘快捷键")

        try:
            while not self._stop:
                task_data = self._poll_task_from_inputs()
                if task_data:
                    self._handle_task(task_data)
                
                if self._is_waiting_for_task:
                    # 等待状态下持续监听键盘/Redis，并定期发布 runtime 心跳供后端判断监听线程是否活着。
                    now = time.time()
                    if now - last_standby_state_publish >= 1.0:
                        self._publish_runtime_state(mode=self._standby_mode)
                        last_standby_state_publish = now
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
        task_data = self._normalize_task_data(task_data)
        logging.info("收到键盘任务: %s - %s", key, task_data["task_name"])
        return task_data

    def _poll_task_from_inputs(
        self,
        *,
        allowed_task_nums: set[str] | None = None,
        keyboard_timeout: float = 0.0,
    ):
        """统一轮询键盘和 Redis 任务输入。"""
        key = self._poll_single_key(timeout=keyboard_timeout)
        task_data = self._build_task_from_key(
            key,
            allowed_task_nums=allowed_task_nums,
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
        self._clear_preempt_task(str(task_num))
        
        logging.info(f"处理语音任务: {task_num} - {task_name}")
        
        if task_num in self._model_task_nums:
            logging.info(f"开始执行任务: {task_name}")
            for subscriber in self._subscribers:
                subscriber.on_episode_start()
            # 设置当前任务
            self._current_task = task_data
            self._is_waiting_for_task = False 
            self._standby_mode = "policy"
            self._publish_runtime_state(mode="policy")
        elif task_num == "6":
            logging.info("收到遥操作体验指令，进入leader-follower演示模式")
            self._current_task = task_data
            self._is_waiting_for_task = False
            self._standby_mode = "leader_follower_prepare"
            self._agent.reset()
            if self._last_action is not None:
                for subscriber in self._subscribers:
                    subscriber.on_episode_end()
            self._publish_runtime_state(mode="leader_follower_prepare")
            self._handle_leader_follower_demo_mode()
        elif task_num == "4":
            logging.info("收到停止指令，回到初始位置并停止agent")
            # 设置等待状态
            self._is_waiting_for_task = True
            self._current_task = None
            self._standby_mode = "waiting"
            # 回到初始位置
            self._environment.stop()
            # 停止agent
            self._agent.reset()   
            # 通知subscriber episode结束
            for subscriber in self._subscribers:
                subscriber.on_episode_end()   
            self._publish_runtime_state(mode="waiting")
        elif task_num == "5":
            logging.info("收到 sleep 指令，回到sleep位置并保持待机")
            self._is_waiting_for_task = True
            self._current_task = None
            self._standby_mode = "sleep"
            self._environment.sleep_arms()
            self._agent.reset()
            for subscriber in self._subscribers:
                subscriber.on_episode_end()
            self._publish_runtime_state(mode="sleep")
        elif task_num == "9":
            logging.info("收到 shutdown 指令，停止 robot runtime")
            self._is_waiting_for_task = True
            self._current_task = None
            self._standby_mode = "shutdown"
            self._agent.reset()
            for subscriber in self._subscribers:
                subscriber.on_episode_end()
            self._publish_runtime_state(mode="shutdown")
            self._stop = True
        else:
            logging.warning(f"未知任务编号: {task_num}")

    def _step(self) -> None:
        """A single step of the runtime loop."""
        self._handle_policy_control_key(self._poll_single_key(timeout=0.0))
        preempt_task = self._take_preempt_task()
        if preempt_task is not None:
            logging.warning("策略 step 开始前收到抢占任务，跳过本次动作: %s", preempt_task.get("task_num"))
            self._handle_task(preempt_task)
            return
        observation = self._environment.get_observation()
        assert self._current_task is not None, "_current_task must be set before calling _step()"
        preempt_task = self._take_preempt_task()
        if preempt_task is not None:
            logging.warning("获取 observation 后收到抢占任务，跳过策略推理: %s", preempt_task.get("task_num"))
            self._handle_task(preempt_task)
            return
        observation_with_task = {
            **observation,
            'prompt': self._current_task.get('task_name'),
            'subtask': {'good_bad_action': self._good_bad_action},
            'rlt_context': self._build_rlt_context(),
        }

        action = self._agent.get_action(observation_with_task)
        preempt_task = self._take_preempt_task()
        if preempt_task is not None:
            logging.warning("策略推理期间收到抢占任务，丢弃本次策略动作: %s", preempt_task.get("task_num"))
            self._handle_task(preempt_task)
            return
        self._update_rlt_actor_status_from_action(action)
        self._environment.apply_action(action)
        # 存储最近 action，用于 6 号 leader/follower demo 对齐。
        self._last_action = action.get("actions") if isinstance(action, dict) and "actions" in action else None
        if self._last_action is not None:
            self._recent_puppet_actions.append(list(self._last_action))
        self._publish_runtime_state(latest_action=self._last_action, mode="policy")

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

        master_left_arm_positions = robot_utils.clip_arm_joint_positions(
            left_arm_pos,
            master_bot_left.arm.group_info.joint_lower_limits,
            master_bot_left.arm.group_info.joint_upper_limits,
            continuous_roll_joints=True,
        )
        master_right_arm_positions = robot_utils.clip_arm_joint_positions(
            right_arm_pos,
            master_bot_right.arm.group_info.joint_lower_limits,
            master_bot_right.arm.group_info.joint_upper_limits,
            continuous_roll_joints=True,
        )
        robot_utils.publish_arm_positions(master_bot_left, master_left_arm_positions)
        robot_utils.publish_arm_positions(master_bot_right, master_right_arm_positions)
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
            continuous_roll_joints=True,
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

    def _handle_leader_follower_demo_mode(self) -> None:
        """Task 6: simple customer-facing leader-follower demo without hdf5 saving."""
        switched_task = False
        try:
            from examples.aloha_real import constants
            from examples.aloha_real import robot_utils
            from examples.aloha_real.real_env import get_action

            if not hasattr(self._environment, "_env"):
                logging.error("无法访问real_env，跳过遥操作体验模式")
                return

            real_env = self._environment._env
            master_bot_left = real_env.master_bot_left
            master_bot_right = real_env.master_bot_right
            puppet_bot_left = real_env.puppet_bot_left
            puppet_bot_right = real_env.puppet_bot_right
            joint_unwrapper = robot_utils.JointPositionUnwrapper()

            reset_position = getattr(real_env, "_reset_position", None)
            if reset_position is None:
                reset_position = [constants.START_ARM_POSE[:6], constants.START_ARM_POSE[8:14]]

            master_gripper_mid = constants.MASTER_GRIPPER_JOINT_MID
            puppet_gripper_mid = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(0.5)

            logging.info("遥操作体验：leader/follower移动到初始位置")
            robot_utils.torque_on(master_bot_left)
            robot_utils.torque_on(master_bot_right)
            robot_utils.torque_on(puppet_bot_left)
            robot_utils.torque_on(puppet_bot_right)
            robot_utils.move_arms(
                [puppet_bot_left, puppet_bot_right],
                reset_position,
                move_time=1,
                continuous_roll_joints=True,
            )
            robot_utils.move_arms(
                [master_bot_left, master_bot_right],
                reset_position,
                move_time=1,
                continuous_roll_joints=True,
            )
            robot_utils.move_grippers(
                [puppet_bot_left, puppet_bot_right],
                [puppet_gripper_mid, puppet_gripper_mid],
                move_time=0.5,
            )
            robot_utils.move_grippers(
                [master_bot_left, master_bot_right],
                [master_gripper_mid, master_gripper_mid],
                move_time=0.5,
            )
            master_bot_left.dxl.robot_torque_enable("single", "gripper", False)
            master_bot_right.dxl.robot_torque_enable("single", "gripper", False)
            self._publish_runtime_state(mode="leader_follower_ready")

            logging.info("遥操作体验：leader爪子力矩已关闭，等待用户闭合任意leader爪子以释放leader手臂力矩")
            trigger_threshold = 0.25
            while not self._stop:
                latest_task = self._take_latest_task(
                    allowed_task_nums=self._model_task_nums | self._stop_task_nums
                )
                if latest_task:
                    logging.info("遥操作体验准备阶段收到Redis任务 %s，切换任务", latest_task["task_num"])
                    switched_task = True
                    self._handle_task(latest_task)
                    return

                key_task = self._build_task_from_key(
                    self._poll_single_key(timeout=0.02),
                    allowed_task_nums=self._model_task_nums | self._stop_task_nums,
                    log_invalid=False,
                )
                if key_task is not None:
                    logging.info("遥操作体验准备阶段收到键盘任务 %s，切换任务", key_task["task_num"])
                    switched_task = True
                    self._handle_task(key_task)
                    return

                left_gripper = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(
                    master_bot_left.dxl.joint_states.position[6]
                )
                right_gripper = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(
                    master_bot_right.dxl.joint_states.position[6]
                )
                if left_gripper <= trigger_threshold or right_gripper <= trigger_threshold:
                    break
                time.sleep(0.02)

            if self._stop:
                return

            logging.info("遥操作体验：检测到leader爪子闭合，关闭leader torque并开始跟随")
            robot_utils.torque_off(master_bot_left)
            robot_utils.torque_off(master_bot_right)
            self._publish_runtime_state(mode="leader_follower")

            while not self._stop:
                t0 = time.time()

                latest_task = self._take_latest_task(
                    allowed_task_nums=self._model_task_nums | self._stop_task_nums
                )
                if latest_task:
                    logging.info("遥操作体验中收到Redis任务 %s，退出并切换任务", latest_task["task_num"])
                    switched_task = True
                    self._handle_task(latest_task)
                    return

                key = self._poll_single_key(timeout=0.0)
                if key and key.lower() == "b":
                    logging.info("遥操作体验收到'b'，退出到等待状态")
                    break
                key_task = self._build_task_from_key(
                    key,
                    allowed_task_nums=self._model_task_nums | self._stop_task_nums,
                    log_invalid=False,
                )
                if key_task is not None:
                    logging.info("遥操作体验中收到键盘任务 %s，退出并切换任务", key_task["task_num"])
                    switched_task = True
                    self._handle_task(key_task)
                    return

                action = get_action(
                    master_bot_left,
                    master_bot_right,
                    joint_unwrapper=joint_unwrapper,
                    use_continuous_joints=True,
                )
                self._environment.apply_action({"actions": action})
                ts = self._environment._ts
                self._publish_runtime_state(qpos=ts.observation.get("qpos"), latest_action=action, mode="leader_follower")
                time.sleep(max(0, self._manual_step_time - (time.time() - t0)))

            self._is_waiting_for_task = True
            self._current_task = None
            self._publish_runtime_state(mode="waiting")

        except Exception as e:
            logging.error("遥操作体验模式出错: %s", e, exc_info=True)
            self._is_waiting_for_task = True
            self._current_task = None
        finally:
            try:
                from examples.aloha_real import robot_utils
                if hasattr(self._environment, "_env"):
                    real_env = self._environment._env
                    robot_utils.torque_on(real_env.master_bot_left)
                    robot_utils.torque_on(real_env.master_bot_right)
            except Exception:
                pass
            if not switched_task:
                self._publish_runtime_state(mode="waiting")
