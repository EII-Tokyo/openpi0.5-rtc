
import sys
import types

sys.modules.setdefault("h5py", types.SimpleNamespace(File=None))

from openpi_client.runtime.runtime import Runtime


class _Env:
    def __init__(self):
        self.actions = []
        self.stop_count = 0
        self.sleep_count = 0

    def get_observation(self):
        return {"origin_observation": {"camera": []}, "qpos": [0.0]}

    def apply_action(self, action):
        self.actions.append(action)

    def stop(self):
        self.stop_count += 1
        return None

    def sleep_arms(self):
        self.sleep_count += 1
        return None


class _Agent:
    def __init__(self):
        self.obs = []
        self.flush_reasons = []
        self.gate_events = []
        self.reset_count = 0

    def get_action(self, obs):
        self.obs.append(obs)
        return {"actions": [1.0]}

    def flush_action_cache(self, reason=None):
        self.flush_reasons.append(reason)

    def set_rlt_gate(self, *, enabled, epoch, reason=None):
        self.gate_events.append({"enabled": enabled, "epoch": epoch, "reason": reason})
        self.flush_action_cache(reason)

    def reset(self):
        self.reset_count += 1


class _PreemptingAgent(_Agent):
    def __init__(self, runtime, task_num="4", task_name="home"):
        super().__init__()
        self._runtime = runtime
        self._task_num = task_num
        self._task_name = task_name

    def get_action(self, obs):
        self.obs.append(obs)
        self._runtime._handle_rlt_control_event(
            {
                "type": "robot_task",
                "task_num": self._task_num,
                "task_name": self._task_name,
            }
        )
        return {"actions": [99.0]}


class _RLTTransitioningAgent(_Agent):
    def __init__(self, runtime):
        super().__init__()
        self._runtime = runtime

    def get_action(self, obs):
        self.obs.append(obs)
        self._runtime._handle_rlt_control_event(
            {
                "type": "key_region_start",
                "key_region_id": "kr-transition",
                "state": {
                    "warmup_target": 1,
                    "warmup_count": 1,
                    "actor_ready": True,
                },
            }
        )
        return {"actions": [77.0], "rlt_context_epoch": obs["rlt_context"]["rlt_context_epoch"]}


class _Subscriber:
    def __init__(self):
        self.episode_starts = 0
        self.episode_ends = 0

    def on_episode_start(self):
        self.episode_starts += 1

    def on_episode_end(self):
        self.episode_ends += 1


def _runtime():
    return Runtime(_Env(), _Agent(), [], max_hz=0, num_episodes=1, max_episode_steps=1)


def test_build_rlt_context_locks_actor_during_warmup():
    runtime = _runtime()
    runtime._rlt_state.update({"warmup_target": 2, "warmup_count": 1, "actor_enabled": True})

    context = runtime._build_rlt_context()

    assert context["actor_requested"] is False
    assert context["actor_locked_reason"] == "warmup"


def test_build_rlt_context_does_not_request_actor_until_left_arrow():
    runtime = _runtime()
    runtime._rlt_state.update(
        {
            "warmup_target": 2,
            "warmup_count": 2,
            "actor_enabled": True,
            "actor_effective": True,
            "actor_ready": True,
        }
    )

    context = runtime._build_rlt_context()

    assert context["actor_requested"] is False
    assert context["actor_effective"] is False
    assert context["manual_actor_requested"] is False
    assert context["actor_locked_reason"] == "manual_arrow_disabled"
    assert context["episode_step"] == 0


def test_policy_arrow_keys_toggle_actor_request():
    runtime = _runtime()
    runtime._rlt_state.update(
        {
            "warmup_target": 2,
            "warmup_count": 2,
            "actor_ready": True,
        }
    )

    assert runtime._handle_policy_control_key(Runtime._KEY_LEFT_ARROW) is True
    enabled_context = runtime._build_rlt_context()
    assert enabled_context["manual_actor_requested"] is True
    assert enabled_context["actor_requested"] is True
    assert enabled_context["actor_effective"] is True

    assert runtime._handle_policy_control_key(Runtime._KEY_RIGHT_ARROW) is True
    disabled_context = runtime._build_rlt_context()
    assert disabled_context["manual_actor_requested"] is False
    assert disabled_context["actor_requested"] is False
    assert disabled_context["actor_effective"] is False
    assert disabled_context["actor_locked_reason"] == "manual_arrow_disabled"


def test_config_update_actor_enabled_toggles_manual_actor_request():
    runtime = _runtime()
    runtime._rlt_state.update(
        {
            "warmup_target": 2,
            "warmup_count": 2,
            "actor_ready": True,
        }
    )

    runtime._handle_rlt_control_event({"type": "config_update", "state": {"actor_enabled": True}})
    enabled_context = runtime._build_rlt_context()
    assert enabled_context["manual_actor_requested"] is True
    assert enabled_context["actor_requested"] is True

    runtime._handle_rlt_control_event({"type": "config_update", "state": {"actor_enabled": False}})
    disabled_context = runtime._build_rlt_context()
    assert disabled_context["manual_actor_requested"] is False
    assert disabled_context["actor_requested"] is False


def test_key_region_events_toggle_manual_actor_request():
    runtime = _runtime()
    runtime._rlt_state.update(
        {
            "warmup_target": 2,
            "warmup_count": 2,
            "actor_ready": True,
        }
    )

    runtime._handle_rlt_control_event({"type": "key_region_start", "key_region_id": "kr-1"})
    enabled_context = runtime._build_rlt_context()
    assert enabled_context["phase"] == "key_region"
    assert enabled_context["manual_actor_requested"] is True
    assert enabled_context["actor_requested"] is True

    runtime._handle_rlt_control_event({"type": "key_region_end", "key_region_id": "kr-1"})
    disabled_context = runtime._build_rlt_context()
    assert disabled_context["phase"] == "await_score"
    assert disabled_context["manual_actor_requested"] is False
    assert disabled_context["actor_requested"] is False


def test_key_region_events_bump_context_epoch_and_flush_action_cache():
    runtime = _runtime()
    runtime._rlt_state.update({"warmup_target": 2, "warmup_count": 2, "actor_ready": True})
    initial_epoch = runtime._build_rlt_context()["rlt_context_epoch"]

    runtime._handle_rlt_control_event({"type": "key_region_start", "key_region_id": "kr-1"})
    start_epoch = runtime._build_rlt_context()["rlt_context_epoch"]

    runtime._handle_rlt_control_event({"type": "key_region_end", "key_region_id": "kr-1"})
    end_epoch = runtime._build_rlt_context()["rlt_context_epoch"]

    assert start_epoch == initial_epoch + 1
    assert end_epoch == start_epoch + 1
    assert runtime._agent.flush_reasons == ["key_region_start", "key_region_end"]
    assert runtime._agent.gate_events == [
        {"enabled": True, "epoch": start_epoch, "reason": "key_region_start"},
        {"enabled": False, "epoch": end_epoch, "reason": "key_region_end"},
    ]


def test_key_region_phase_requests_actor_even_if_manual_flag_is_stale():
    runtime = _runtime()
    runtime._rlt_state.update(
        {
            "phase": "key_region",
            "warmup_target": 2,
            "warmup_count": 2,
            "actor_ready": True,
            "actor_enabled": False,
            "actor_effective": False,
        }
    )
    runtime._manual_actor_requested = False

    context = runtime._build_rlt_context()

    assert context["actor_requested"] is True
    assert context["actor_effective"] is True


def test_step_passes_rlt_context_to_agent():
    runtime = _runtime()
    runtime._current_task = {"task_name": "Twist off the bottle cap"}
    runtime._rlt_state.update({"warmup_target": 1, "warmup_count": 1, "actor_ready": True})
    runtime._handle_policy_control_key(Runtime._KEY_LEFT_ARROW)

    runtime._step()

    obs = runtime._agent.obs[0]
    assert obs["prompt"] == "Twist off the bottle cap"
    assert obs["rlt_context"]["actor_requested"] is True
    assert obs["rlt_context"]["current_task"] == {"task_name": "Twist off the bottle cap"}


def test_stop_task_during_get_action_preempts_policy_action_before_apply():
    env = _Env()
    runtime = Runtime(env, _Agent(), [], max_hz=0, num_episodes=1, max_episode_steps=1)
    agent = _PreemptingAgent(runtime, task_num="4", task_name="home")
    runtime._agent = agent
    runtime._current_task = {"task_num": "1", "task_name": "Twist off the bottle cap"}
    runtime._is_waiting_for_task = False

    runtime._step()

    assert env.actions == []
    assert env.stop_count == 1
    assert runtime._current_task is None
    assert runtime._is_waiting_for_task is True
    assert agent.flush_reasons == ["preempt_task_4"]
    assert agent.reset_count == 1


def test_rlt_epoch_change_during_get_action_discards_stale_policy_action_before_apply():
    env = _Env()
    runtime = Runtime(env, _Agent(), [], max_hz=0, num_episodes=1, max_episode_steps=1)
    agent = _RLTTransitioningAgent(runtime)
    runtime._agent = agent
    runtime._current_task = {"task_num": "1", "task_name": "Twist off the bottle cap"}
    runtime._is_waiting_for_task = False
    runtime._rlt_state.update({"warmup_target": 1, "warmup_count": 1, "actor_ready": True})

    runtime._step()

    assert env.actions == []
    assert runtime._rlt_state["phase"] == "key_region"
    assert runtime._rlt_state["active_key_region_id"] == "kr-transition"
    assert agent.flush_reasons == ["key_region_start"]


def test_sleep_task_moves_to_sleep_and_keeps_runtime_listening():
    env = _Env()
    agent = _Agent()
    subscriber = _Subscriber()
    runtime = Runtime(env, agent, [subscriber], max_hz=0, num_episodes=1, max_episode_steps=1)
    runtime._current_task = {"task_num": "1", "task_name": "Twist off the bottle cap"}
    runtime._is_waiting_for_task = False

    runtime._handle_task({"task_num": "5", "task_name": "sleep"})

    assert env.sleep_count == 1
    assert env.stop_count == 0
    assert agent.reset_count == 1
    assert subscriber.episode_ends == 1
    assert runtime._current_task is None
    assert runtime._is_waiting_for_task is True
    assert runtime._stop is False


def test_shutdown_task_stops_runtime_without_reusing_sleep():
    env = _Env()
    agent = _Agent()
    subscriber = _Subscriber()
    runtime = Runtime(env, agent, [subscriber], max_hz=0, num_episodes=1, max_episode_steps=1)

    runtime._handle_task({"task_num": "9", "task_name": "shutdown"})

    assert env.sleep_count == 0
    assert agent.reset_count == 1
    assert subscriber.episode_ends == 1
    assert runtime._is_waiting_for_task is True
    assert runtime._stop is True



def test_control_event_preserves_backend_actor_effective_gate():
    runtime = _runtime()
    runtime._handle_rlt_control_event(
        {
            "type": "config_update",
            "state": {
                "warmup_target": 1,
                "warmup_count": 1,
                "actor_enabled": True,
                "actor_effective": False,
                "actor_ready": False,
                "actor_locked_reason": "actor_not_ready",
            },
        }
    )

    context = runtime._build_rlt_context()
    assert context["actor_effective"] is False
    assert context["actor_requested"] is False
    assert context["actor_locked_reason"] == "actor_not_ready"


def test_build_rlt_context_honors_backend_actor_effective_gate():
    runtime = _runtime()
    runtime._handle_rlt_control_event(
        {
            "type": "key_region_start",
            "state": {
                "warmup_target": 1,
                "warmup_count": 1,
                "actor_enabled": True,
                "actor_effective": True,
                "actor_ready": False,
                "actor_locked_reason": None,
            },
            "key_region_id": "kr-1",
        }
    )

    context = runtime._build_rlt_context()
    assert context["actor_requested"] is True
    assert context["actor_effective"] is True
    assert context["actor_locked_reason"] is None


def test_control_event_passes_critic_gate_config_to_context():
    runtime = _runtime()
    runtime._handle_rlt_control_event(
        {
            "type": "config_update",
            "state": {
                "critic_gate_enabled": True,
                "critic_gate_margin": 0.15,
                "critic_gate_temperature": 0.2,
            },
        }
    )

    context = runtime._build_rlt_context()
    assert context["critic_gate_enabled"] is True
    assert context["critic_gate_margin"] == 0.15
    assert context["critic_gate_temperature"] == 0.2


def test_control_event_passes_projected_blend_config_to_context():
    runtime = _runtime()
    runtime._handle_rlt_control_event(
        {
            "type": "config_update",
            "state": {
                "rlt_blend_mode": "projected_slow_push",
                "rlt_blend_preset": "align",
                "lambda_push": 0.2,
                "lambda_vla_align": 0.3,
                "lambda_actor": 0.5,
                "push_joint_indices": [0, 1, 2, 3, 4, 5],
                "push_axis": [-0.53, 0.2, -0.78, 0.23, -0.08, 0.06],
            },
        }
    )

    context = runtime._build_rlt_context()
    assert context["rlt_blend_mode"] == "projected_slow_push"
    assert context["rlt_blend_preset"] == "align"
    assert context["lambda_push"] == 0.2
    assert context["lambda_vla_align"] == 0.3
    assert context["lambda_actor"] == 0.5
    assert context["push_joint_indices"] == [0, 1, 2, 3, 4, 5]
    assert context["push_axis"] == [-0.53, 0.2, -0.78, 0.23, -0.08, 0.06]


def test_update_rlt_actor_status_records_inference_metrics():
    runtime = _runtime()
    epoch = runtime._build_rlt_context()["rlt_context_epoch"]
    runtime._update_rlt_actor_status_from_action(
        {
            "rlt_context_epoch": epoch,
            "rlt_actor_applied": True,
            "rlt_actor_reason": None,
            "rlt_actor_step": 32,
            "rlt_actor_dir": "/tmp/actor",
            "rlt_actor_delta_norm": 0.03,
            "rlt_reference_q": 0.2,
            "rlt_actor_q": 0.7,
            "rlt_q_advantage": 0.5,
            "rlt_key_region_probability": 0.9,
            "rlt_gate_reason": "critic_gate_actor_active",
            "rlt_critic_ready": True,
            "rlt_critic_gate_enabled": True,
        }
    )

    assert runtime._rlt_state["inference_actor_active"] is True
    assert runtime._rlt_state["inference_delta_norm"] == 0.03
    assert runtime._rlt_state["loaded_actor_step"] == 32
    assert runtime._rlt_state["inference_reference_q_value"] == 0.2
    assert runtime._rlt_state["inference_actor_q_value"] == 0.7
    assert runtime._rlt_state["inference_q_advantage"] == 0.5
    assert runtime._rlt_state["key_region_probability"] == 0.9
    assert runtime._rlt_state["inference_gate_reason"] == "critic_gate_actor_active"
    assert runtime._rlt_state["critic_ready"] is True


def test_update_rlt_actor_status_ignores_stale_context_epoch():
    runtime = _runtime()
    runtime._rlt_state["inference_actor_active"] = False
    runtime._rlt_state["inference_gate_reason"] = "actor_not_requested"
    runtime._bump_rlt_context_epoch_locked()

    runtime._update_rlt_actor_status_from_action(
        {
            "rlt_context_epoch": 0,
            "rlt_actor_applied": True,
            "rlt_actor_reason": None,
            "rlt_actor_delta_norm": 0.08,
            "rlt_gate_reason": None,
        }
    )

    assert runtime._rlt_state["inference_actor_active"] is False
    assert runtime._rlt_state["inference_delta_norm"] is None
    assert runtime._rlt_state["inference_gate_reason"] == "stale_actor_context"


class _StatusAgent(_Agent):
    def rlt_actor_status(self):
        return {
            "actor_ready": True,
            "critic_ready": True,
            "actor_dir": "/tmp/actor",
            "actor_step": 15000,
            "actor_load_error": None,
        }


def test_publish_rlt_state_includes_loaded_actor_runtime_status():
    runtime = Runtime(_Env(), _StatusAgent(), [], max_hz=0, num_episodes=1, max_episode_steps=1)
    runtime._redis_client = type("Redis", (), {"publish": lambda self, channel, payload: None})()

    runtime._publish_rlt_state()

    assert runtime._rlt_state["actor_ready"] is True
    assert runtime._rlt_state["critic_ready"] is True
    assert runtime._rlt_state["loaded_actor_step"] == 15000
    assert runtime._rlt_state["inference_gate_reason"] == "waiting_for_inference"
