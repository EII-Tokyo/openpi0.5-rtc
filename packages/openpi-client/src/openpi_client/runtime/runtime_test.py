
import sys
import types

sys.modules.setdefault("h5py", types.SimpleNamespace(File=None))

from openpi_client.runtime.runtime import Runtime


class _Env:
    def __init__(self):
        self.actions = []

    def get_observation(self):
        return {"origin_observation": {"camera": []}, "qpos": [0.0]}

    def apply_action(self, action):
        self.actions.append(action)


class _Agent:
    def __init__(self):
        self.obs = []

    def get_action(self, obs):
        self.obs.append(obs)
        return {"actions": [1.0]}

    def reset(self):
        pass


def _runtime():
    return Runtime(_Env(), _Agent(), [], max_hz=0, num_episodes=1, max_episode_steps=1)


def test_build_rlt_context_locks_actor_during_warmup():
    runtime = _runtime()
    runtime._rlt_state.update({"warmup_target": 2, "warmup_count": 1, "actor_enabled": True})

    context = runtime._build_rlt_context()

    assert context["actor_requested"] is False
    assert context["actor_locked_reason"] == "warmup"


def test_build_rlt_context_requests_actor_after_warmup_when_enabled_and_ready():
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

    assert context["actor_requested"] is True
    assert context["episode_step"] == 0


def test_step_passes_rlt_context_to_agent():
    runtime = _runtime()
    runtime._current_task = {"task_name": "Twist off the bottle cap"}
    runtime._rlt_state.update({"warmup_target": 1, "warmup_count": 1, "actor_enabled": True, "actor_effective": True})

    runtime._step()

    obs = runtime._agent.obs[0]
    assert obs["prompt"] == "Twist off the bottle cap"
    assert obs["rlt_context"]["actor_requested"] is True
    assert obs["rlt_context"]["current_task"] == {"task_name": "Twist off the bottle cap"}



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


def test_update_rlt_actor_status_records_inference_metrics():
    runtime = _runtime()
    runtime._update_rlt_actor_status_from_action(
        {
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
