
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
